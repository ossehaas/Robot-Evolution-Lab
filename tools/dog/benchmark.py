"""One dog, one test scenario: the standard evaluation (every dog one episode with real walking commands, spread over
all difficulty rows of the course) under changed conditions, always with heading hold (as the dog would be driven).
Appends one JSON line {run, scenario, stats} to the output file.

    lab tools/dog/benchmark.py runs/exp2_dog/11_imitation/amp_w6m_slow_s1 slippery --out runs/exp2_dog/benchmark.jsonl

Scenarios:
    flat      a flat floor
    course    the dog course (bumps, slopes, stairs up and down, rubble; 10 difficulty rows)
    slippery  the course with half the grip (TPU paw on a smooth tiled floor: friction ~0.5 instead of 1.0)
    grippy    the course with 1.4x the grip (soft silicone-like sole)
    pushes    the course with a shove every 2-4 s (0.4 m/s, 0.8 rad/s) plus the training randomization (masses
              +-15 %, servo gains, ankle damping)
    dark      the course with terrain-map sensor blackouts (on average every 2 s, 0.3-1.5 s long): only memory helps.
              Same as "course" for a dog with the perfect scan
    forward   the course, every walking command within +-30 deg of straight ahead (where a chest sensor looks)
    forward_dark  forward with the blackouts of "dark"
    stairs_straight  walking straight on: every command within +-6 deg of straight ahead, on stair tiles the dog starts
              facing the steps (+-15 deg) and is never asked to turn there (see type_fall_rates for the stairs)

A memory student (core/evolution/distill_memory.py, student.pt) runs with its hidden state.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from core.evolution.train import build_robot_files, evaluate
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

SCENARIOS = ("flat", "course", "slippery", "grippy", "pushes", "dark", "forward", "forward_dark", "stairs_straight")
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("scenario", choices=SCENARIOS)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--heading_hold", type=float, default=2.0)
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run, sc = args.run, args.scenario
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, sc == "pushes", task="command",
                   terrain="flat" if sc == "flat" else res.get("terrain_arg", "dog_rough"),
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False), terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.heading_hold_gain = args.heading_hold
if sc in ("slippery", "grippy"):
    f = 0.5 if sc == "slippery" else 1.4
    mat = cfg.terrain.physics_material
    mat.static_friction, mat.dynamic_friction = f, f
if sc == "pushes":
    cfg.push_interval, cfg.push_velocity = (2.0, 4.0), 0.4  # the environment builds the shove from these
if sc.startswith("forward"):
    cfg.eval_heading_range = (-0.52, 0.52)
if sc == "stairs_straight":
    cfg.eval_heading_range = (-0.1, 0.1)
    cfg.stairs_square_yaw, cfg.stairs_turn_scale = 0.26, 0.0
cfg.map_dropout_rate = 0.5 if sc.endswith("dark") else 0.0  # the course itself: no blackouts, whatever training used

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    if (run / "student.pt").exists():
        from core.evolution.distill_memory import RecurrentPolicy

        policy = RecurrentPolicy(torch.jit.load(str(run / "student.pt"), map_location=env.unwrapped.device).eval(),
                                 env.unwrapped)
    else:
        actor = torch.jit.load(str(run / "policy.pt"), map_location=env.unwrapped.device).eval()

        class Policy:
            def __call__(self, obs):
                return actor(obs["policy"])

            def reset(self, dones):
                pass

        policy = Policy()
    stats = evaluate(env, policy, int(env.unwrapped.max_episode_length))
    env.close()
    keep = ("fall_rate", "walk_score", "lin_vel_error_m_s", "yaw_rate_error_rad_s", "cost_of_transport",
            "mechanical_power_w", "type_fall_rates", "level_fall_rates", "type_level_scores", "type_level_climb_m")
    line = {"run": str(run).replace("\\", "/"), "scenario": sc, **{k: stats[k] for k in keep if k in stats}}
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(f"[benchmark] {run.name} {sc}: falls {stats['fall_rate']:.1%}, walk {stats['walk_score']:.3f}, "
          f"CoT {stats.get('cost_of_transport', float('nan')):.2f}")
sys.exit(0)
