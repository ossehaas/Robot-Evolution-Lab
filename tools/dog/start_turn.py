"""Does the dog keep its heading when it starts walking? Stand 2 s, then "straight ahead" for 3 s (as the video tour
starts); per dog the heading change in degrees (+ = left) while it should be zero. Flat ground, 64 dogs; with
--course LEVEL on the run's own course at that difficulty row, dogs spread over all tile types (reported per type).

    lab tools/dog/start_turn.py runs/exp2_dog/11_imitation/amp_w6 [speed] [--course 5] [--expr]

--expr: head and tail move on their own as in training and in the videos (default: held still).
--hold G: heading hold with gain G (corrective turn commands, cfg.heading_hold_gain), 0 = as trained.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

run = Path(sys.argv[1])
argv = sys.argv[2:]
course = int(argv.pop(argv.index("--course") + 1)) if "--course" in argv else None
expr = "--expr" in argv
hold = float(argv.pop(argv.index("--hold") + 1)) if "--hold" in argv else None
argv = [a for a in argv if a not in ("--course", "--expr", "--hold")]
speed = float(argv[0]) if argv else 0.25
parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args([])
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 64 if course is None else 120, False, task="command",
                   terrain="flat" if course is None else res.get("terrain_arg", "dog_rough"),
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False), terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, res.get("env_overrides"))
cfg.reset_yaw_range = 0.0
cfg.episode_length_s = 30.0
if hold is not None:
    cfg.heading_hold_gain = hold
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv

    env = EvoLocomotionEnv(cfg)
    if not expr:
        env.expression_override = torch.zeros(1, len(env._expr_ids), device=env.device)
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    env.command_override = cmd
    policy = torch.jit.load(str(run / "policy.pt"), map_location=env.device).eval()
    type_of = None
    if course is not None:
        env.curriculum_enabled = False
        t, gen = env.terrain, cfg.terrain.terrain_generator
        names = list(gen.sub_terrains)
        cum = torch.cumsum(torch.tensor([gen.sub_terrains[n].proportion for n in names]), 0)
        cum = cum / cum[-1]
        col_type = [names[int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
        t.terrain_levels[:] = course
        t.terrain_types[:] = torch.arange(env.num_envs, device=env.device) % gen.num_cols
        t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
        type_of = [col_type[int(c)] for c in t.terrain_types.tolist()]
    obs, _ = env.reset()

    from isaaclab.utils.math import quat_apply

    def heading():
        # from the body's forward axis (the library knows the quaternion layout: Isaac Lab 3 stores x, y, z, w)
        q = env.robot.data.root_quat_w.torch
        fwd = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(q.shape[0], 3))
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    with torch.inference_mode():
        for _ in range(100):  # stand
            obs, *_ = env.step(policy(obs["policy"]))
        h0 = heading().clone()
        p0 = env.robot.data.root_pos_w.torch[:, :2].clone()
        total = torch.zeros_like(h0)
        prev = h0.clone()
        cmd[:, 0] = speed
        peak = torch.zeros_like(h0)
        for _ in range(150):  # walk straight ahead
            obs, *_ = env.step(policy(obs["policy"]))
            h = heading()
            total += torch.remainder(h - prev + math.pi, 2 * math.pi) - math.pi
            prev = h.clone()
            peak = torch.maximum(peak, total.abs())
        dist = torch.linalg.norm(env.robot.data.root_pos_w.torch[:, :2] - p0, dim=-1)
    deg = torch.rad2deg(total)
    print(f"start at {speed:.2f} m/s, after 3 s: heading change mean {float(deg.mean()):+.1f} deg, mean |change| "
          f"{float(deg.abs().mean()):.1f} deg, worst {float(deg.abs().max()):.1f} deg, largest swing during the start "
          f"{float(torch.rad2deg(peak).mean()):.1f} deg (mean), distance {float(dist.mean()):.2f} m")
    if type_of is not None:
        for name in dict.fromkeys(type_of):
            m = torch.tensor([n == name for n in type_of], device=deg.device)
            print(f"  level {course} {name:>16s}: mean |change| {float(deg[m].abs().mean()):5.1f} deg, worst "
                  f"{float(deg[m].abs().max()):5.1f} deg, distance {float(dist[m].mean()):.2f} m")
    env.close()
