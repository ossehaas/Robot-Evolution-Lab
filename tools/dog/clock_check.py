"""Smoke check of gait_clock_idle (journal step 44): half the dogs told to stand still, half to walk at 0.2 m/s, on the
flat, driven by a trained policy for 3 s. Per group: how much the clock input (sin, cos of the gait phase) changes per
step. With "hold" / "zero" the standing dogs (upright) must see a constant clock ("zero": 0, 0), the walking dogs a
running one; with "run" both run. Exit code 1 if not.

    lab tools/dog/clock_check.py runs/exp2_dog/18_stairs_goal/free_s2 --clock hold
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.evolution.train import build_robot_files  # noqa: E402
from core.morphology import Genome, build_robot  # noqa: E402
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--clock", choices=("run", "hold", "zero"), required=True)
ap.add_argument("--num_envs", type=int, default=64)
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, False, task="command", terrain="flat",
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False),
                   terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.gait_clock_idle = args.clock
cfg.recover, cfg.topple_push = True, ()
cfg.recover_max_down_s = 1e9

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    dev, n = base.device, base.num_envs
    actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()
    walk = torch.arange(n, device=dev) % 2 == 1
    cmd = torch.zeros(n, 3, device=dev)
    cmd[walk, 0] = 0.2
    base.command_override = cmd
    obs, _ = env.reset()
    change = torch.zeros(n, device=dev)
    magnitude = torch.zeros(n, device=dev)
    fallen = torch.zeros(n, dtype=torch.bool, device=dev)
    steps = int(3.0 / base.step_dt)
    with torch.inference_mode():
        last = base._clock().clone()
        for _ in range(steps):
            obs, _, _, _ = env.step(actor(obs["policy"]))
            clock = base._clock()
            change += (clock - last).abs().sum(-1)
            magnitude = torch.maximum(magnitude, clock.abs().amax(-1))
            fallen |= base.fallen
            last = clock.clone()
    env.close()
    change = change / steps
    stand = ~walk & ~fallen
    s_change, w_change = float(change[stand].mean()), float(change[walk & ~fallen].mean())
    s_mag = float(magnitude[stand].max())
    print(f"[clock_check] {run.name}, gait_clock_idle={args.clock}: {int(fallen.sum())} of {n} fell")
    print(f"  clock change per step: standing {s_change:.4f}, walking {w_change:.4f}; largest clock value standing {s_mag:.3f}")
    bad = []
    if w_change < 0.05:
        bad.append("the walking dogs' clock does not run")
    if args.clock == "run" and s_change < 0.05:
        bad.append("the standing dogs' clock stopped with 'run'")
    if args.clock != "run" and s_change > 1e-4:
        bad.append("the standing dogs' clock still runs")
    if args.clock == "zero" and s_mag > 1e-6:
        bad.append("the standing dogs' clock is not 0, 0")
    print("  " + ("OK" if not bad else "PROBLEMS: " + "; ".join(bad)))
sys.exit(1 if bad else 0)
