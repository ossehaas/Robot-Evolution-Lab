"""Smoke check of the tilt pad tiles in the simulator (base skills, journal step 43): every robot on a tilt pad
(rows 0-30 deg, the training terrain dog_rough_stairs_tilt), placed by the environment's own reset, driven by a trained
policy for 3 s. Per tilt: share fallen, standing height above the pad (ground_height) against the target, tilt of the
body against the pad, and whether the commands stay "stand still". Exit code 1 if the placement looks wrong (fallen
on the gentle pads, height off by more than 3 cm, or moving commands).

    lab tools/dog/tilt_pad_check.py runs/exp2_dog/18_stairs_goal/free_s2
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.evolution.train import build_robot_files  # noqa: E402
from core.morphology import Genome, build_robot  # noqa: E402
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg  # noqa: E402
from core.sim.terrain import TILT_TRAIN_ANGLES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--num_envs", type=int, default=300)
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, False, task="command",
                   terrain="dog_rough_stairs_tilt", sensors=res["sensors"], obs_history=res["obs_history"],
                   height_scan=res.get("height_scan", False), terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.recover, cfg.topple_push = True, ()
cfg.recover_max_down_s = 1e9
cfg.curriculum = False

with launch_simulation(cfg, args):
    from isaaclab.utils.math import quat_apply
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    dev, n = base.device, base.num_envs
    actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()
    base.spread_over_terrain_levels()
    t = base.terrain
    cols = torch.nonzero(base._tilt_col).squeeze(-1)
    t.terrain_types[:] = cols[torch.arange(n, device=dev) % len(cols)]
    t.terrain_levels[:] = torch.arange(n, device=dev) % t.terrain_origins.shape[0]
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
    obs, _ = env.reset()
    fallen = torch.zeros(n, dtype=torch.bool, device=dev)
    moving_cmd = torch.zeros(n, dtype=torch.bool, device=dev)
    height0 = base.ground_height().clone()
    with torch.inference_mode():
        for _ in range(int(3.0 / base.step_dt)):
            obs, _, _, _ = env.step(actor(obs["policy"]))
            fallen |= base.fallen
            moving_cmd |= base.commands.abs().amax(-1) > 1e-6
        height = base.ground_height()
        up = quat_apply(base.robot.data.root_quat_w.torch, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        rel_tilt = torch.rad2deg(torch.arccos((up * base.tilt_normal()).sum(-1).clamp(-1, 1)))
    env.close()
    target = cfg.base_height_target
    bad = []
    print(f"[tilt_pad_check] {run.name}: {n} dogs on {len(cols)} tilt pad columns; target height {100 * target:.1f} cm")
    print("  tilt   fallen   height at start / after 3 s [cm]   body vs pad [deg]")
    for row, ang in enumerate(TILT_TRAIN_ANGLES):
        sel = (t.terrain_levels == row) & ~fallen
        f = float(fallen[t.terrain_levels == row].float().mean())
        h0 = float(height0[t.terrain_levels == row].mean())
        h = float(height[sel].mean()) if bool(sel.any()) else math.nan
        tl = float(rel_tilt[sel].mean()) if bool(sel.any()) else math.nan
        print(f"  {ang:4.0f}   {100 * f:5.0f} %   {100 * h0:6.1f} / {100 * h:6.1f}   {tl:6.1f}")
        if ang <= 12.0 and f > 0.2:
            bad.append(f"{ang:.0f} deg: {100 * f:.0f} % fallen")
        if abs(h0 - spec.spawn_height) > 0.03 or (ang <= 12.0 and abs(h - target) > 0.03):
            bad.append(f"{ang:.0f} deg: height {100 * h0:.1f} -> {100 * h:.1f} cm")
    if bool(moving_cmd.any()):
        bad.append(f"{int(moving_cmd.sum())} dogs got moving commands on the pads")
    print("  " + ("OK" if not bad else "PROBLEMS: " + "; ".join(bad)))
sys.exit(1 if bad else 0)
