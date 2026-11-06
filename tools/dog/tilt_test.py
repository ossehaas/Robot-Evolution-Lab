"""Tilt test (journal step 42): at what tilt can the dog stand, per direction, and what does standing still cost?

Every dog stands on a flat plate tilted by an angle (rows 0-45 deg, terrain.TILT_COURSE), facing +x; the plate goes
down towards 0 (nose down, as on stairs going down), 45 (diagonal), 90 (left side low) or 180 deg (nose up). It is
placed standing on the plate (perpendicular to it) and told to stand still for 12 s (command zero; no shoves, no
randomization). Per angle and direction:

    stood        not fallen and moved less than 5 cm along the plate between 2 s and the end
    fell         tilted more than 35 deg against the plate, or the body touching it (falls do not end the episode)
    slid         moved more than 5 cm (slipping or stepping away) without falling
    moved_cm     median distance moved in those 10 s by the dogs that did not fall (stepping about, sliding)
    power_w      mechanical power of the servos (mean of the dogs that did not fall, last 10 s)
    joint_rad_s  summed joint speeds (how still it stands)
    still        share of the time after the first 2 s that the dog is still (summed joint speed < 0.5 rad/s), dogs
                 that did not fall (base skill B1: >= 90 % on the flat)
    settle_s     median time until it is still for 1 s in a row (B1: <= 2 s)

--shove v (base skill B2): every dog gets one shove of v m/s in a random horizontal direction at 6 s. Then also:
    fell_after   share falling after the shove
    resettle_s   median time from the shove until it is still for 1 s in a row again (B2: <= 3 s), dogs not fallen
    still_after  share of the last 3 s (9-12 s) that it is still

--hold: no policy, the servos hold the standing pose (the bare body: compare tools/dog/tilt_limits.py).
Appends one JSON line {run, hold, cells} to --out and prints the tables.

    lab tools/dog/tilt_test.py runs/exp2_dog/18_stairs_goal/free_s2 --out runs/exp2_dog/20_balance_and_style/tilt.jsonl
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
from core.sim.terrain import TILT_ANGLES, TILT_COURSE, TILT_DIRECTIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--per_cell", type=int, default=10, help="dogs per angle and direction")
ap.add_argument("--hold", action="store_true", help="no policy: the servos hold the standing pose")
ap.add_argument("--shove", type=float, default=0.0, help="[m/s] one shove at 6 s (0 = none)")
ap.add_argument("--set", action="append", default=[], help="extra environment setting, e.g. gait_clock_idle=hold")
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
n_cells = len(TILT_ANGLES) * len(TILT_DIRECTIONS)
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), n_cells * args.per_cell, False, task="command",
                   terrain="dog_stairs", sensors=res["sensors"], obs_history=res["obs_history"],
                   height_scan=res.get("height_scan", False), terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
for item in args.set:
    import ast

    key, value = item.split("=", 1)
    if not hasattr(cfg, key):
        raise SystemExit(f"--set: unknown environment setting {key!r}")
    try:
        setattr(cfg, key, ast.literal_eval(value))
    except (ValueError, SyntaxError):
        setattr(cfg, key, value)  # plain text, e.g. gait_clock_idle=hold
cfg.terrain.terrain_generator = TILT_COURSE
cfg.recover, cfg.topple_push, cfg.getup = True, (), False
cfg.recover_max_down_s = 1e9
cfg.episode_length_s = 14.0
cfg.curriculum = False
SECONDS, SETTLE, SHOVE_T, STILL = 12.0, 2.0, 6.0, 0.5  # [s], [s], [s], [rad/s]

with launch_simulation(cfg, args):
    from isaaclab.utils.math import quat_apply, quat_from_angle_axis
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    dev, n = base.device, base.num_envs
    if args.hold:
        def policy(obs):
            return torch.zeros(n, env.num_actions, device=dev)
    else:
        actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()

        def policy(obs):
            return actor(obs["policy"])
    base.command_override = torch.zeros(3, device=dev)
    base.spread_over_terrain_levels()
    t = base.terrain
    obs, _ = env.reset()
    # stand each dog on its plate: rotated with the plate (about the horizontal axis that lowers the downhill side)
    angle = torch.tensor(TILT_ANGLES, device=dev)[t.terrain_levels] * math.pi / 180
    phi = torch.tensor(TILT_DIRECTIONS, device=dev)[t.terrain_types] * math.pi / 180
    axis = torch.stack((-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)), -1)
    quat = quat_from_angle_axis(angle, axis)  # the library's quaternion order (this Isaac Lab: x, y, z, w)
    normal = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
    pos = t.env_origins + normal * (spec.spawn_height + 0.003)
    ids = torch.arange(n, device=dev)
    base.robot.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), -1), env_ids=ids)
    base.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    start = torch.zeros(n, 3, device=dev)
    power = torch.zeros(n, device=dev)
    motion = torch.zeros(n, device=dev)
    count = 0
    dt = base.step_dt
    motors = base._obs_ids
    steps = int(SECONDS / dt)
    still_log = torch.zeros(steps, n, dtype=torch.bool, device=dev)
    fell_before = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        for k in range(steps):
            if args.shove > 0 and k == int(SHOVE_T / dt):
                fell_before = fell.clone()
                a = torch.rand(n, device=dev) * 2 * math.pi
                vel = torch.cat((base.robot.data.root_lin_vel_w.torch, base.robot.data.root_ang_vel_w.torch), -1).clone()
                vel[:, 0] += args.shove * torch.cos(a)
                vel[:, 1] += args.shove * torch.sin(a)
                base.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
            obs, _, _, _ = env.step(policy(obs))
            d = base.robot.data
            still_log[k] = torch.sum(torch.abs(d.joint_vel.torch[:, motors]), -1) < STILL
            up = quat_apply(d.root_quat_w.torch, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
            tilt = torch.rad2deg(torch.arccos((up * normal).sum(-1).clamp(-1, 1)))
            body = (torch.linalg.norm(base.contact.data.net_forces_w.torch[:, base._chassis_sensor_id], dim=-1) > 1.0).any(-1)
            fell |= (tilt > 35.0) | body
            if k == int(SETTLE / dt):
                start = d.root_pos_w.torch.clone()
            if k > int(SETTLE / dt):
                power += torch.sum(torch.abs(d.applied_torque.torch[:, motors] * d.joint_vel.torch[:, motors]), -1)
                motion += torch.sum(torch.abs(d.joint_vel.torch[:, motors]), -1)
                count += 1
        moved = torch.linalg.norm(base.robot.data.root_pos_w.torch - start, dim=-1)
    env.close()
    slid = ~fell & (moved > 0.05)
    stood = ~fell & ~slid
    run_len = int(1.0 / dt)

    def settle_time(from_step: int) -> torch.Tensor:
        """(n,) seconds from step `from_step` until the dog is still for 1 s in a row (inf if never)."""
        s = still_log[from_step:].float()
        if s.shape[0] < run_len:
            return torch.full((n,), float("inf"), device=dev)
        window = s.unfold(0, run_len, 1).sum(-1) >= run_len  # (steps - run_len + 1, n)
        first = torch.where(window.any(0), window.float().argmax(0).float(), torch.full((n,), float("inf"), device=dev))
        return first * dt

    settle = settle_time(0)
    still = still_log[int(SETTLE / dt):].float().mean(0)
    shoved = args.shove > 0
    if shoved:
        resettle = settle_time(int(SHOVE_T / dt))
        still_after = still_log[int(9.0 / dt):].float().mean(0)
        fell_after = fell & ~fell_before

    def med(x: torch.Tensor):
        x = x[torch.isfinite(x)]
        return round(float(x.median()), 2) if x.numel() else None

    cells = []
    for li, a in enumerate(TILT_ANGLES):
        for ti, direc in enumerate(TILT_DIRECTIONS):
            sel = (t.terrain_levels == li) & (t.terrain_types == ti)
            ok = sel & ~fell  # standing cost of every dog still up (also those that stepped about)
            cells.append({"angle_deg": a, "direction_deg": direc, "n": int(sel.sum()),
                          "stood": round(float(stood[sel].float().mean()), 3), "fell": round(float(fell[sel].float().mean()), 3),
                          "slid": round(float(slid[sel].float().mean()), 3),
                          "moved_cm": round(float(100 * moved[ok].median()), 1) if bool(ok.any()) else None,
                          "power_w": round(float(power[ok].mean() / count), 2) if bool(ok.any()) else None,
                          "joint_rad_s": round(float(motion[ok].mean() / count), 3) if bool(ok.any()) else None,
                          "still": round(float(still[ok].mean()), 3) if bool(ok.any()) else None,
                          "settle_s": med(settle[ok]), "never_settled": round(float((~torch.isfinite(settle[ok])).float().mean()), 3)
                          if bool(ok.any()) else None})
            if shoved:
                ok_b = sel & ~fell_before
                cells[-1].update({"fell_after": round(float(fell_after[ok_b].float().mean()), 3) if bool(ok_b.any()) else None,
                                  "resettle_s": med(resettle[ok]),
                                  "still_after": round(float(still_after[ok].mean()), 3) if bool(ok.any()) else None})
    line = {"run": str(run).replace("\\", "/"), "hold": args.hold, "shove_m_s": args.shove, "set": args.set, "cells": cells}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    names = {0.0: "nose down", 45.0: "diagonal", 90.0: "sideways", 180.0: "nose up"}
    print(f"[tilt_test] {run.name}{' (servos holding the standing pose)' if args.hold else ''}: stood % "
          f"(fell % / slid %), power W on the flat")
    print("  tilt " + "".join(f"{names[d]:>20}" for d in TILT_DIRECTIONS))
    for a in TILT_ANGLES:
        row = [c for c in cells if c["angle_deg"] == a]
        print(f"  {a:4.0f} " + "".join(f"{100 * c['stood']:>9.0f} ({100 * c['fell']:3.0f}/{100 * c['slid']:3.0f})" for c in row))
    flat = [c for c in cells if c["angle_deg"] == 0.0 and c["power_w"] is not None]
    if flat:
        print(f"  standing still on the flat: {sum(c['power_w'] for c in flat) / len(flat):.2f} W, "
              f"{sum(c['joint_rad_s'] for c in flat) / len(flat):.3f} rad/s summed joint speed, moved "
              f"{sum(c['moved_cm'] for c in flat) / len(flat):.1f} cm in 10 s, still "
              f"{100 * sum(c['still'] for c in flat) / len(flat):.0f} % of the time")
    print("  still % (after 2 s, dogs up) / settle s per tilt and direction:")
    for a in TILT_ANGLES:
        row = [c for c in cells if c["angle_deg"] == a]
        cols = []
        for c in row:
            s_txt = "-" if c["still"] is None else f"{100 * c['still']:.0f}"
            t_txt = "-" if c["settle_s"] is None else f"{c['settle_s']:.1f}"
            cols.append(f"{s_txt:>12} / {t_txt:>5}")
        print(f"  {a:4.0f} " + "".join(cols))
sys.exit(0)
