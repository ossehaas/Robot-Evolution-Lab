"""Shove and recovery: robots walk straight on at 20 cm/s on a flat floor (heading hold); after 3 s each gets one shove
(random direction, no other randomization); the robots are split over shove strengths. In the 5 s after the shove a
robot either stays up (never fallen), falls and gets up (fallen, then upright and off the ground again for 1 s), or
stays down. Falls do not end the episode (recover mode), so dogs without get-up training can be tested the same way.
Appends one JSON line.

    lab tools/dog/recover_test.py runs/exp2_dog/16_recover/recover --out runs/exp2_dog/16_recover/recover_test.jsonl
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

STRENGTHS = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)  # [m/s]
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=1022)  # a multiple of len(STRENGTHS)
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
cfg.recover, cfg.topple_push, cfg.getup = True, (), False
cfg.heading_hold_gain = 2.0
cfg.map_dropout_rate = 0.0
cfg.episode_length_s = 30.0
cfg.recover_max_down_s = 100.0  # never end the episode during the test

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    dev = base.device
    actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()
    base.command_override = torch.tensor([0.2, 0.0, 0.0], device=dev)
    n = base.num_envs
    strength = torch.tensor(STRENGTHS, device=dev)[torch.arange(n, device=dev) % len(STRENGTHS)]
    obs, _ = env.reset()
    dt = base.step_dt
    push_step, end_step = int(3.0 / dt), int(8.0 / dt)
    down_before = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    up_for = torch.zeros(n, device=dev)
    recovered = torch.zeros(n, dtype=torch.bool, device=dev)
    recover_time = torch.full((n,), float("nan"), device=dev)
    with torch.inference_mode():
        for step in range(end_step):
            if step == push_step:
                angle = torch.rand(n, device=dev) * 2 * math.pi
                d = base.robot.data
                vel = torch.cat((d.root_lin_vel_w.torch, d.root_ang_vel_w.torch), dim=-1).clone()
                vel[:, 0] += strength * torch.cos(angle)
                vel[:, 1] += strength * torch.sin(angle)
                base.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=torch.arange(n, device=dev))
            obs, _, _, _ = env.step(actor(obs["policy"]))
            if step < push_step:
                down_before |= base.fallen
                continue
            fell |= base.fallen
            up_for = torch.where(base.fallen, torch.zeros_like(up_for), up_for + dt)
            new = fell & ~recovered & (up_for >= 1.0)
            recover_time[new] = (step - push_step + 1) * dt - 1.0
            recovered |= new
    ok = ~down_before
    rows = {}
    for s in STRENGTHS:
        sel = (strength == s) & ok
        stayed_up = (~fell & sel).float().sum()
        got_up = (fell & recovered & sel).float().sum()
        times = recover_time[fell & recovered & sel]
        total = sel.float().sum().clamp(min=1)
        rows[f"{s:.1f}"] = {"stayed_up": round(float(stayed_up / total), 3), "fell_got_up": round(float(got_up / total), 3),
                            "stayed_down": round(float((total - stayed_up - got_up) / total), 3),
                            "median_recovery_s": round(float(times.median()), 2) if len(times) else None}
    line = {"run": str(run).replace("\\", "/"), "walking_m_s": 0.2, "down_before_shove": round(float(down_before.float().mean()), 3),
            "by_shove_m_s": rows}
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print("[recover_test]", run.name, {k: (v["stayed_up"], v["fell_got_up"], v["stayed_down"]) for k, v in rows.items()})
    env.close()
sys.exit(0)
