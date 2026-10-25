"""Gait analysis of a trained dog: per forward speed, how many paws are down, how the footfalls are timed, and what
it costs.

    lab tools/dog/gait_check.py runs/exp2_dog/4_stairs_seeing/seeing_v3 [speeds ...]

Per speed: achieved speed, mean number of paws on the ground (walk ~3, trot ~2), stance share per paw, how often
diagonal pairs touch down together (trot) vs one paw at a time (walk), stride frequency (touchdowns per paw per second)
cost of transport and the mean turn rate (rad/s, + = left; the command asks for none: a drift).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

run = Path(sys.argv[1])
on_course = "--course" in sys.argv  # on the run's own terrain (easiest level) instead of a flat plane
speeds = [float(v) for v in sys.argv[2:] if not v.startswith("--")] or [0.1, 0.2, 0.3, 0.45]
parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args([])
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
n_per = 16
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), n_per * len(speeds), "--dr" in sys.argv, task="command",
                   terrain=res.get("terrain_arg", "flat") if on_course else "flat",
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False), terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, res.get("env_overrides"))
cfg.reset_yaw_range = 0.0
cfg.episode_length_s = 30.0
cfg.command_accel = 0.0
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv

    env = EvoLocomotionEnv(cfg)
    env.expression_override = torch.zeros(1, len(env._expr_ids), device=env.device)
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    for k, v in enumerate(speeds):
        cmd[k * n_per:(k + 1) * n_per, 0] = v
    env.command_override = cmd
    if on_course and cfg.terrain_kind == "rough":
        env.curriculum_enabled = False
        t = env.terrain
        t.terrain_levels[:] = 0
        t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
    policy = torch.jit.load(str(run / "policy.pt"), map_location=env.device).eval()
    obs, _ = env.reset()
    feet = env._foot_sensor_ids
    names = [env.contact.body_names[i] for i in feet]
    contacts, vel, yaw, power, geo = [], [], [], [], []
    with torch.inference_mode():
        for step in range(400):
            obs, *_ = env.step(policy(obs["policy"]))
            if step < 100:
                continue
            f = torch.linalg.norm(env.contact.data.net_forces_w.torch[:, feet], dim=-1) > 1.0
            contacts.append(f.clone())
            # cross-check by geometry: paw sole within 5 mm of the floor
            from isaaclab.utils.math import quat_apply
            dd_ = env.robot.data
            pos = dd_.body_pos_w.torch[:, env._foot_body_ids]
            quat = dd_.body_quat_w.torch[:, env._foot_body_ids]
            off = env._foot_offsets.unsqueeze(0).expand(env.num_envs, -1, -1)
            tip = pos + quat_apply(quat.reshape(-1, 4), off.reshape(-1, 3)).reshape(pos.shape)
            sole = tip[..., 2] - env._foot_radius - env.terrain.env_origins[:, 2:3]
            geo.append(sole < 0.005)
            d = env.robot.data
            vel.append(d.root_lin_vel_b.torch[:, 0].clone())
            yaw.append(d.root_ang_vel_b.torch[:, 2].clone())
            m = env._obs_ids
            power.append(torch.sum(torch.abs(d.applied_torque.torch[:, m] * d.joint_vel.torch[:, m]), dim=-1))
    c = torch.stack(contacts)  # (T, envs, 4)
    gc = torch.stack(geo)
    v = torch.stack(vel)
    w = torch.stack(yaw)
    p = torch.stack(power)
    down = c[1:] & ~c[:-1]  # touchdown events
    diag = {"A": [names.index(n) for n in names if n[:2] in ("L0", "R1")], "B": [names.index(n) for n in names if n[:2] in ("R0", "L1")]}
    mass = cfg.robot_mass
    print(f"{'speed':>6s} {'achieved':>8s} {'paws down':>9s} {'stance per paw (' + ' '.join(n[:2] for n in names) + ')':>32s} "
          f"{'paired':>7s} {'strides/s':>9s} {'CoT':>5s} {'turn':>6s}")
    for k, sp in enumerate(speeds):
        sl = slice(k * n_per, (k + 1) * n_per)
        cc, dd = c[:, sl], down[:, sl]
        stance = cc.float().mean(dim=(0, 1))
        # a touchdown is "paired" if its diagonal partner touches down within 2 steps (40 ms): trot-like
        paired = total = 0
        a, b = diag["A"], diag["B"]
        for i, j in ((a[0], a[1]), (a[1], a[0]), (b[0], b[1]), (b[1], b[0])):
            ev = dd[:, :, i]
            near = torch.zeros_like(ev)
            for sh in range(-2, 3):
                near |= torch.roll(dd[:, :, j], sh, dims=0)
            paired += int((ev & near).sum())
            total += int(ev.sum())
        # touchdowns per paw per second = stride frequency (the mean already averages over the four paws)
        steps_per_s = float(dd.float().sum(dim=0).mean()) / (cc.shape[0] * env.step_dt)
        achieved = float(v[:, sl].mean())
        cot = float(p[:, sl].mean()) / (mass * 9.81 * max(abs(achieved), 1e-3))
        print(f"   (geometry: paws within 5 mm of the floor {float(gc[:, sl].float().sum(-1).mean()):.2f}, per paw "
              f"{' '.join(f'{x:.2f}' for x in gc[:, sl].float().mean(dim=(0, 1)).tolist())})")
        print(f"{sp:6.2f} {achieved:8.3f} {float(cc.float().sum(-1).mean()):9.2f} "
              f"{' '.join(f'{x:.2f}' for x in stance.tolist()):>32s} {paired / max(total, 1):7.0%} {steps_per_s:9.2f} {cot:5.2f} {float(w[:, sl].mean()):+6.2f}")
    env.close()
