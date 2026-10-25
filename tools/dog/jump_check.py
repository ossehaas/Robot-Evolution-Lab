"""Is a trained jumper really in the air? Per step: lowest paw sole, body height, contact forces, lowest link."""
import argparse, json, sys, torch
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

run = Path(sys.argv[1])
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args([])
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 16, False, task="jump", sensors=res["sensors"],
                   obs_history=res["obs_history"])
apply_overrides(cfg, res.get("env_overrides"))
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv
    from isaaclab.utils.math import quat_apply
    env = EvoLocomotionEnv(cfg)
    policy = torch.jit.load(str(run / "policy.pt"), map_location=env.device).eval()
    obs, _ = env.reset()
    names = env.robot.body_names
    rows = []
    with torch.inference_mode():
        for step in range(190):
            obs, *_ = env.step(policy(obs["policy"]))
            d = env.robot.data
            pos = d.body_pos_w.torch[:, env._foot_body_ids]
            quat = d.body_quat_w.torch[:, env._foot_body_ids]
            off = env._foot_offsets.unsqueeze(0).expand(env.num_envs, -1, -1)
            tip = pos + quat_apply(quat.reshape(-1, 4), off.reshape(-1, 3)).reshape(pos.shape)
            sole = (tip[..., 2] - env._foot_radius).min(-1).values
            f = torch.linalg.norm(env.contact.data.net_forces_w.torch, dim=-1)
            lowest_link = d.body_pos_w.torch[..., 2].min(-1)
            rows.append((step, sole[0].item(), d.root_pos_w.torch[0, 2].item(), f[0].sum().item(),
                         names[lowest_link.indices[0].item()], lowest_link.values[0].item()))
    air = [r for r in rows if r[3] < 1.0 and r[1] > 0.003]
    print(f"steps with no sensed contact and lowest paw > 3 mm: {len(air)} of {len(rows)}")
    for r in rows[40:190:6]:
        print(f"step {r[0]:3d}: lowest paw sole {1000 * r[1]:6.1f} mm, body {1000 * r[2]:5.0f} mm, sensed force {r[3]:6.1f} N, "
              f"lowest link frame {r[4]} at {1000 * r[5]:.0f} mm")
    env.close()
