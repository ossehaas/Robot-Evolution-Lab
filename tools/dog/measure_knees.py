"""Heights of knees and paws above the floor while the trained dog walks forward at 25 cm/s."""
import argparse, json, sys, torch
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import make_env_cfg

run = Path(sys.argv[1])
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args([])
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 16, False, task="command", sensors=res["sensors"],
                   obs_history=res["obs_history"])
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv
    env = EvoLocomotionEnv(cfg)
    policy = torch.jit.load(str(run / "policy.pt"), map_location=env.device).eval()
    env.command_override = torch.tensor([0.25, 0.0, 0.0], device=env.device)
    obs, _ = env.reset()
    shins = [env.robot.find_bodies(f"{s}{i}_s2")[0][0] for i in (0, 1) for s in "LR"]
    knee_min, paw_min, low_knee = [], [], 0
    sat, absmax, cnt = 0, 0.0, 0
    with torch.inference_mode():
        for step in range(400):
            raw = policy(obs["policy"])
            if step >= 100:
                sat += int((raw.abs() > 1.0).sum().item()); cnt += raw.numel(); absmax = max(absmax, raw.abs().mean().item())
            obs, _, _, _, _ = env.step(raw)
            if step < 100:
                continue
            z = env.robot.data.body_pos_w.torch[:, shins, 2] - env.terrain.env_origins[:, 2:3]  # knee joint heights
            knee_min.append(z.min().item())
            low_knee += int((z < 0.04).sum().item())
    n = 300 * 16 * 4
    print(f"lowest knee: {1000 * min(knee_min):.0f} mm; knee below 4 cm in {100 * low_knee / n:.1f}% of leg-steps")
    print(f"actions beyond the +-1 limit: {100 * sat / cnt:.0f}%, mean |action| up to {absmax:.2f}")
    print(f"rest-pose knee height: {1000 * (spec.link_map['L0_s2'].origin[2] + spec.spawn_height):.0f} mm")
    env.close()
