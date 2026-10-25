"""Where does the power go? Mechanical power and joint speeds of a trained dog standing still and walking."""
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
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 64, False, task="command", sensors=res["sensors"],
                   obs_history=res["obs_history"])
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv
    env = EvoLocomotionEnv(cfg)
    env.expression_override = torch.zeros(1, len(env._expr_ids), device=env.device)  # head and tail still
    policy = torch.jit.load(str(run / "policy.pt"), map_location=env.device).eval()
    names = env.robot.joint_names
    for label, cmd in [("standing", [0.0, 0.0, 0.0]), ("walking 0.25 m/s", [0.25, 0.0, 0.0])]:
        env.command_override = torch.tensor(cmd, device=env.device)
        obs, _ = env.reset()
        p_sum, w_sum, a_rate, n = 0.0, torch.zeros(len(names), device=env.device), 0.0, 0
        with torch.inference_mode():
            for step in range(300):
                act = policy(obs["policy"])
                obs, _, _, _, _ = env.step(act)
                if step < 100:
                    continue
                d = env.robot.data
                p = (d.applied_torque.torch * d.joint_vel.torch).abs()
                p_sum += p.sum(-1).mean().item()
                w_sum += d.joint_vel.torch.abs().mean(0)
                a_rate += (env.actions - env.prev_actions).abs().mean().item()
                n += 1
        w = (w_sum / n).tolist()
        print(f"{label}: mechanical power {p_sum / n:.1f} W, mean action change per step {a_rate / n:.3f}")
        print("   mean |joint speed| rad/s:", {k: round(v, 2) for k, v in zip(names, w) if not k.startswith(("neck", "head", "tail"))})
    env.close()
