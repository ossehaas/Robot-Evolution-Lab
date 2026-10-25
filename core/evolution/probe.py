"""Measure what a steerable robot actually does for fixed commands (forward, backward, sideways, turning).

    python -m core.evolution.probe --run_dir runs/crawler_q4_steer
    python -m core.evolution.probe --run_dir runs/crawler_q4_steer --speeds 0.1 0.2 0.3 0.4

Every environment gets one fixed command for a whole episode; after a settling period the achieved velocity in
the robot's own frame is averaged. Prints a table and writes <run_dir>/probe.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Probe a steerable policy with fixed commands.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--turn_rates", type=float, nargs="+", default=[0.6, 1.2])
    parser.add_argument("--repeats", type=int, default=32, help="robots per command (domain randomization on)")
    parser.add_argument("--seconds", type=float, default=6.0)
    add_launcher_args(parser)
    args = parser.parse_args(argv)

    # (label, vx, vy, yaw rate)
    probes = []
    for v in args.speeds:
        probes += [(f"forward {v:.2f}", v, 0.0, 0.0), (f"backward {v:.2f}", -v, 0.0, 0.0)]
        probes += [(f"left {v:.2f}", 0.0, v, 0.0), (f"right {v:.2f}", 0.0, -v, 0.0)]
    for w in args.turn_rates:
        probes += [(f"turn left {w:.1f}", 0.0, 0.0, w), (f"turn right {w:.1f}", 0.0, 0.0, -w)]

    from core.evolution.train import build_robot_files
    from core.morphology import Genome, build_robot
    from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

    genome_file = args.run_dir / "genome.json"
    usd = build_robot_files(genome_file.resolve(), args.run_dir / "robot")
    spec = build_robot(Genome.load(genome_file))
    num_envs = len(probes) * args.repeats
    result_file = args.run_dir / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    env_cfg = make_env_cfg(
        spec, str(usd.resolve()).replace("\\", "/"), num_envs, True, task="command",
        sensors=result.get("sensors", "ideal"), obs_history=result.get("obs_history", 1),
        height_scan=result.get("height_scan", False), terrain_map=result.get("terrain_map", ""),
    )
    apply_overrides(env_cfg, result.get("env_overrides"))
    env_cfg.episode_length_s = args.seconds + 1.0

    from isaaclab.app import launch_simulation

    with launch_simulation(env_cfg, args):
        import importlib.metadata as metadata

        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from core.evolution.agent_cfg import EvoPPORunnerCfg
        from core.sim.locomotion_env import EvoLocomotionEnv

        env = RslRlVecEnvWrapper(EvoLocomotionEnv(env_cfg), clip_actions=1.0)
        base = env.unwrapped
        device = base.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(EvoPPORunnerCfg(), metadata.version("rsl-rl-lib"))
        agent_cfg.device = device
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(str(args.checkpoint or args.run_dir / "model_final.pt"))
        policy = runner.get_inference_policy(device=device)

        commands = torch.tensor([p[1:] for p in probes], device=device).repeat_interleave(args.repeats, dim=0)
        base.command_override = commands
        obs, _ = env.reset()
        steps = int(args.seconds / base.step_dt)
        settle = int(1.5 / base.step_dt)
        vel_sum = torch.zeros(num_envs, 3, device=device)
        alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        with torch.inference_mode():
            for k in range(steps):
                obs, _, dones, _ = env.step(policy(obs))
                alive &= ~dones.bool()
                if k >= settle:
                    d = base.robot.data
                    vel_sum += torch.cat((d.root_lin_vel_b.torch[:, :2], d.root_ang_vel_b.torch[:, 2:3]), dim=-1)
        vel = (vel_sum / (steps - settle)).view(len(probes), args.repeats, 3)
        alive = alive.view(len(probes), args.repeats).float().mean(dim=1)

        rows = []
        print(f"\n{'command':<16}{'target':>10}{'achieved':>10}{'ratio':>8}{'cross':>8}{'upright':>9}")
        for i, (label, vx, vy, wz) in enumerate(probes):
            mean = vel[i].mean(dim=0)
            axis = 0 if vx else (1 if vy else 2)
            target = (vx, vy, wz)[axis]
            achieved = float(mean[axis])
            cross = float(torch.linalg.norm(mean[:2])) if axis == 2 else float(abs(mean[1 - axis]))
            unit = "rad/s" if axis == 2 else "m/s"
            print(f"{label:<16}{target:>10.2f}{achieved:>10.3f}{achieved / target:>8.2f}{cross:>8.3f}{float(alive[i]):>9.2f}  {unit}")
            rows.append({"command": label, "target": target, "achieved": achieved, "ratio": achieved / target,
                         "cross_axis": cross, "survived": float(alive[i])})
        (args.run_dir / "probe.json").write_text(json.dumps(rows, indent=2))
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
