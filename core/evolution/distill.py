"""Combine specialist policies into one student policy (multi-teacher distillation, DAgger style).

    python -m core.evolution.distill --genome <genome.json> --out runs/student \\
        --teacher stairs=runs/spec_stairs/model_final.pt:stairs_climb,stairs_descend \\
        --teacher blocks=runs/spec_blocks/model_final.pt:blocks \\
        --fallback runs/generalist/model_final.pt --sensors imu --obs_history 8

How it works
    * the robots walk the full rough terrain (all difficulty levels, curriculum frozen) with the normal random
      commands; on each tile the teacher assigned to that terrain type says what it would do
    * the student only gets the robot's own senses: the terrain label is used to pick the teacher, never given to
      the student. It must learn to recognize the situation itself.
    * early on, the teachers drive the robots part of the time (so the student sees good states); later the student
      drives and is corrected wherever it ends up, including transitions no single teacher practised (DAgger)
    * the student is a standard policy network: polish it afterwards with RL via
      ``core.evolution.train --init_checkpoint <out>/model_final.pt``

Teachers must use the same senses (--sensors, --obs_history) as the student. Writes model_final.pt, policy.onnx and
result.json (same evaluation as core.evolution.train) into --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_teacher(spec: str) -> tuple[str, Path, list[str]]:
    """'name=path/model.pt:type1,type2' -> (name, path, [types])."""
    name, rest = spec.split("=", 1)
    path, types = rest.rsplit(":", 1)
    return name, Path(path), [t for t in types.split(",") if t]


def main(argv=None) -> int:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Distil specialist policies into one student.")
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", action="append", default=[], help="name=checkpoint:terrain_type[,type...]")
    parser.add_argument("--fallback", type=Path, required=True, help="teacher for terrain types no specialist covers")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--steps_per_iteration", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher_drive_until", type=float, default=0.3,
                        help="fraction of training during which teachers drive part of the time (linearly fading)")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--sensors", choices=("ideal", "imu", "imu_flow"), default="ideal")
    parser.add_argument("--obs_history", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    teachers = [parse_teacher(t) for t in args.teacher]

    from core.evolution.train import build_robot_files
    from core.morphology import Genome, build_robot
    from core.sim.locomotion_env_cfg import make_env_cfg

    genome_path = args.genome.resolve()
    genome = Genome.load(genome_path)
    genome.save(args.out / "genome.json")
    usd = build_robot_files(genome_path, args.out / "robot")
    spec = build_robot(genome)
    env_cfg = make_env_cfg(
        spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, True, task="command", terrain="rough",
        sensors=args.sensors, obs_history=args.obs_history,
    )  # fmt: skip
    env_cfg.curriculum = False
    env_cfg.seed = args.seed

    from isaaclab.app import launch_simulation

    started = time.time()
    with launch_simulation(env_cfg, args):
        import importlib.metadata as metadata

        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from core.evolution.agent_cfg import EvoPPORunnerCfg
        from core.evolution.train import evaluate, export_policy
        from core.sim.locomotion_env import EvoLocomotionEnv

        env = RslRlVecEnvWrapper(EvoLocomotionEnv(env_cfg), clip_actions=1.0)
        base = env.unwrapped
        device = base.device

        def make_runner(log_dir=None):
            cfg = handle_deprecated_rsl_rl_cfg(EvoPPORunnerCfg(), metadata.version("rsl-rl-lib"))
            cfg.device = device
            return OnPolicyRunner(env, cfg.to_dict(), log_dir=log_dir, device=device)

        # ---------------------------------------------------------------- teachers, routed by terrain type
        gen = env_cfg.terrain.terrain_generator
        names = list(gen.sub_terrains)
        cum = torch.cumsum(torch.tensor([gen.sub_terrains[n].proportion for n in names]), 0)
        cum = cum / cum[-1]
        col_type = torch.tensor(
            [int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0]) for c in range(gen.num_cols)], device=device
        )
        teacher_policies, type_to_teacher = [], torch.full((len(names),), len(teachers), device=device, dtype=torch.long)
        for k, (tname, path, types) in enumerate(teachers):
            runner = make_runner()
            runner.load(str(path), load_cfg={"actor": True})
            teacher_policies.append(runner.get_inference_policy(device=device))
            for t in types:
                if t not in names:
                    raise SystemExit(f"teacher {tname}: unknown terrain type {t!r}, choose from {names}")
                type_to_teacher[names.index(t)] = k
            print(f"[distill] teacher {tname}: {path} for {types}")
        fallback = make_runner()
        fallback.load(str(args.fallback), load_cfg={"actor": True})
        teacher_policies.append(fallback.get_inference_policy(device=device))
        print(f"[distill] fallback teacher: {args.fallback}")

        # ---------------------------------------------------------------- student (a fresh standard policy)
        student_runner = make_runner(log_dir=str(args.out))
        student = student_runner.alg.get_policy()
        # start with the fallback teacher's exploration noise, not the initial 0.6: matters for a later RL polish
        student.distribution.load_state_dict(teacher_policies[-1].distribution.state_dict())
        student.train()
        optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.lr)

        base.spread_over_terrain_levels()  # every difficulty level, curriculum frozen
        obs, _ = env.reset()
        log = []
        for it in range(args.iterations):
            # teachers drive with probability beta (fading to 0), otherwise the student
            beta = max(0.0, 1.0 - it / max(1.0, args.teacher_drive_until * args.iterations))
            buf_obs, buf_act = [], []
            with torch.inference_mode():
                for _ in range(args.steps_per_iteration):
                    route = type_to_teacher[col_type[base.terrain.terrain_types]]
                    target = torch.zeros(base.num_envs, env.num_actions, device=device)
                    for k, pol in enumerate(teacher_policies):
                        sel = route == k
                        if torch.any(sel):
                            target[sel] = pol(obs)[sel]
                    target = target.clamp(-1.0, 1.0)
                    student_act = student(obs).clamp(-1.0, 1.0)
                    drive = torch.rand(base.num_envs, 1, device=device) < beta
                    buf_obs.append(obs.clone())
                    buf_act.append(target.clone())
                    student.update_normalization(obs)
                    obs, _, _, _ = env.step(torch.where(drive, target, student_act))
            data_obs = torch.cat(buf_obs, dim=0)
            data_act = torch.cat(buf_act, dim=0)
            n = data_act.shape[0]
            batch = n // args.minibatches
            losses = []
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=device)
                for b in range(args.minibatches):
                    idx = perm[b * batch : (b + 1) * batch]
                    pred = student(data_obs[idx])
                    loss = torch.nn.functional.mse_loss(pred, data_act[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss))
            mean_loss = sum(losses) / len(losses)
            log.append({"iteration": it, "loss": mean_loss, "beta": beta})
            if it % 10 == 0 or it == args.iterations - 1:
                print(f"[distill] it {it:4d}/{args.iterations}  imitation loss {mean_loss:.4f}  teacher drives {beta:.2f}",
                      flush=True)

        student_runner.save(str(args.out / "model_final.pt"))
        train_time = time.time() - started
        student.eval()
        policy = student_runner.get_inference_policy(device=device)
        stats = evaluate(env, policy, int(base.max_episode_length))
        export_policy(student_runner, args.out)
        result = {
            "name": genome.name,
            "genome": str(genome_path),
            "topology": genome.topology_key(),
            "num_servos": len(spec.joints),
            "mass_kg": round(spec.total_mass, 4),
            "method": "distillation",
            "teachers": {t[0]: {"checkpoint": str(t[1]), "types": t[2]} for t in teachers},
            "fallback": str(args.fallback),
            "iterations": args.iterations,
            "num_envs": args.num_envs,
            "terrain": "rough",
            "sensors": args.sensors,
            "obs_history": args.obs_history,
            "train_time_s": round(train_time, 1),
            "checkpoint": str((args.out / "model_final.pt").resolve()),
            "loss_curve": log[:: max(1, len(log) // 50)],
            **stats,
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(f"[distill] student: walk score {stats['walk_score']:.3f}, progress {stats['progress_ratio']:.2f}, "
              f"falls {stats['fall_rate']:.2f}")
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
