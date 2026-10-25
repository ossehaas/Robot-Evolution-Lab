"""One dog that walks and gets up after a fall: taught by two teachers, chosen by its posture (DAgger).

    lab -m core.evolution.distill_recover --walker runs/exp2_dog/11_imitation/amp_w6m_slow_s1 \\
        --getup runs/exp2_dog/15_getup_design/getup_abd29_head --out runs/exp2_dog/16_recover/distill

How it works
    * recover mode: a fall does not end the episode; topple shoves knock the dogs over regularly
    * while the dog is upright the walking teacher says what to do; from the moment it is fallen (tilted > 35 deg or
      the body on the ground) until it stands again the get-up teacher does
    * the student drives all joints incl. neck, head and tail (the get-up trick needs them; the walker held them at
      rest), at the get-up teacher's full joint range: the walker's actions are rescaled from its own action_scale
    * each teacher sees what it was trained with: the walker its last actions in its own scale, the get-up teacher a
      zero (stand still) command
    * early on the teachers drive part of the time, later the student drives and is corrected wherever it ends up

Writes model_final.pt (an RSL-RL checkpoint: polish it with core.evolution.train --init_checkpoint), policy.pt /
policy.onnx and result.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPRESSION = ("neck_j0", "head_j1", "tail_j0")


def main(argv=None) -> int:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Teach one dog to walk and to get up after a fall.")
    parser.add_argument("--walker", type=Path, required=True, help="run directory of the walking teacher")
    parser.add_argument("--getup", type=Path, required=True, help="run directory of the get-up teacher")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--steps_per_iteration", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher_drive_until", type=float, default=0.3)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--terrain", default="dog_rough")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_override", action="append", default=[], help="extra environment setting key=value")
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True
    args.out.mkdir(parents=True, exist_ok=True)

    import ast

    from core.evolution.train import build_robot_files
    from core.morphology import Genome, build_robot
    from core.sim.locomotion_env_cfg import make_env_cfg

    walker_res = json.loads((args.walker / "result.json").read_text())
    getup_res = json.loads((args.getup / "result.json").read_text())
    walker_scale = float(walker_res.get("env_overrides", {}).get("action_scale", 0.5))
    genome_path = (args.getup / "genome.json").resolve()  # the body the get-up teacher learned on
    genome = Genome.load(genome_path)
    genome.save(args.out / "genome.json")
    usd = build_robot_files(genome_path, args.out / "robot")
    spec = build_robot(genome)
    env_cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, True, task="command",
                           terrain=args.terrain, sensors=walker_res["sensors"], obs_history=walker_res["obs_history"],
                           height_scan=True)
    # the walker's recipe (gait, commands, rewards; no style judge here), then recover mode at full range
    overrides = {k: v for k, v in walker_res.get("env_overrides", {}).items() if not k.startswith("amp_")}
    overrides.update({"recover": True, "policy_drives_expression": True, "self_collisions": True, "action_scale": 1.0,
                      "heading_hold_gain": 2.0, "topple_push": (0.6, 1.2), "topple_interval": (4.0, 8.0)})
    for item in args.env_override:
        key, value = item.split("=", 1)
        overrides[key] = ast.literal_eval(value)
    for key, value in overrides.items():  # set directly: recover and topple_push are training-only for apply_overrides
        if not hasattr(env_cfg, key):
            raise SystemExit(f"unknown environment setting {key!r}")
        setattr(env_cfg, key, tuple(value) if isinstance(value, list) else value)
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
        from core.evolution.train import export_policy
        from core.sim.locomotion_env import EvoLocomotionEnv

        env = RslRlVecEnvWrapper(EvoLocomotionEnv(env_cfg), clip_actions=1.0)
        base = env.unwrapped
        device = base.device
        walker = torch.jit.load(str(args.walker / "policy.pt"), map_location=device).eval()
        getup = torch.jit.load(str(args.getup / "policy.pt"), map_location=device).eval()

        # observation layout: obs_history frames of [gravity 3, gyro 3, accel 3, joint angles n, last actions n,
        # clock 2, command 3], then the height scan
        names = list(base.obs_joint_names)
        n = len(names)
        frame = 9 + 2 * n + 2 + 3
        hist = env_cfg.obs_history
        if frame * hist + env_cfg.height_scan_points != env_cfg.observation_space:
            raise SystemExit(f"unexpected observation layout: {frame} x {hist} + scan != {env_cfg.observation_space}")
        last_act = torch.cat([torch.arange(k * frame + 9 + n, k * frame + 9 + 2 * n) for k in range(hist)]).to(device)
        command = torch.cat([torch.arange(k * frame + frame - 3, k * frame + frame) for k in range(hist)]).to(device)
        expr_pos = torch.tensor([i for i, j in enumerate(names) if j in EXPRESSION], device=device)
        leg_pos = torch.tensor([i for i, j in enumerate(names) if j not in EXPRESSION], device=device)
        leg_last_act = last_act.view(hist, n)[:, leg_pos].flatten()
        print(f"[recover] student drives {n} joints; walker action_scale {walker_scale} -> 1.0; teachers: "
              f"{args.walker.name} (upright), {args.getup.name} (fallen until standing)", flush=True)

        def teacher_actions(obs: torch.Tensor, recovering: torch.Tensor) -> torch.Tensor:
            w_obs = obs.clone()
            w_obs[:, leg_last_act] = w_obs[:, leg_last_act] / walker_scale  # back to the walker's own scale
            target = torch.zeros(obs.shape[0], n, device=device)
            target[:, leg_pos] = walker(w_obs).clamp(-1.0, 1.0) * walker_scale  # neck, head, tail: at rest
            g_obs = obs.clone()
            g_obs[:, command] = 0.0  # the get-up teacher only knows "stand still"
            return torch.where(recovering.unsqueeze(-1), getup(g_obs).clamp(-1.0, 1.0), target)

        cfg = handle_deprecated_rsl_rl_cfg(EvoPPORunnerCfg(), metadata.version("rsl-rl-lib"))
        cfg.device = device
        runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=str(args.out), device=device)
        student = runner.alg.get_policy()
        for name, p in student.named_parameters():
            if "std" in name:
                p.data.fill_(0.3)  # moderate exploration noise for the RL polish that follows
        student.train()
        optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.lr)

        base.spread_over_terrain_levels()
        obs, _ = env.reset()
        recovering = torch.zeros(base.num_envs, dtype=torch.bool, device=device)
        log = []
        for it in range(args.iterations):
            beta = max(0.0, 1.0 - it / max(1.0, args.teacher_drive_until * args.iterations))
            buf_obs, buf_act = [], []
            share = 0.0
            with torch.inference_mode():
                for _ in range(args.steps_per_iteration):
                    policy_obs = obs["policy"]
                    target = teacher_actions(policy_obs, recovering)
                    student_act = student(obs).clamp(-1.0, 1.0)
                    drive = torch.rand(base.num_envs, 1, device=device) < beta
                    buf_obs.append(obs.clone())
                    buf_act.append(target.clone())
                    student.update_normalization(obs)
                    obs, _, dones, _ = env.step(torch.where(drive, target, student_act))
                    # the get-up teacher takes over at a fall and hands back once the dog stands again
                    recovering = (recovering | base.fallen) & ~base.getup_standing & ~dones.bool()
                    share += float(recovering.float().mean()) / args.steps_per_iteration
            data_act = torch.cat(buf_act, dim=0)
            data_obs = torch.cat(buf_obs, dim=0)
            m = data_act.shape[0]
            batch = m // args.minibatches
            losses = []
            for _ in range(args.epochs):
                perm = torch.randperm(m, device=device)
                for b in range(args.minibatches):
                    idx = perm[b * batch : (b + 1) * batch]
                    loss = torch.nn.functional.mse_loss(student(data_obs[idx]), data_act[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss))
            entry = {"iteration": it, "loss": sum(losses) / len(losses), "teacher_drives": beta, "recovering": share}
            log.append(entry)
            if it % 10 == 0 or it == args.iterations - 1:
                print(f"[recover] it {it:4d}/{args.iterations}  imitation loss {entry['loss']:.4f}  teacher drives "
                      f"{beta:.2f}  share recovering {share:.2f}", flush=True)
        runner.save(str(args.out / "model_final.pt"))
        student.eval()
        export_policy(runner, args.out)
        result = {
            "name": genome.name, "genome": str(genome_path), "method": "distill_recover",
            "walker": str(args.walker), "getup": str(args.getup), "iterations": args.iterations,
            "num_envs": args.num_envs, "task": "command", "terrain_arg": args.terrain,
            "sensors": walker_res["sensors"], "obs_history": walker_res["obs_history"], "height_scan": True,
            "env_overrides": overrides, "train_time_s": round(time.time() - started, 1),
            "loss_curve": log[:: max(1, len(log) // 50)],
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2, default=list))
        print(f"[recover] done: final imitation loss {log[-1]['loss']:.4f}")
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
