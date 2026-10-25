"""Train (and score) one robot: genome -> build -> PPO in thousands of parallel copies -> fitness.

    python -m core.evolution.train --genome prototypes/crawler_q4/genome.json --out runs/crawler_q4
    python -m core.evolution.train --genome ... --out ... --iterations 500 --num_envs 4096 --physics newton
    python -m core.evolution.train --genome ... --out ... --viz kit          # watch it learn (slower)
    python -m core.evolution.train --genome ... --out ... --task command     # steerable: follow velocity commands
    python -m core.evolution.train --genome ... --out ... --task command --terrain rough   # blind, rough terrain

Writes into --out:
    robot/                  generated USD, URDF, STL parts, BOM, preview.png (see core.morphology)
    model_<iter>.pt         RSL-RL checkpoints (model_final.pt at the end), TensorBoard logs
    policy.pt / policy.onnx exported actor for deployment
    result.json             fitness and statistics (read by the evolution loop)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def build_robot_files(genome_path: Path, robot_dir: Path) -> Path:
    """Build the robot in a separate process: the standalone USD library must not load next to Isaac Sim's."""
    usd = next(robot_dir.glob("*.usda"), None) if robot_dir.exists() else None
    if usd is None:
        subprocess.run(
            [sys.executable, "-m", "core.morphology", "build", str(genome_path), "--out", str(robot_dir)],
            cwd=REPO,
            check=True,
        )
        usd = next(robot_dir.glob("*.usda"))
    return usd


def parse_args(argv=None):
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Train one generated robot with PPO.")
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=("forward", "command", "goal", "jump"), default="forward")
    parser.add_argument("--terrain", choices=("flat", "rough", "obstacles", "obstacles_rough", "enclosure", "enclosure_pocket", "cup", "dog_rough"), default="flat",
                        help="rough requires --task command, obstacles requires --task goal")
    parser.add_argument("--terrain_types", default=None,
                        help="rough terrain: comma-separated subset, e.g. stairs_climb,stairs_descend (a specialist)")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--physics", dest="evo_physics", choices=("physx", "newton"), default="physx")
    parser.add_argument("--servo_model", choices=("dc_motor", "implicit"), default="dc_motor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="warm start (same topology only)")
    parser.add_argument("--no_dr", action="store_true", help="disable domain randomization")
    parser.add_argument("--sensors", choices=("ideal", "imu", "imu_flow"), default="ideal",
                        help="body senses: ideal, realistic IMU, or IMU + optical flow (see roadmap M2b)")
    parser.add_argument("--obs_history", type=int, default=1, help="sensor frames the policy remembers")
    parser.add_argument("--height_scan", action="store_true", help="give the policy the terrain heights around it")
    parser.add_argument("--foot_clearance", action="store_true", help="reward lifting swinging feet (rough terrain)")
    parser.add_argument("--terrain_map", choices=("", "depth", "tof"), default="",
                        help="with --height_scan (dog): the policy reads the scan from a terrain map built by a chest "
                             "sensor (depth camera or 8x8 time-of-flight) instead of the perfect scan")
    parser.add_argument(
        "--pretrain_flat",
        type=int,
        default=0,
        help="rough terrain without --init_checkpoint: first learn to walk on flat ground for this many iterations",
    )
    parser.add_argument("--skip_eval", action="store_true", help="only train and save (used for the flat stage)")
    parser.add_argument("--symmetry", action="store_true",
                        help="also train on left-right mirrored samples (symmetric gaits; command task)")
    parser.add_argument("--memory", choices=("none", "gru", "lstm"), default="none",
                        help="recurrent policy: remembers the whole episode (use with --obs_history 1)")
    parser.add_argument("--low_level_policy", type=Path, default=None,
                        help="goal task: train only a navigator that steers this frozen walking policy (policy.pt)")
    parser.add_argument("--init_action_scale", type=float, default=None,
                        help="with --init_checkpoint: the action_scale it was trained with. If the run uses another one, "
                             "the actor's output is rescaled so the robot starts out moving exactly as before")
    parser.add_argument("--bc_walker", type=Path, default=None,
                        help="with --bc_getup: keep copying these teachers during RL, chosen by posture "
                             "(core/evolution/gated_bc.py)")
    parser.add_argument("--bc_getup", type=Path, default=None)
    parser.add_argument("--critic_warmup", type=int, default=0,
                        help="with --init_checkpoint: first train only the critic for this many iterations (actor frozen)")
    parser.add_argument("--env_override", action="append", default=[],
                        help="set an environment config value, e.g. --env_override collision_weight=6.0 (repeatable)")
    add_launcher_args(parser)
    return parser.parse_args(argv)


def evaluate(env, policy, max_steps: int) -> dict:
    """Run every env for one full episode with the deterministic policy.

    Always reports the distance travelled along +x. In the command task every command asks for real movement
    (10-30 cm/s in a random direction plus a turn rate; no standing still), and it reports:
        walk_score      fitness, 0..1: per step, the share of the commanded speed delivered along the commanded
                        direction (clipped to 0..1), discounted for sideways drift and multiplied by
                        (0.5 + 0.5 * turn-rate accuracy). Standing still scores 0.
        tracking_score  mean of the velocity and turn-rate tracking kernels, 0..1 (the training reward's view)
        progress_ratio  share of the commanded speed achieved along the commanded direction (0 = not moving)
        lin_vel_error_m_s, yaw_rate_error_rad_s
    On rough terrain robots are spread evenly over all difficulty rows and scores are also given per row/type.
    """
    import torch

    base = env.unwrapped
    command_task = base.cfg.task == "command"
    base.prepare_evaluation()
    obs, _ = env.reset()
    base.finished_episodes.zero_()
    done_once = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    distance = torch.zeros(base.num_envs, device=base.device)
    fell = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    goal_task = base.cfg.task == "goal"
    jump = torch.zeros(base.num_envs, device=base.device)
    rise = torch.zeros_like(jump)
    air = torch.zeros_like(jump)
    goals = torch.zeros(base.num_envs, device=base.device)
    collision = torch.zeros(base.num_envs, device=base.device)
    seconds_done = torch.zeros(base.num_envs, device=base.device)
    enclosure = base.cfg.terrain_kind == "enclosure"
    end_xy = torch.zeros(base.num_envs, 2, device=base.device)
    start_xy = torch.zeros_like(end_xy)
    goal_xy = torch.zeros_like(end_xy)
    unreachable = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    lin_sum = torch.zeros(base.num_envs, device=base.device)
    ang_sum = torch.zeros_like(lin_sum)
    score_sum = torch.zeros_like(lin_sum)
    progress_sum = torch.zeros_like(lin_sum)
    walk_sum = torch.zeros_like(lin_sum)
    power_sum = torch.zeros_like(lin_sum)
    speed_sum = torch.zeros_like(lin_sum)
    steps = torch.zeros_like(lin_sum)
    style_sum = torch.zeros_like(lin_sum)
    climb = torch.zeros_like(lin_sum)  # largest height change from the start [m]: steps actually climbed / descended
    if base.amp is not None:
        base.amp.frozen = True  # score against the discriminator as trained
    with torch.inference_mode():
        for _ in range(max_steps + 2):
            if command_task:
                # measured before stepping, against the command the policy is about to act on
                lin, ang = base.tracking_errors()
                active = (~done_once).float()
                lin_sum += lin * active
                ang_sum += ang * active
                score_sum += 0.5 * active * (
                    torch.exp(-torch.square(lin / base.cfg.tracking_lin_sigma))
                    + torch.exp(-torch.square(ang / base.cfg.tracking_ang_sigma))
                )
                cmd = base.commands[:, :2]
                vel = base.robot.data.root_lin_vel_b.torch[:, :2]
                along = torch.sum(cmd * vel, dim=-1) / torch.sum(cmd * cmd, dim=-1).clamp(min=1e-4)
                progress_sum += active * along.clamp(-1.0, 2.0)
                # sideways drift relative to the commanded direction, in m/s
                unit = cmd / torch.linalg.norm(cmd, dim=-1, keepdim=True).clamp(min=1e-4)
                drift = torch.abs(unit[:, 0] * vel[:, 1] - unit[:, 1] * vel[:, 0])
                turn_ok = torch.exp(-torch.square(ang / base.cfg.tracking_ang_sigma))
                walk = along.clamp(0.0, 1.0) * torch.exp(-torch.square(drift / 0.1)) * (0.5 + 0.5 * turn_ok)
                walk_sum += active * walk
                d = base.robot.data
                power_sum += active * torch.sum(torch.abs(d.applied_torque.torch * d.joint_vel.torch), dim=-1)
                speed_sum += active * torch.linalg.norm(d.root_lin_vel_w.torch[:, :2], dim=-1)
                climb = torch.maximum(climb, active * (d.root_pos_w.torch[:, 2] - base.start_pos[:, 2]).abs())
                steps += active
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)  # recurrent policies: forget the finished episode
            if base.amp is not None:
                style_sum += (~done_once).float() * base.amp_style
            newly = (base.finished_episodes >= 1) & ~done_once
            if base.cfg.task == "jump":
                jump[newly] = base.last_episode_jump[newly]
                rise[newly] = base.last_episode_rise[newly]
                air[newly] = base.last_episode_air[newly]
            distance[newly] = base.last_episode_distance[newly]
            fell[newly] = base.last_episode_fell[newly]
            if goal_task:
                goals[newly] = base.last_episode_goals[newly]
                collision[newly] = base.last_episode_collision[newly]
                seconds_done[newly] = base.last_episode_seconds[newly]
                if enclosure:
                    end_xy[newly] = base.last_episode_end_xy[newly]
                    start_xy[newly] = base.last_episode_start_xy[newly]
                    goal_xy[newly] = base.last_episode_goal_xy[newly]
                    unreachable[newly] = base.last_episode_unreachable[newly]
            done_once |= newly
            if bool(done_once.all()):
                break
    seconds = base.max_episode_length_s
    stats = {
        "task": base.cfg.task,
        "distance_m": float(distance[done_once].mean()),
        "distance_std_m": float(distance[done_once].std()) if int(done_once.sum()) > 1 else 0.0,
        "speed_m_s": float(distance[done_once].mean()) / seconds,
        "fall_rate": float(fell[done_once].float().mean()),
        "episodes": int(done_once.sum()),
    }
    if command_task:
        steps = steps.clamp(min=1.0)
        stats["lin_vel_error_m_s"] = float((lin_sum / steps).mean())
        stats["yaw_rate_error_rad_s"] = float((ang_sum / steps).mean())
        per_env = score_sum / steps
        stats["tracking_score"] = float(per_env.mean())
        stats["progress_ratio"] = float((progress_sum / steps).mean())
        walk_env = walk_sum / steps
        stats["walk_score"] = float(walk_env.mean())
        if base.amp is not None:
            stats["style_score"] = float((style_sum / steps).mean())  # 0..1: how dog-like (AMP discriminator)
        if base.cfg.robot_mass > 0:
            # cost of transport: mechanical power / (weight * speed); dimensionless (walking dogs ~0.2-0.5)
            ok = done_once & ~fell
            power, speed = (power_sum / steps)[ok].mean(), (speed_sum / steps)[ok].mean()
            stats["cost_of_transport"] = float(power / (base.cfg.robot_mass * 9.81 * speed.clamp(min=1e-3)))
            stats["mechanical_power_w"] = float(power)
        if base.cfg.terrain_kind == "rough":
            levels = base.terrain.terrain_levels
            stats["level_scores"] = [
                round(float(walk_env[levels == lv].mean()), 3) for lv in range(int(levels.max()) + 1)
            ]
            stats["level_fall_rates"] = [
                round(float(fell[levels == lv].float().mean()), 3) for lv in range(int(levels.max()) + 1)
            ]
            # score per terrain type and level: columns of the terrain grid hold the sub-terrain types
            gen = base.cfg.terrain.terrain_generator
            names = list(gen.sub_terrains)
            cum = torch.cumsum(torch.tensor([gen.sub_terrains[n].proportion for n in names]), 0)
            cum = cum / cum[-1]
            col_type = [int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0]) for c in range(gen.num_cols)]
            types = torch.tensor(col_type, device=base.device)[base.terrain.terrain_types]
            stats["type_fall_rates"] = {
                name: [round(float(fell[(types == k) & (levels == lv)].float().mean()), 3) for lv in range(int(levels.max()) + 1)]
                for k, name in enumerate(names)
            }
            # how far up or down the dog really got (few falls alone also fits a dog that never sets foot on the stairs)
            stats["type_level_climb_m"] = {
                name: [round(float(climb[(types == k) & (levels == lv)].mean()), 3) for lv in range(int(levels.max()) + 1)]
                for k, name in enumerate(names)
            }
            stats["type_level_scores"] = {
                name: [
                    round(float(walk_env[(types == k) & (levels == lv)].mean()), 3)
                    for lv in range(int(levels.max()) + 1)
                ]
                for k, name in enumerate(names)
            }
    if goal_task:
        per_min = goals / seconds_done.clamp(min=1e-3) * 60.0
        stats["goals_per_minute"] = float(per_min[done_once].mean())
        stats["goal_success_rate"] = float((goals[done_once] >= 1).float().mean())
        stats["collision_fraction"] = float(collision[done_once].mean())
        if base.cfg.terrain_kind == "obstacles":
            levels = base.terrain.terrain_levels
            stats["level_goals_per_minute"] = [
                round(float(per_min[done_once & (levels == lv)].mean()), 2) for lv in range(int(levels.max()) + 1)
            ]
            stats["level_collision_fraction"] = [
                round(float(collision[done_once & (levels == lv)].mean()), 3) for lv in range(int(levels.max()) + 1)
            ]
    if goal_task and enclosure and base.cfg.cup:
        stats.update(cup_stats(done_once, fell, goals, end_xy, unreachable, base.terrain.terrain_levels,
                               base.cfg.terrain.terrain_generator.sub_terrains["ring"]))
    elif goal_task and enclosure:
        stats.update(enclosure_stats(done_once, fell, goals, end_xy, start_xy, goal_xy, unreachable,
                                     base.cfg.terrain.terrain_generator.sub_terrains["ring"].radius, base.cfg.pocket))
    if base.cfg.task == "jump":
        # jump height = how high the LOWEST paw got (0 without a real jump); rise = how far the body got above standing
        ok = done_once
        stats["jump_height_m_mean"] = float(jump[ok].mean())
        stats["jump_height_m_p90"] = float(torch.quantile(jump[ok], 0.9))
        stats["jump_height_m_max"] = float(jump[ok].max())
        stats["jump_share"] = float((jump[ok] > 0.005).float().mean())
        stats["body_rise_m_mean"] = float(rise[ok].mean())
        stats["airborne_s_mean"] = float(air[ok].mean())
    stats["terrain"] = base.cfg.terrain_kind
    return stats


def cup_stats(done, fell, goals, end_xy, inside, levels, ring_cfg) -> dict:
    """Detour test: goal inside an open ring whose closed side faces the robot's start.
        inside_success_rate      reached the goal inside the cup (within the 30 s episode)
        stuck_at_wall_rate       did not, and ended against the closed side (the local minimum)
        level_*                  the same per arc length (cup_arcs, degrees of wall)
    Robots with a goal outside the cup are scored as usual (outside_goal_success_rate)."""
    import torch

    sel = done & inside
    reached = goals >= 1
    dist = torch.linalg.norm(end_xy, dim=-1)
    # stuck: outside the wall on the closed side (within 25 cm of it, left of the cup centre)
    stuck = ~reached & (dist > ring_cfg.radius - 0.02) & (dist < ring_cfg.radius + 0.25) & (end_xy[:, 0] < 0)
    arcs = list(ring_cfg.cup_arcs)
    out = {
        "inside_success_rate": float(reached[sel].float().mean()),
        "stuck_at_wall_rate": float(stuck[sel].float().mean()),
        "outside_goal_success_rate": float(reached[done & ~inside].float().mean()),
        "cup_arcs_deg": arcs,
        "level_inside_success_rate": [round(float(reached[sel & (levels == lv)].float().mean()), 3) for lv in range(len(arcs))],
        "level_stuck_at_wall_rate": [round(float(stuck[sel & (levels == lv)].float().mean()), 3) for lv in range(len(arcs))],
    }
    return out


def enclosure_stats(done, fell, goals, end_xy, start_xy, goal_xy, unreachable, radius: float, pocket=None) -> dict:
    """Unreachable-goal test: did the robot end at the point outside the ring wall closest to the goal inside it?

    For a goal at offset o from the ring centre, the closest reachable spot is on the outside of the wall in the
    goal's direction, at distance (radius + gap - o), where gap is how close the robot's centre gets to the wall.
        angle_error_deg     angle (seen from the ring centre) between where the robot ended and the goal direction
        success_rate        ended at the wall (gap < 15 cm) within 30 degrees of the right direction
        far_side_*          the same for robots that started on the far side (> 120 degrees away from the goal side)
        within_5cm_rate     ended within 5 cm of the closest reachable distance (any shape)
    With a pocket (a slot into the ring along +x) the slot bottom is a second candidate; when the outer wall is
    closer, ending in the slot means the robot was caught in the local minimum (trapped_rate).
    Reachable goals (outside the ring) are scored as usual: the share of robots that reached one."""
    import math

    import torch

    sel = done & unreachable & ~fell
    out = {"reachable_goal_success_rate": float((goals[done & ~unreachable] >= 1).float().mean())}
    if int(sel.sum()) == 0:
        return out

    def angle_between(a, b):
        cos = torch.sum(a * b, -1) / (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)).clamp(min=1e-6)
        return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))

    end, goal, start = end_xy[sel], goal_xy[sel], start_xy[sel]
    err = angle_between(end, goal)
    start_err = angle_between(start, goal)
    gap = torch.linalg.norm(end, dim=-1) - radius
    final = torch.linalg.norm(end - goal, dim=-1)
    in_slot = torch.zeros_like(gap, dtype=torch.bool)
    if pocket is not None:
        _, width, depth = pocket
        in_slot = (end[:, 0] > radius - depth - 0.02) & (end[:, 1].abs() < width / 2 + 0.02) & (gap < 0.0)
    outside = ~in_slot & (gap > -0.01)
    # how close this robot's centre gets to the outer wall
    min_gap = float(torch.quantile(gap[outside], 0.05)) if bool(outside.any()) else 0.1
    offset = torch.linalg.norm(goal, dim=-1)
    outer = radius + min_gap - offset
    worst = radius + min_gap + offset  # same gap, opposite side of the ring
    ideal = outer
    pocket_better = torch.zeros_like(in_slot)
    if pocket is not None:
        # outer-wall spot straight out from the goal does not exist where the slot cuts the ring
        cut = (goal[:, 0] > 0) & (goal[:, 1].abs() / offset.clamp(min=1e-6) * radius < width / 2 + 0.06)
        outer = torch.where(cut, torch.full_like(outer, float("inf")), outer)
        slot_bottom = torch.stack((torch.full_like(offset, radius - depth + min_gap), torch.zeros_like(offset)), -1)
        slot = torch.linalg.norm(slot_bottom - goal, dim=-1)
        ideal = torch.minimum(outer, slot)
        pocket_better = slot < outer - 0.05
        worst = torch.maximum(worst, slot)
    ok = (err < 30.0) & (gap < 0.15)
    far = start_err > 120.0
    out.update({
        "unreachable_episodes": int(sel.sum()),
        "unreachable_fall_rate": float(fell[done & unreachable].float().mean()),
        "angle_error_deg_mean": float(err.mean()),
        "angle_error_deg_median": float(err.median()),
        "start_angle_error_deg_mean": float(start_err.mean()),
        "success_rate": float(ok.float().mean()),
        "far_side_success_rate": float(ok[far].float().mean()) if bool(far.any()) else math.nan,
        "near_side_success_rate": float(ok[start_err < 60.0].float().mean()) if bool((start_err < 60.0).any()) else math.nan,
        "final_distance_m": float(final.mean()),
        "ideal_distance_m": float(ideal.mean()),
        "worst_wall_distance_m": float(worst.mean()),
        "wall_gap_m_median": float(gap.median()),
        # 1 = ended at the closest reachable point, 0 = at the worst point along the wall (or farther away)
        "closeness_score": float(((worst - final) / (worst - ideal).clamp(min=1e-3)).clamp(0.0, 1.0).mean()),
        "within_5cm_rate": float((final <= ideal + 0.05).float().mean()),
        "min_wall_gap_m": min_gap,
    })
    if pocket is not None:
        trap = ~pocket_better
        out.update({
            "pocket_best_share": float(pocket_better.float().mean()),
            "trapped_rate": float(in_slot[trap].float().mean()) if bool(trap.any()) else math.nan,
            "within_5cm_rate_outer_best": float((final <= ideal + 0.05)[trap].float().mean()) if bool(trap.any()) else math.nan,
            "within_5cm_rate_pocket_best": float((final <= ideal + 0.05)[pocket_better].float().mean())
            if bool(pocket_better.any()) else math.nan,
        })
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    genome_path = args.genome.resolve()
    usd_path = build_robot_files(genome_path, args.out / "robot")

    from core.morphology import Genome, build_robot
    from core.sim.locomotion_env_cfg import make_env_cfg

    genome = Genome.load(genome_path)
    if genome_path != (args.out / "genome.json").resolve():
        genome.save(args.out / "genome.json")
    spec = build_robot(genome)

    # stage 1 (own process, before this one starts the simulator): learn to walk on flat ground
    if args.pretrain_flat > 0 and args.terrain == "rough" and args.init_checkpoint is None:
        stage_dir = args.out / "pretrain_flat"
        cmd = [
            sys.executable, "-m", "core.evolution.train", "--genome", str(genome_path), "--out", str(stage_dir),
            "--task", "command", "--terrain", "flat", "--iterations", str(args.pretrain_flat),
            "--num_envs", str(args.num_envs), "--seed", str(args.seed), "--servo_model", args.servo_model,
            "--sensors", args.sensors, "--obs_history", str(args.obs_history), "--skip_eval",
            *(["--height_scan"] if args.height_scan else []),
        ]  # fmt: skip
        print(f"[evolution] stage 1: {args.pretrain_flat} iterations on flat ground", flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
        args.init_checkpoint = stage_dir / "model_final.pt"
    env_cfg = make_env_cfg(
        spec, str(usd_path.resolve()).replace("\\", "/"), args.num_envs, not args.no_dr, physics=args.evo_physics,
        servo_model=args.servo_model, task=args.task, terrain=args.terrain,
        sensors=args.sensors, obs_history=args.obs_history,
        terrain_types=args.terrain_types.split(",") if args.terrain_types else None,
        height_scan=args.height_scan, foot_clearance=args.foot_clearance,
        low_level_policy=str(args.low_level_policy.resolve()) if args.low_level_policy else None,
        terrain_map=args.terrain_map,
    )
    env_cfg.seed = args.seed
    import ast

    overrides = {}
    for item in args.env_override:
        key, value = item.split("=", 1)
        if not hasattr(env_cfg, key):
            raise SystemExit(f"--env_override: unknown environment setting {key!r}")
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value  # plain text, e.g. goal_signal=camera (cmd strips the quotes)
        setattr(env_cfg, key, parsed)
        overrides[key] = parsed

    from isaaclab.app import launch_simulation

    started = time.time()
    with launch_simulation(env_cfg, args):
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from core.evolution.agent_cfg import EvoPPORunnerCfg, apply_memory
        from core.sim.locomotion_env import EvoLocomotionEnv

        import importlib.metadata as metadata

        agent_cfg = apply_memory(EvoPPORunnerCfg(), args.memory)
        if args.symmetry:
            from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

            from core.sim.symmetry import mirror_augment

            agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=True,
                                                                data_augmentation_func=mirror_augment)
        agent_cfg.max_iterations = args.iterations
        agent_cfg.seed = args.seed
        agent_cfg.device = env_cfg.sim.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        if env_cfg.action_limit_weight > 0:
            agent_cfg.clip_actions = None  # the environment clips and penalizes what lies beyond the limit
        render_mode = "rgb_array" if getattr(args, "video", False) else None
        env = EvoLocomotionEnv(env_cfg, render_mode=render_mode)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        agent_dict = agent_cfg.to_dict()
        if args.bc_walker is not None and args.bc_getup is not None:
            import torch

            from core.evolution import gated_bc

            dev = env.unwrapped.device
            walker_res = json.loads((args.bc_walker / "result.json").read_text())
            gated_bc.CONFIG.update(
                walker=torch.jit.load(str(args.bc_walker / "policy.pt"), map_location=dev).eval(),
                getup=torch.jit.load(str(args.bc_getup / "policy.pt"), map_location=dev).eval(),
                walker_scale=float(walker_res.get("env_overrides", {}).get("action_scale", 0.5)),
                student_scale=float(env_cfg.action_scale), obs_joint_names=list(env.unwrapped.obs_joint_names),
                obs_history=env_cfg.obs_history, scan_points=env_cfg.height_scan_points)
            agent_dict["algorithm"]["class_name"] = "core.evolution.gated_bc:GatedBCPPO"
            print(f"[evolution] gated behaviour cloning: walker {args.bc_walker.name}, get-up {args.bc_getup.name}")
        runner = OnPolicyRunner(env, agent_dict, log_dir=str(args.out), device=agent_cfg.device)
        if args.init_checkpoint is not None and args.init_checkpoint.exists():
            print(f"[evolution] warm start from {args.init_checkpoint}")
            checkpoint = widen_checkpoint(args.init_checkpoint, env.unwrapped, args.out)
            if args.init_action_scale is not None and args.init_action_scale != env_cfg.action_scale:
                checkpoint = rescale_actions(checkpoint, args.init_action_scale / env_cfg.action_scale, args.out)
            # weights only: fresh optimizer and iteration counter
            runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True})
        setup_time = time.time() - started

        if args.critic_warmup > 0 and args.init_checkpoint is not None:
            # a policy that comes from elsewhere (e.g. distillation) has no matching critic yet: let the critic learn
            # first, otherwise its bad value estimates push the actor around
            actor = runner.alg.get_policy()
            for p in actor.parameters():
                p.requires_grad_(False)
            print(f"[evolution] critic warm-up: {args.critic_warmup} iterations with the actor frozen", flush=True)
            runner.learn(num_learning_iterations=args.critic_warmup, init_at_random_ep_len=True)
            for p in actor.parameters():
                p.requires_grad_(True)
            # the adaptive learning rate drifted up while the actor could not change: reset it
            runner.alg.learning_rate = agent_cfg.algorithm.learning_rate
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = agent_cfg.algorithm.learning_rate
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=args.critic_warmup == 0)
        runner.save(str(args.out / "model_final.pt"))
        if env.unwrapped.amp is not None and not env.unwrapped.amp.frozen:
            env.unwrapped.amp.save(args.out / "amp_disc.pt")
        train_time = time.time() - started - setup_time
        if args.skip_eval:
            env.close()
            return 0

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        stats = evaluate(env, policy, int(env.unwrapped.max_episode_length))
        export_policy(runner, args.out)

        result = {
            "name": genome.name,
            "genome": str(genome_path),
            "topology": genome.topology_key(),
            "num_servos": len(spec.joints),
            "mass_kg": round(spec.total_mass, 4),
            "iterations": args.iterations,
            "num_envs": args.num_envs,
            "physics": args.evo_physics,
            "servo_model": args.servo_model,
            "terrain": args.terrain,
            "terrain_types": args.terrain_types,
            "sensors": args.sensors,
            "obs_history": args.obs_history,
            "height_scan": args.height_scan,
            "terrain_map": args.terrain_map,
            "foot_clearance": args.foot_clearance,
            "env_overrides": overrides,
            "low_level_policy": str(args.low_level_policy.resolve()) if args.low_level_policy else None,
            "terrain_arg": args.terrain,
            "critic_warmup": args.critic_warmup,
            "bc_teachers": [str(args.bc_walker), str(args.bc_getup)] if args.bc_walker else None,
            "init_action_scale": args.init_action_scale,
            "memory": args.memory,
            "symmetry": args.symmetry,
            "setup_time_s": round(setup_time, 1),
            "train_time_s": round(train_time, 1),
            "checkpoint": str((args.out / "model_final.pt").resolve()),
            "warm_start": str(args.init_checkpoint) if args.init_checkpoint else None,
            "pretrain_flat_iterations": args.pretrain_flat if "pretrain_flat" in str(args.init_checkpoint) else 0,
            **stats,
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        summary = f"{stats['speed_m_s'] * 100:.1f} cm/s along +x, fall rate {stats['fall_rate']:.2f}"
        if "goals_per_minute" in stats:
            summary = (f"{stats['goals_per_minute']:.1f} goals/min, {100 * stats['goal_success_rate']:.0f}% reach a goal, "
                       f"in collision {100 * stats['collision_fraction']:.1f}% of the time, fall rate {stats['fall_rate']:.2f}")
        if "inside_success_rate" in stats:
            per = ", ".join(f"{a:.0f} deg {100 * s:.0f}%" for a, s in zip(stats["cup_arcs_deg"], stats["level_inside_success_rate"]))
            summary = (f"goal inside the cup reached {100 * stats['inside_success_rate']:.0f}% ({per}); stuck at the closed "
                       f"side {100 * stats['stuck_at_wall_rate']:.0f}%; outside goals {100 * stats['outside_goal_success_rate']:.0f}%")
        if "jump_height_m_mean" in stats:
            summary = (f"jump: lowest paw up to {100 * stats['jump_height_m_mean']:.1f} cm on average "
                       f"(90th percentile {100 * stats['jump_height_m_p90']:.1f}, best {100 * stats['jump_height_m_max']:.1f} cm), "
                       f"{100 * stats['jump_share']:.0f}% of attempts leave the ground, body rises "
                       f"{100 * stats['body_rise_m_mean']:.1f} cm, {stats['airborne_s_mean']:.2f} s airborne per 4 s, "
                       f"fall rate {stats['fall_rate']:.2f}")
        if "angle_error_deg_mean" in stats:
            summary = (f"unreachable goal: {100 * stats['success_rate']:.0f}% end at the closest side "
                       f"(far-side starts {100 * stats['far_side_success_rate']:.0f}%), median angle error "
                       f"{stats['angle_error_deg_median']:.0f} deg, final distance {stats['final_distance_m']:.2f} m "
                       f"(ideal {stats['ideal_distance_m']:.2f}); reachable goals reached "
                       f"{100 * stats['reachable_goal_success_rate']:.0f}%")
        if "tracking_score" in stats:
            summary = (
                f"walk score {stats['walk_score']:.2f}, tracking {stats['tracking_score']:.2f}, "
                f"progress {stats['progress_ratio']:.2f} "
                f"(velocity error {100 * stats['lin_vel_error_m_s']:.1f} cm/s, "
                f"turn-rate error {stats['yaw_rate_error_rad_s']:.2f} rad/s), fall rate {stats['fall_rate']:.2f}"
            )
            if "cost_of_transport" in stats:
                summary += f", cost of transport {stats['cost_of_transport']:.2f} ({stats['mechanical_power_w']:.1f} W)"
        print(f"[evolution] {genome.name}: {summary}")
        env.close()
    return 0


def widen_checkpoint(path: Path, base, out_dir: Path) -> Path:
    """A checkpoint trained without the height scan, loaded into a network with it: the scan inputs get zero weights
    (and a neutral normalizer), so the network starts out acting exactly as before and learns to use the new sense.
    Likewise a checkpoint trained with the perfect scan, loaded into a terrain-map network: the map's heights take the
    scan's place and only the new known flags get zero weights."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if base.height_scanner is None or base.cfg.task != "command":
        return path
    from core.sim.symmetry import known_masks, scan_masks

    scans, knowns = scan_masks(base), known_masks(base)
    changed = False
    for part, key in (("actor_state_dict", "policy"), ("critic_state_dict", "critic")):
        sd = ck[part]
        w = sd["mlp.0.weight"]
        if w.shape[1] == scans[key].numel():
            continue
        mask = next((m for m in (knowns[key], scans[key]) if w.shape[1] == int((~m).sum())), None)
        if mask is None:
            raise RuntimeError(f"cannot widen {part}: {w.shape[1]} inputs, expected {int((~scans[key]).sum())} "
                               f"(no scan) or {int((~knowns[key]).sum())} (perfect scan)")
        new_w = torch.zeros(w.shape[0], mask.numel(), dtype=w.dtype)
        new_w[:, ~mask] = w
        sd["mlp.0.weight"] = new_w
        for stat, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            name = f"obs_normalizer.{stat}"
            old = sd[name]
            new = torch.full((1, mask.numel()), fill, dtype=old.dtype)
            new[:, ~mask] = old
            sd[name] = new
        changed = True
    if not changed:
        return path
    widened = out_dir / "init_widened.pt"
    ck.pop("optimizer_state_dict", None)
    torch.save(ck, widened)
    print(f"[evolution] widened {path.name} with zero-weight height-scan inputs -> {widened.name}")
    return widened


def rescale_actions(path: Path, factor: float, out_dir: Path) -> Path:
    """Scale the actor's output layer (and its exploration noise) by ``factor``: with a larger action_scale the same
    servo targets need smaller actions, so a warm-started policy keeps its gait instead of suddenly moving bigger."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    last = max((k for k in sd if k.startswith("mlp.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    sd[last] = sd[last] * factor
    sd[last.replace("weight", "bias")] = sd[last.replace("weight", "bias")] * factor
    if "distribution.std_param" in sd:
        sd["distribution.std_param"] = sd["distribution.std_param"] * factor
    ck.pop("optimizer_state_dict", None)
    out = out_dir / "init_rescaled.pt"
    torch.save(ck, out)
    print(f"[evolution] actor output scaled by {factor:.3f} (action_scale change) -> {out.name}")
    return out


def export_policy(runner, out: Path) -> None:
    """Save the actor (with its observation normalizer) as TorchScript and ONNX for deployment."""
    try:
        runner.export_policy_to_jit(str(out), filename="policy.pt")
        runner.export_policy_to_onnx(str(out), filename="policy.onnx")
        return
    except AttributeError:
        pass
    try:
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

        policy = runner.alg.get_policy() if hasattr(runner.alg, "get_policy") else runner.alg.actor
        export_policy_as_jit(policy, None, str(out), filename="policy.pt")
        export_policy_as_onnx(policy, str(out), filename="policy.onnx")
    except Exception as exc:  # export is a convenience, never fail the run for it
        print(f"[evolution] policy export skipped: {exc}")


if __name__ == "__main__":
    sys.exit(main())
