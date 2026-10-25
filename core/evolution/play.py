"""Watch a trained robot walk, or drive a steerable one.

    python -m core.evolution.play --run_dir runs/crawler_q4                     # Isaac Sim viewport
    python -m core.evolution.play --run_dir runs/crawler_q4 --video --num_envs 4    # headless, writes an mp4
    python -m core.evolution.play --run_dir runs/crawler_q4 --random            # untrained: random servo commands

Steerable robots (trained with --task command; detected from result.json):

    python -m core.evolution.play --run_dir runs/crawler_q4_steer               # drive with the keyboard
    python -m core.evolution.play --run_dir runs/crawler_q4_steer --video       # scripted tour, captioned video

Keyboard (click the viewport first): arrow keys / numpad 8,2 = forward/back, numpad 4,6 = sideways,
Z / X = turn left/right, L = stop. Green arrow = command, blue arrow = actual velocity.

The run directory is the --out of core.evolution.train (genome.json, robot/*.usda, model_final.pt).
"""

from __future__ import annotations

import argparse
import math
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# scripted tour for steerable robots: (seconds, vx [m/s], vy [m/s], yaw rate [rad/s], caption)
TOUR = [
    (2.0, 0.00, 0.00, 0.0, "stand still"),
    (3.0, 0.10, 0.00, 0.0, "slow 10 cm/s"),
    (3.0, 0.25, 0.00, 0.0, "forward 25 cm/s"),
    (3.0, 0.20, 0.00, 0.8, "curve left"),
    (3.0, 0.00, 0.00, -1.0, "turn in place (right)"),
    (2.5, 0.00, 0.20, 0.0, "sidestep left 20 cm/s"),
    (2.5, 0.00, -0.20, 0.0, "sidestep right 20 cm/s"),
    (3.0, -0.20, 0.00, 0.0, "backward 20 cm/s"),
    (2.5, 0.35, 0.00, 0.0, "full speed 35 cm/s"),
]
# straight on (--drive straight): stairs videos, the dog starts facing the steps
STRAIGHT = [
    (1.5, 0.00, 0.00, 0.0, "stand still"),
    (6.0, 0.15, 0.00, 0.0, "straight on 15 cm/s"),
    (6.0, 0.25, 0.00, 0.0, "straight on 25 cm/s"),
]
# standing still (--drive stand): get-up videos
STAND = [(8.0, 0.00, 0.00, 0.0, "stand still")]


def tour_command(t: float):
    """Command and caption of the scripted tour at time t [s] (holds the last segment)."""
    elapsed = 0.0
    for seconds, vx, vy, wz, caption in TOUR:
        elapsed += seconds
        if t < elapsed:
            return (vx, vy, wz), caption
    _, vx, vy, wz, caption = TOUR[-1]
    return (vx, vy, wz), caption


def show_goals(base) -> None:
    """Mark the current goal of every robot with a red sphere."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    if not hasattr(base, "_goal_markers"):
        base._goal_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/goals",
                markers={"goal": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.1, 0.1)))},
            )
        )
    pos = base.goal_pos.clone()
    pos[:, 2] += 0.03
    base._goal_markers.visualize(translations=pos)


def show_trail(base, step: int, every: int = 10) -> None:
    """Drop a small blue dot at every robot's position every ``every`` steps (cleared when its episode restarts)."""
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    if not hasattr(base, "_trail_markers"):
        base._trail_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/trail",
                markers={"dot": sim_utils.SphereCfg(radius=0.008, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.4, 0.95)))},
            )
        )
        base._trail = [[] for _ in range(base.num_envs)]
    if step % every:
        return
    pos = base.robot.data.root_pos_w.torch.clone()
    pos[:, 2] = base.terrain.env_origins[:, 2] + 0.01
    for i in range(base.num_envs):
        if base.episode_length_buf[i] < every:
            base._trail[i].clear()
        base._trail[i].append(pos[i])
    base._trail_markers.visualize(translations=torch.stack([p for t in base._trail for p in t]))


def caption_video(path: Path, fps: float) -> Path:
    """Write a copy of the recorded tour with the active command printed on every frame."""
    import cv2
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(str(path))

    def annotate(get_frame, t):
        frame = get_frame(t).copy()
        (vx, vy, wz), caption = tour_command(t)
        lines = [caption, f"command: vx {100 * vx:+.0f} cm/s  vy {100 * vy:+.0f} cm/s  turn {wz:+.1f} rad/s"]
        for i, text in enumerate(lines):
            org, size = (24, 44 + 34 * i), 0.9 - 0.25 * i
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2, cv2.LINE_AA)
        legend = "green arrow = command, blue arrow = actual velocity"
        org = (24, frame.shape[0] - 24)
        cv2.putText(frame, legend, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, legend, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    out = path.with_name(path.stem + "_captioned.mp4")
    clip.fl(annotate).write_videofile(str(out), fps=fps, codec="libx264", audio=False, logger=None)
    clip.close()
    return out


def main(argv=None) -> int:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Play a trained robot policy.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: <run_dir>/model_final.pt")
    parser.add_argument("--num_envs", type=int, default=9)
    parser.add_argument("--physics", dest="evo_physics", choices=("physx", "newton"), default="physx")
    parser.add_argument("--servo_model", choices=("dc_motor", "implicit"), default="dc_motor")
    parser.add_argument("--task", choices=("forward", "command", "goal", "jump"), default=None, help="default: from result.json")
    parser.add_argument("--terrain", choices=("flat", "rough", "obstacles", "obstacles_rough", "enclosure", "enclosure_pocket", "cup", "dog_rough"), default=None,
                        help="default: from result.json")
    parser.add_argument("--level", type=int, default=None, help="rough terrain: put every robot on this difficulty row")
    parser.add_argument(
        "--terrain_type",
        default=None,
        help="rough terrain: put every robot on this terrain type (bumpy, slope_up, slope_down, stairs_descend, stairs_climb, blocks)",
    )
    parser.add_argument("--far_start", action="store_true", help="enclosure: start opposite the goal side")
    parser.add_argument("--heading_hold", type=float, default=None,
                        help="steerable robots: keep the heading while walking straight (gain, e.g. 2.0; 0 = off)")
    parser.add_argument("--out_name", default=None, help="file name for the recorded video (default: clip_0000...)")
    parser.add_argument(
        "--drive",
        choices=("keyboard", "tour", "straight", "stand", "random"),
        default=None,
        help="command source for steerable robots (default: tour with --video, keyboard otherwise; straight = "
             "only walking straight on)",
    )
    parser.add_argument("--random", action="store_true", help="ignore the policy, send random servo targets")
    parser.add_argument("--still", action="store_true", help="no policy: hold the rest pose (does the body stand?)")
    parser.add_argument("--eye", type=float, nargs=3, default=None, help="camera position relative to the robot [m]")
    parser.add_argument("--steps", type=int, default=0, help="stop after this many policy steps (0 = until closed)")
    parser.add_argument("--video", action="store_true", help="record an mp4 to <run_dir>/videos/play and exit")
    parser.add_argument("--video_length", type=int, default=None, help="policy steps to record (50 steps = 1 s)")
    parser.add_argument("--env_set", action="append", default=[],
                        help="environment setting after the run's own, e.g. --env_set getup_category=3 (repeatable)")
    add_launcher_args(parser)
    args = parser.parse_args(argv)

    result_file = args.run_dir / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    if args.task is None:
        args.task = result.get("task", "forward")
    if args.terrain is None:
        args.terrain = result.get("terrain_arg", result.get("terrain", "flat"))
    sensors, obs_history = result.get("sensors", "ideal"), result.get("obs_history", 1)
    steer = args.task == "command"
    if args.drive is None:
        args.drive = "tour" if (steer and args.video) else ("keyboard" if steer else "random")
    if args.drive in ("straight", "stand"):
        TOUR[:] = STRAIGHT if args.drive == "straight" else STAND  # tour, captions and video length all follow TOUR
        args.drive = "tour"
    if args.video_length is None:
        args.video_length = int(round(sum(seg[0] for seg in TOUR) * 50)) if (steer and args.drive == "tour") else 400
        if args.task == "goal":
            args.video_length = 900  # 18 s: several goals
            if args.terrain in ("enclosure_pocket", "cup"):
                args.video_length = 1500  # 30 s: one episode on the pocket course
    args.video_interval = None
    if args.video:
        args.steps = args.steps or args.video_length + 5
    elif args.visualizer is None:
        args.visualizer = ["kit"]

    from core.evolution.train import build_robot_files
    from core.morphology import Genome, build_robot
    from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

    genome_file = args.run_dir / "genome.json"
    if not genome_file.exists():
        genome_file = args.run_dir / "robot" / "genome.json"
    usd = build_robot_files(genome_file.resolve(), args.run_dir / "robot")
    spec = build_robot(Genome.load(genome_file))
    env_cfg = make_env_cfg(
        spec,
        str(usd.resolve()).replace("\\", "/"),
        args.num_envs,
        False,
        physics=args.evo_physics,
        servo_model=args.servo_model,
        task=args.task,
        terrain=args.terrain,
        sensors=sensors,
        obs_history=obs_history,
        height_scan=result.get("height_scan", False), terrain_map=result.get("terrain_map", ""),
        low_level_policy=result.get("low_level_policy"),
    )
    apply_overrides(env_cfg, result.get("env_overrides"))
    for item in args.env_set:
        import ast

        key, value = item.split("=", 1)
        if not hasattr(env_cfg, key):
            raise SystemExit(f"--env_set: unknown environment setting {key!r}")
        setattr(env_cfg, key, ast.literal_eval(value))
    env_cfg.curriculum = False
    env_cfg.episode_length_s = 60.0
    if args.task == "goal":
        env_cfg.viewer.eye = (-0.6, -0.8, 1.0)  # high view: see the pillars, the goal and the way around
    if args.terrain.startswith("enclosure") or args.terrain == "cup":
        # the test itself: every goal inside the ring; camera fixed above the ring of env 0
        env_cfg.unreachable_prob = 1.0
        env_cfg.enclosure_far_start = args.far_start
        env_cfg.episode_length_s = 30.0 if (env_cfg.pocket or env_cfg.cup) else 20.0
        env_cfg.cup_start_spread = env_cfg.cup_eval_start_spread  # cup: start on the closed side
        env_cfg.viewer.eye = (0.0, -0.9, 1.4)  # high above and behind the robot: the whole ring in view
    if steer:
        env_cfg.reset_yaw_range = 0.0  # start facing +x, the camera's reference direction
        env_cfg.viewer.eye = (-0.35, -0.55, 0.60)  # higher chase view so turns are easy to follow
    if args.eye is not None:
        env_cfg.viewer.eye = tuple(args.eye)
    if args.heading_hold is not None:
        env_cfg.heading_hold_gain = args.heading_hold
    if args.video:
        from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video

        enable_cameras_for_video(args)
        apply_video_recording(env_cfg, str(args.run_dir), args, subdir="play")

    from isaaclab.app import launch_simulation

    with launch_simulation(env_cfg, args):
        import importlib.metadata as metadata

        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from core.evolution.agent_cfg import EvoPPORunnerCfg, apply_memory
        from core.sim.locomotion_env import EvoLocomotionEnv

        render_mode = "rgb_array" if args.video else None
        env = RslRlVecEnvWrapper(EvoLocomotionEnv(env_cfg, render_mode=render_mode), clip_actions=1.0)
        base = env.unwrapped
        device = base.device
        if args.still:

            def policy(obs):
                return torch.zeros(env.num_envs, env.num_actions, device=device)

        elif args.random:

            def policy(obs):
                return 2.0 * torch.rand(env.num_envs, env.num_actions, device=device) - 1.0

        else:
            agent_cfg = handle_deprecated_rsl_rl_cfg(apply_memory(EvoPPORunnerCfg(), result.get("memory")),
                                                     metadata.version("rsl-rl-lib"))
            agent_cfg.device = device
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(args.checkpoint or args.run_dir / "model_final.pt"))
            policy = runner.get_inference_policy(device=device)

        keyboard = None
        if steer and args.drive == "keyboard":
            from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

            keyboard = Se2Keyboard(
                Se2KeyboardCfg(v_x_sensitivity=0.25, v_y_sensitivity=0.25, omega_z_sensitivity=1.0, sim_device=device)
            )
            print(keyboard)
        if steer and args.drive != "random":
            base.command_override = torch.zeros(3, device=device)

        if args.terrain in ("rough", "dog_rough", "obstacles", "obstacles_rough", "cup"):
            base.spread_over_terrain_levels()
            t = base.terrain
            if args.level is not None:
                t.terrain_levels[:] = args.level
            if args.terrain_type is not None:
                gen = env_cfg.terrain.terrain_generator
                names = list(gen.sub_terrains)
                if args.terrain_type not in names:
                    raise SystemExit(f"--terrain_type must be one of {names}")
                cum = torch.cumsum(torch.tensor([gen.sub_terrains[n].proportion for n in names]), 0)
                cum = cum / cum[-1]
                cols = [c for c in range(gen.num_cols) if names[int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0])] == args.terrain_type]
                t.terrain_types[:] = torch.tensor(cols, device=device)[torch.arange(env.num_envs, device=device) % len(cols)]
            t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
        obs, _ = env.reset()
        step, last_caption = 0, None
        while base.sim.is_playing() or step == 0:
            if steer and args.drive == "tour":
                cmd, caption = tour_command(step * base.step_dt)
                base.command_override = torch.tensor(cmd, device=device)
                if caption != last_caption:
                    # heading [deg, + = left] and speed of robot 0 when a segment starts: drift shows in the log
                    from isaaclab.utils.math import quat_apply

                    q = base.robot.data.root_quat_w.torch[:1]
                    fwd = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
                    yaw = math.degrees(math.atan2(float(fwd[1]), float(fwd[0])))
                    v = base.robot.data.root_lin_vel_b.torch[0]
                    print(f"[{step * base.step_dt:5.1f}s] {caption}  (heading {yaw:+.0f} deg, velocity "
                          f"{float(v[0]):+.2f} {float(v[1]):+.2f} m/s)")
                    last_caption = caption
            elif keyboard is not None:
                base.command_override = keyboard.advance()
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                if hasattr(policy, "reset"):
                    policy.reset(dones)
            if steer:
                base.show_command_arrows()
            if args.task == "goal":
                show_goals(base)
            if args.terrain.startswith("enclosure") or args.terrain == "cup":
                show_trail(base, step)
            step += 1
            if step % 250 == 0 and not steer:
                x = (base.robot.data.root_pos_w.torch[:, 0] - base.start_pos[:, 0]).mean().item()
                print(f"step {step}: mean distance this episode {100 * x:.1f} cm")
            if args.steps and step >= args.steps:
                break
        step_dt = base.step_dt
        env.close()  # finalizes the video file

        # inside the launch context: closing the app on exit may end the process
        if args.video and args.out_name and not (steer and args.drive == "tour"):
            clips = sorted((args.run_dir / "videos" / "play").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            if clips:
                final = args.run_dir / "videos" / args.out_name
                clips[-1].replace(final)
                print(f"video: {final}")
        if args.video and steer and args.drive == "tour":
            clips = sorted((args.run_dir / "videos" / "play").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            clips = [c for c in clips if not c.stem.endswith("_captioned")]
            if clips:
                out = caption_video(clips[-1], fps=round(1.0 / step_dt))
                if args.out_name:
                    final = out.parent.parent / args.out_name
                    out.replace(final)
                    out = final
                print(f"captioned video: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
