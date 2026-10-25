"""Flip test video: the dog stands; each time it has stood steadily for 1 s it is kicked over (thrown up with a spin that
tumbles it in one direction), the next direction 36 deg further round (relative to where the dog faces): 10 flips in
10 directions. If it has not stood up 8 s after a flip it is kicked anyway (a failure). Captions show the flip, the
direction and the time to stand up; the times are also written to <out>.json.

    lab tools/dog/flip_video.py runs/exp2_dog/15_getup_design/getup_abd29_head --out runs/exp2_dog/15_getup_design/videos/flips.mp4
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

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--flips", type=int, default=10)
ap.add_argument("--spin", type=float, default=14.0, help="[rad/s] tumbling rate of the kick")
ap.add_argument("--lift", type=float, default=1.3, help="[m/s] upward speed of the kick")
ap.add_argument("--push", type=float, default=0.4, help="[m/s] horizontal speed of the kick (a shove: --spin 0)")
ap.add_argument("--vx", type=float, default=0.0, help="[m/s] walking command between kicks (dogs trained to recover)")
ap.add_argument("--eye", type=float, nargs=3, default=(0.9, -1.1, 0.6))
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 1, False, task="command", terrain="flat",
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
recover_run = bool(res.get("env_overrides", {}).get("recover", False))
if recover_run:  # a walking dog that gets up: recover mode, no shoves of its own, never ended for lying down
    cfg.getup, cfg.recover, cfg.topple_push, cfg.recover_max_down_s = False, True, (), 1000.0
else:
    cfg.getup, cfg.getup_category = True, 0  # start standing
LABEL = "flip" if args.spin > 0 else "shove"
cfg.episode_length_s = 300.0  # one long episode
cfg.viewer.eye = tuple(args.eye)
STEPS_MAX = int(args.flips * 11.0 / (cfg.sim.dt * cfg.decimation)) + 200
args.video, args.video_length, args.video_interval = True, STEPS_MAX, None
from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video  # noqa: E402

enable_cameras_for_video(args)
work = args.out.parent / "flip_frames"
apply_video_recording(cfg, str(work), args, subdir="raw")

with launch_simulation(cfg, args):
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg, render_mode="rgb_array"), clip_actions=None)
    base = env.unwrapped
    dev, dt = base.device, base.step_dt
    actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()
    obs, _ = env.reset()
    if recover_run:
        base.command_override = torch.tensor([args.vx, 0.0, 0.0], device=dev)
    events = []  # (flip number, direction [deg], kick time [s], stood-up time [s] or None; 0 = never went down)
    downs = []  # per kick: did the dog go down
    held, flip, kicked_at, step = 0.0, 0, None, 0
    with torch.inference_mode():
        while step < STEPS_MAX:
            t = step * dt
            upright = (~base.fallen[0]) if recover_run else base.getup_standing[0]  # walking: not fallen
            held = held + dt if bool(upright) else 0.0
            if kicked_at is not None and not bool(upright):
                downs[-1] = True
            if kicked_at is not None and events[-1][3] is None and held >= 0.5:
                if downs[-1]:
                    events[-1] = (*events[-1][:3], round(t - 0.5 - kicked_at, 2))  # up (held 0.5 s) since then
                elif t - kicked_at >= 1.5:
                    events[-1] = (*events[-1][:3], 0.0)  # never went down: stayed on its feet
            ready = held >= 1.0 if kicked_at is None or events[-1][3] is not None else t - kicked_at > 8.0
            if flip == args.flips and (events[-1][3] is not None or t - kicked_at > 8.0):
                if held >= 1.5 or t - kicked_at > 8.0:
                    break
            elif ready and (kicked_at is None or t - kicked_at > 1.5):
                direction = 36.0 * flip  # relative to the dog's heading: 0 = forwards, 90 = to its left
                _, _, yaw = euler_xyz_from_quat(base.robot.data.root_quat_w.torch)
                phi = float(yaw[0]) + math.radians(direction)
                v = torch.zeros(1, 6, device=dev)
                v[0, :3] = torch.tensor([args.push * math.cos(phi), args.push * math.sin(phi), args.lift])
                v[0, 3:] = args.spin * torch.tensor([-math.sin(phi), math.cos(phi), 0.0])  # tumble towards phi
                base.robot.write_root_velocity_to_sim_index(root_velocity=v, env_ids=torch.zeros(1, dtype=torch.long, device=dev))
                flip += 1
                kicked_at, held = t, 0.0
                events.append((flip, direction, round(t, 2), None))
                downs.append(False)
                print(f"[flip] {LABEL} {flip} at {t:.1f} s, direction {direction:.0f} deg")
            obs, _, _, _ = env.step(actor(obs["policy"]))
            step += 1
    env.close()  # finalizes the raw video
    raw = sorted((work / "videos" / "raw").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not raw:
        print("[flip] no video recorded")
        sys.exit(1)
    ok = [e for e, down in zip(events, downs) if e[3] is not None and down]
    summary = {"run": str(run).replace("\\", "/"), "flips": [
        {"flip": f, "direction_deg": d, "kick_s": k, "stood_up_after_s": s} for f, d, k, s in events],
        "went_down": sum(downs), "stayed_on_its_feet": sum(1 for e, down in zip(events, downs) if not down),
        "stood_up": f"{len(ok)}/{sum(downs)}",
        "mean_s": round(sum(e[3] for e in ok) / len(ok), 2) if ok else None}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print("[flip]", summary["stood_up"], "stood up, mean", summary["mean_s"], "s")

    import cv2
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(str(raw[-1]))

    def caption(get_frame, t):
        frame = get_frame(t).copy()
        current = [e for e in events if e[2] <= t]
        if not current:
            lines = [f"waiting for the first {LABEL}"]
        else:
            f, d, k, s = current[-1]
            state = ("stayed on its feet" if s == 0.0 and t >= k + 1.5 else
                     f"stood up after {s:.1f} s" if s is not None and s > 0 and t >= k + s else
                     ("getting up..." if s is not None or t - k <= 8.0 else "did not stand up within 8 s"))
            lines = [f"{LABEL} {f}/{len(events)}: direction {d:.0f} deg (0 = forwards, 90 = to its left)", state]
        done = sum(1 for e in events if e[3] is not None and t >= e[2] + e[3])
        lines.append(f"stood up {done} of {len(current)} so far")
        for i, text in enumerate(lines):
            org, size = (24, 44 + 34 * i), 0.85 - 0.15 * min(i, 1)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    clip.fl(caption).write_videofile(str(args.out), fps=round(1.0 / dt), codec="libx264", audio=False, logger=None)
    clip.close()
    print(f"[flip] video: {args.out}")
sys.exit(0)
