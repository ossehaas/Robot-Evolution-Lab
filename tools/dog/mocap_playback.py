"""Play retargeted dog motion on the robot (kinematic body, servos follow the joint angles) and record a video.

    lab tools/dog/mocap_playback.py data/mocap/puppy_dog_motion.npz --clip 0 --seconds 20 --out runs/exp2_dog/11_imitation/videos/replay.mp4
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation

from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import make_env_cfg

ap = argparse.ArgumentParser()
ap.add_argument("motion", type=Path)
ap.add_argument("--genome", default="prototypes/puppy/genome.json")
ap.add_argument("--clip", type=int, default=0)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--out", type=Path, default=Path("runs/scratch/mocap_replay.mp4"))
add_launcher_args(ap)
args = ap.parse_args()
args.video, args.video_length, args.video_interval = True, int(args.seconds / 0.02), None

data = np.load(args.motion)
sel = data["clip"] == args.clip
J, H, V, W, G = (data[k][sel] for k in ("joints", "height", "lin_vel_b", "ang_vel_b", "gravity_b"))
names = list(data["joint_names"])
work = Path("runs/scratch/mocap_replay")
work.mkdir(parents=True, exist_ok=True)
usd = build_robot_files(Path(args.genome).resolve(), work / "robot")
spec = build_robot(Genome.load(args.genome))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 1, False, task="command", sensors="imu", obs_history=1)
cfg.action_scale = 1.0
cfg.episode_length_s = args.seconds + 5
cfg.fall_up_threshold = -2.0
cfg.viewer.eye = (0.9, -1.0, 0.45)
from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video  # noqa: E402

enable_cameras_for_video(args)
apply_video_recording(cfg, str(work), args, subdir="video")
with launch_simulation(cfg, args):
    from isaaclab.utils.math import quat_from_matrix

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = EvoLocomotionEnv(cfg, render_mode="rgb_array")
    env.expression_override = torch.zeros(1, len(env._expr_ids), device=env.device)
    env.command_override = torch.zeros(3, device=env.device)
    env.reset()
    act_names = [env.robot.joint_names[i] for i in env._act_ids.tolist()]
    col = [names.index(n) for n in act_names]
    rng = (env.joint_range * env._target_scale)[env._act_ids].cpu().numpy()
    hip_z = float(spec.link_map["L1_s1"].origin[2])  # hip axis relative to the chassis centre
    x = y = yaw = 0.0
    for t in range(min(len(H), args.video_length + 5)):
        # integrate the planar path from the body-frame velocities; tilt from the gravity direction
        yaw += W[t, 2] * 0.02
        c, s = math.cos(yaw), math.sin(yaw)
        x += (c * V[t, 0] - s * V[t, 1]) * 0.02
        y += (s * V[t, 0] + c * V[t, 1]) * 0.02
        g = G[t] / np.linalg.norm(G[t])
        z_b = -g  # world up in body coordinates
        x_b = np.cross([0.0, 1.0, 0.0], z_b); x_b /= np.linalg.norm(x_b)
        y_b = np.cross(z_b, x_b)
        R_tilt = np.stack((x_b, y_b, z_b), 0)  # rows: world axes in body coords -> body->world = R_tilt.T ... (small)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        R = R_yaw @ R_tilt.T.T
        quat = quat_from_matrix(torch.tensor(R, dtype=torch.float32).unsqueeze(0)).to(env.device)
        pose = torch.zeros(1, 7, device=env.device)
        pose[0, :3] = torch.tensor([x, y, H[t] - hip_z], device=env.device)
        pose[0, 3:] = quat[0]
        env.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=env.device))
        env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device),
                                                  env_ids=torch.tensor([0], device=env.device))
        a = torch.tensor(np.clip(J[t, col] / rng, -1, 1), dtype=torch.float32, device=env.device).unsqueeze(0)
        env.step(a)
    env.close()
clips = sorted((work / "videos" / "video").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
if clips:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clips[-1].replace(args.out)
    print(f"video: {args.out}")
