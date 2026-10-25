"""Imitation round 3: replicate the best of round 2, amp_w6m_slow (style 6, mirrored recordings plus 0.75x / 0.5x
playback copies): fewest falls of all imitation runs (0.4 %), walks at 0.1 m/s on CoT 1.6, clean trot at 0.45 m/s.
  amp_w6m_slow_s2  same recipe, seed 2
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAUNCHER = "lab.cmd" if os.name == "nt" else "lab.sh"  # the lab's Python with this repository on the path
HERE = ROOT / "runs" / "exp2_dog" / "11_imitation"
START = ROOT / "runs" / "exp2_dog" / "10_printable_dog" / "D_print_s2"
COMMON = ["-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough",
          "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8", "--symmetry", "--height_scan",
          "--genome", str(START / "genome.json"), "--init_checkpoint", str(START / "model_final.pt"),
          "--iterations", "1500", "--critic_warmup", "50",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)",
          "--env_override", "command_accel=0.5", "--env_override", "command_yaw_accel=1.5",
          "--env_override=command_vx_range=(-0.3, 0.45)", "--env_override", "gait_freq_speed=0.45",
          "--env_override", "gait=free", "--env_override", "gait_trot_weight=3.0", "--env_override", "flight_weight=2.0",
          "--env_override", "action_scale=0.75", "--env_override", "action_rate_weight=3.0",
          "--env_override", "stand_still_weight=0.3"]
AMP = ["--env_override", "amp_motion_file=" + str(ROOT / "data" / "mocap" / "puppy_dog_motion.npz").replace("\\", "/")]
SLOW = ["--env_override", "amp_motion_file=" + str(ROOT / "data" / "mocap" / "puppy_dog_motion_slow.npz").replace("\\", "/")]
W6 = ["--env_override", "amp_weight=6.0", "--env_override", "amp_mirror=True"]
RUNS = [("amp_w6m_slow_s2", "2", [*SLOW, *W6])]
for name, seed, extra in RUNS:
    out = HERE / name
    if not (out / "result.json").exists():
        with open(HERE / f"{name}.log", "w") as log:
            subprocess.run([str(ROOT / LAUNCHER), *COMMON, "--seed", seed, *extra, "--out", str(out)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(HERE / "batch3.done").write_text("done")
