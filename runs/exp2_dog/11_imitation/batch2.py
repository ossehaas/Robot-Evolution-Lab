"""Imitation round 2: style weight 6 (round 1's best: most dog-like by both judges, ~1/3 of the energy at slow
speeds, front paws longer on the ground like a real dog) but it drifted right when told to go straight: the recorded
dog turns right more than left (-0.09 rad/s on average while walking). The judge now also learns the left-right
mirrored recordings (core/sim/amp.py, amp_mirror).
  amp_w6m_s1       style 6, mirrored recordings, seed 1
  amp_w6m_slow_s1  + the recordings played at 0.75x and 0.5x speed (examples of slow walking; round 1 at style 3
                   stood still when asked for 0.1 m/s)
  amp_w6m_s2       style 6, mirrored recordings, seed 2 (replicate)
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
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
RUNS = [("amp_w6m_s1", "1", [*AMP, *W6]),
        ("amp_w6m_slow_s1", "1", [*SLOW, *W6]),
        ("amp_w6m_s2", "2", [*AMP, *W6])]
for name, seed, extra in RUNS:
    out = HERE / name
    if not (out / "result.json").exists():
        with open(HERE / f"{name}.log", "w") as log:
            subprocess.run([str(ROOT / "lab.cmd"), *COMMON, "--seed", seed, *extra, "--out", str(out)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(HERE / "batch2.done").write_text("done")
