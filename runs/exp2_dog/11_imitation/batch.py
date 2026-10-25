"""List step 4: imitate a real dog (AMP style reward, core/sim/amp.py) on the printable puppy D_print_s2.

Motion: AI4Animation dog mocap (8 clips, ~12 min walk + ~2 min trot), retargeted to the puppy (core/mocap/retarget.py,
Froude scaling, average spine pitch removed), transitions up to 0.55 m/s. A discriminator learns real dog transitions
(leg angles + speeds, body velocity, turn rate, gravity direction) against the robot's; the robot earns 0..1 style per
second times amp_weight. No prescribed gait rhythm (gait=free): the dog's style decides walk or trot.
Fine-tune from D_print_s2 (stage-2 settings), 50 critic warm-up iterations (new reward term), 1500 iterations.
  amp_w3     style weight 3 (task rewards are ~8 per second in total)
  amp_w6     style weight 6
  free_ctrl  the same without the style reward: separates "imitation" from "free gait + more training"
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
RUNS = [("amp_w3", "1", [*AMP, "--env_override", "amp_weight=3.0"]),
        ("amp_w6", "1", [*AMP, "--env_override", "amp_weight=6.0"]),
        ("free_ctrl", "1", [])]
for name, seed, extra in RUNS:
    out = HERE / name
    if not (out / "result.json").exists():
        with open(HERE / f"{name}.log", "w") as log:
            subprocess.run([str(ROOT / LAUNCHER), *COMMON, "--seed", seed, *extra, "--out", str(out)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(HERE / "batch.done").write_text("done")
