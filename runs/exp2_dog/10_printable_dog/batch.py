"""The printable puppy as chosen (variant D): Spot knees, printed TPU ankle flexure (1.5 N*m/rad), soft TPU sole
(1e4 N/m), real masses (2.45 kg); ankle damping randomized 0.005-0.03 (printed TPU damps little). Same two-stage recipe
as 9_gait_body, two seeds (a replicate and a check of the printable version).

Gait and body (list steps 1-3 combined), all from scratch with one two-stage recipe on the dog course with the
height scan, symmetry, walk-when-slow / trot-when-fast, no flight phase, smooth speed changes, top speed 0.45 m/s.
The paw is its own link (ankle hinge) so the contact sensor sees paws only and shins on the ground count as body contact.
  stage 1 (2000 it): learn to walk - mild smoothness penalties, 50 % joint range
  stage 2 (2500 it): make it calm - action rate 3.0, stand-still 0.3, 75 % joint range
(The strict calmness penalties from the start make falling at once the cheapest option for a flailing new network.)
  A_stiff       knees as now (front back, rear forward), stiff ankle (50 N*m/rad, effectively rigid)
  B_springy     knees as now, springy ankle (1.5 N*m/rad) + soft TPU paws (compliant contact)
  C_spot_knees  all knees backwards (as Spot), stiff ankle"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAUNCHER = "lab.cmd" if os.name == "nt" else "lab.sh"  # the lab's Python with this repository on the path
HERE = ROOT / "runs" / "exp2_dog" / "10_printable_dog"
COMMON = ["-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough",
          "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8", "--symmetry", "--height_scan",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)",
          "--env_override", "command_accel=0.5", "--env_override", "command_yaw_accel=1.5",
          "--env_override=command_vx_range=(-0.3, 0.45)", "--env_override", "gait_freq_speed=0.45",
          "--env_override", "gait=walk_trot", "--env_override", "gait_trot_weight=3.0", "--env_override", "flight_weight=2.0"]
STAGE1 = ["--iterations", "2000", "--skip_eval", "--env_override", "action_scale=0.5"]
STAGE2 = ["--iterations", "2500", "--env_override", "action_scale=0.75", "--env_override", "action_rate_weight=3.0",
          "--env_override", "stand_still_weight=0.3"]
for name, seed in (("D_print_s1", "1"), ("D_print_s2", "2")):
    genome = str(HERE / "puppy_print.json")
    s1, s2 = HERE / f"{name}_stage1", HERE / name
    if not (s1 / "model_final.pt").exists():
        with open(HERE / f"{name}_stage1.log", "w") as log:
            subprocess.run([str(ROOT / LAUNCHER), *COMMON, "--seed", seed, *STAGE1, "--genome", genome, "--out", str(s1)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if not (s2 / "result.json").exists():
        with open(HERE / f"{name}.log", "w") as log:
            subprocess.run([str(ROOT / LAUNCHER), *COMMON, "--seed", seed, *STAGE2, "--genome", genome, "--out", str(s2),
                            "--init_checkpoint", str(s1 / "model_final.pt")], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(HERE / "batch.done").write_text("done")
