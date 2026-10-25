"""Natural gait (list step 1): smooth speed changes (0.5 m/s^2), top speed 0.45 m/s, and the gait either prescribed by
speed (4-beat walk when slow, trot when fast, no flight phase) or free (no rhythm, energy weighted 3x).
Each +3000 it from seeing_v3 (dog course, height scan)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
HERE = ROOT / "runs" / "exp2_dog" / "6_natural_gait"
COMMON = ["-m", "core.evolution.train", "--genome", r"prototypes\puppy\genome.json", "--task", "command",
          "--terrain", "dog_rough", "--iterations", "3000", "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8",
          "--symmetry", "--height_scan", "--init_checkpoint", r"runs\exp2_dog\4_stairs_seeing\seeing_v3\model_final.pt",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)",
          "--env_override", "action_rate_weight=3.0", "--env_override", "stand_still_weight=0.3",
          "--env_override", "action_scale=0.75", "--env_override", "command_accel=0.5",
          "--env_override", "command_yaw_accel=1.5", "--env_override=command_vx_range=(-0.3, 0.45)",
          "--env_override", "gait_freq_speed=0.45"]
RUNS = [("walk_trot", ["--env_override", "gait=walk_trot", "--env_override", "gait_trot_weight=3.0",
                       "--env_override", "flight_weight=2.0"]),
        ("free", ["--env_override", "gait=free", "--env_override", "energy_weight=0.057"])]
for name, extra in RUNS:
    out = HERE / name
    if (out / "result.json").exists():
        continue
    with open(HERE / f"{name}.log", "w") as log:
        subprocess.run([str(ROOT / "lab.cmd"), *COMMON, "--out", str(out), *extra], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(HERE / "batch.done").write_text("done")
