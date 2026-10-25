"""Calm the jitter: action rate weight x20, and a stand-still penalty; each +1000 it from rough_v5."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
COMMON = ["-m", "core.evolution.train", "--genome", r"prototypes\puppy\genome.json", "--task", "command",
          "--terrain", "rough", "--iterations", "1000", "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8",
          "--symmetry", "--init_checkpoint", r"runs\dog_walk\puppy_rough_v5\model_final.pt",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)"]
RUNS = [("calm_rate", ["--env_override", "action_rate_weight=1.0"]),
        ("calm_stand", ["--env_override", "stand_still_weight=0.3"]),
        ("calm_both", ["--env_override", "action_rate_weight=1.0", "--env_override", "stand_still_weight=0.3"])]
for name, extra in RUNS:
    out = ROOT / "runs" / "dog_walk" / name
    if (out / "result.json").exists():
        continue
    with open(ROOT / "runs" / "dog_walk" / f"{name}.log", "w") as log:
        subprocess.run([str(ROOT / "lab.cmd"), *COMMON, "--out", str(out), *extra], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(ROOT / "runs" / "dog_walk" / "calm.done").write_text("done")
