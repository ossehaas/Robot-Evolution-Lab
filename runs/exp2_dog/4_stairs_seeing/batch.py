"""Stairs on the dog-sized course: joint range 75 % vs 50 %, each +3000 it from calm_v2."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
COMMON = ["-m", "core.evolution.train", "--genome", r"prototypes\puppy\genome.json", "--task", "command",
          "--terrain", "dog_rough", "--iterations", "3000", "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8",
          "--symmetry", "--init_checkpoint", r"runs\dog_walk\calm_v2\model_final.pt",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)",
          "--env_override", "action_rate_weight=3.0", "--env_override", "stand_still_weight=0.3"]
for name, scale in [("range75", 0.75), ("range50", 0.5)]:
    out = ROOT / "runs" / "dog_stairs" / name
    if (out / "result.json").exists():
        continue
    with open(ROOT / "runs" / "dog_stairs" / f"{name}.log", "w") as log:
        subprocess.run([str(ROOT / "lab.cmd"), *COMMON, "--out", str(out), "--env_override", f"action_scale={scale}"],
                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(ROOT / "runs" / "dog_stairs" / "batch.done").write_text("done")
