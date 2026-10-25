"""Seeing vs blind on the dog-sized course: +3000 it from range75 with and without the height scan."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
COMMON = ["-m", "core.evolution.train", "--genome", r"prototypes\puppy\genome.json", "--task", "command",
          "--terrain", "dog_rough", "--iterations", "3000", "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8",
          "--symmetry", "--init_checkpoint", r"runs\dog_stairs\range75\model_final.pt",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)",
          "--env_override", "action_rate_weight=3.0", "--env_override", "stand_still_weight=0.3",
          "--env_override", "action_scale=0.75"]
for name, extra in [("seeing", ["--height_scan"]), ("blind", [])]:
    out = ROOT / "runs" / "dog_stairs" / name
    if (out / "result.json").exists():
        continue
    with open(ROOT / "runs" / "dog_stairs" / f"{name}.log", "w") as log:
        subprocess.run([str(ROOT / "lab.cmd"), *COMMON, "--out", str(out), *extra], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(ROOT / "runs" / "dog_stairs" / "batch_scan.done").write_text("done")
