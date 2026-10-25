"""Energy sweep: the dog's energy weight (0.3 * 0.15 / 2.37 kg = 0.019 per W) x3 and x10, +1000 it from rough_v5."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
COMMON = ["-m", "core.evolution.train", "--genome", r"prototypes\puppy\genome.json", "--task", "command",
          "--terrain", "rough", "--iterations", "1000", "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8",
          "--symmetry", "--init_checkpoint", r"runs\dog_walk\puppy_rough_v5\model_final.pt",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)"]
for name, w in [("energy_x3", 0.057), ("energy_x10", 0.19)]:
    out = ROOT / "runs" / "dog_walk" / name
    if (out / "result.json").exists():
        continue
    with open(ROOT / "runs" / "dog_walk" / f"{name}.log", "w") as log:
        subprocess.run([str(ROOT / "lab.cmd"), *COMMON, "--out", str(out), "--env_override", f"energy_weight={w}"],
                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(ROOT / "runs" / "dog_walk" / "energy.done").write_text("done")
