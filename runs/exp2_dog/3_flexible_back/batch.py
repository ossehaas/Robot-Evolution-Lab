"""Flexible back experiment (roadmap D2b): every variant gets +1000 iterations from the stiff dog's rough_v5."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAB = str(ROOT / "lab.cmd")
COMMON = ["-m", "core.evolution.train", "--task", "command", "--terrain", "rough", "--iterations", "1000",
          "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8", "--symmetry",
          "--init_checkpoint", r"runs\dog_walk\puppy_rough_v5\model_final.pt",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)"]
for name in ["stiff", "roll_k8", "pitch_k8", "both_k8", "roll_k4", "roll_k16"]:
    out = ROOT / "runs" / "dog_spine" / name
    if (out / "result.json").exists():
        continue
    with open(ROOT / "runs" / "dog_spine" / f"{name}.log", "w") as log:
        subprocess.run([LAB, *COMMON, "--genome", rf"runs\dog_spine\{name}.json", "--out", rf"runs\dog_spine\{name}"],
                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(ROOT / "runs" / "dog_spine" / "batch.done").write_text("done")
