"""Jump training: the reward is the best height of the lowest paw. Real STS3215 servos vs hypothetical fast motors."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAUNCHER = "lab.cmd" if os.name == "nt" else "lab.sh"  # the lab's Python with this repository on the path
COMMON = ["-m", "core.evolution.train", "--task", "jump", "--iterations", "1500", "--num_envs", "2048",
          "--sensors", "imu", "--obs_history", "8", "--symmetry"]
for name, genome in [("sts3215", r"prototypes\puppy\genome.json"), ("qdd", r"runs\dog_jump\puppy_qdd.json")]:
    out = ROOT / "runs" / "dog_jump" / name
    if (out / "result.json").exists():
        continue
    with open(ROOT / "runs" / "dog_jump" / f"{name}.log", "w") as log:
        subprocess.run([str(ROOT / LAUNCHER), *COMMON, "--genome", genome, "--out", str(out)],
                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
(ROOT / "runs" / "dog_jump" / "batch.done").write_text("done")
