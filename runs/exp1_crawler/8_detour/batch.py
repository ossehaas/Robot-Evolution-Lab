import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
lab = str(ROOT / "lab.cmd")
common = ["-m", "core.evolution.train", "--genome", r"runs\poc\nav_camera\genome.json", "--task", "goal", "--terrain", "cup",
          "--num_envs", "2048", "--sensors", "imu", "--obs_history", "4", "--env_override", "goal_signal=camera"]
runs = [("zeroshot_nav_camera", ["--iterations", "0", "--init_checkpoint", r"runs\poc\nav_camera\model_final.pt"]),
        ("cup_v1", ["--iterations", "1500", "--init_checkpoint", r"runs\poc\nav_camera\model_final.pt"])]
for name, extra in runs:
    with open(rf"runs\detour\{name}.log", "w") as log:
        subprocess.run([lab, *common, "--out", rf"runs\detour\{name}", *extra], stdout=log, stderr=subprocess.STDOUT,
                       cwd=ROOT)
open(r"runs\detour\batch.done", "w").write("done")
