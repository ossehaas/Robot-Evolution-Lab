"""Steps 44 and 45 back to back (owner, 2026-10-28): the still-clock batch, then, if it finished, the stairs expert
on its best dog. Each batch logs to its own batch.log; this prints only which one runs.

    IsaacLab/.venv/Scripts/python.exe runs/exp2_dog/chain_44_45.py > runs/exp2_dog/chain_44_45.log 2>&1
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
for step, folder in ((44, "22_still_clock"), (45, "23_stairs_expert")):
    here = ROOT / "runs" / "exp2_dog" / folder
    if (here / "batch.done").exists():
        print(f"step {step}: already done", flush=True)
        continue
    print(f"step {step}: {folder}/batch.py", flush=True)
    with open(here / "batch.log", "a", encoding="utf-8") as log:
        code = subprocess.run([PY, str(here / "batch.py")], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    if code != 0 or not (here / "batch.done").exists():
        print(f"step {step}: stopped (exit code {code}), see {folder}/batch.failed and batch.log", flush=True)
        sys.exit(1)
    print(f"step {step}: done", flush=True)
