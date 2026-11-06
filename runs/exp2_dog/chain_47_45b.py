"""Step 47 then step 45b (owner, 2026-10-31: "if everything goes fine do 1 to 4"). Step 47 (26_ring_base/batch.py) is
already running; this waits for it, then runs 27_stairs_expert_b/batch.py only if step 47 passed (best.txt "ok").

    IsaacLab/.venv/Scripts/python.exe runs/exp2_dog/chain_47_45b.py > runs/exp2_dog/chain_47_45b.log 2>&1
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S47 = ROOT / "runs" / "exp2_dog" / "26_ring_base"
S45B = ROOT / "runs" / "exp2_dog" / "27_stairs_expert_b"
while not (S47 / "batch.done").exists():
    if (S47 / "batch.failed").exists():
        print(f"step 47 failed: {(S47 / 'batch.failed').read_text().strip()}", flush=True)
        sys.exit(1)
    time.sleep(60)
best = json.loads((S47 / "best.txt").read_text())
print(f"step 47 done: {best}", flush=True)
if not best.get("ok"):
    print("step 47 did not pass the mark: step 45b not started", flush=True)
    sys.exit(2)
print("step 45b: 27_stairs_expert_b/batch.py", flush=True)
with open(S45B / "batch.log", "a", encoding="utf-8") as log:
    code = subprocess.run([sys.executable, str(S45B / "batch.py")], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
print(f"step 45b: exit code {code}, done: {(S45B / 'batch.done').exists()}", flush=True)
sys.exit(0 if (S45B / "batch.done").exists() else 1)
