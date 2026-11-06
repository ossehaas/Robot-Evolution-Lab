"""The tests step 51's batch lost to a crash (plan_target without plans, fixed 2026-11-05): walking, recovery, get-up and tilt
for copy_s1 and copy_s2. Is the down teacher of step 53 still a whole dog? Rewrites results.md. ~30 min on the GPU.

    lab runs/exp2_dog/31_copy_plans/retest.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("copy_plans_batch", HERE / "batch.py")
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)
s47 = batch.s50.s47

for name, _ in batch.s50.RUNS:
    s47.test(name)
missing = [f"{name}: {file}" for name, _ in batch.s50.RUNS for file in ("benchmark.jsonl", "recover_test.jsonl", "getup_test.jsonl", "tilt.jsonl")
           if not batch.s50.has(HERE / file, name)]
batch.s50.run([batch.s50.PY, str(HERE / "summary.py")], HERE / "summary.log")
print("missing: " + ", ".join(missing) if missing else "all tests there")
sys.exit(1 if missing else 0)
