"""Step 50: control for the replays. Today's code on step 41's plans, on stairs where they climbed then (2/15, 2/9: 100 %,
4/12: 75 %, randomized, 4 dogs), next to the new free plans on the same stairs. A traced nominal run of each shows when the dog falls.

    lab runs/exp2_dog/30_level_body/replay_control.py
"""
import importlib.util
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "replay" / "control"
_spec = importlib.util.spec_from_file_location("still_clock_batch", HERE.parent / "22_still_clock" / "batch.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)
SETS = {"old": HERE.parent / "19_stairs_proof" / "plans_all", "new": HERE / "plans_p45_m15"}
CELLS = ("up_r2_t15", "up_r2_t9", "up_r4_t12")


def main() -> int:
    for name, source in SETS.items():
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        for cell in CELLS:
            shutil.copy2(source / f"{cell}.json", folder / f"{cell}.json")
        s44.run([s44.LAB, "tools/dog/stairs_replay.py", str(folder), "--out", str(OUT / f"{name}_random.jsonl"), "--copies", "4"], OUT / f"{name}_random.log")
        s44.run([s44.LAB, "tools/dog/stairs_replay.py", str(folder / "up_r2_t15.json"), "--out", str(OUT / f"{name}_trace.jsonl"), "--trace"], OUT / f"{name}_trace.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
