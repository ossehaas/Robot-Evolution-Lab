"""Follow-up of step 41 (queued overnight, runs after batch.py): the descent. The dogs refuse at the top edge of the
stairs they could physically go down (few falls, almost no steps made), and one curriculum level per staircase kept
the high descents untrained where climbing up failed. desc_s1: as rsi (half of the stairs episodes mid-climb), plus a
curriculum per direction (stairs_dir_curriculum) and half of the mid-climb descents in the first 40 % of the plan (the
first steps off the landing). One seed; tests as in batch.py; a descent comparison video; results.md again."""
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# this step's batch.py under its own name (it imports step 38's module "batch" itself)
_spec = importlib.util.spec_from_file_location("stairs_proof_batch", HERE / "batch.py")
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)


def main() -> int:
    while not (HERE / "batch.done").exists():
        if (HERE / "batch.failed").exists():
            return 1
        time.sleep(60)
    name = "desc_s1"
    if not (HERE / name / "result.json").exists():
        batch.run(batch.train_cmd(HERE / name, "desc", 1), HERE / f"{name}.log")
    batch.test(name)
    if (HERE / name / "result.json").exists():
        for d, rise, tread in (("down", 4, 15), ("down", 3, 9)):
            out = HERE / "videos" / f"followup_{d}_r{rise}_t{tread}.mp4"
            if not out.exists():
                batch.run([batch.PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs",
                           "--terrain_type", f"goal_stairs_t{tread}", "--level", str(rise - 1), "--drive", "goal",
                           "--env_set", "stairs_goal_dir=-1",
                           "--subtitle", f"down {rise} cm steps, {tread} cm treads (physically possible): descent follow-up",
                           "--run", f"start (free_s2)={batch.START}", "--run", f"rsi (rsi_s1)={HERE / 'rsi_s1'}",
                           "--run", f"descent follow-up ({name})={HERE / name}"], HERE / "videos.log")
    batch.run([batch.PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "followup.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
