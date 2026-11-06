"""Hard paw front (journal step 46, owner 2026-10-31): the paw housing in PLA, the TPU sole disc the only grippy part.

In compare_up_r4_t9.mp4 (step 45) a paw that ends up right against the next riser struggles to lift: the soft TPU
front grips the riser. With a PLA housing it should slide up. The simulated paw changes from one TPU sphere (18 mm)
to a TPU sole sphere (9 mm, same sole height) + a PLA capsule over the printed housing (toes 27 mm ahead of the centre,
the old sphere only 18 mm: the real paw is longer than the simulated one was).

Run 1 (kept in run1_full_pla/): the whole housing in PLA: no more stuck paws, but the flat base (1.5 mm above the
sole) slips: walking and stairs down worse. Run 2 (owner: split the housing): four paws, the printed shape modelled as
a TPU sole sphere + two base rails (the flat base + 1.5 mm) + a shell capsule above them:
  tpu         the old model (one TPU sphere), for reference
  tpu_shaped  the printed shape all in TPU (the paw as designed: is a change from the shape or the material?)
  ring        PLA shell on a low TPU base ring (hard front, grippy base)
  pla         all PLA (as run 1, on the new shape model)

No training: the same policies with each paw, two dogs (hold_s2, zero_s1 from step 44):
  lunge test (stairs up: climbed, stumble = a paw pushed into a riser, stuck = > 0.3 s like that), stairs goal test
  (limit table), walking benchmark (forward, course), getting up, shoves, tilt (a hard rim may slip on the plates);
  videos TPU / PLA from the side up the stairs of the owner's screenshot (4 cm, 9 cm treads) and 3 cm / 15 cm, and
  down 4 cm / 15 cm. ~1.5 h. Smoke first.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "25_hard_paw"
_spec = importlib.util.spec_from_file_location("still_clock_batch", ROOT / "runs" / "exp2_dog" / "22_still_clock" / "batch.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)
LAB, PY, SHOVE, SIDE_EYE, lines, has = s44.LAB, s44.PY, s44.SHOVE, s44.SIDE_EYE, s44.lines, s44.has
PAWS = ["tpu", "tpu_shaped", "ring", "pla"]
DOGS = [f"{d}_{p}" for d in ("hold_s2", "zero_s1") for p in PAWS]
# (name, rise row = rise cm - 1, tread column, direction)
VIDEOS = [("up_r4_t9", 3, "goal_stairs_t9", 1), ("up_r3_t15", 2, "goal_stairs_t15", 1),
          ("down_r4_t15", 3, "goal_stairs_t15", -1)]


def run(cmd: list[str], log: Path) -> int:
    return s44.run(cmd, log)


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def lunge(dog: Path, log: Path, out: Path | None = None, num_envs: int = 1000) -> None:
    out = out or HERE / "lunge.jsonl"
    if not has(out, dog.name, mode="stairs"):
        run([LAB, "tools/dog/lunge_test.py", str(dog), "--out", str(out), "--num_envs", str(num_envs)], log)


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    lunge(HERE / "hold_s2_ring", log, out=out / "lunge.jsonl", num_envs=200)
    rows = lines(out / "lunge.jsonl")
    if not rows or not rows[-1].get("cells") or rows[-1]["cells"][0].get("stumble") is None:
        return fail("smoke (lunge test with the ring paw): no cells or no stumble count, see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def test(name: str) -> None:
    dog = HERE / name
    log = HERE / "tests.log"
    lunge(dog, log)
    if not has(HERE / "stairs_goal.jsonl", name):
        run([LAB, "tools/dog/stairs_goal_test.py", str(dog), "--out", str(HERE / "stairs_goal.jsonl")], log)
    for sc in ("forward", "course"):
        if not has(HERE / "benchmark.jsonl", name, scenario=sc):
            run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], log)
    for tool, out in (("recover_test", "recover_test.jsonl"), ("getup_test", "getup_test.jsonl")):
        for _ in range(2):  # once more after a start-up hiccup
            if not has(HERE / out, name):
                run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], log)
    if not any(Path(r["run"]).name == name and not r.get("shove_m_s") for r in lines(HERE / "tilt.jsonl")):
        run([LAB, "tools/dog/tilt_test.py", str(dog), "--per_cell", "10", "--out", str(HERE / "tilt.jsonl")], log)
    if not any(Path(r["run"]).name == name and r.get("shove_m_s") for r in lines(HERE / "tilt.jsonl")):
        run([LAB, "tools/dog/tilt_test.py", str(dog), "--per_cell", "10", "--shove", str(SHOVE), "--out",
             str(HERE / "tilt.jsonl")], log)


def main() -> int:
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    for name in DOGS:
        test(name)
    for vname, row, ttype, direction in VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if not out.exists():
            what = "up" if direction > 0 else "down"
            run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs",
                 "--terrain_type", ttype, "--level", str(row), "--drive", "goal", "--env_set",
                 f"stairs_goal_dir={direction}", "--eye", *SIDE_EYE, "--subtitle",
                 f"{what} {row + 1} cm steps, {ttype.split('_t')[-1]} cm treads: old paw / printed shape TPU / "
                 "PLA shell + TPU ring / all PLA",
                 *[a for paw in PAWS for a in ("--run", f"{paw}={HERE / f'hold_s2_{paw}'}")]],
                HERE / "videos.log")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
