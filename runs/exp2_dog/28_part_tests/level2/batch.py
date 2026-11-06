"""Step 48, level 2: the longer soles on the trained dogs, no training (the funnel after the test stand).

Same policies as step 46 (hold_s2, zero_s1 from step 44), same tests (25_hard_paw/batch.py), three paws:
  ring  the current paw, sole out 1.5 mm (control: must match step 46's numbers)
  p3    sole out 3 mm
  p45   sole out 4.5 mm
Colliders only, no CAD for the longer soles yet. The old ball comes from step 46's results. Videos of hold_s2: old ball /
current / 3 mm / 4.5 mm. Smoke first.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STEP46 = ROOT / "runs" / "exp2_dog" / "25_hard_paw"
_spec = importlib.util.spec_from_file_location("hard_paw_batch", STEP46 / "batch.py")
s46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s46)
s46.HERE = HERE  # its test() and lunge() write next to this batch
PY, SIDE_EYE, run = s46.PY, s46.SIDE_EYE, s46.run
PAWS = {"ring": 1.5, "p3": 3.0, "p45": 4.5}
DOGS = [f"{d}_{p}" for d in ("hold_s2", "zero_s1") for p in PAWS]
VIDEOS = [("stop_go", "flat", None, 0, "stop_go", None, "walk 25 cm/s, stop, walk, stop"),
          ("up_r4_t9", "dog_stairs", "goal_stairs_t9", 3, "goal", "stairs_goal_dir=1", "up 4 cm steps, 9 cm treads"),
          ("down_r4_t15", "dog_stairs", "goal_stairs_t15", 3, "goal", "stairs_goal_dir=-1", "down 4 cm steps, 15 cm treads")]


def make_dog(name: str) -> None:
    """A copy of step 46's ring dog (policy, genome), only the sole's protrusion changed; its robot is built on first use."""
    dog, paw = HERE / name, name.rsplit("_", 1)[1]
    if dog.exists():
        return
    dog.mkdir()
    src = STEP46 / f"{name.rsplit('_', 1)[0]}_ring"
    for file in ("model_final.pt", "policy.pt", "result.json"):
        shutil.copy2(src / file, dog / file)
    g = json.loads((src / "genome.json").read_text(encoding="utf-8"))
    if paw != "ring":
        g["legs"]["paw_protrusion"] = PAWS[paw]
    (dog / "genome.json").write_text(json.dumps(g, indent=2), encoding="utf-8")


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(exist_ok=True)
    for name in ("hold_s2_p3", "hold_s2_p45"):
        s46.lunge(HERE / name, out / "smoke.log", out=out / "lunge.jsonl", num_envs=200)
    rows = s46.lines(out / "lunge.jsonl")
    if len(rows) < 2 or any(not r.get("cells") or r["cells"][0].get("stumble") is None for r in rows):
        return s46.fail("smoke (lunge test on the longer soles): no cells or no stumble count, see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def main() -> int:
    (HERE / "batch.failed").unlink(missing_ok=True)
    for name in DOGS:
        make_dog(name)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    if sys.argv[1:] == ["smoke"]:
        return 0
    for name in DOGS:
        for _ in range(2):  # once more after the CAD build's random crash: tests already done are skipped
            s46.test(name)
    panels = [f"old ball={STEP46 / 'hold_s2_tpu'}", f"now={HERE / 'hold_s2_ring'}", f"sole 3 mm={HERE / 'hold_s2_p3'}",
              f"sole 4.5 mm={HERE / 'hold_s2_p45'}"]
    for vname, terrain, ttype, level, drive, env_set, subtitle in VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if not out.exists():
            cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", terrain, "--level", str(level),
                   "--drive", drive, "--eye", *SIDE_EYE, "--subtitle", subtitle]
            cmd += ["--terrain_type", ttype] if ttype else []
            cmd += ["--env_set", env_set] if env_set else []
            run(cmd + [a for p in panels for a in ("--run", p)], HERE / "videos.log")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
