"""The base on the 4.5 mm sole (journal step 49, owner 2026-11-01: draw the sole with a stop, retrain on it).

Step 48: the sole 4.5 mm below the rim gave the best walk of the untrained dogs (0.36, current paw 0.30, old ball 0.45).
The paw as designed: genome legs.paw_protrusion = 4.5, sole_stop = 2 mm (a PLA pin, in the simulation a rigid sphere
behind the sole). hold_s2 (step 44) trained on with it, step 47's recipe, 3000 it, seeds 1 and 2, two ways:
  teach   copies the walking teacher while upright, as in step 47 (a dog that learned on the sphere paw)
  free    only the get-up teacher: nothing holds the sphere paw's gait in place (the style judge and the rewards stay)
First the untrained dog on this paw (start): the stop's own effect against step 48's hold_s2_p45.

Tests as step 47; the same pass mark for step 45b (best.txt). Videos old ball / current paw retrained / 4.5 mm teach /
4.5 mm free. Smoke first. One training at a time; ~11 h.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S46, S47 = HERE.parent / "25_hard_paw", HERE.parent / "26_ring_base"
_spec = importlib.util.spec_from_file_location("ring_base_batch", S47 / "batch.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)
s47.HERE = HERE  # its test() and scores() read and write next to this batch
s47.GENOME = HERE / "genome.json"
PY, SIDE_EYE, run, OK, START = s47.PY, s47.SIDE_EYE, s47.run, s47.OK, s47.START
RUNS = [("teach_s1", "teach", 1), ("free_s1", "free", 1), ("teach_s2", "teach", 2), ("free_s2", "free", 2)]


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = s47.ITERATIONS) -> list[str]:
    cmd = s47.train_cmd(out, seed, num_envs, iterations)
    return cmd + ["--bc_free_walk"] if variant == "free" else cmd


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(exist_ok=True)
    log = out / "smoke.log"
    for variant in ("teach", "free"):
        dog = out / variant
        code = run(train_cmd(dog, variant, 1, 64, 4), log)
        if code != 0 or not (dog / "result.json").exists():
            return fail(f"smoke (training, {variant}): exit code {code}, see smoke/smoke.log")
        legs = json.loads((dog / "genome.json").read_text())["legs"]
        if legs.get("paw_protrusion") != 4.5 or legs.get("sole_stop") != 2.0:
            return fail(f"smoke ({variant}): the genome has no 4.5 mm sole with its stop")
        res = json.loads((dog / "result.json").read_text())
        if res.get("bc_free_walk") != (variant == "free"):
            return fail(f"smoke ({variant}): bc_free_walk not recorded as expected")
        spheres = next((dog / "robot").glob("*.usda")).read_text(encoding="utf-8").count('def Sphere')
        old = next((S46 / "hold_s2_ring" / "robot").glob("*.usda")).read_text(encoding="utf-8").count('def Sphere')
        if spheres != old + 4:
            return fail(f"smoke ({variant}): {spheres} spheres in the USD, expected {old + 4} (one stop per paw)")
    (out / "ok").write_text("ok")
    return True


def make_start() -> None:
    dog = HERE / "start"
    if not dog.exists():
        dog.mkdir()
        for file in ("model_final.pt", "policy.pt", "result.json"):
            shutil.copy2(START / file, dog / file)
        shutil.copy2(HERE / "genome.json", dog / "genome.json")


def main() -> int:
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    if sys.argv[1:] == ["smoke"]:
        return 0
    make_start()
    s47.test("start")
    for name, variant, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            code = run(train_cmd(HERE / name, variant, seed), HERE / f"{name}.log")
            if not (HERE / name / "result.json").exists():
                fail(f"training {name} ended without result.json (exit code {code}), see {name}.log")
                return 1
        s47.test(name)
    total = lambda n: (s47.scores(n)["walk_score"] or 0) + (s47.scores(n)["stairs_up"] or 0)  # noqa: E731
    best = {v: max((n for n, w, _ in RUNS if w == v), key=total) for v in ("teach", "free")}
    panels = [f"old ball, untrained={S46 / 'hold_s2_tpu'}", f"current paw, retrained={S47 / 'ring_s1'}",
              f"4.5 mm with teacher ({best['teach']})={HERE / best['teach']}",
              f"4.5 mm no teacher ({best['free']})={HERE / best['free']}"]
    for vname, terrain, ttype, level, drive, env_set, subtitle in s47.VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if out.exists():
            continue
        cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", terrain, "--level", str(level),
               "--drive", drive, "--eye", *SIDE_EYE, "--subtitle", subtitle]
        cmd += ["--terrain_type", ttype] if ttype else []
        cmd += ["--env_set", env_set] if env_set else []
        run(cmd + [a for p in panels for a in ("--run", p)], HERE / "videos.log")
    top = max(best.values(), key=total)
    s = s47.scores(top)
    (HERE / "best.txt").write_text(json.dumps({"best": top, "ok": s47.passes(s), "scores": s, "pass_mark": OK}) + "\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
