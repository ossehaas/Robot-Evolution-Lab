"""The base on the ring paw (journal step 47, owner 2026-10-31: the ring paw, then retrain the base).

Step 46 showed that every policy so far learned on a sphere paw; on the printed paw shape (flat base, toes 27 mm ahead)
hold_s2 walks worse (walk score 0.45 -> 0.30) and climbs worse. The ring paw (PLA shell on a TPU base ring, genome
legs.paw_housing = "ring") is the chosen paw. Here hold_s2 (step 44) is trained on, with that paw: step 44's recipe
(clock held while standing, the walking and get-up teachers, stand-still terms), 3000 it, seeds 1 and 2.

Tests as step 46 (lunge / stuck, stairs goal, walking, get-up, shoves, tilt calm and shoved) + stairs straight; the
references are step 46's rows (hold_s2 with the sphere paw and with the ring paw, untrained). Videos sphere paw /
ring untrained / ring retrained: stop_go and standing shoved on the flat, up 4 cm / 9 cm treads, down 4 cm / 15 cm.
best.txt: the dog with the best walk score forward + clean share up the stairs, and whether it passes ("ok": walks
as hold_s2 did on the sphere, gets up, climbs): the gate for step 45b (27_stairs_expert_b). Smoke first. ~4.5 h.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "26_ring_base"
S46 = ROOT / "runs" / "exp2_dog" / "25_hard_paw"
START = ROOT / "runs" / "exp2_dog" / "22_still_clock" / "hold_s2"
_spec = importlib.util.spec_from_file_location("still_clock_batch", ROOT / "runs" / "exp2_dog" / "22_still_clock" / "batch.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)
LAB, PY, O, SHOVE, SIDE_EYE, lines, has = s44.LAB, s44.PY, s44.O, s44.SHOVE, s44.SIDE_EYE, s44.lines, s44.has
RUNS = [("ring_s1", 1), ("ring_s2", 2)]
ITERATIONS = 3000
GENOME = HERE / "genome_ring.json"
# the pass mark for going on to step 45b: hold_s2 on the sphere paw walked 0.45 / fell 1 %, got up 99-100 %, climbed
# 0.41 clean (step 46 run 2); a little slack for seed noise
OK = {"walk_score": 0.40, "fall_rate": 0.02, "getup": 0.95, "stairs_up": 0.37}
# (name, terrain, terrain type, level, drive, env_set, subtitle)
VIDEOS = [("stop_go", "flat", None, 0, "stop_go", None, "walk 25 cm/s, stop, walk, stop"),
          ("stand_flat", "flat", None, 0, "stand_shove", None, "standing on the flat, shoved 0.3 m/s sideways at 6 s"),
          ("up_r4_t9", "dog_stairs", "goal_stairs_t9", 3, "goal", "stairs_goal_dir=1", "up 4 cm steps, 9 cm treads"),
          ("down_r4_t15", "dog_stairs", "goal_stairs_t15", 3, "goal", "stairs_goal_dir=-1", "down 4 cm steps, 15 cm treads")]


def write_genome() -> None:
    g = json.loads((START / "genome.json").read_text())
    g["legs"]["paw_housing"] = "ring"
    GENOME.write_text(json.dumps(g, indent=2))


def train_cmd(out: Path, seed: int, num_envs: int = 2048, iterations: int = ITERATIONS) -> list[str]:
    cmd = s44.train_cmd(out, "hold", seed, num_envs, iterations)
    cmd[cmd.index("--genome") + 1] = str(GENOME)
    cmd[cmd.index("--init_checkpoint") + 1] = str(START / "model_final.pt")
    return cmd


def run(cmd: list[str], log: Path) -> int:
    return s44.run(cmd, log)


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    code = run(train_cmd(out / "train", 1, 64, 4), log)
    if code != 0 or not (out / "train" / "result.json").exists():
        return fail(f"smoke (training on the ring paw): exit code {code}, see smoke/smoke.log")
    g = json.loads((out / "train" / "genome.json").read_text())
    if g["legs"].get("paw_housing") != "ring":
        return fail("smoke: the trained dog's genome has no ring paw")
    res = json.loads((out / "train" / "result.json").read_text())
    if res.get("env_overrides", {}).get("gait_clock_idle") != "hold":
        return fail("smoke: gait_clock_idle=hold not recorded")
    (out / "ok").write_text("ok")
    return True


def test(name: str) -> None:
    dog = HERE / name
    if not (dog / "result.json").exists():
        return
    log = HERE / "tests.log"
    if not has(HERE / "lunge.jsonl", name, mode="stairs"):
        run([LAB, "tools/dog/lunge_test.py", str(dog), "--out", str(HERE / "lunge.jsonl")], log)
    if not has(HERE / "stairs_goal.jsonl", name):
        run([LAB, "tools/dog/stairs_goal_test.py", str(dog), "--out", str(HERE / "stairs_goal.jsonl")], log)
    for sc in ("forward", "course", "stairs_straight"):
        if not has(HERE / "benchmark.jsonl", name, scenario=sc):
            run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], log)
    for tool, out in (("recover_test", "recover_test.jsonl"), ("getup_test", "getup_test.jsonl")):
        for _ in range(2):  # once more after a start-up hiccup
            if not has(HERE / out, name):
                run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], log)
    for shove in (0.0, SHOVE):
        if not has(HERE / "tilt.jsonl", name, hold=False, shove_m_s=shove):
            run([LAB, "tools/dog/tilt_test.py", str(dog), "--per_cell", "10", "--shove", str(shove), "--out",
                 str(HERE / "tilt.jsonl")], log)


def scores(name: str) -> dict:
    """The pass-mark numbers of one dog (None where a test is missing)."""
    bench = {r["scenario"]: r for r in lines(HERE / "benchmark.jsonl") if Path(r["run"]).name == name}
    gu = [r for r in lines(HERE / "getup_test.jsonl") if Path(r["run"]).name == name]
    sg = [r for r in lines(HERE / "stairs_goal.jsonl") if Path(r["run"]).name == name]
    up = [c["clean"] for c in sg[-1]["cells"] if c["dir"] == "up" and c["walls"] is None] if sg else []
    fwd = bench.get("forward")
    return {"walk_score": fwd["walk_score"] if fwd else None, "fall_rate": fwd["fall_rate"] if fwd else None,
            "getup": min(gu[-1][c]["success"] for c in ("standing", "dropped upright", "on its side", "on its back"))
            if gu else None,
            "stairs_up": sum(up) / len(up) if up else None}


def passes(s: dict) -> bool:
    return (None not in s.values() and s["walk_score"] >= OK["walk_score"] and s["fall_rate"] <= OK["fall_rate"]
            and s["getup"] >= OK["getup"] and s["stairs_up"] >= OK["stairs_up"])


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    write_genome()
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    for name, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            code = run(train_cmd(HERE / name, seed), HERE / f"{name}.log")
            if not (HERE / name / "result.json").exists():
                fail(f"training {name} ended without result.json (exit code {code}), see {name}.log")
                return 1
        test(name)
    best = max((n for n, _ in RUNS), key=lambda n: (scores(n)["walk_score"] or 0) + (scores(n)["stairs_up"] or 0))
    panels = [f"sphere paw (hold_s2)={S46 / 'hold_s2_tpu'}", f"ring paw, untrained={S46 / 'hold_s2_ring'}",
              f"ring paw, retrained ({best})={HERE / best}"]
    for vname, terrain, ttype, level, drive, env_set, subtitle in VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if out.exists():
            continue
        cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", terrain, "--level", str(level),
               "--drive", drive, "--eye", *SIDE_EYE, "--subtitle", subtitle]
        if ttype:
            cmd += ["--terrain_type", ttype]
        if env_set:
            cmd += ["--env_set", env_set]
        run(cmd + [a for p in panels for a in ("--run", p)], HERE / "videos.log")
    s = scores(best)
    (HERE / "best.txt").write_text(json.dumps({"best": best, "ok": passes(s), "scores": s, "pass_mark": OK}) + "\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
