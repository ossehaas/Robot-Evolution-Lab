"""Teach the stairs expert (journal step 45b; owner 2026-10-31: after the ring-paw base, steps 1-4).

Step 45: the gate switched cleanly but the expert learned nothing: its correction stayed ~0.01 because the quiet loss
(the correction pulled to zero on the ~90 % of samples outside the stairs zone) outweighed the sparse plan reward. Here,
on the new base (step 47's best dog on the ring paw, frozen), the quiet loss is off (the gate already shuts the expert
outside the zone) and the expert gets the plans as a demonstration:

  copy      plan reward (plan_track 2.0, 3 paws down 1.0) + copy loss (weight 1.0): in the zone the whole action
            (base + gate x correction) is pulled to the plan's joint pose at the nearest keyframe (x and the leg up),
            the standing legs' hips and knees and the neck (the swinging leg's pose holds only before it lifts)
  noquiet   the plan reward only, quiet loss off (is the quiet loss alone the cause?)
  2000 it each, seeds 1 and 2; the tests and videos of step 45 (lunge: weight shift against the plan, stuck paws,
  gate before / in / after the zone and on the course; stairs goal; walking; shoves; get-up; tilt).
Starts only when step 47 passed (26_ring_base/best.txt "ok"). Smoke first. ~6 h.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "27_stairs_expert_b"
S47 = ROOT / "runs" / "exp2_dog" / "26_ring_base"
_spec = importlib.util.spec_from_file_location("stairs_expert_batch", ROOT / "runs" / "exp2_dog" / "23_stairs_expert" / "batch.py")
s45 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s45)
LAB, PY, O, lines, has, SIDE_EYE = s45.LAB, s45.PY, s45.O, s45.lines, s45.has, s45.SIDE_EYE
RUNS = [("copy_s1", "copy", 1), ("noquiet_s1", "noquiet", 1), ("copy_s2", "copy", 2), ("noquiet_s2", "noquiet", 2)]
ITERATIONS = 2000


def base_dog() -> tuple[Path, str] | None:
    """Step 47's best dog if it passed, with its clock setting."""
    best = S47 / "best.txt"
    if not (S47 / "batch.done").exists() or not best.exists():
        return None
    b = json.loads(best.read_text())
    return (S47 / b["best"], "hold") if b.get("ok") else None


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = ITERATIONS) -> list[str]:
    s45.base_dog = base_dog  # step 45's recipe on this base
    cmd = s45.train_cmd(out, "plan", seed, num_envs, iterations) + ["--expert_quiet", "0"]
    if variant == "copy":
        cmd += ["--expert_copy", "1.0"]
    return cmd


def run(cmd: list[str], log: Path) -> int:
    return s45.run(cmd, log)


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def lunge(dog: Path, log: Path, course: bool = False, out: Path | None = None, num_envs: int = 1000) -> None:
    out = out or HERE / "lunge.jsonl"
    if not has(out, dog.name, mode="course" if course else "stairs"):
        run([LAB, "tools/dog/lunge_test.py", str(dog), "--out", str(out), "--num_envs", str(num_envs)]
            + (["--course"] if course else []), log)


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    code = run(train_cmd(out / "train", "copy", 1, 64, 4), log)
    text = log.read_text(encoding="utf-8", errors="ignore")
    if code != 0 or not (out / "train" / "result.json").exists():
        return fail(f"smoke (expert training with the copy loss): exit code {code}, see smoke/smoke.log")
    for key in ("expert_copy", "gate_bce", "plan_track"):
        if key not in text:
            return fail(f"smoke (expert training): {key!r} never logged, see smoke/smoke.log")
    res = json.loads((out / "train" / "result.json").read_text())
    if res.get("expert_copy") != 1.0 or res.get("expert_quiet") != 0.0:
        return fail("smoke: expert_copy / expert_quiet not recorded")
    lunge(out / "train", log, out=out / "lunge.jsonl", num_envs=200)
    if not lines(out / "lunge.jsonl") or not lines(out / "lunge.jsonl")[-1]["gate"]:
        return fail("smoke (lunge test on the expert): no result or no gate, see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def test(name: str) -> None:
    dog = HERE / name
    if not (dog / "result.json").exists():
        return
    log = HERE / "tests.log"
    lunge(dog, log)
    lunge(dog, log, course=True)
    if not has(HERE / "stairs_goal.jsonl", name):
        run([LAB, "tools/dog/stairs_goal_test.py", str(dog), "--out", str(HERE / "stairs_goal.jsonl")], log)
    for sc in ("forward", "course"):
        if not has(HERE / "benchmark.jsonl", name, scenario=sc):
            run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], log)
    for tool, out in (("recover_test", "recover_test.jsonl"), ("getup_test", "getup_test.jsonl")):
        for _ in range(2):
            if not has(HERE / out, name):
                run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], log)
    if not has(HERE / "tilt.jsonl", name, hold=False, shove_m_s=0.0):
        run([LAB, "tools/dog/tilt_test.py", str(dog), "--per_cell", "10", "--out", str(HERE / "tilt.jsonl")], log)


def climb_score(name: str) -> float:
    rows = [r for r in lines(HERE / "stairs_goal.jsonl") if Path(r["run"]).name == name]
    if not rows:
        return -1.0
    cells = [c for c in rows[-1]["cells"] if c["dir"] == "up" and c["walls"] is None and c["tread_cm"] in (15, 9)]
    return sum(c["clean"] for c in cells) / max(len(cells), 1)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if base_dog() is None:
        fail("step 47 (26_ring_base) has not finished or did not pass: no base")
        return 1
    base, _ = base_dog()
    (HERE / "base.txt").write_text(f"{base.name} (gait_clock_idle=hold, ring paw)\n")
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    log = HERE / "measure.log"
    lunge(base, log)
    lunge(base, log, course=True)
    if not has(HERE / "stairs_goal.jsonl", base.name):
        run([LAB, "tools/dog/stairs_goal_test.py", str(base), "--out", str(HERE / "stairs_goal.jsonl")], log)
    for name, variant, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            code = run(train_cmd(HERE / name, variant, seed), HERE / f"{name}.log")
            if not (HERE / name / "result.json").exists():
                fail(f"training {name} ended without result.json (exit code {code}), see {name}.log")
                return 1
        test(name)
    best = {}
    for name, variant, _ in RUNS:
        if climb_score(name) > climb_score(best.get(variant, "")):
            best[variant] = name
    panels = [f"base ({base.name})={base}"] + [f"{v} ({best[v]})={HERE / best[v]}" for v in ("copy", "noquiet") if v in best]
    for vname, row, ttype in s45.VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if not out.exists() and len(panels) >= 2:
            run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs", "--terrain_type",
                 ttype, "--level", str(row), "--drive", "goal", "--env_set", "stairs_goal_dir=1", "--eye", *SIDE_EYE,
                 "--subtitle", f"up {row + 1} cm steps, {ttype.split('_t')[-1]} cm treads; bar = expert gate",
                 *[a for p in panels for a in ("--run", p)]], HERE / "videos.log")
    (HERE / "best.txt").write_text(json.dumps({v: [n, round(climb_score(n), 3)] for v, n in best.items()}
                                              | {"base": [base.name, round(climb_score(base.name), 3)]}) + "\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
