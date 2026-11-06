"""Base skills B1 + B2, second try (journal step 44): stop the gait clock while standing. Step 43 found no dog ever still
(~3.8 rad/s summed joint speed while told to stand, before and after training): the gait clock kept ticking while the
command was "stand still", and the policy, trained for thousands of iterations to step with it, kept stepping.

  1  measurements without training: the start dog free_s2 with the clock stopped (hold / zero), calm and shoved, and
     getting up (must not change: the clock runs while fallen)
  2  training from free_s2 with step 43's "still" recipe (--bc_free_idle, stand_still_weight 1.0 not while fallen,
     stand commands 25 %) plus the clock while standing, 3000 it (a new input, so longer than step 43), seeds 1 and 2:
        hold    the clock stops: a constant clock input while standing
        zero    the clock input reads 0, 0 while standing: a clear "no rhythm" signal
     each dog tested right after its training: tilt test (still, and with the shove), stairs goal test, walking
     benchmark, shoves while walking, getting up (nothing may be forgotten)
  3  comparison videos (start / best hold / best zero): standing with a shove on the flat and on a 15 deg plate, walk
     and stop twice, a descent (stairs kept?); results.md (step 43's summary tables)
Smoke tests first (stop at the first failure). ~12 h; one training at a time (RAM).
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "22_still_clock"
STEP43 = ROOT / "runs" / "exp2_dog" / "21_base_skills"
_spec = importlib.util.spec_from_file_location("base_skills_batch", STEP43 / "batch.py")
s43 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s43)
LAB, PY, O, START, SHOVE, SIDE_EYE, lines = s43.LAB, s43.PY, s43.O, s43.START, s43.SHOVE, s43.SIDE_EYE, s43.lines
RUNS = [("hold_s1", "hold", 1), ("zero_s1", "zero", 1), ("hold_s2", "hold", 2), ("zero_s2", "zero", 2)]
ITERATIONS = 3000
# (name, terrain, terrain type, level, drive, subtitle)
VIDEOS = [("stand_flat", "flat", None, 0, "stand_shove", "standing still on the flat, shoved 0.3 m/s sideways at 6 s"),
          ("stop_go", "flat", None, 0, "stop_go", "walk 25 cm/s, stop, walk, stop: still when stopped?"),
          ("tilt15_nose_down", "dog_tilt", "tilt_0", 3, "stand_shove", "15 deg plate, nose down, shoved at 6 s"),
          ("tilt15_sideways", "dog_tilt", "tilt_90", 3, "stand_shove", "15 deg plate, sideways, shoved at 6 s")]


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = ITERATIONS) -> list[str]:
    return s43.train_cmd(out, "still", seed, num_envs, iterations) + [O + f"gait_clock_idle={variant}"]


def run(cmd: list[str], log: Path) -> int:
    return s43.run(cmd, log)


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def has(path: Path, name: str, **match) -> bool:
    return any(Path(r.get("run", "")).name == name and all(r.get(k) == v for k, v in match.items()) for r in lines(path))


def tilt(dog: Path, log: Path, shove: float = 0.0, clock: str | None = None, out: Path | None = None,
         per_cell: int = 10) -> None:
    """tools/dog/tilt_test.py; clock: run the dog with this gait_clock_idle (else its own)."""
    out = out or HERE / "tilt.jsonl"
    sets = [f"gait_clock_idle={clock}"] if clock else []
    if not has(out, dog.name, hold=False, shove_m_s=shove, set=sets):
        run([LAB, "tools/dog/tilt_test.py", str(dog), "--per_cell", str(per_cell), "--shove", str(shove), "--out", str(out)]
            + [a for s in sets for a in ("--set", s)], log)


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    for clock in ("run", "hold", "zero"):
        if run([LAB, "tools/dog/clock_check.py", str(START), "--clock", clock], log) != 0:
            return fail(f"smoke (clock check, {clock}): see smoke/smoke.log")
    tilt(START, log, shove=SHOVE, clock="hold", out=out / "tilt.jsonl", per_cell=1)
    rows = lines(out / "tilt.jsonl")
    if not rows or rows[-1].get("set") != ["gait_clock_idle=hold"]:
        return fail("smoke (tilt test with --set): no result or setting not recorded, see smoke/smoke.log")
    code = run(train_cmd(out / "train", "zero", 1, 64, 4), log)
    if code != 0 or not (out / "train" / "result.json").exists():
        return fail(f"smoke (training with the clock stopped): exit code {code}, see smoke/smoke.log")
    res = json.loads((out / "train" / "result.json").read_text())
    if res.get("env_overrides", {}).get("gait_clock_idle") != "zero" or not res.get("bc_free_idle"):
        return fail("smoke (training): gait_clock_idle or bc_free_idle not recorded")
    # benchmark.py was changed by another session (2026-10-30: --physics, a "physics" field): check it once
    run([LAB, "tools/dog/benchmark.py", str(out / "train"), "forward", "--num_envs", "64", "--out",
         str(out / "benchmark.jsonl")], log)
    rows = lines(out / "benchmark.jsonl")
    if not rows or rows[-1].get("physics") != "physx" or "fall_rate" not in rows[-1]:
        return fail("smoke (benchmark): no result, or no physics / fall_rate field, see smoke/smoke.log")
    video = out / "videos" / "smoke_stop_go.mp4"
    run([PY, "-m", "core.evolution.compare_video", "--out", str(video), "--terrain", "flat", "--level", "0",
         "--drive", "stop_go", "--eye", *SIDE_EYE, "--subtitle", "smoke", "--run", f"smoke={out / 'train'}"], log)
    if not video.exists():
        return fail("smoke (stop_go video): see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def test(name: str) -> None:
    dog = HERE / name
    if not (dog / "result.json").exists():
        return
    log = HERE / "tests.log"
    tilt(dog, log)
    tilt(dog, log, shove=SHOVE)
    if not has(HERE / "stairs_goal.jsonl", name):
        run([LAB, "tools/dog/stairs_goal_test.py", str(dog), "--out", str(HERE / "stairs_goal.jsonl")], log)
    for sc in ("forward", "course", "stairs_straight"):
        if not has(HERE / "benchmark.jsonl", name, scenario=sc):
            run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], log)
    for tool, out in (("recover_test", "recover_test.jsonl"), ("getup_test", "getup_test.jsonl")):
        for _ in range(2):  # once more after a start-up hiccup
            if not has(HERE / out, name):
                run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], log)


def base_score(name: str) -> float:
    """B1 + B2 in one number: on plates up to 15 deg, share up x share of the time still (calm and shoved)."""
    rows = [r for r in lines(HERE / "tilt.jsonl") if Path(r["run"]).name == name and not r["hold"] and not r.get("set")]
    if not rows:
        return -1.0
    vals = [(1 - c["fell"]) * (c["still"] or 0.0) for r in rows for c in r["cells"] if c["angle_deg"] <= 15.0]
    return sum(vals) / len(vals)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    # 1 without training: is the stepping only the clock?
    log = HERE / "measure.log"
    for clock in ("hold", "zero"):
        tilt(START, log, clock=clock)
        tilt(START, log, shove=SHOVE, clock=clock)
    if not has(HERE / "getup_test_start.jsonl", START.name):
        run([LAB, "tools/dog/getup_test.py", str(START), "--set", "gait_clock_idle=hold",
             "--out", str(HERE / "getup_test_start.jsonl")], log)
    # 2 training, each dog tested right away
    for name, variant, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            code = run(train_cmd(HERE / name, variant, seed), HERE / f"{name}.log")
            if not (HERE / name / "result.json").exists():  # e.g. a Windows Update restart: stop, don't go on
                fail(f"training {name} ended without result.json (exit code {code}), see {name}.log")
                return 1
        test(name)
    # 3 videos
    best = {}
    for name, variant, _ in RUNS:
        if base_score(name) > base_score(best.get(variant, "")):
            best[variant] = name
    panels = [f"start (free_s2)={START}"] + [f"{v} ({best[v]})={HERE / best[v]}" for v in ("hold", "zero") if v in best]
    for vname, terrain, ttype, level, drive, subtitle in VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if out.exists() or len(panels) < 2:
            continue
        cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", terrain, "--level", str(level),
               "--drive", drive, "--eye", *SIDE_EYE, "--subtitle", subtitle]
        if ttype:
            cmd += ["--terrain_type", ttype, "--env_set", "tilt_reset_yaw_random=False"]
        run(cmd + [a for p in panels for a in ("--run", p)], HERE / "videos.log")
    out = HERE / "videos" / "compare_down_r4_t15.mp4"
    if not out.exists() and len(panels) >= 2:
        run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs",
             "--terrain_type", "goal_stairs_t15", "--level", "3", "--drive", "goal", "--env_set", "stairs_goal_dir=-1",
             "--subtitle", "down 4 cm steps, 15 cm treads (stairs kept?)", *[a for p in panels for a in ("--run", p)]],
            HERE / "videos.log")
    (HERE / "best.txt").write_text(json.dumps({v: [n, round(base_score(n), 3)] for v, n in best.items()}
                                              | {"start free_s2": round(base_score(START.name), 3)}) + "\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
