"""One model, many skills, first proof (journal step 45, next-steps section 0b): the stairs-up expert on a frozen base.

  base     step 44's best dog (22_still_clock/best.txt: the higher base score of hold / zero), frozen
  expert   a small MLP (128, 64) that reads the observations and the base's hidden layers and adds a correction to the
           base's action; a gate (64, 32) says how much, trained on the plan zone as its label (a stairs goal tile, going
           up, the body between the plan's first and last keyframe) and inferred from the sensors at run time
  teacher  the stair plans (19_stairs_proof/plans_all, the forward lunge of replay_up_r5_t15 / r4_t9 / r3_t15): in the
           zone a reward for the plan's body height and pitch at this point and its weight shift (body ahead of the
           paws' centre while a rear paw is up, behind it while a front paw is up); a reward for 3 paws down there;
           mid-climb starts from the plans (50 % of the stairs goal episodes)

  1  the base without expert: lunge test (weight shift, 3 paws down, climbed) for reference
  2  training on the base's course (dog_rough_stairs), 2000 it, seeds 1 and 2:
        plan    plan teacher (plan_track_weight 2.0) + 3 paws down (plan_support_weight 1.0)
        own     3 paws down only (the expert finds its own way: does the teacher matter?)
     each dog tested right after: lunge test on the stairs (weight shift vs plan, gate before / in / after the zone) and
     on the course (gate must stay shut), stairs goal test (the limit table), walking benchmark, shoves, getting up,
     tilt test (the base's skills must be unchanged)
  3  comparison videos (base / best plan / best own), from the side, up the three stairs of the replays, with the gate
     drawn as a bar; results.md
Starts only after step 44 (22_still_clock/batch.done). Smoke tests first (stop at the first failure). ~10 h.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "23_stairs_expert"
STEP44 = ROOT / "runs" / "exp2_dog" / "22_still_clock"
_spec = importlib.util.spec_from_file_location("still_clock_batch", STEP44 / "batch.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)
LAB, PY, O, SHOVE, SIDE_EYE, lines, has = s44.LAB, s44.PY, s44.O, s44.SHOVE, s44.SIDE_EYE, s44.lines, s44.has
PLANS = ROOT / "runs" / "exp2_dog" / "19_stairs_proof" / "plans_all"
RUNS = [("plan_s1", "plan", 1), ("own_s1", "own", 1), ("plan_s2", "plan", 2), ("own_s2", "own", 2)]
ITERATIONS = 2000
# the owner's three replays (19_stairs_proof/videos): (name, rise row = rise cm - 1, tread column)
VIDEOS = [("up_r5_t15", 4, "goal_stairs_t15"), ("up_r4_t9", 3, "goal_stairs_t9"), ("up_r3_t15", 2, "goal_stairs_t15")]


def base_dog() -> tuple[Path, str] | None:
    """Step 44's best dog and its clock setting, or None if step 44 has not finished."""
    if not (STEP44 / "batch.done").exists():
        return None
    best = json.loads((STEP44 / "best.txt").read_text())
    name = max((v for k, v in best.items() if k in ("hold", "zero")), key=lambda v: v[1])[0]
    return STEP44 / name, json.loads((STEP44 / name / "result.json").read_text())["env_overrides"]["gait_clock_idle"]


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = ITERATIONS) -> list[str]:
    base, clock = base_dog()
    cmd = s44.train_cmd(out, clock, seed, num_envs, iterations)  # the base's own recipe (course, recover, clock)
    for flag in ("--bc_walker", "--bc_getup", "--init_checkpoint"):  # the base is frozen: no teachers to copy
        i = cmd.index(flag)
        del cmd[i : i + 2]
    for flag in ("--bc_free_stairs", "--bc_free_idle"):
        cmd.remove(flag)
    cmd[cmd.index("--genome") + 1] = str(base / "genome.json")
    cmd += ["--expert_base", str(base), "--expert_dims", "128,64",
            O + f"stairs_rsi_plans={PLANS.as_posix()}", O + "stairs_rsi_prob=0.5", O + "plan_support_weight=1.0"]
    if variant == "plan":
        cmd += [O + "plan_track_weight=2.0"]
    return cmd


def run(cmd: list[str], log: Path) -> int:
    return s44.run(cmd, log)


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def lunge(dog: Path, log: Path, course: bool = False, out: Path | None = None, num_envs: int = 1000) -> None:
    out = out or HERE / "lunge.jsonl"
    if not has(out, dog.name, mode="course" if course else "stairs"):
        run([LAB, "tools/dog/lunge_test.py", str(dog), "--out", str(out), "--num_envs", str(num_envs)]
            + (["--course"] if course else []), log)


def video(out: Path, panels: list[str], row: int, ttype: str, subtitle: str, log: Path) -> None:
    if not out.exists():
        run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs", "--terrain_type",
             ttype, "--level", str(row), "--drive", "goal", "--env_set", "stairs_goal_dir=1", "--eye", *SIDE_EYE,
             "--subtitle", subtitle, *[a for p in panels for a in ("--run", p)]], log)


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    base, _ = base_dog()
    lunge(base, log, out=out / "lunge.jsonl", num_envs=200)
    rows = lines(out / "lunge.jsonl")
    if not rows or not rows[-1].get("cells") or rows[-1]["gate"]:
        return fail("smoke (lunge test on the base): no cells, or a gate on the base, see smoke/smoke.log")
    if not any((c["shift_front_cm"] is not None) for c in rows[-1]["cells"]):
        return fail("smoke (lunge test on the base): never one front paw up in a plan zone, see smoke/smoke.log")
    code = run(train_cmd(out / "train", "plan", 1, 64, 4), log)
    text = log.read_text(encoding="utf-8", errors="ignore")
    if code != 0 or not (out / "train" / "result.json").exists():
        return fail(f"smoke (expert training): exit code {code}, see smoke/smoke.log")
    for key in ("plan_track", "plan_zone", "gate_bce", "gate_on_mean"):
        if key not in text:
            return fail(f"smoke (expert training): {key!r} never logged, see smoke/smoke.log")
    res = json.loads((out / "train" / "result.json").read_text())
    if not res.get("expert_base"):
        return fail("smoke (expert training): expert_base not recorded")
    lunge(out / "train", log, out=out / "lunge.jsonl", num_envs=200)
    if not lines(out / "lunge.jsonl")[-1]["gate"]:
        return fail("smoke (lunge test on the expert): policy.pt has no gate_value, see smoke/smoke.log")
    run([LAB, "tools/dog/benchmark.py", str(out / "train"), "forward", "--num_envs", "64", "--out",
         str(out / "benchmark.jsonl")], log)
    rows = lines(out / "benchmark.jsonl")
    if not rows or rows[-1].get("physics") != "physx" or "fall_rate" not in rows[-1]:
        return fail("smoke (benchmark on the expert): no result, see smoke/smoke.log")
    v = out / "videos" / "smoke_up.mp4"
    video(v, [f"base={base}", f"expert={out / 'train'}"], 2, "goal_stairs_t15", "smoke", log)
    if not v.exists():
        return fail("smoke (video with the gate bar): see smoke/smoke.log")
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
        for _ in range(2):  # once more after a start-up hiccup
            if not has(HERE / out, name):
                run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], log)
    if not has(HERE / "tilt.jsonl", name):
        run([LAB, "tools/dog/tilt_test.py", str(dog), "--per_cell", "10", "--out", str(HERE / "tilt.jsonl")], log)


def climb_score(name: str) -> float:
    """Stairs up, the clean share on the treads of the three replays (15 and 9 cm), from the stairs goal test."""
    rows = [r for r in lines(HERE / "stairs_goal.jsonl") if Path(r["run"]).name == name]
    if not rows:
        return -1.0
    cells = [c for c in rows[-1]["cells"] if c["dir"] == "up" and c["walls"] is None and c["tread_cm"] in (15, 9)]
    return sum(c["clean"] for c in cells) / max(len(cells), 1)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if base_dog() is None:
        fail("step 44 (22_still_clock) has not finished: no base yet")
        return 1
    base, clock = base_dog()
    (HERE / "base.txt").write_text(f"{base.name} (gait_clock_idle={clock})\n")
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    # 1 the base for reference
    log = HERE / "measure.log"
    lunge(base, log)
    if not has(HERE / "stairs_goal.jsonl", base.name):
        run([LAB, "tools/dog/stairs_goal_test.py", str(base), "--out", str(HERE / "stairs_goal.jsonl")], log)
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
        if climb_score(name) > climb_score(best.get(variant, "")):
            best[variant] = name
    panels = [f"base ({base.name})={base}"] + [f"{v} ({best[v]})={HERE / best[v]}" for v in ("plan", "own") if v in best]
    if len(panels) >= 2:
        for vname, row, ttype in VIDEOS:
            video(HERE / "videos" / f"compare_{vname}.mp4", panels, row, ttype,
                  f"up {row + 1} cm steps, {ttype.split('_t')[-1]} cm treads (the replay's stairs); bar = expert gate",
                  HERE / "videos.log")
    (HERE / "best.txt").write_text(json.dumps({v: [n, round(climb_score(n), 3)] for v, n in best.items()}
                                              | {"base": [base.name, round(climb_score(base.name), 3)]}) + "\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
