"""A level body on the stairs (journal step 50, owner 2026-11-02: the dog does not balance going up, its front is too high, its
rear legs should stretch more, it flips onto its back if it moves).

teach_s1 (step 49, the 4.5 mm sole) trained on, the whole policy (step 45's expert on a frozen base learned nothing), step
49's recipe, 3000 it, seeds 1 and 2, on the stairs going up rewarded for the capped plans (plan_caps.py): nose up at most
15 deg, the free plan only on 5 / 15, which has no capped one. Keeps the body low and level (plan_track height, pitch), the
weight ahead of the paws with a rear paw up and behind them with a front paw up, at least 3 paws down (plan_support), half
of the stairs goal episodes start mid-climb in a plan's pose. plan_shift_sigma 4 cm (step 45: 2 cm): the dog is ~5 cm off,
where 2 cm pays nothing.

Plans are from the sphere paw of the same sole height: a reference, the dog keeps its real paw. Tests as step 49 + lunge test
against the capped plans (teach_s1 for reference). Videos teach_s1 / level_s1 / level_s2. Smoke first. ~5.5 h.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S49 = HERE.parent / "29_long_sole"
_spec = importlib.util.spec_from_file_location("long_sole_batch", S49 / "batch.py")
s49 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s49)
s47 = s49.s47
s47.HERE = HERE  # its test() and scores() read and write next to this batch
PY, LAB, O, SIDE_EYE, run, lines, has = s47.s44.PY, s47.LAB, s47.O, s47.SIDE_EYE, s47.run, s47.lines, s47.has
START = S49 / "teach_s1"
PLANS = ",".join((HERE / f"plans_{p}_m15").as_posix() for p in ("p15", "p45"))
RUNS = [("level_s1", 1), ("level_s2", 2)]
DIRECTION, LABEL = 1, "level body"
VIDEOS = [("up_r3_t12", "goal_stairs_t12", 2, "up 3 cm steps, 12 cm treads"), ("up_r4_t15", "goal_stairs_t15", 3, "up 4 cm steps, 15 cm treads"),
          ("up_r4_t9", "goal_stairs_t9", 3, "up 4 cm steps, 9 cm treads"), ("up_r5_t15", "goal_stairs_t15", 4, "up 5 cm steps, 15 cm treads")]


def train_cmd(out: Path, seed: int, num_envs: int = 2048, iterations: int = s47.ITERATIONS) -> list[str]:
    cmd = s49.train_cmd(out, "teach", seed, num_envs, iterations)
    cmd[cmd.index("--init_checkpoint") + 1] = str(START / "model_final.pt")
    return cmd + [O + f"stairs_rsi_plans={PLANS}", O + "stairs_rsi_prob=0.5", O + "plan_track_weight=2.0",
                  O + "plan_shift_sigma=0.04", O + "plan_support_weight=1.0"]


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def lunge(dog: Path, log: Path, out: Path | None = None, num_envs: int = 1000) -> None:
    out = out or HERE / "lunge_capped.jsonl"
    if not has(out, dog.name, mode="stairs"):
        run([LAB, "tools/dog/lunge_test.py", str(dog), "--plans", PLANS, "--out", str(out), "--num_envs", str(num_envs)], log)


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(exist_ok=True)
    log = out / "smoke.log"
    dog = out / "train"
    code = run(train_cmd(dog, 1, 64, 4), log)
    if code != 0 or not (dog / "result.json").exists():
        return fail(f"smoke (training): exit code {code}, see smoke/smoke.log")
    text = log.read_text(encoding="utf-8", errors="ignore")
    for key in ("plan_track", "plan_support", "mid-climb starts"):
        if key not in text:
            return fail(f"smoke: {key!r} never logged, see smoke/smoke.log")
    env = json.loads((dog / "result.json").read_text()).get("env_overrides", {})
    if env.get("plan_shift_sigma") != 0.04 or "plans_p15_m15" not in str(env.get("stairs_rsi_plans")):
        return fail("smoke: the plan settings are not recorded in result.json")
    legs = json.loads((dog / "genome.json").read_text())["legs"]
    if legs.get("paw_protrusion") != 4.5 or legs.get("sole_stop") != 2.0:
        return fail("smoke: the genome has no 4.5 mm sole with its stop")
    lunge(dog, log, out / "lunge.jsonl", 200)
    if not lines(out / "lunge.jsonl"):
        return fail("smoke (lunge test against the capped plans): see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def main() -> int:
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    if sys.argv[1:] == ["smoke"]:
        return 0
    lunge(START, HERE / "tests.log")
    for name, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            code = run(train_cmd(HERE / name, seed), HERE / f"{name}.log")
            if not (HERE / name / "result.json").exists():
                fail(f"training {name} ended without result.json (exit code {code}), see {name}.log")
                return 1
        s47.test(name)
        lunge(HERE / name, HERE / "tests.log")
    panels = [f"before (teach_s1)={START}"] + [f"{LABEL} ({n})={HERE / n}" for n, _ in RUNS]
    for vname, ttype, row, subtitle in VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if not out.exists():
            run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs", "--terrain_type", ttype, "--level", str(row),
                 "--drive", "goal", "--env_set", f"stairs_goal_dir={DIRECTION}", "--eye", *SIDE_EYE, "--subtitle", subtitle, *[a for p in panels for a in ("--run", p)]],
                HERE / "videos.log")
    for vname, terrain, ttype, level, drive, env_set, subtitle in s47.VIDEOS[:1] + s47.VIDEOS[-1:]:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if not out.exists():
            cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", terrain, "--level", str(level), "--drive", drive, "--eye", *SIDE_EYE, "--subtitle", subtitle]
            cmd += ["--terrain_type", ttype] if ttype else []
            cmd += ["--env_set", env_set] if env_set else []
            run(cmd + [a for p in panels for a in ("--run", p)], HERE / "videos.log")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
