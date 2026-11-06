"""One model, many skills (journal step 53, owner 2026-11-05: walking, up and down in one model; a recognizer outside it only
sends a signal. Does the signal help the model tell the skills apart?).

No rewards: the student copies teach_s1 (walking, up, tilted) and copy_s1 (down), equal shares per kind of sample
(core/evolution/distill_skills.py). A perfect recognizer for now. Three arms, seeds 1 and 2:
  none   one output, no signal (the control)
  input  one output, the signal as an input
  heads  the signal as an input, one output section per skill
The mark: up about teach_s1's 0.432 clean, down about copy_s1's 0.336, walking and getting up no worse than teach_s1. The teachers'
flaws come along: passing says the skills can be combined, not that they are good. Tests as step 50; videos show hand-overs
(the goal flips at the top and bottom). Smoke first, with every test tool: step 51 lost its tests to a crash no smoke saw.

    lab runs/exp2_dog/33_one_model/batch.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UP = HERE.parent / "29_long_sole" / "teach_s1"
DOWN = HERE.parent / "31_copy_plans" / "copy_s1"
_spec = importlib.util.spec_from_file_location("level_body_batch", HERE.parent / "30_level_body" / "batch.py")
s50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s50)
s47, LAB, PY, SIDE_EYE, run, has, lines = s50.s47, s50.LAB, s50.PY, s50.SIDE_EYE, s50.run, s50.has, s50.lines
s50.HERE = s47.HERE = HERE
ARMS = ("none", "input", "heads")
RUNS = [(f"{arm}_s{seed}", arm, seed) for seed in (1, 2) for arm in ARMS]
VIDEOS = [("up_r4_t15", "goal_stairs_t15", 3, 1, UP, "teach_s1", "up 4 cm steps, 15 cm treads"), ("up_r3_t12", "goal_stairs_t12", 2, 1, UP, "teach_s1", "up 3 cm steps, 12 cm treads"),
          ("down_r4_t15", "goal_stairs_t15", 3, -1, DOWN, "copy_s1", "down 4 cm steps, 15 cm treads"), ("down_r3_t15", "goal_stairs_t15", 2, -1, DOWN, "copy_s1", "down 3 cm steps, 15 cm treads")]
TEST_FILES = ("stairs_goal.jsonl", "benchmark.jsonl", "recover_test.jsonl", "getup_test.jsonl", "tilt.jsonl", "lunge_capped.jsonl")


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    print(msg, flush=True)
    return False


def train_cmd(out: Path, arm: str, seed: int, num_envs: int = 2048, iterations: int = 1000) -> list[str]:
    return [LAB, "-m", "core.evolution.distill_skills", "--up_teacher", str(UP), "--down_teacher", str(DOWN), "--arm", arm, "--out", str(out),
            "--seed", str(seed), "--num_envs", str(num_envs), "--iterations", str(iterations)]


def tested(name: str) -> list[str]:
    """The test files without a line for this dog: a test that crashes must stop the batch, not leave a hole."""
    return [f for f in TEST_FILES if not has(HERE / f, name)]


def test(name: str) -> None:
    s47.test(name)
    s50.lunge(s50.HERE / name, s50.HERE / "tests.log")


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(exist_ok=True)
    log = out / "smoke.log"
    for arm in ("none", "heads"):
        dog = out / arm
        code = run(train_cmd(dog, arm, 1, 64, 4), log)
        if code != 0 or not (dog / "policy.pt").exists() or not (dog / "result.json").exists():
            return fail(f"smoke ({arm}): exit code {code}, see smoke/smoke.log")
    text = log.read_text(encoding="utf-8", errors="ignore")
    for key in ("arm none", "arm heads", "output section", "loss walk"):
        if key not in text:
            return fail(f"smoke: {key!r} never logged, see smoke/smoke.log")
    # every test tool, on the model with output sections, as the real runs do
    s47.HERE = s50.HERE = out
    test("heads")
    s47.HERE = s50.HERE = HERE
    holes = [f for f in TEST_FILES if not has(out / f, "heads")]
    if holes:
        return fail(f"smoke: no result from {holes}, see smoke/tests.log")
    video = out / "video.mp4"
    run([PY, "-m", "core.evolution.compare_video", "--out", str(video), "--terrain", "dog_stairs", "--terrain_type", "goal_stairs_t15", "--level", "2", "--drive", "goal",
         "--env_set", "stairs_goal_dir=-1", "--eye", *SIDE_EYE, "--subtitle", "smoke", "--run", f"heads={out / 'heads'}"], log)
    if not video.exists():
        return fail("smoke: no video, see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def main() -> int:
    (HERE / "batch.failed").unlink(missing_ok=True)
    for dog in (UP, DOWN):
        if not (dog / "policy.pt").exists():
            fail(f"no teacher at {dog}")
            return 1
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    if sys.argv[1:] == ["smoke"]:
        return 0
    s50.lunge(UP, HERE / "tests.log")
    for name, arm, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            code = run(train_cmd(HERE / name, arm, seed), HERE / f"{name}.log")
            if not (HERE / name / "result.json").exists():
                fail(f"training {name} ended without result.json (exit code {code}), see {name}.log")
                return 1
        test(name)
        holes = tested(name)
        if holes:
            fail(f"{name}: no result from {holes}, see tests.log")
            return 1
    for vname, ttype, row, direction, teacher, tname, subtitle in VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if not out.exists():
            panels = [f"teacher ({tname})={teacher}"] + [f"{arm} (s1)={HERE / f'{arm}_s1'}" for arm in ARMS]
            run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs", "--terrain_type", ttype, "--level", str(row), "--drive", "goal",
                 "--env_set", f"stairs_goal_dir={direction}", "--eye", *SIDE_EYE, "--subtitle", subtitle, *[a for p in panels for a in ("--run", p)]], HERE / "videos.log")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
