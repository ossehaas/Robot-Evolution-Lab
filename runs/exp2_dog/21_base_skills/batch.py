"""Base skills B1 + B2 (journal step 43, .documentation/base-skills.md): stand balanced and still, stay balanced on a
tilt and move only when unstable. Dog v1's line fails both (step 42: it never stands still, ~2 W, ~3.5 rad/s).

  1  measurements before training (tools/dog/tilt_test.py, now with stillness and a 0.3 m/s shove at 6 s): the start
     dog free_s2 (the owner's pick from step 42's videos), the walking teacher (does it march on the spot? its
     copying holds the start dog's gait while standing), the bare body (servos holding the pose) with the shove
  2  training from free_s2 (its recipe, 2000 it, seeds 1 and 2):
        still   stand still asked for: no copying of the walking teacher while the command is "stand still"
                (--bc_free_idle), stand_still_weight 0.3 -> 1.0 per rad/s (not while fallen), stand commands 10 ->
                25 % of the time
        tilt    still + tilted pads in the course (3 more columns, 0-30 deg by row, a curriculum: steeper after a whole
                episode without a fall), the dog placed standing on the pad facing a random way, told to stand
                still; the usual shoves (0.2 m/s every 4-8 s, topple shoves 0.8-1.3 m/s every 15-25 s) knock it
     each dog tested right after its training: tilt test (still, and with the shove), stairs goal test, walking
     benchmark, shoves while walking, getting up (nothing may be forgotten)
  3  comparison videos (start / best still / best tilt): standing with a shove on the flat and on 20-25 deg plates,
     and a descent (stairs kept?); results.md
Smoke tests first (stop at the first failure). ~8 h; one training at a time (RAM).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "21_base_skills"
STEP41 = ROOT / "runs" / "exp2_dog" / "19_stairs_proof"
_spec = importlib.util.spec_from_file_location("stairs_proof_batch", STEP41 / "batch.py")
s41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s41)
LAB, PY, O, WALKER = s41.LAB, s41.PY, s41.O, s41.WALKER
START = s41.START  # 18_stairs_goal/free_s2
RUNS = [("still_s1", "still", 1), ("tilt_s1", "tilt", 1), ("still_s2", "still", 2), ("tilt_s2", "tilt", 2)]
SHOVE = 0.3  # [m/s] base skill B2
SIDE_EYE = ("0.15", "-0.95", "0.45")
# (name, terrain, terrain type, level, subtitle)
VIDEOS = [("stand_flat", "flat", None, 0, "standing still on the flat, shoved 0.3 m/s sideways at 6 s"),
          ("tilt20_nose_down", "dog_tilt", "tilt_0", 4, "20 deg plate, nose down, shoved at 6 s"),
          ("tilt25_sideways", "dog_tilt", "tilt_90", 5, "25 deg plate, sideways, shoved at 6 s"),
          ("tilt25_nose_up", "dog_tilt", "tilt_180", 5, "25 deg plate, nose up, shoved at 6 s")]


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = 2000) -> list[str]:
    cmd = s41.train_cmd(out, "cont", seed, num_envs, iterations)  # free_s2's own recipe, continued
    cmd += ["--bc_free_idle", O + "stand_still_weight=1.0", O + "stand_still_upright_only=True",
            O + "command_standing_prob=0.25"]
    if variant == "tilt":
        cmd[cmd.index("--terrain") + 1] = "dog_rough_stairs_tilt"
        # no style judge on the pads either (the recorded dog only walked on a flat floor)
        skip = next(i for i, a in enumerate(cmd) if a.startswith(O + "amp_skip_types="))
        cmd[skip] = O + f"amp_skip_types={s41.GOAL_TYPES + ('tilt_pad',)}"
    return cmd


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def has(path: Path, name: str, **match) -> bool:
    return any(Path(r.get("run", "")).name == name and all(r.get(k) == v for k, v in match.items()) for r in lines(path))


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def tilt(dog: Path, log: Path, hold: bool = False, shove: float = 0.0, out: Path | None = None, per_cell: int = 10) -> None:
    out = out or HERE / "tilt.jsonl"
    if not has(out, dog.name, hold=hold, shove_m_s=shove):
        run([LAB, "tools/dog/tilt_test.py", str(dog), "--per_cell", str(per_cell), "--shove", str(shove), "--out", str(out)]
            + (["--hold"] if hold else []), log)


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    tilt(START, log, shove=SHOVE, out=out / "tilt.jsonl", per_cell=1)
    rows = lines(out / "tilt.jsonl")
    if not rows or "still" not in rows[-1]["cells"][0] or "resettle_s" not in rows[-1]["cells"][0]:
        return fail("smoke (tilt test with stillness and shove): no result, see smoke/smoke.log")
    if run([LAB, "tools/dog/tilt_pad_check.py", str(START)], log) != 0:
        return fail("smoke (tilt pads: placement / height / commands): see smoke/smoke.log")
    code = run(train_cmd(out / "train", "tilt", 1, 64, 4), log)
    text = log.read_text(encoding="utf-8", errors="ignore")
    if code != 0 or not (out / "train" / "result.json").exists() or "tilt_level" not in text:
        return fail(f"smoke (training with the tilt pads): exit code {code}, see smoke/smoke.log")
    if not json.loads((out / "train" / "result.json").read_text()).get("bc_free_idle"):
        return fail("smoke (training): bc_free_idle not recorded")
    video = out / "videos" / "smoke_tilt.mp4"
    run([PY, "-m", "core.evolution.compare_video", "--out", str(video), "--terrain", "dog_tilt", "--terrain_type", "tilt_0",
         "--level", "3", "--drive", "stand_shove", "--env_set", "tilt_reset_yaw_random=False", "--eye", *SIDE_EYE,
         "--subtitle", "smoke", "--run", f"start={START}"], log)
    if not video.exists():
        return fail("smoke (video on a tilted plate): see smoke/smoke.log")
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
    """B1 + B2 in one number: on plates up to 20 deg, share up x share of the time still (no shove and shoved)."""
    rows = [r for r in lines(HERE / "tilt.jsonl") if Path(r["run"]).name == name and not r["hold"]]
    if not rows:
        return -1.0
    vals = [(1 - c["fell"]) * (c["still"] or 0.0) for r in rows for c in r["cells"] if c["angle_deg"] <= 20.0]
    return sum(vals) / len(vals)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    # 1 before training
    log = HERE / "measure.log"
    tilt(START, log)
    tilt(START, log, shove=SHOVE)
    tilt(WALKER, log)
    tilt(START, log, hold=True, shove=SHOVE)
    # 2 training, each dog tested right away
    for name, variant, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            run(train_cmd(HERE / name, variant, seed), HERE / f"{name}.log")
        test(name)
    # 3 videos
    best = {}
    for name, variant, _ in RUNS:
        if base_score(name) > base_score(best.get(variant, "")):
            best[variant] = name
    panels = [f"start (free_s2)={START}"] + [f"{v} ({best[v]})={HERE / best[v]}" for v in ("still", "tilt") if v in best]
    for vname, terrain, ttype, level, subtitle in VIDEOS:
        out = HERE / "videos" / f"compare_{vname}.mp4"
        if out.exists() or len(panels) < 2:
            continue
        cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", terrain, "--level", str(level),
               "--drive", "stand_shove", "--eye", *SIDE_EYE, "--subtitle", subtitle]
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
