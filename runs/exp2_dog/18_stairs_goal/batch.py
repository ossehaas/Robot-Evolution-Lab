"""Stairs as a goal task, and the climbing-limit table (journal step 40).

The course (core/sim/terrain.py, DOG_STAIRS): one straight staircase per tile, 5 risers to a landing, rise 1-10 cm over
the rows, treads 25 / 20 / 15 / 12 / 9 cm with open sides or walls over the columns. The dog walks it up (from ~0.8 m
in front of the first riser to the landing) or down (from the landing to the floor); a navigation layer turns the goal
into walking commands (turn towards it, then walk at it at 20 cm/s), so the walker stays one policy and cannot avoid
the stairs. A fall on the stairs is soft (the dog gets up and goes on); falling down the stairs (more than one step
lower while fallen, or off an open side) is a hard fail.

  1  zero-shot: dog v1 (17_recover_bc/bc_s2) and, for reference, the original walker (cannot get up) on the stairs goal
     test (tools/dog/stairs_goal_test.py: 2000 dogs x 2 rounds, every tile both ways) -> the climbing-limit table
  2  training from dog v1 on the dog course plus the stairs goal tiles (terrain dog_rough_stairs: 12 + 10 columns; on the
     stairs tiles the navigation layer drives, the goal flips at each arrival so it goes up and down; curriculum over
     the rise; no style reward and no topple shoves there), RL with gated behaviour cloning (step 39), 2000 iterations,
     seeds 1 and 2, two variants:
        anchor  the walker teacher is copied everywhere while upright (as dog v1 was trained)
        free    no walker copying on the stairs goal tiles (RL may change the stair technique) and the standing-height
                reward there at 0.3 (crouching allowed); the get-up teacher still applies everywhere
  3  tests of the trained dogs: stairs goal test, walking benchmark (forward, course, stairs_straight), shove and
     recovery, get-up (nothing may be forgotten)
  4  videos (tread 15 cm, open sides; dog v1 / best anchor / best free): up and down at the sweet spot and at the limit
     of the best dog, and up beyond its limit
  5  results.md (summary.py)
A smoke test first (stops at the first failure). ~6.5 h.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
HERE = ROOT / "runs" / "exp2_dog" / "18_stairs_goal"
PREV = ROOT / "runs" / "exp2_dog" / "16_recover"
sys.path.insert(0, str(PREV))
from batch import GETUP, LAB, O, PY, WALKER, walker_recipe  # noqa: E402

DOG_V1 = ROOT / "runs" / "exp2_dog" / "17_recover_bc" / "bc_s2"
GOAL_TYPES = tuple(f"goal_stairs_t{t}{w}" for t in (25, 20, 15, 12, 9) for w in ("", "_walls"))
RUNS = [("anchor_s1", "anchor", 1), ("anchor_s2", "anchor", 2), ("free_s1", "free", 1), ("free_s2", "free", 2)]
VIDEO_TREAD = 15


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = 2000) -> list[str]:
    cmd = [LAB, "-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough_stairs", "--sensors", "imu",
           "--obs_history", "8", "--symmetry", "--height_scan", "--genome", str(DOG_V1 / "genome.json"),
           "--init_checkpoint", str(DOG_V1 / "model_final.pt"), "--bc_walker", str(WALKER), "--bc_getup", str(GETUP),
           *walker_recipe(), O + "action_scale=1.0", O + "recover=True", O + "policy_drives_expression=True",
           O + "self_collisions=True", O + "heading_hold_gain=2.0", O + "topple_push=(0.8, 1.3)",
           O + "topple_interval=(15.0, 25.0)", O + f"amp_skip_types={GOAL_TYPES}",
           "--num_envs", str(num_envs), "--iterations", str(iterations),
           "--critic_warmup", "100" if iterations > 10 else "2", "--seed", str(seed), "--out", str(out)]
    if variant == "free":
        cmd += ["--bc_free_stairs", O + "stairs_goal_height_scale=0.3"]
    return cmd


def stairs_test_cmd(dog: Path, out: Path, num_envs: int = 2000, rounds: int = 2) -> list[str]:
    return [LAB, "tools/dog/stairs_goal_test.py", str(dog), "--out", str(out), "--num_envs", str(num_envs),
            "--rounds", str(rounds)]


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def last_line(path: Path, name: str, scenario: str | None = None) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [r for r in rows if Path(r["run"]).name == name and (scenario is None or r.get("scenario") == scenario)]
    return rows[-1] if rows else None


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    # 1 the test on dog v1 (small): it must reach the goal on the 1-2 cm stairs, up and down
    code = run(stairs_test_cmd(DOG_V1, out / "stairs_goal.jsonl", 200, 1), log)
    line = last_line(out / "stairs_goal.jsonl", DOG_V1.name)
    if code != 0 or line is None:
        return fail(f"smoke (stairs goal test): exit code {code}, see smoke/smoke.log")
    for d in ("up", "down"):
        low = [c for c in line["cells"] if c["dir"] == d and c["walls"] is None and c["rise_cm"] <= 2]
        reached = sum(c["reached"] * c["n"] for c in low) / max(1, sum(c["n"] for c in low))
        if reached < 0.6:
            return fail(f"smoke (stairs goal test): dog v1 reached the goal on 1-2 cm stairs going {d} in only "
                        f"{100 * reached:.0f} % (expected nearly all): navigation, start or goal placement is off")
    # 2 training on the mixed course with copying (the free variant covers both the stairs mask and the crouch setting)
    code = run(train_cmd(out / "train", "free", 1, 64, 4), log)
    text = log.read_text(encoding="utf-8", errors="ignore")
    if code != 0 or not (out / "train" / "result.json").exists() or "gated behaviour cloning" not in text:
        return fail(f"smoke (training on dog_rough_stairs): exit code {code}, see smoke/smoke.log")
    # 3 a short stairs goal video
    code = run([PY, "-m", "core.evolution.play", "--run_dir", str(out / "train"), "--video", "--num_envs", "1",
                "--drive", "goal", "--terrain", "dog_stairs", "--terrain_type", f"goal_stairs_t{VIDEO_TREAD}",
                "--level", "3", "--video_length", "100", "--out_name", "smoke_goal.mp4"], log)
    if code != 0 or not (out / "train" / "videos" / "smoke_goal.mp4").exists():
        return fail(f"smoke (stairs goal video): exit code {code}, see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def stairs_score(name: str) -> float:
    """Mean share of clean arrivals (no fall) over all rises, treads and both directions."""
    line = last_line(HERE / "stairs_goal.jsonl", name)
    if line is None:
        return -1.0
    cells = [c for c in line["cells"] if c["walls"] is None]
    return sum(c["clean"] for c in cells) / len(cells)


def column(line: dict, d: str, tread: int) -> dict:
    return {int(c["rise_cm"]): c for c in line["cells"] if c["dir"] == d and c["walls"] is None and int(c["tread_cm"]) == tread}


def sweet_spot(line: dict, d: str, tread: int) -> int:
    """Highest rise [cm] up to which every rise is done clean by >= 98 % at most 1.3x slower than on 1 cm steps."""
    col = column(line, d, tread)
    base = (col.get(1) or {}).get("time_s")
    best = 1
    for r in sorted(col):
        c = col[r]
        if c["clean"] < 0.98 or c["time_s"] is None or base is None or c["time_s"] > 1.3 * base:
            break
        best = r
    return best


def videos(best: str | None, anchor: str | None, free: str | None) -> None:
    ref = best or DOG_V1.name
    line = last_line(HERE / "stairs_goal.jsonl", ref)
    if line is None:
        return
    panels = [f"dog v1={DOG_V1}"] + [f"{lbl} ({n})={HERE / n}" for lbl, n in (("anchor", anchor), ("free", free)) if n]
    shots = []
    for d, sign in (("up", 1), ("down", -1)):
        limit = max(1, int(line["limits_rise_cm"].get(f"{d}_t{VIDEO_TREAD}", 1)))
        shots += [(d, sign, sweet_spot(line, d, VIDEO_TREAD), "sweet spot"), (d, sign, limit, "limit")]
    up_limit = max(1, int(line["limits_rise_cm"].get(f"up_t{VIDEO_TREAD}", 1)))
    shots.append(("up", 1, min(10, up_limit + 2), "beyond the limit"))
    done = set()
    for d, sign, rise, what in shots:
        if (d, rise) in done:
            continue
        done.add((d, rise))
        slope = round(math.degrees(math.atan2(rise, VIDEO_TREAD)))
        out = HERE / "videos" / f"{d}_r{rise}_t{VIDEO_TREAD}.mp4"
        if out.exists():
            continue
        run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs",
             "--terrain_type", f"goal_stairs_t{VIDEO_TREAD}", "--level", str(rise - 1), "--drive", "goal",
             "--env_set", f"stairs_goal_dir={sign}",
             "--subtitle", f"{d} {rise} cm steps, {VIDEO_TREAD} cm treads ({slope} deg), open sides: {what} of {ref}",
             *[a for p in panels for a in ("--run", p)]], HERE / "videos.log")


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    # 1 zero-shot
    for dog in (DOG_V1, WALKER):
        if last_line(HERE / "stairs_goal.jsonl", dog.name) is None:
            run(stairs_test_cmd(dog, HERE / "stairs_goal.jsonl"), HERE / "tests.log")
    # 2 training
    for name, variant, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            run(train_cmd(HERE / name, variant, seed), HERE / f"{name}.log")
    # 3 tests
    for name, _, _ in RUNS:
        dog = HERE / name
        if not (dog / "result.json").exists():
            continue
        if last_line(HERE / "stairs_goal.jsonl", name) is None:
            run(stairs_test_cmd(dog, HERE / "stairs_goal.jsonl"), HERE / "tests.log")
        for sc in ("forward", "course", "stairs_straight"):
            if last_line(HERE / "benchmark.jsonl", name, sc) is None:
                run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], HERE / "tests.log")
        if last_line(HERE / "recover_test.jsonl", name) is None:
            run([LAB, "tools/dog/recover_test.py", str(dog), "--out", str(HERE / "recover_test.jsonl")], HERE / "tests.log")
        if last_line(HERE / "getup_test.jsonl", name) is None:
            run([LAB, "tools/dog/getup_test.py", str(dog), "--out", str(HERE / "getup_test.jsonl")], HERE / "tests.log")
    # 4 videos: the best seed of each variant, levels from the best dog overall
    scored = {name: stairs_score(name) for name, _, _ in RUNS}
    best_of = {}
    for name, variant, _ in RUNS:
        if scored[name] > 0 and scored[name] > scored.get(best_of.get(variant, ""), -1.0):
            best_of[variant] = name
    best = max(best_of.values(), key=scored.get) if best_of else None
    if best is not None and scored[best] <= stairs_score(DOG_V1.name):
        best = None  # training did not beat dog v1 on the stairs: film at dog v1's levels
    (HERE / "best.txt").write_text(f"{best or DOG_V1.name} (clean share {scored.get(best, stairs_score(DOG_V1.name)):.3f}; "
                                   f"all: {scored}; dog v1 {stairs_score(DOG_V1.name):.3f})\n")
    videos(best, best_of.get("anchor"), best_of.get("free"))
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
