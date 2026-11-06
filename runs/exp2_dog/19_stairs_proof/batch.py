"""Physically possible, then learned (journal step 41). Approved by the owner for the night of 2026-10-25.

  0  the proof (CPU, done before this batch: tools/dog/stairs_plan.py): quasi-static stair plans per rise and tread,
     up and down (plans/, plans_retry/ with a wider search, plans_fullrange/ with the full joint range)
  1  proof in physics (tools/dog/stairs_replay.py): every plan played open loop with the real servo model, nominal and
     with the training randomization (4 dogs per plan); side videos of four plans
  2  the plans as trainer: from the best stairs dog of step 40 (18_stairs_goal/free_s2), same recipe (dog course + stairs
     goal tiles, gated behaviour cloning, no walker copying on the stairs, crouch allowed), 2000 iterations:
        rsi        half of the stairs episodes start mid-climb, in a random four-paw pose of the plan for the tile
                   (seeds 1, 2)
        rsi_terms  rsi + the stair technique terms of step 35 on the stairs tiles (lift target from the step ahead,
                   lift before pushing forward, no toe against a riser, no paw on an edge) (seeds 1, 2)
        cont       the same training without mid-climb starts (control: is it the trainer or just more training?)
     each dog tested right after its training: stairs goal test, walking benchmark, shoves, getting up
  3  comparison videos on the stairs goal course (free_s2 / cont / best rsi / best rsi_terms)
  4  results.md (summary.py)
A smoke test first (stops at the first failure). ~10 h.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "19_stairs_proof"
PREV = ROOT / "runs" / "exp2_dog" / "16_recover"
sys.path.insert(0, str(PREV))
from batch import GETUP, LAB, O, PY, WALKER, walker_recipe  # noqa: E402

START = ROOT / "runs" / "exp2_dog" / "18_stairs_goal" / "free_s2"
GOAL_TYPES = tuple(f"goal_stairs_t{t}{w}" for t in (25, 20, 15, 12, 9) for w in ("", "_walls"))
PLANS = HERE / "plans_all"  # plans/ plus the cells only the wider search solved
TERMS = [O + f"stairs_terms_types={GOAL_TYPES}", O + "foot_clearance_terrain=True", O + "stumble_weight=1.0",
         O + "foothold_weight=0.5", O + "lift_first_weight=1.0"]
RUNS = [("rsi_s1", "rsi", 1), ("rsi_terms_s1", "rsi_terms", 1), ("rsi_s2", "rsi", 2), ("rsi_terms_s2", "rsi_terms", 2),
        ("cont_s1", "cont", 1)]
VIDEO_PLANS = ("up_r5_t9.json", "up_r5_t15.json", "down_r4_t15.json", "down_r5_t25.json")
COMPARE = [("up", 5, 15), ("up", 5, 9), ("down", 4, 15), ("down", 3, 9)]


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = 2000) -> list[str]:
    cmd = [LAB, "-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough_stairs", "--sensors", "imu",
           "--obs_history", "8", "--symmetry", "--height_scan", "--genome", str(START / "genome.json"),
           "--init_checkpoint", str(START / "model_final.pt"), "--bc_walker", str(WALKER), "--bc_getup", str(GETUP),
           "--bc_free_stairs", *walker_recipe(), O + "action_scale=1.0", O + "recover=True",
           O + "policy_drives_expression=True", O + "self_collisions=True", O + "heading_hold_gain=2.0",
           O + "topple_push=(0.8, 1.3)", O + "topple_interval=(15.0, 25.0)", O + f"amp_skip_types={GOAL_TYPES}",
           O + "stairs_goal_height_scale=0.3", "--num_envs", str(num_envs), "--iterations", str(iterations),
           "--critic_warmup", "100" if iterations > 10 else "2", "--seed", str(seed), "--out", str(out)]
    if variant == "desc":  # follow-up: the descent (a curriculum per direction, mid-climb starts at the first steps down)
        cmd += [O + "stairs_dir_curriculum=True", O + "stairs_rsi_down_early=0.5"]
    if variant in ("rsi", "rsi_terms", "desc"):
        cmd += [O + f"stairs_rsi_plans={PLANS.as_posix()}", O + "stairs_rsi_prob=0.5"]
    if variant == "rsi_terms":
        cmd += TERMS
    return cmd


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def last_line(path: Path, name: str, scenario: str | None = None) -> dict | None:
    rows = [r for r in lines(path) if Path(r.get("run", "")).name == name and (scenario is None or r.get("scenario") == scenario)]
    return rows[-1] if rows else None


def fail(msg: str) -> bool:
    (HERE / "batch.failed").write_text(msg + "\n")
    return False


def collect_plans() -> None:
    """plans_all = plans/ plus the cells plans/ has no plan for but plans_retry/ has (same limits, wider search)."""
    PLANS.mkdir(exist_ok=True)
    for f in (HERE / "plans").glob("*_r*_t*.json"):
        shutil.copy(f, PLANS / f.name)
    for f in (HERE / "plans_retry").glob("*_r*_t*.json"):
        have = PLANS / f.name
        solved_before = have.exists() and json.loads(have.read_text()).get("feasible", False)
        if json.loads(f.read_text()).get("feasible") and not solved_before:
            shutil.copy(f, have)


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    code = run([LAB, "tools/dog/stairs_replay.py", str(PLANS / "up_r3_t15.json"), "--open_loop", "--out", str(out / "replay.jsonl")], log)
    if code != 0 or not lines(out / "replay.jsonl"):
        return fail(f"smoke (replay of one plan): exit code {code}, see smoke/smoke.log")
    code = run([LAB, "tools/dog/rsi_check.py", "--plans", str(PLANS), "--out", str(out / "rsi_check.jsonl")], log)
    if code != 0:
        return fail(f"smoke (mid-climb starts): exit code {code} (paws off the steps or dogs falling from plan poses), "
                    f"see smoke/rsi_check.jsonl and smoke/smoke.log")
    code = run(train_cmd(out / "train", "rsi_terms", 1, 64, 4), log)
    text = log.read_text(encoding="utf-8", errors="ignore")
    if code != 0 or not (out / "train" / "result.json").exists() or "mid-climb starts from" not in text:
        return fail(f"smoke (training with mid-climb starts): exit code {code}, see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def test(name: str) -> None:
    dog = HERE / name
    if not (dog / "result.json").exists():
        return
    if last_line(HERE / "stairs_goal.jsonl", name) is None:
        run([LAB, "tools/dog/stairs_goal_test.py", str(dog), "--out", str(HERE / "stairs_goal.jsonl")], HERE / "tests.log")
    for sc in ("forward", "course", "stairs_straight"):
        if last_line(HERE / "benchmark.jsonl", name, sc) is None:
            run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], HERE / "tests.log")
    for tool, out in (("recover_test", "recover_test.jsonl"), ("getup_test", "getup_test.jsonl")):
        if last_line(HERE / out, name) is None:
            if run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], HERE / "tests.log") != 0:
                run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], HERE / "tests.log")  # once more


def stairs_score(name: str) -> float:
    line = last_line(HERE / "stairs_goal.jsonl", name) or last_line(START.parent / "stairs_goal.jsonl", name)
    if line is None:
        return -1.0
    cells = [c for c in line["cells"] if c["walls"] is None]
    return sum(c["clean"] for c in cells) / len(cells)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    collect_plans()
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    # 1 proof in physics
    if not lines(HERE / "replay.jsonl"):
        run([LAB, "tools/dog/stairs_replay.py", str(PLANS), "--open_loop", "--out", str(HERE / "replay.jsonl")], HERE / "replay.log")
    if not lines(HERE / "replay_randomized.jsonl"):
        run([LAB, "tools/dog/stairs_replay.py", str(PLANS), "--open_loop", "--copies", "4", "--out", str(HERE / "replay_randomized.jsonl")],
            HERE / "replay.log")
    for plan in VIDEO_PLANS:
        if (PLANS / plan).exists() and not (HERE / "videos" / f"replay_{Path(plan).stem}.mp4").exists():
            run([LAB, "tools/dog/stairs_replay.py", str(PLANS), "--open_loop", "--video", plan, "--out", str(HERE / "replay_video.jsonl")],
                HERE / "videos.log")
    # 2 training, each dog tested right away
    for name, variant, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            run(train_cmd(HERE / name, variant, seed), HERE / f"{name}.log")
        test(name)
    # 3 comparison videos: the start dog, the control, the best of each trainer variant
    best = {}
    for name, variant, _ in RUNS:
        if stairs_score(name) > stairs_score(best.get(variant, "")):
            best[variant] = name
    panels = [f"start (free_s2)={START}"] + [f"{v} ({best[v]})={HERE / best[v]}" for v in ("cont", "rsi", "rsi_terms") if v in best]
    for d, rise, tread in COMPARE:
        out = HERE / "videos" / f"compare_{d}_r{rise}_t{tread}.mp4"
        if out.exists():
            continue
        run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs",
             "--terrain_type", f"goal_stairs_t{tread}", "--level", str(rise - 1), "--drive", "goal",
             "--env_set", f"stairs_goal_dir={1 if d == 'up' else -1}",
             "--subtitle", f"{d} {rise} cm steps, {tread} cm treads, open sides (physically possible: a plan exists)",
             *[a for p in panels for a in ("--run", p)]], HERE / "videos.log")
    (HERE / "best.txt").write_text(json.dumps({v: [n, round(stairs_score(n), 3)] for v, n in best.items()}
                                              | {"start free_s2": round(stairs_score(START.name), 3)}) + "\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
