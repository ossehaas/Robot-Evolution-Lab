"""Balance first, then the descent (journal step 42). The owner's list after step 41 (2026-10-26):

  1  evidence: why do the dogs refuse to go down stairs they could physically descend? tools/dog/descent_terms.py reads
     the training reward terms per phase of a descent (landing / at the edge / on the stairs / refusing at the edge)
     for the start dog free_s2 (step 40), rsi_s2 and the control cont_s1 (step 41)
  2  the tilting plateau: tools/dog/tilt_test.py, standing still on plates tilted 0-45 deg nose down / diagonal /
     sideways / nose up, for the bare body (servos holding the standing pose), dog v1 and free_s2; also what standing
     still costs (power, joint motion; for the later "stand still until unstable" tour). Static limits on paper:
     tools/dog/tilt_limits.py (standing pose tips at 20 deg facing downhill, 28 nose up, 29 sideways)
  3  a stricter proof: stair plans with a 3 cm balance margin (plans_m30, CPU, made before this batch), replayed in
     physics open loop, nominal and randomized (4 dogs per plan), and videos of two plans
  4  the plans' knowledge in training, from free_s2 (its recipe; mid-climb starts from step 41's plans), 2000 it,
     seeds 1 and 2:
        desc     a curriculum per direction on the stairs tiles (up and down move separately: one shared level kept
                 high descents untrained where climbing failed) + half of the mid-climb descents in the first 40 % of
                 the plan (the first steps off the landing)
        margin   desc + the plans' rule as a reward: the centre of gravity >= 3 cm inside the paws on the ground on the
                 stairs tiles (stairs_margin_weight 2.0 per second)
     each dog tested right after its training (stairs goal test, walking benchmark, shoves, getting up, tilt test)
  5  comparison videos (start / best desc / best margin), results.md
Smoke tests first (stop at the first failure). ~9 h; one training at a time (RAM).
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "20_balance_and_style"
STEP41 = ROOT / "runs" / "exp2_dog" / "19_stairs_proof"
_spec = importlib.util.spec_from_file_location("stairs_proof_batch", STEP41 / "batch.py")
s41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s41)
LAB, PY, O = s41.LAB, s41.PY, s41.O
START = s41.START  # 18_stairs_goal/free_s2
DOG_V1 = ROOT / "runs" / "exp2_dog" / "17_recover_bc" / "bc_s2"
RUNS = [("desc_s1", "desc", 1), ("margin_s1", "margin", 1), ("desc_s2", "desc", 2), ("margin_s2", "margin", 2)]
COMPARE = [("down", 4, 15), ("down", 3, 9), ("up", 5, 15), ("up", 4, 9)]
VIDEO_PLANS = ("up_r4_t12.json", "down_r3_t15.json")  # the hardest with a 3 cm margin


def train_cmd(out: Path, variant: str, seed: int, num_envs: int = 2048, iterations: int = 2000) -> list[str]:
    cmd = s41.train_cmd(out, "desc", seed, num_envs, iterations)  # mid-climb starts + curriculum per direction
    if variant == "margin":
        cmd += [O + "stairs_margin_weight=2.0", O + "stairs_margin_target=0.03"]
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


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    code = run([LAB, "tools/dog/tilt_test.py", str(START), "--per_cell", "1", "--out", str(out / "tilt.jsonl")], log)
    rows = lines(out / "tilt.jsonl")
    if code != 0 or not rows:
        return fail(f"smoke (tilt test): exit code {code}, see smoke/smoke.log")
    flat = [c for c in rows[-1]["cells"] if c["angle_deg"] == 0.0]
    if any(c["fell"] > 0 for c in flat):  # on the flat plate it must not fall (placement / plate check)
        return fail(f"smoke (tilt test): the dog fell on the flat plates ({flat}): placement is off")
    code = run([LAB, "tools/dog/descent_terms.py", str(START), "--per_cell", "4", "--out", str(out / "descent.jsonl")], log)
    if code != 0 or not lines(out / "descent.jsonl"):
        return fail(f"smoke (descent terms): exit code {code}, see smoke/smoke.log")
    code = run(train_cmd(out / "train", "margin", 1, 64, 4), log)
    text = log.read_text(encoding="utf-8", errors="ignore")
    if code != 0 or not (out / "train" / "result.json").exists() or "stairs_down_level" not in text \
            or "sg_margin" not in text:
        return fail(f"smoke (training with the margin reward): exit code {code}, see smoke/smoke.log")
    (out / "ok").write_text("ok")
    return True


def test(name: str) -> None:
    dog = HERE / name
    if not (dog / "result.json").exists():
        return
    if not has(HERE / "stairs_goal.jsonl", name):
        run([LAB, "tools/dog/stairs_goal_test.py", str(dog), "--out", str(HERE / "stairs_goal.jsonl")], HERE / "tests.log")
    for sc in ("forward", "course", "stairs_straight"):
        if not has(HERE / "benchmark.jsonl", name, scenario=sc):
            run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], HERE / "tests.log")
    for tool, out in (("recover_test", "recover_test.jsonl"), ("getup_test", "getup_test.jsonl")):
        for _ in range(2):  # once more after a start-up hiccup
            if not has(HERE / out, name):
                run([LAB, f"tools/dog/{tool}.py", str(dog), "--out", str(HERE / out)], HERE / "tests.log")
    if not has(HERE / "tilt.jsonl", name, hold=False):
        run([LAB, "tools/dog/tilt_test.py", str(dog), "--out", str(HERE / "tilt.jsonl")], HERE / "tests.log")
    if not has(HERE / "descent_terms.jsonl", name):
        run([LAB, "tools/dog/descent_terms.py", str(dog), "--out", str(HERE / "descent_terms.jsonl")], HERE / "tests.log")


def stairs_score(name: str) -> float:
    for path in (HERE / "stairs_goal.jsonl", STEP41 / "stairs_goal.jsonl", START.parent / "stairs_goal.jsonl"):
        rows = [r for r in lines(path) if Path(r["run"]).name == name]
        if rows:
            cells = [c for c in rows[-1]["cells"] if c["walls"] is None]
            return sum(c["clean"] for c in cells) / len(cells)
    return -1.0


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    # 1 + 2 measurements on the existing dogs
    for dog in (START, STEP41 / "rsi_s2", STEP41 / "cont_s1"):
        if not has(HERE / "descent_terms.jsonl", dog.name):
            run([LAB, "tools/dog/descent_terms.py", str(dog), "--out", str(HERE / "descent_terms.jsonl")], HERE / "measure.log")
    if not has(HERE / "tilt.jsonl", START.name, hold=True):
        run([LAB, "tools/dog/tilt_test.py", str(START), "--hold", "--out", str(HERE / "tilt.jsonl")], HERE / "measure.log")
    for dog in (DOG_V1, START):
        if not has(HERE / "tilt.jsonl", dog.name, hold=False):
            run([LAB, "tools/dog/tilt_test.py", str(dog), "--out", str(HERE / "tilt.jsonl")], HERE / "measure.log")
    # 3 the stricter proof in physics
    plans = HERE / "plans_m30"
    if any(plans.glob("*_r*_t*.json")):
        if not lines(HERE / "replay.jsonl"):
            run([LAB, "tools/dog/stairs_replay.py", str(plans), "--open_loop", "--out", str(HERE / "replay.jsonl")], HERE / "replay.log")
        if not lines(HERE / "replay_randomized.jsonl"):
            run([LAB, "tools/dog/stairs_replay.py", str(plans), "--open_loop", "--copies", "4",
                 "--out", str(HERE / "replay_randomized.jsonl")], HERE / "replay.log")
        for plan in VIDEO_PLANS:
            if (plans / plan).exists() and not (HERE / "videos" / f"replay_{Path(plan).stem}.mp4").exists():
                run([LAB, "tools/dog/stairs_replay.py", str(plans), "--open_loop", "--video", plan,
                     "--out", str(HERE / "replay_video.jsonl")], HERE / "videos.log")
    # 4 training, each dog tested right away
    for name, variant, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            run(train_cmd(HERE / name, variant, seed), HERE / f"{name}.log")
        test(name)
    # 5 comparison videos
    best = {}
    for name, variant, _ in RUNS:
        if stairs_score(name) > stairs_score(best.get(variant, "")):
            best[variant] = name
    panels = [f"start (free_s2)={START}"] + [f"{v} ({best[v]})={HERE / best[v]}" for v in ("desc", "margin") if v in best]
    for d, rise, tread in COMPARE:
        out = HERE / "videos" / f"compare_{d}_r{rise}_t{tread}.mp4"
        if out.exists() or len(panels) < 2:
            continue
        run([PY, "-m", "core.evolution.compare_video", "--out", str(out), "--terrain", "dog_stairs",
             "--terrain_type", f"goal_stairs_t{tread}", "--level", str(rise - 1), "--drive", "goal",
             "--env_set", f"stairs_goal_dir={1 if d == 'up' else -1}",
             "--subtitle", f"{d} {rise} cm steps, {tread} cm treads, open sides (physically possible)",
             *[a for p in panels for a in ("--run", p)]], HERE / "videos.log")
    (HERE / "best.txt").write_text(json.dumps({v: [n, round(stairs_score(n), 3)] for v, n in best.items()}
                                              | {"start free_s2": round(stairs_score(START.name), 3)}) + "\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
