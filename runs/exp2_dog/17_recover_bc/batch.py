"""Balance training that does not forget (journal step 39). Step 38's RL polish made the dog steadier under shoves but
it partly forgot getting up from its side (70 %) and walked worse; it had not converged and was down ~19 % of the time.

Here, from step 38's distilled dog (walks and gets up), RL polish with gated behaviour cloning (core/evolution/gated_bc.py):
while learning balance it keeps copying the original walker when upright (tilt < 25 deg) and the get-up dog when down
(tilt > 35 deg). Far fewer topple shoves: 0.8-1.3 m/s every 15-25 s (was 6-12 s). 3000 iterations, seeds 1 and 2.

Then: shove-and-recovery test, get-up test, walking benchmark (forward, course, stairs_straight), and for the better
seed the walking shove/flip videos and a flat tour (original / step 38 polish / this one). results.md includes step 38's
dogs for comparison. A smoke test first. ~4.5 h.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
HERE = ROOT / "runs" / "exp2_dog" / "17_recover_bc"
PREV = ROOT / "runs" / "exp2_dog" / "16_recover"
sys.path.insert(0, str(PREV))
from batch import GETUP, LAB, O, PY, WALKER, walker_recipe  # noqa: E402

START = PREV / "distill"
RUNS = [("bc_s1", 1), ("bc_s2", 2)]


def polish_cmd(out: Path, seed: int, num_envs: int = 2048, iterations: int = 3000) -> list[str]:
    return [LAB, "-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough", "--sensors", "imu",
            "--obs_history", "8", "--symmetry", "--height_scan", "--genome", str(GETUP / "genome.json"),
            "--init_checkpoint", str(START / "model_final.pt"), "--bc_walker", str(WALKER), "--bc_getup", str(GETUP),
            *walker_recipe(), O + "action_scale=1.0", O + "recover=True", O + "policy_drives_expression=True",
            O + "self_collisions=True", O + "heading_hold_gain=2.0", O + "topple_push=(0.8, 1.3)",
            O + "topple_interval=(15.0, 25.0)", "--num_envs", str(num_envs), "--iterations", str(iterations),
            "--critic_warmup", "100" if iterations > 10 else "2", "--seed", str(seed), "--out", str(out)]


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


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    code = run(polish_cmd(out / "bc", 1, 64, 4), out / "smoke.log")  # 2 warm-up + 2 with copying
    ok = code == 0 and (out / "bc" / "result.json").exists()
    if ok:
        ok = "gated behaviour cloning" in (out / "smoke.log").read_text(encoding="utf-8", errors="ignore")
    if not ok:
        (HERE / "batch.failed").write_text(f"smoke (polish with copying): exit code {code}, see smoke/smoke.log\n")
        return False
    (out / "ok").write_text("ok")
    return True


def score(name: str) -> float:
    """Kept on its feet or got up after a shove (mean over strengths) + got up from side and back + walk score."""
    rec, gu = last_line(HERE / "recover_test.jsonl", name), last_line(HERE / "getup_test.jsonl", name)
    fw = last_line(HERE / "benchmark.jsonl", name, "forward")
    if not (rec and gu and fw):
        return -1.0
    shove = sum(v["stayed_up"] + v["fell_got_up"] for v in rec["by_shove_m_s"].values()) / len(rec["by_shove_m_s"])
    return shove + 0.5 * (gu["on its side"]["success"] + gu["on its back"]["success"]) + fw["walk_score"]


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    for name, seed in RUNS:
        if not (HERE / name / "result.json").exists():
            run(polish_cmd(HERE / name, seed), HERE / f"{name}.log")
    for name, _ in RUNS:
        dog = HERE / name
        if not (dog / "result.json").exists():
            continue
        if last_line(HERE / "recover_test.jsonl", name) is None:
            run([LAB, "tools/dog/recover_test.py", str(dog), "--out", str(HERE / "recover_test.jsonl")], HERE / "tests.log")
        if last_line(HERE / "getup_test.jsonl", name) is None:
            run([LAB, "tools/dog/getup_test.py", str(dog), "--out", str(HERE / "getup_test.jsonl")], HERE / "tests.log")
        for sc in ("forward", "course", "stairs_straight"):
            if last_line(HERE / "benchmark.jsonl", name, sc) is None:
                run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")], HERE / "tests.log")
    scored = {name: score(name) for name, _ in RUNS}
    best = max(scored, key=scored.get)
    if scored[best] > 0:
        dog, videos = HERE / best, HERE / "videos"
        run([LAB, "tools/dog/flip_video.py", str(dog), "--vx", "0.2", "--spin", "0", "--lift", "0.2", "--push", "1.0",
             "--out", str(videos / "shoves_walking.mp4")], HERE / "videos.log")
        run([LAB, "tools/dog/flip_video.py", str(dog), "--vx", "0.2", "--out", str(videos / "flips_walking.mp4")],
            HERE / "videos.log")
        run([PY, "-m", "core.evolution.compare_video", "--out", str(videos / "tour_flat.mp4"), "--terrain", "flat",
             "--heading_hold", "2.0", "--subtitle", "walking tour on a flat floor",
             "--run", f"original walker={WALKER}", "--run", f"step 38 polish={PREV / 'recover'}",
             "--run", f"with teachers as anchor ({best})={dog}"], HERE / "videos.log")
        (HERE / "best.txt").write_text(f"{best} (score {scored[best]:.3f}; all: {scored})\n")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
