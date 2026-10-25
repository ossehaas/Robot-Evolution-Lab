"""Keep balance, and get up after a fall (journal step 38): one dog that walks, catches itself after a shove and, if it
does go over, gets up and walks on.

  1  distill: one student from two teachers, chosen by its posture: the original walking dog (amp_w6m_slow_s1) while
     upright, the get-up dog (getup_abd29_head: neck, head and tail driven) from a fall until it stands again.
     Recover mode (a fall does not end the episode), topple shoves 0.6-1.2 m/s every 4-8 s, self-collision on, the
     student drives all 15 joints at full range. 400 iterations.
  2  recover: RL polish from the student with the walker's recipe (gait, commands, style judge while upright) plus
     recover mode: get-up rewards while fallen, a fall costs once, topple shoves 0.8-1.3 m/s every 6-12 s (mostly calm
     walking; Microduck's lesson: constant hard shoves make a nervous gait). 2000 iterations.
  3  tests: shove and recovery (stays up / falls and gets up / stays down per shove strength) for the original walker,
     the student and the polished dog; get-up test from fallen starts; walking benchmark (forward, course,
     stairs_straight with the climb measure) for the original and the polished dog; videos: 10 shoves and 10 flips
     while walking (10 directions), and a walking tour (original vs polished) to judge the gait by eye.
A smoke test first. ~3.5 h.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
HERE = ROOT / "runs" / "exp2_dog" / "16_recover"
WALKER = ROOT / "runs" / "exp2_dog" / "11_imitation" / "amp_w6m_slow_s1"
GETUP = ROOT / "runs" / "exp2_dog" / "15_getup_design" / "getup_abd29_head"
LAB = str(ROOT / "lab.cmd")
PY = str(ROOT / "IsaacLab" / ".venv" / "Scripts" / "python.exe")
O = "--env_override="


def walker_recipe() -> list[str]:
    """The walker's own training settings as --env_override arguments (gait, commands, rewards, style judge)."""
    res = json.loads((WALKER / "result.json").read_text())
    out = []
    for key, value in res.get("env_overrides", {}).items():
        if key == "action_scale":
            continue  # full range: set below
        out.append(f"{O}{key}={tuple(value) if isinstance(value, list) else value}")
    return out


def distill_cmd(out: Path, num_envs: int = 2048, iterations: int = 400) -> list[str]:
    return [LAB, "-m", "core.evolution.distill_recover", "--walker", str(WALKER), "--getup", str(GETUP), "--out", str(out),
            "--num_envs", str(num_envs), "--iterations", str(iterations)]


def polish_cmd(start: Path, out: Path, num_envs: int = 2048, iterations: int = 2000) -> list[str]:
    return [LAB, "-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough", "--sensors", "imu",
            "--obs_history", "8", "--symmetry", "--height_scan", "--genome", str(GETUP / "genome.json"),
            "--init_checkpoint", str(start / "model_final.pt"), *walker_recipe(),
            O + "action_scale=1.0", O + "recover=True", O + "policy_drives_expression=True", O + "self_collisions=True",
            O + "heading_hold_gain=2.0", O + "topple_push=(0.8, 1.3)", O + "topple_interval=(6.0, 12.0)",
            "--num_envs", str(num_envs), "--iterations", str(iterations),
            "--critic_warmup", "100" if iterations > 10 else "2", "--seed", "1", "--out", str(out)]


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def measured(path: Path, name: str, key: str = "run") -> bool:
    return path.exists() and any(Path(json.loads(line)[key]).name == name for line in path.read_text().splitlines())


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    steps = [("distill", distill_cmd(out / "distill", 64, 3), out / "distill" / "result.json"),
             ("polish", polish_cmd(out / "distill", out / "recover", 64, 3), out / "recover" / "result.json"),
             ("recover test", [LAB, "tools/dog/recover_test.py", str(out / "recover"), "--num_envs", "70", "--out",
                               str(out / "recover_test.jsonl")], out / "recover_test.jsonl"),
             ("benchmark", [LAB, "tools/dog/benchmark.py", str(out / "recover"), "forward", "--num_envs", "64", "--out",
                            str(out / "benchmark.jsonl")], out / "benchmark.jsonl")]
    for name, cmd, expect in steps:
        code = run(cmd, log)
        if code != 0 or not expect.exists():
            (HERE / "batch.failed").write_text(f"smoke ({name}): exit code {code}, see smoke/smoke.log\n")
            return False
    (out / "ok").write_text("ok")
    return True


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    if not (HERE / "distill" / "result.json").exists():
        run(distill_cmd(HERE / "distill"), HERE / "distill.log")
    if (HERE / "distill" / "result.json").exists() and not (HERE / "recover" / "result.json").exists():
        run(polish_cmd(HERE / "distill", HERE / "recover"), HERE / "recover.log")
    dogs = [WALKER, HERE / "distill", HERE / "recover"]
    for dog in dogs:
        if (dog / "result.json").exists() and not measured(HERE / "recover_test.jsonl", dog.name):
            run([LAB, "tools/dog/recover_test.py", str(dog), "--out", str(HERE / "recover_test.jsonl")], HERE / "tests.log")
    for dog in (HERE / "distill", HERE / "recover"):
        if (dog / "result.json").exists() and not measured(HERE / "getup_test.jsonl", dog.name):
            run([LAB, "tools/dog/getup_test.py", str(dog), "--out", str(HERE / "getup_test.jsonl")], HERE / "tests.log")
    for dog in (WALKER, HERE / "recover"):
        if (dog / "result.json").exists():
            for sc in ("forward", "course", "stairs_straight"):
                run([LAB, "tools/dog/benchmark.py", str(dog), sc, "--out", str(HERE / "benchmark.jsonl")],
                    HERE / "tests.log")
    best = HERE / "recover"
    if (best / "result.json").exists():
        videos = HERE / "videos"
        run([LAB, "tools/dog/flip_video.py", str(best), "--vx", "0.2", "--spin", "0", "--lift", "0.2", "--push", "1.0",
             "--out", str(videos / "shoves_walking.mp4")], HERE / "videos.log")
        run([LAB, "tools/dog/flip_video.py", str(best), "--vx", "0.2", "--out", str(videos / "flips_walking.mp4")],
            HERE / "videos.log")
        run([PY, "-m", "core.evolution.compare_video", "--out", str(videos / "tour_flat.mp4"), "--terrain", "flat",
             "--heading_hold", "2.0", "--subtitle", "walking tour on a flat floor: before / after the balance training",
             "--run", f"before (original)={WALKER}", "--run", f"after (walks + gets up)={best}"], HERE / "videos.log")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
