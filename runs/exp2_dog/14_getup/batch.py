"""Getting up and being pushed (journal step 36), plus a re-measurement of the stairs dogs.

  0  smoke test of every new piece (stops at a failure: batch.failed)
  1  climb measure: the step 35 stairs dogs on stairs_straight again, now with the height actually climbed per level
     (the owner saw in the videos that some dogs circle or never get a rear paw on the first step: few falls alone
     does not show that)
  2  balance: one shove of 0.2-1.0 m/s while walking at 20 cm/s, falls per shove strength (original candidate, base,
     stairs, support, memory student)
  3  what the walking dogs do when they find themselves on the floor (get-up test of two walkers: no get-up training)
  4  get-up training from scratch, perfect scan, same senses as the walkers, full joint range, flat floor, 8 s
     episodes starting standing / dropped upright / on the side / on the back (harder starts later), 3000 iterations,
     seeds 1 and 2
  5  get-up test per start, a video of the four starts (best seed), results.md
Takes ~4 h.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
HERE = ROOT / "runs" / "exp2_dog" / "14_getup"
STAIRS = ROOT / "runs" / "exp2_dog" / "13_stairs"
ORIGINAL = ROOT / "runs" / "exp2_dog" / "11_imitation" / "amp_w6m_slow_s1"
BEFORE = ROOT / "runs" / "exp2_dog" / "12_terrain_map" / "map_depth"
LAB = str(ROOT / "lab.cmd")
PY = str(ROOT / "IsaacLab" / ".venv" / "Scripts" / "python.exe")
O = "--env_override="
ITERATIONS = 3000
GETUP = ["-m", "core.evolution.train", "--task", "command", "--terrain", "flat", "--sensors", "imu", "--obs_history", "8",
         "--symmetry", "--height_scan", "--genome", str(ORIGINAL / "genome.json"),
         O + "getup=True", O + "command_standing_prob=1.0", O + "episode_length_s=8.0", O + "action_scale=1.0",
         O + "action_rate_weight=0.1", O + f"getup_curriculum_steps={ITERATIONS * 24}"]
RUNS = [("getup_s1", 1), ("getup_s2", 2)]
STAIRS_DOGS = [ORIGINAL, BEFORE] + [STAIRS / n for n in ("base_s1", "base_s2", "stairs_s1", "stairs_s2", "support_s1",
                                                         "support_s2", "student")]
PUSHED = [ORIGINAL] + [STAIRS / n for n in ("base_s1", "stairs_s2", "support_s2", "student")]
CATEGORIES = ("standing", "dropped upright", "on its side", "on its back")


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def measured(path: Path, name: str) -> bool:
    """True if a JSON-lines result file already has a line for this run (a restarted batch skips it)."""
    return path.exists() and any(Path(json.loads(line)["run"]).name == name for line in path.read_text().splitlines())


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    steps = [
        ("get-up training", [LAB, *GETUP, "--num_envs", "64", "--iterations", "3", "--seed", "1", "--out", str(out / "getup")],
         out / "getup" / "result.json"),
        ("get-up test", [LAB, "tools/dog/getup_test.py", str(out / "getup"), "--num_envs", "64", "--seconds", "2",
                         "--out", str(out / "getup_test.jsonl")], out / "getup_test.jsonl"),
        ("push test", [LAB, "tools/dog/push_test.py", str(STAIRS / "base_s1"), "--num_envs", "70", "--out",
                       str(out / "push_test.jsonl")], out / "push_test.jsonl"),
        ("climb measure", [LAB, "tools/dog/benchmark.py", str(STAIRS / "base_s1"), "stairs_straight", "--num_envs", "64",
                           "--out", str(out / "benchmark.jsonl")], out / "benchmark.jsonl"),
    ]
    for name, cmd, expect in steps:
        code = run(cmd, log)
        if code != 0 or not expect.exists():
            (HERE / "batch.failed").write_text(f"smoke ({name}): exit code {code}, see smoke/smoke.log\n")
            return False
    if "type_level_climb_m" not in (out / "benchmark.jsonl").read_text():
        (HERE / "batch.failed").write_text("smoke (climb measure): no type_level_climb_m in the benchmark line\n")
        return False
    (out / "ok").write_text("ok")
    return True


def record_getup_video(run_dir: Path) -> None:
    """One recording per start (standing, dropped upright, side, back), side by side."""
    sys.path.insert(0, str(ROOT))
    from core.evolution.compare_video import compose

    clips = []
    for k, name in enumerate(CATEGORIES):
        clip = f"getup_{k}.mp4"
        run([PY, "-m", "core.evolution.play", "--run_dir", str(run_dir), "--video", "--num_envs", "1", "--drive", "stand",
             "--out_name", clip, "--env_set", f"getup_category={k}", "--eye", "0.6", "-0.7", "0.35"], HERE / "videos.log")
        path = run_dir / "videos" / clip
        if path.exists():
            clips.append((name, path))
    if clips:
        compose(clips, f"getting up ({run_dir.name}): 8 s from each start", HERE / "videos" / f"getup_{run_dir.name}.mp4")


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    for dog in STAIRS_DOGS:  # 1
        if (dog / "result.json").exists() and not measured(HERE / "climb.jsonl", dog.name):
            run([LAB, "tools/dog/benchmark.py", str(dog), "stairs_straight", "--out", str(HERE / "climb.jsonl")],
                HERE / "climb.log")
    for dog in PUSHED:  # 2
        if (dog / "result.json").exists() and not measured(HERE / "push_test.jsonl", dog.name):
            run([LAB, "tools/dog/push_test.py", str(dog), "--out", str(HERE / "push_test.jsonl")], HERE / "push.log")
    for dog in (ORIGINAL, STAIRS / "stairs_s2"):  # 3
        if not measured(HERE / "getup_test.jsonl", dog.name):
            run([LAB, "tools/dog/getup_test.py", str(dog), "--out", str(HERE / "getup_test.jsonl")], HERE / "getup_test.log")
    for name, seed in RUNS:  # 4
        if not (HERE / name / "result.json").exists():
            run([LAB, *GETUP, "--num_envs", "2048", "--iterations", str(ITERATIONS), "--seed", str(seed), "--out",
                 str(HERE / name)], HERE / f"{name}.log")
    scores = {}
    for name, _ in RUNS:  # 5
        if (HERE / name / "result.json").exists():
            run([LAB, "tools/dog/getup_test.py", str(HERE / name), "--out", str(HERE / "getup_test.jsonl")],
                HERE / "getup_test.log")
            rows = [json.loads(line) for line in (HERE / "getup_test.jsonl").read_text().splitlines()]
            mine = [r for r in rows if Path(r["run"]).name == name]
            if mine:
                scores[name] = sum(mine[-1][c]["success"] for c in CATEGORIES)
    if scores:
        record_getup_video(HERE / max(scores, key=scores.get))
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
