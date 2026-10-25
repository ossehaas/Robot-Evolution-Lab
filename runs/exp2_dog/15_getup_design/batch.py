"""Which design change lets the dog get up from its side? (journal step 37)

Step 36: with the sideways (abduction) joint limited to +-29 deg the dog rolls from its back onto its side but never
gets from its side onto its belly: no limb can push the body over, and neck, head and tail move randomly. Here the same
get-up training (from scratch, 3000 it, flat floor, perfect scan, full joint range) with:

  abduction range   29 / 50 / 80 deg (genomes/abd*.json; the current shoulder collides beyond ~29 deg: the
                    mechanics follow once the needed range is known)
  neck/head/tail    random as before, or driven by the policy (levers for momentum)
  self-collision    ON in every run: legs passing through each other or the body would be a fake solution

  getup_abd29_head, getup_abd50, getup_abd50_head, getup_abd80, getup_abd80_head (abd29 without the head = step 36)

Then the get-up test per start, a video of the four starts for the best run and results.md. A smoke test first (incl.
a check that self-collision does not break a standing walker). ~5 h.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
HERE = ROOT / "runs" / "exp2_dog" / "15_getup_design"
ORIGINAL = ROOT / "runs" / "exp2_dog" / "11_imitation" / "amp_w6m_slow_s1"
LAB = str(ROOT / "lab.cmd")
PY = str(ROOT / "IsaacLab" / ".venv" / "Scripts" / "python.exe")
O = "--env_override="
ITERATIONS = 3000
CATEGORIES = ("standing", "dropped upright", "on its side", "on its back")


def getup_cmd(abd: int, head: bool, out: Path, num_envs: int = 2048, iterations: int = ITERATIONS) -> list[str]:
    return [LAB, "-m", "core.evolution.train", "--task", "command", "--terrain", "flat", "--sensors", "imu",
            "--obs_history", "8", "--symmetry", "--height_scan", "--genome", str(HERE / "genomes" / f"abd{abd}.json"),
            O + "getup=True", O + "command_standing_prob=1.0", O + "episode_length_s=8.0", O + "action_scale=1.0",
            O + "action_rate_weight=0.1", O + f"getup_curriculum_steps={iterations * 24}", O + "self_collisions=True",
            *([O + "policy_drives_expression=True"] if head else []),
            "--num_envs", str(num_envs), "--iterations", str(iterations), "--seed", "1", "--out", str(out)]


RUNS = [("getup_abd29_head", 29, True), ("getup_abd50", 50, False), ("getup_abd50_head", 50, True),
        ("getup_abd80", 80, False), ("getup_abd80_head", 80, True)]


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def last_line(path: Path, name: str) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mine = [r for r in rows if Path(r["run"]).name == name]
    return mine[-1] if mine else None


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"

    def fail(msg: str) -> bool:
        (HERE / "batch.failed").write_text(msg + f", see {log}\n")
        return False

    # a walker standing with self-collision on must still stand (no parts starting inside each other)
    code = run([LAB, "tools/dog/getup_test.py", str(ORIGINAL), "--num_envs", "64", "--seconds", "8", "--set",
                "self_collisions=True", "--out", str(out / "getup_test.jsonl")], log)
    line = last_line(out / "getup_test.jsonl", ORIGINAL.name)
    if code != 0 or line is None:
        return fail(f"smoke (walker with self-collision): exit code {code}")
    if line["standing"]["success"] < 0.8:
        return fail(f"smoke: with self-collision only {line['standing']['success']:.0%} of standing walkers stand")
    code = run(getup_cmd(80, True, out / "getup", 64, 3), log)
    if code != 0 or not (out / "getup" / "result.json").exists():
        return fail(f"smoke (get-up training abd80 + head): exit code {code}")
    code = run([LAB, "tools/dog/getup_test.py", str(out / "getup"), "--num_envs", "64", "--seconds", "2", "--out",
                str(out / "getup_test.jsonl")], log)
    if code != 0 or last_line(out / "getup_test.jsonl", "getup") is None:
        return fail(f"smoke (get-up test of the abd80 + head dog): exit code {code}")
    (out / "ok").write_text("ok")
    return True


def record_getup_video(run_dir: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from core.evolution.compare_video import compose

    clips = []
    for k, name in enumerate(CATEGORIES):
        clip = f"getup_{k}.mp4"
        run([PY, "-m", "core.evolution.play", "--run_dir", str(run_dir), "--video", "--num_envs", "1", "--drive", "stand",
             "--out_name", clip, "--env_set", f"getup_category={k}", "--eye", "0.6", "-0.7", "0.35"], HERE / "videos.log")
        if (run_dir / "videos" / clip).exists():
            clips.append((name, run_dir / "videos" / clip))
    if clips:
        compose(clips, f"getting up ({run_dir.name}): 8 s from each start", HERE / "videos" / f"{run_dir.name}.mp4")


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not (HERE / "smoke" / "ok").exists() and not smoke():
        return 1
    for name, abd, head in RUNS:
        if not (HERE / name / "result.json").exists():
            run(getup_cmd(abd, head, HERE / name), HERE / f"{name}.log")
    scores = {}
    for name, _, _ in RUNS:
        if (HERE / name / "result.json").exists():
            if last_line(HERE / "getup_test.jsonl", name) is None:
                run([LAB, "tools/dog/getup_test.py", str(HERE / name), "--out", str(HERE / "getup_test.jsonl")],
                    HERE / "getup_test.log")
            line = last_line(HERE / "getup_test.jsonl", name)
            if line:
                scores[name] = line["on its side"]["success"] + line["on its back"]["success"]
    if scores:
        best = max(scores, key=scores.get)
        record_getup_video(HERE / best)
        if best != "getup_abd80_head" and (HERE / "getup_abd80_head" / "result.json").exists():
            record_getup_video(HERE / "getup_abd80_head")  # the most capable variant, for comparison
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
