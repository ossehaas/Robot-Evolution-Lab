"""Forward walking (journal step 34). A chest sensor only looks ahead, but the benchmark walked in random directions.

  A (question 1, no training, ~35 min): how much of the gap to the perfect scan is the walking direction? The five
    dogs of batch.py on the course with every command within +-30 deg of straight ahead ("forward"), with blackouts
    ("forward_dark"), and the map quality when walking forwards.
  B (question 2, ~3.5 h): train for it. +1500 iterations each with mostly forward commands (forward up to 0.45 m/s,
    backwards at most 0.1 m/s, sideways at most 0.1 m/s; turning unchanged), same robust recipe as batch.py:
      fwd_scan    from robust_scan (perfect scan, the reference)
      fwd_depth   from map_depth (depth camera + memory)
      fwd_tof     from map_tof (8x8 ToF + memory)
      student_fwd_depth   GRU memory student taught by fwd_depth
  C (~1 h): benchmark forward / forward_dark / course / pushes, map quality, stairs videos, tables:
    results_forward_test.md (A), results_forward_training.md (B + C).

A smoke test first (the forward scenario with 64 dogs); stops at a failure (batch2.failed).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import COMMON, HERE, LAB, MAP, PY, run  # noqa: E402

FWD = ["--env_override=command_vx_range=(-0.1, 0.45)", "--env_override=command_vy_range=(-0.1, 0.1)"]
OLD = ["robust_scan", "map_depth", "map_tof", "map_depth_nomem", "student_depth"]
TRAIN = [("fwd_scan", "robust_scan", []),
         ("fwd_depth", "map_depth", ["--terrain_map", "depth", *MAP]),
         ("fwd_tof", "map_tof", ["--terrain_map", "tof", *MAP])]
STUDENTS = [("student_fwd_depth", "fwd_depth")]
NEW = [n for n, _, _ in TRAIN] + [n for n, _ in STUDENTS]


def recipe(start: str) -> list[str]:
    """batch.py's recipe, warm-started from one of its runs instead of amp_w6m_slow_s1."""
    cmd = list(COMMON)
    cmd[cmd.index("--init_checkpoint") + 1] = str(HERE / start / "model_final.pt")
    return cmd


def bench(names: list[str], scenarios: tuple[str, ...]) -> None:
    for name in names:
        if (HERE / name / "result.json").exists():
            for sc in scenarios:
                run([LAB, "tools/dog/benchmark.py", str(HERE / name), sc, "--out", str(HERE / "benchmark.jsonl")],
                    HERE / "benchmark2.log")


def map_checks(names: list[str]) -> None:
    for name in names:
        if (HERE / name / "result.json").exists():
            run([LAB, "tools/dog/map_check.py", str(HERE / name), "--forward", "--num_envs", "256", "--out",
                 str(HERE / "map_check")], HERE / "map_check2.log")


def main() -> int:
    (HERE / "batch2.failed").unlink(missing_ok=True)
    smoke = HERE / "smoke" / "benchmark_forward.jsonl"
    if not smoke.exists():
        code = run([LAB, "tools/dog/benchmark.py", str(HERE / "map_depth"), "forward", "--num_envs", "64", "--out",
                    str(smoke)], HERE / "smoke" / "smoke2.log")
        if code != 0 or not smoke.exists():
            (HERE / "batch2.failed").write_text(f"smoke (forward benchmark): exit code {code}, see smoke/smoke2.log\n")
            return 1
    # A: the existing dogs, walking forwards
    bench(OLD, ("forward", "forward_dark"))
    map_checks(["map_depth", "map_tof"])
    run([PY, str(HERE / "summary.py"), "--runs", ",".join(OLD), "--scenarios", "forward,forward_dark,course,dark",
         "--out", "results_forward_test.md", "--title", "Walking forwards: the batch.py dogs"], HERE / "summary2.log")
    # B: train for forward walking
    for name, start, extra in TRAIN:
        out = HERE / name
        if not (out / "result.json").exists():
            run([LAB, *recipe(start), *FWD, "--num_envs", "2048", "--iterations", "1500", "--critic_warmup", "50",
                 "--seed", "1", *extra, "--out", str(out)], HERE / f"{name}.log")
    for name, teacher in STUDENTS:
        out = HERE / name
        if not (out / "result.json").exists() and (HERE / teacher / "result.json").exists():
            run([LAB, "-m", "core.evolution.distill_memory", "--teacher", str(HERE / teacher), "--out", str(out)],
                HERE / f"{name}.log")
    # C: benchmark, map quality, videos, tables
    bench(NEW, ("forward", "forward_dark", "course", "pushes"))
    map_checks(["fwd_depth", "fwd_tof"])
    labels = {"fwd_scan": "perfect scan", "fwd_depth": "depth camera + memory", "fwd_tof": "8x8 ToF + memory"}
    for terrain in ("stairs_climb", "stairs_descend"):
        cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(HERE / "videos" / f"fwd_{terrain}_l7.mp4"),
               "--terrain_type", terrain, "--level", "7", "--heading_hold", "2.0", "--eye", "0.9", "-1.1", "0.45",
               "--subtitle", f"trained for forward walking: {terrain}, level 7 (~5 cm steps)"]
        for name, label in labels.items():
            if (HERE / name / "result.json").exists():
                cmd += ["--run", f"{label}={HERE / name}"]
        run(cmd, HERE / "videos2.log")
    run([PY, str(HERE / "summary.py"), "--runs", ",".join(["robust_scan", "map_depth", "student_depth", *NEW]),
         "--scenarios", "forward,forward_dark,course,pushes", "--out", "results_forward_training.md",
         "--title", "Trained for forward walking (batch.py dogs for comparison)"], HERE / "summary2.log")
    (HERE / "batch2.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
