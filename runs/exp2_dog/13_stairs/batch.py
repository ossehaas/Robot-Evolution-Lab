"""Deliberate stair stepping (journal step 35). The owner saw in the videos that the dogs walk into the stairs without
lifting their paws high enough, and that a step up or down should be a controlled move (three paws steady, the fourth
placed on the tread). Stairs are continuous climbing for a dog: ~16-20 cm paw span on 15 cm treads.

All from amp_w6m_slow_s1 (the candidate, 0.4 % falls on the course) WITHOUT the robust recipe of steps 32-34 (the
main suspect for the worse absolute numbers there), depth camera + terrain map with memory, sensor blackouts 0.1/s,
heading hold, mostly forward commands (forward up to 0.45 m/s, backwards 0.1, sideways 0.15), 2000 iterations,
seeds 1 and 2:

  base_s*      no new rewards: does dropping the robust recipe bring the quality back?
  stairs_s*    + stair treads shortening over the rows (40 cm: a curb, all paws fit -> 15 cm: continuous climbing)
               + terrain-aware paw lift (clear the ground just ahead by 1.5 cm, up to 10 cm)
               + toe-stubbing penalty, penalty for landing on a step edge
               + no style reward on stairs and rubble (the recorded dog walked on a flat floor)
               + joint range 85 % (was 75 %: ~64 mm paw lift instead of 56)
               + straight onto the stairs: on stair tiles the dog starts facing the steps (+-15 deg) and is not asked
                 to turn there (owner: it goes wrong when it turns on the stairs)
               + lift first: a paw right in front of a riser must rise, not push forwards into it (owner)
  support_s*   stairs + reward for three paws on the ground while moving on stairs

Then: benchmark (stairs_straight, forward, forward_dark, course; always the standard 15 cm stairs; step 33's map_depth
as "before" on stairs_straight), a GRU memory student from the stairs or support dog with the fewest falls walking
forwards, stairs videos (level 7, straight on) and results.md. A smoke test first.
Takes ~10 h.
"""
import os
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAUNCHER = "lab.cmd" if os.name == "nt" else "lab.sh"  # the lab's Python with this repository on the path
HERE = ROOT / "runs" / "exp2_dog" / "13_stairs"
START = ROOT / "runs" / "exp2_dog" / "11_imitation" / "amp_w6m_slow_s1"
LAB = str(ROOT / LAUNCHER)
PY = str(ROOT / "IsaacLab" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
O = "--env_override="
# the recipe amp_w6m_slow_s1 was trained with (11_imitation/batch*.py), plus heading hold
RECIPE = ["-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough",
          "--sensors", "imu", "--obs_history", "8", "--symmetry", "--height_scan",
          "--genome", str(START / "genome.json"), "--init_checkpoint", str(START / "model_final.pt"),
          O + "tracking_ang_weight=4.0", O + "tracking_ang_sigma=0.3", O + "tracking_lin_weight=6.0",
          O + "gait_freq_range=(1.0, 2.6)", O + "command_accel=0.5", O + "command_yaw_accel=1.5",
          O + "gait_freq_speed=0.45", O + "gait=free", O + "gait_trot_weight=3.0", O + "flight_weight=2.0",
          O + "action_scale=0.75", O + "action_rate_weight=3.0", O + "stand_still_weight=0.3",
          O + "amp_motion_file=" + str(ROOT / "data" / "mocap" / "puppy_dog_motion_slow.npz").replace("\\", "/"),
          O + "amp_weight=6.0", O + "amp_mirror=True", O + "heading_hold_gain=2.0"]
SEEING = ["--terrain_map", "depth", O + "map_dropout_rate=0.1"]
FORWARD = [O + "command_vx_range=(-0.1, 0.45)", O + "command_vy_range=(-0.15, 0.15)"]
STAIRS = [O + "stair_tread_range=(0.40, 0.15)", O + "foot_clearance_terrain=True", O + "stumble_weight=1.0",
          O + "foothold_weight=0.5", O + "amp_skip_types=('stairs_climb', 'stairs_descend', 'blocks')",
          # the stairs terms only on the tiles without style: stair technique there, the dog's own style elsewhere
          O + "stairs_terms_types=('stairs_climb', 'stairs_descend', 'blocks')",
          O + "action_scale=0.85", O + "stairs_square_yaw=0.26", O + "stairs_turn_scale=0.0",
          O + "lift_first_weight=1.0",
          # the start dog was trained with action_scale 0.75: rescale its output so it starts with the same gait
          # (without this the style discriminator wins at once and the style reward stays 0: first stairs_s1 attempt)
          "--init_action_scale", "0.75"]
SUPPORT = [*STAIRS, O + "stairs_support_weight=1.0"]
VARIANTS = {"base": [], "stairs": STAIRS, "support": SUPPORT}
RUNS = [(f"{v}_s{seed}", seed, extra) for v, extra in VARIANTS.items() for seed in (1, 2)]
SCENARIOS = ("stairs_straight", "forward", "forward_dark", "course")
BEFORE = ROOT / "runs" / "exp2_dog" / "12_terrain_map" / "map_depth"


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def train_cmd(extra: list[str], seed: int, out: Path, num_envs: int = 2048, iterations: int = 2000) -> list[str]:
    return [LAB, *RECIPE, *SEEING, *FORWARD, *extra, "--num_envs", str(num_envs), "--iterations", str(iterations),
            "--critic_warmup", "50" if iterations > 10 else "2", "--seed", str(seed), "--out", str(out)]


def forward_falls(name: str) -> float:
    rows = [json.loads(line) for line in (HERE / "benchmark.jsonl").read_text().splitlines()]
    falls = [r["fall_rate"] for r in rows if Path(r["run"]).name == name and r["scenario"] == "forward"]
    return falls[-1] if falls else 1.0


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    smoke = HERE / "smoke"
    if not (smoke / "ok").exists():
        smoke.mkdir(exist_ok=True)
        steps = [("train with the stairs rewards", train_cmd(SUPPORT, 1, smoke / "train", 64, 3), smoke / "train" / "result.json"),
                 ("forward benchmark", [LAB, "tools/dog/benchmark.py", str(smoke / "train"), "forward", "--num_envs", "64",
                                        "--out", str(smoke / "benchmark.jsonl")], smoke / "benchmark.jsonl")]
        for name, cmd, expect in steps:
            code = run(cmd, smoke / "smoke.log")
            if code != 0 or not expect.exists():
                (HERE / "batch.failed").write_text(f"smoke ({name}): exit code {code}, see smoke/smoke.log\n")
                return 1
        (smoke / "ok").write_text("ok")
    for name, seed, extra in RUNS:
        if not (HERE / name / "result.json").exists():
            run(train_cmd(extra, seed, HERE / name), HERE / f"{name}.log")
    run([LAB, "tools/dog/benchmark.py", str(BEFORE), "stairs_straight", "--out", str(HERE / "benchmark.jsonl")],
        HERE / "benchmark.log")
    for name, _, _ in RUNS:
        if (HERE / name / "result.json").exists():
            for sc in SCENARIOS:
                run([LAB, "tools/dog/benchmark.py", str(HERE / name), sc, "--out", str(HERE / "benchmark.jsonl")],
                    HERE / "benchmark.log")
    # the memory student from the best stairs-trained dog (fewest falls walking forwards)
    candidates = [n for n, _, _ in RUNS if not n.startswith("base") and (HERE / n / "result.json").exists()]
    if candidates and (HERE / "benchmark.jsonl").exists():
        teacher = min(candidates, key=forward_falls)
        student = HERE / "student"
        if not (student / "result.json").exists():
            run([LAB, "-m", "core.evolution.distill_memory", "--teacher", str(HERE / teacher), "--out", str(student)],
                HERE / "student.log")
        if (student / "result.json").exists():
            for sc in SCENARIOS:
                run([LAB, "tools/dog/benchmark.py", str(student), sc, "--out", str(HERE / "benchmark.jsonl")],
                    HERE / "benchmark.log")
    labels = {"before (step 33 map_depth)": BEFORE,
              "base": HERE / "base_s1", "stairs rewards": HERE / "stairs_s1", "stairs + 3 paws down": HERE / "support_s1"}
    for terrain in ("stairs_climb", "stairs_descend"):
        cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(HERE / "videos" / f"{terrain}_l7.mp4"),
               "--terrain_type", terrain, "--level", "7", "--heading_hold", "2.0", "--eye", "0.9", "-1.1", "0.45",
               "--drive", "straight", "--subtitle", f"{terrain}, level 7 (~5 cm steps, 15 cm treads), straight on"]
        for label, path in labels.items():
            if (path / "result.json").exists():
                cmd += ["--run", f"{label}={path}"]
        run(cmd, HERE / "videos.log")
    run([PY, str(ROOT / "runs" / "exp2_dog" / "12_terrain_map" / "summary.py"), "--dir", str(HERE),
         "--runs", ",".join(["map_depth"] + [n for n, _, _ in RUNS] + ["student"]), "--scenarios", ",".join(SCENARIOS),
         "--out", "results.md", "--title", "Deliberate stair stepping"], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
