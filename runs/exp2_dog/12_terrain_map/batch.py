"""Terrain map experiments (journal step 33). From amp_w6m_slow_s1 (imitation + slow walks), all with the robust
recipe of 11_imitation/batch4.py (paw grip 0.5-1.4, shoves 0.3 m/s every 3-6 s, heading hold):

  robust_scan       the perfect height scan (the control; = batch4's robust_slow_s1)
  map_depth         terrain map from a chest depth camera, with memory
  map_tof           terrain map from an 8x8 time-of-flight sensor (VL53L5CX), with memory
  map_depth_nomem   depth camera, no memory: only what the latest frame covers
  student_depth     learned memory: a GRU student that sees only the latest depth frame, taught by map_depth

Map runs train with sensor blackouts (0.1 per second, 0.3-1.5 s). Then: map quality (map_check), the benchmark
(course, dark = blackouts, slippery, pushes) and stairs videos. Smoke tests first: the batch stops at the first one
that fails (smoke.failed names it).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAUNCHER = "lab.cmd" if os.name == "nt" else "lab.sh"  # the lab's Python with this repository on the path
HERE = ROOT / "runs" / "exp2_dog" / "12_terrain_map"
IMIT = ROOT / "runs" / "exp2_dog" / "11_imitation"
START = IMIT / "amp_w6m_slow_s1"
LAB = str(ROOT / LAUNCHER)
PY = str(ROOT / "IsaacLab" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
COMMON = ["-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough",
          "--sensors", "imu", "--obs_history", "8", "--symmetry", "--height_scan",
          "--genome", str(START / "genome.json"), "--init_checkpoint", str(START / "model_final.pt"),
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)",
          "--env_override", "command_accel=0.5", "--env_override", "command_yaw_accel=1.5",
          "--env_override=command_vx_range=(-0.3, 0.45)", "--env_override", "gait_freq_speed=0.45",
          "--env_override", "gait=free", "--env_override", "gait_trot_weight=3.0", "--env_override", "flight_weight=2.0",
          "--env_override", "action_scale=0.75", "--env_override", "action_rate_weight=3.0",
          "--env_override", "stand_still_weight=0.3",
          "--env_override", "amp_motion_file=" + str(ROOT / "data" / "mocap" / "puppy_dog_motion_slow.npz").replace("\\", "/"),
          "--env_override", "amp_weight=6.0", "--env_override", "amp_mirror=True",
          "--env_override=paw_friction_range=(0.5, 1.4)", "--env_override", "push_velocity=0.3",
          "--env_override=push_interval=(3.0, 6.0)", "--env_override", "heading_hold_gain=2.0"]
MAP = ["--env_override", "map_dropout_rate=0.1"]
RUNS = [("robust_scan", []),
        ("map_depth", ["--terrain_map", "depth", *MAP]),
        ("map_tof", ["--terrain_map", "tof", *MAP]),
        ("map_depth_nomem", ["--terrain_map", "depth", *MAP, "--env_override", "map_memory_s=0.0"])]
STUDENTS = [("student_depth", "map_depth")]
SCENARIOS = ("course", "dark", "slippery", "pushes")


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n>>> {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def smoke() -> bool:
    """Short end-to-end runs of every new piece; False (and smoke.failed) at the first failure."""
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "smoke.log"
    steps = [
        ("map unit checks", [PY, "tools/dog/test_terrain_map.py"], None),
        ("map check depth", [LAB, "tools/dog/map_check.py", str(START), "--terrain_map", "depth", "--num_envs", "32",
                             "--steps", "300", "--out", str(out)], out / f"map_{START.name}_depth.png"),
        ("map check tof", [LAB, "tools/dog/map_check.py", str(START), "--terrain_map", "tof", "--num_envs", "32",
                           "--steps", "300", "--out", str(out)], out / f"map_{START.name}_tof.png"),
        ("train on the map", [LAB, *COMMON, "--terrain_map", "depth", *MAP, "--num_envs", "64", "--iterations", "3",
                              "--critic_warmup", "2", "--seed", "1", "--out", str(out / "train")],
         out / "train" / "result.json"),
        ("memory student", [LAB, "-m", "core.evolution.distill_memory", "--teacher", str(out / "train"), "--out",
                            str(out / "student"), "--num_envs", "64", "--iterations", "3", "--seq_len", "16"],
         out / "student" / "result.json"),
        ("benchmark student", [LAB, "tools/dog/benchmark.py", str(out / "student"), "dark", "--num_envs", "64", "--out",
                               str(out / "benchmark.jsonl")], out / "benchmark.jsonl"),
    ]
    for name, cmd, expect in steps:
        code = run(cmd, log)
        if code != 0 or (expect is not None and not expect.exists()):
            (HERE / "smoke.failed").write_text(f"{name}: exit code {code}, see {log}\n")
            return False
    (HERE / "smoke.ok").write_text("ok")
    return True


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "smoke.failed").unlink(missing_ok=True)
    if not (HERE / "smoke.ok").exists() and not smoke():
        return 1
    for name, extra in RUNS:
        out = HERE / name
        if not (out / "result.json").exists():
            run([LAB, *COMMON, "--num_envs", "2048", "--iterations", "1500", "--critic_warmup", "50", "--seed", "1",
                 *extra, "--out", str(out)], HERE / f"{name}.log")
    for name, teacher in STUDENTS:
        out = HERE / name
        if not (out / "result.json").exists() and (HERE / teacher / "result.json").exists():
            run([LAB, "-m", "core.evolution.distill_memory", "--teacher", str(HERE / teacher), "--out", str(out)],
                HERE / f"{name}.log")
    # map quality: the original dog walking with each sensor's map built alongside, and the map dogs on their own map
    checks = [(START, "depth"), (START, "tof")] + [(HERE / n, None) for n, e in RUNS if "--terrain_map" in e]
    for path, kind in checks:
        if (path / "result.json").exists():
            run([LAB, "tools/dog/map_check.py", str(path), "--num_envs", "256", "--out", str(HERE / "map_check"),
                 *(["--terrain_map", kind] if kind else [])], HERE / "map_check.log")
    for name in [n for n, _ in RUNS] + [n for n, _ in STUDENTS]:
        if (HERE / name / "result.json").exists():
            for sc in SCENARIOS:
                run([LAB, "tools/dog/benchmark.py", str(HERE / name), sc, "--out", str(HERE / "benchmark.jsonl")],
                    HERE / "benchmark.log")
    labels = {"robust_scan": "perfect scan", "map_depth": "depth camera + memory", "map_tof": "8x8 ToF + memory",
              "map_depth_nomem": "depth camera, no memory"}
    for terrain in ("stairs_climb", "stairs_descend"):
        cmd = [PY, "-m", "core.evolution.compare_video", "--out", str(HERE / "videos" / f"{terrain}_l7.mp4"),
               "--terrain_type", terrain, "--level", "7", "--heading_hold", "2.0", "--eye", "0.9", "-1.1", "0.45",
               "--subtitle", f"{terrain}, level 7 (~5 cm steps)"]
        for name, label in labels.items():
            if (HERE / name / "result.json").exists():
                cmd += ["--run", f"{label}={HERE / name}"]
        run(cmd, HERE / "videos.log")
    run([PY, str(HERE / "summary.py")], HERE / "summary.log")
    (HERE / "batch.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
