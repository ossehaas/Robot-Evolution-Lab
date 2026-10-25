"""Robust imitation: the benchmark (benchmark.md) showed the imitation dogs topple more under shoves (2.4-11 % vs
0.4 %) and on slippery rubble; the grip was never varied in training. Fine-tune with paw grip 0.5-1.4 (slippery tiles
to grippy rubber), shoves of 0.3 m/s every 3-6 s (training had 0.2 every 4-8 s) and heading hold during training.
  robust_slow_s1/s2  from amp_w6m_slow_s1 (style 6, mirrored + slowed recordings), seeds 1 and 2
  robust_D           from D_print_s2 with its own recipe (walk/trot rhythm, no imitation): the fair comparison
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
HERE = ROOT / "runs" / "exp2_dog" / "11_imitation"
START = ROOT / "runs" / "exp2_dog" / "10_printable_dog" / "D_print_s2"
COMMON = ["-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough",
          "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8", "--symmetry", "--height_scan",
          "--genome", str(START / "genome.json"),
          "--iterations", "1500", "--critic_warmup", "50",
          "--env_override", "tracking_ang_weight=4.0", "--env_override", "tracking_ang_sigma=0.3",
          "--env_override", "tracking_lin_weight=6.0", "--env_override=gait_freq_range=(1.0, 2.6)",
          "--env_override", "command_accel=0.5", "--env_override", "command_yaw_accel=1.5",
          "--env_override=command_vx_range=(-0.3, 0.45)", "--env_override", "gait_freq_speed=0.45",
          "--env_override", "gait=free", "--env_override", "gait_trot_weight=3.0", "--env_override", "flight_weight=2.0",
          "--env_override", "action_scale=0.75", "--env_override", "action_rate_weight=3.0",
          "--env_override", "stand_still_weight=0.3"]
AMP = ["--env_override", "amp_motion_file=" + str(ROOT / "data" / "mocap" / "puppy_dog_motion.npz").replace("\\", "/")]
SLOW = ["--env_override", "amp_motion_file=" + str(ROOT / "data" / "mocap" / "puppy_dog_motion_slow.npz").replace("\\", "/")]
W6 = ["--env_override", "amp_weight=6.0", "--env_override", "amp_mirror=True"]
ROBUST = ["--env_override=paw_friction_range=(0.5, 1.4)", "--env_override", "push_velocity=0.3",
          "--env_override=push_interval=(3.0, 6.0)", "--env_override", "heading_hold_gain=2.0"]
SLOW_START = ["--init_checkpoint", str(HERE / "amp_w6m_slow_s1" / "model_final.pt")]
D_START = ["--init_checkpoint", str(START / "model_final.pt"), "--env_override", "gait=walk_trot"]
RUNS = [("robust_slow_s1", "1", [*SLOW_START, *SLOW, *W6, *ROBUST]),
        ("robust_D", "1", [*D_START, *ROBUST]),
        ("robust_slow_s2", "2", [*SLOW_START, *SLOW, *W6, *ROBUST])]
for name, seed, extra in RUNS:
    out = HERE / name
    if not (out / "result.json").exists():
        with open(HERE / f"{name}.log", "w") as log:
            subprocess.run([str(ROOT / "lab.cmd"), *COMMON, "--seed", seed, *extra, "--out", str(out)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
for name, _, _ in RUNS:  # the benchmark for the new dogs
    for sc in ("flat", "course", "slippery", "grippy", "pushes"):
        subprocess.run([str(ROOT / "lab.cmd"), "tools/dog/benchmark.py", str(HERE / name), sc, "--out",
                        str(HERE / "benchmark.jsonl")], stdout=open(HERE / "benchmark.log", "a"),
                       stderr=subprocess.STDOUT, cwd=ROOT)
(HERE / "batch4.done").write_text("done")
