"""Score every dog with every trained judge (AMP discriminator, frozen): evaluation only (--iterations 0 warm start).
Each judge was trained against its own policy, so its scores of the others are a cross-check, not an absolute scale.
Writes cross_score.json: {judge: {dog: style_score}}."""
import os
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAUNCHER = "lab.cmd" if os.name == "nt" else "lab.sh"  # the lab's Python with this repository on the path
HERE = ROOT / "runs" / "exp2_dog" / "11_imitation"
DOGS = {"D_print_s2": ROOT / "runs" / "exp2_dog" / "10_printable_dog" / "D_print_s2",
        "free_ctrl": HERE / "free_ctrl", "amp_w3": HERE / "amp_w3", "amp_w6": HERE / "amp_w6",
        "amp_w6m_s1": HERE / "amp_w6m_s1", "amp_w6m_slow_s1": HERE / "amp_w6m_slow_s1", "amp_w6m_s2": HERE / "amp_w6m_s2"}
MOTION = str(ROOT / "data" / "mocap" / "puppy_dog_motion.npz").replace("\\", "/")
# judge -> (motion file, mirrored): the judge's input normalization comes from the data it was trained on
JUDGES = {"amp_w3": (MOTION, False), "amp_w6": (MOTION, False), "amp_w6m_s1": (MOTION, True)}
work = HERE / "cross_score"
work.mkdir(exist_ok=True)
table = json.loads((HERE / "cross_score.json").read_text()) if (HERE / "cross_score.json").exists() else {}
for judge, (motion, mirror) in JUDGES.items():
    disc = str(HERE / judge / "amp_disc.pt").replace("\\", "/")
    for dog, run in DOGS.items():
        if dog in table.get(judge, {}):
            continue
        res = json.loads((run / "result.json").read_text())
        cmd = [str(ROOT / LAUNCHER), "-m", "core.evolution.train", "--task", "command", "--terrain", "dog_rough",
               "--num_envs", "1024", "--sensors", "imu", "--obs_history", "8", "--symmetry", "--height_scan",
               "--genome", str(run / "genome.json"), "--init_checkpoint", str(run / "model_final.pt"), "--iterations", "0",
               "--seed", "7", "--out", str(work / f"{judge}__{dog}")]
        for k, v in res["env_overrides"].items():
            if not k.startswith("amp_"):
                cmd += [f"--env_override={k}={tuple(v) if isinstance(v, list) else v}"]
        cmd += ["--env_override", f"amp_motion_file={motion}", "--env_override", f"amp_mirror={mirror}",
                "--env_override", f"amp_disc_file={disc}"]
        with open(work / f"{judge}__{dog}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        out = work / f"{judge}__{dog}" / "result.json"
        if out.exists():
            table.setdefault(judge, {})[dog] = round(json.loads(out.read_text())["style_score"], 4)
            (HERE / "cross_score.json").write_text(json.dumps(table, indent=1))
            print(judge, dog, table[judge][dog], flush=True)
(HERE / "cross_score.done").write_text("done")
