"""Flexible back experiment: re-evaluate every variant with extra seeds and print falls per level, walk score and energy.

    python runs/dog_spine/compare.py            (runs the extra evaluations that are missing, then prints the table)
"""
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAB = str(ROOT / "lab.cmd")
VARIANTS = ["stiff", "roll_k4", "roll_k8", "roll_k16", "pitch_k8", "both_k8"]
SEEDS = [1, 2]


def evaluate(name: str, seed: int) -> Path:
    out = ROOT / "runs" / "dog_spine" / "eval" / f"{name}_s{seed}"
    if not (out / "result.json").exists():
        cmd = [LAB, "-m", "core.evolution.train", "--task", "command", "--terrain", "rough", "--iterations", "0",
               "--num_envs", "2048", "--sensors", "imu", "--obs_history", "8", "--seed", str(seed),
               "--genome", rf"runs\dog_spine\{name}.json", "--out", str(out),
               "--init_checkpoint", rf"runs\dog_spine\{name}\model_final.pt"]
        with open(out.parent / f"{name}_s{seed}.log", "w") as log:
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return out / "result.json"


def main() -> None:
    (ROOT / "runs" / "dog_spine" / "eval").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in VARIANTS:
        files = [ROOT / "runs" / "dog_spine" / name / "result.json"] + [evaluate(name, s) for s in SEEDS]
        results = [json.loads(f.read_text()) for f in files if f.exists()]
        if not results:
            continue
        n_levels = len(results[0]["level_fall_rates"])
        levels = [sum(r["level_fall_rates"][lv] for r in results) / len(results) for lv in range(n_levels)]
        falls = sum(r["fall_rate"] for r in results) / len(results)
        episodes = sum(r["episodes"] for r in results)
        se = math.sqrt(falls * (1 - falls) / episodes)
        types = {}
        for t in results[0]["type_level_scores"]:
            types[t] = sum(sum(r["type_level_scores"][t]) / n_levels for r in results) / len(results)
        rows.append({
            "variant": name, "evaluations": len(results), "episodes": episodes,
            "fall_rate": falls, "fall_rate_se": se, "level_fall_rates": levels,
            "walk_score": sum(r["walk_score"] for r in results) / len(results),
            "cost_of_transport": sum(r.get("cost_of_transport", float("nan")) for r in results) / len(results),
            "type_walk_scores": types,
        })
    (ROOT / "runs" / "dog_spine" / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(f"{'variant':10s} {'falls':>12s} {'walk':>6s} {'CoT':>5s}   falls per level 0..9")
    for r in rows:
        lv = " ".join(f"{100 * v:3.0f}" for v in r["level_fall_rates"])
        print(f"{r['variant']:10s} {100 * r['fall_rate']:5.1f} +- {100 * r['fall_rate_se']:3.1f}% "
              f"{r['walk_score']:6.3f} {r['cost_of_transport']:5.2f}   {lv}")
    print("\nwalk score per terrain type (mean over levels):")
    types = list(rows[0]["type_walk_scores"])
    print(f"{'variant':10s} " + " ".join(f"{t[:12]:>12s}" for t in types))
    for r in rows:
        print(f"{r['variant']:10s} " + " ".join(f"{r['type_walk_scores'][t]:12.3f}" for t in types))


if __name__ == "__main__":
    main()
