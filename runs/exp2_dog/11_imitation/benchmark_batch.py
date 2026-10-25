"""Every candidate dog through every benchmark scenario (tools/dog/benchmark.py); results in benchmark.jsonl,
summary table in benchmark.md."""
import os
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
LAUNCHER = "lab.cmd" if os.name == "nt" else "lab.sh"  # the lab's Python with this repository on the path
HERE = ROOT / "runs" / "exp2_dog" / "11_imitation"
OUT = HERE / "benchmark.jsonl"
DOGS = {"now: D_print_s2": "runs/exp2_dog/10_printable_dog/D_print_s2",
        "imitation (style 6)": "runs/exp2_dog/11_imitation/amp_w6m_s2",
        "imitation + slow, seed 1": "runs/exp2_dog/11_imitation/amp_w6m_slow_s1",
        "imitation + slow, seed 2": "runs/exp2_dog/11_imitation/amp_w6m_slow_s2"}
SCENARIOS = ["flat", "course", "slippery", "grippy", "pushes"]


def done() -> set:
    if not OUT.exists():
        return set()
    return {(r["run"], r["scenario"]) for r in map(json.loads, OUT.read_text().splitlines())}


for sc in SCENARIOS:
    for run in DOGS.values():
        if (run, sc) in done():
            continue
        with open(HERE / "benchmark.log", "a") as log:
            subprocess.run([str(ROOT / LAUNCHER), "tools/dog/benchmark.py", run, sc, "--out", str(OUT)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)

rows = {(r["run"], r["scenario"]): r for r in map(json.loads, OUT.read_text().splitlines())}
lines = ["| dog | " + " | ".join(SCENARIOS) + " |", "|---|" + "---|" * len(SCENARIOS)]
for label, run in DOGS.items():
    cells = []
    for sc in SCENARIOS:
        r = rows.get((run, sc))
        cells.append("-" if r is None else f"{r['fall_rate']:.1%} / {r['walk_score']:.2f}")
    lines.append(f"| {label} | " + " | ".join(cells) + " |")
lines.append("")
lines.append("Cells: falls / walk score. Rubble (blocks) falls per scenario:")
for label, run in DOGS.items():
    parts = []
    for sc in SCENARIOS[1:]:
        r = rows.get((run, sc))
        if r and "type_fall_rates" in r:
            v = r["type_fall_rates"]["blocks"]
            parts.append(f"{sc} {sum(v) / len(v):.1%}")
    lines.append(f"- {label}: " + ", ".join(parts))
(HERE / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
