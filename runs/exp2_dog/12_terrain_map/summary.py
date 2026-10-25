"""Tables of the terrain map experiments -> results.md (benchmark per scenario, falls per terrain type, map quality,
memory student training).

    python summary.py                          the first batch (batch.py)
    python summary.py --runs a,b --scenarios forward,course --out results_forward.md --title "..."   (batch2.py)
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="robust_scan,map_depth,map_tof,map_depth_nomem,student_depth")
ap.add_argument("--scenarios", default="course,dark,slippery,pushes")
ap.add_argument("--out", default="results.md")
ap.add_argument("--title", default="Terrain map experiments")
ap.add_argument("--dir", type=Path, default=None, help="experiment folder (default: this one)")
args = ap.parse_args()
HERE = args.dir.resolve() if args.dir else HERE
RUNS = args.runs.split(",")
SCENARIOS = args.scenarios.split(",")
lines = [f"# {args.title}", ""]

bench = {}
if (HERE / "benchmark.jsonl").exists():
    for row in map(json.loads, (HERE / "benchmark.jsonl").read_text().splitlines()):
        bench[(Path(row["run"]).name, row["scenario"])] = row  # the latest line per run and scenario counts
lines += ["## Falls per scenario (walk score)", "", "| run | " + " | ".join(SCENARIOS) + f" | power {SCENARIOS[0]} |",
          "|---|" + "---|" * (len(SCENARIOS) + 1)]
for r in RUNS:
    cells = []
    for sc in SCENARIOS:
        b = bench.get((r, sc))
        cells.append(f"{100 * b['fall_rate']:.1f} % ({b['walk_score']:.2f})" if b else "-")
    c = bench.get((r, SCENARIOS[0]))
    cells.append(f"{c['mechanical_power_w']:.1f} W" if c and "mechanical_power_w" in c else "-")
    lines.append(f"| {r} | " + " | ".join(cells) + " |")

# gap to the perfect scan: extra falls of each sensor dog over the perfect-scan dog of the same training
REFERENCE = {"map_depth": "robust_scan", "map_tof": "robust_scan", "map_depth_nomem": "robust_scan",
             "student_depth": "robust_scan", "fwd_depth": "fwd_scan", "fwd_tof": "fwd_scan",
             "student_fwd_depth": "fwd_scan"}
gaps = [r for r in RUNS if REFERENCE.get(r) in RUNS]
if gaps:
    lines += ["", "## Gap to the perfect scan (extra falls, percentage points; perfect-scan dog of the same training)",
              "", "| run | reference | " + " | ".join(SCENARIOS) + " |", "|---|---|" + "---|" * len(SCENARIOS)]
    for r in gaps:
        ref = REFERENCE[r]
        cells = []
        for sc in SCENARIOS:
            a, b = bench.get((r, sc)), bench.get((ref, sc))
            cells.append(f"{100 * (a['fall_rate'] - b['fall_rate']):+.1f}" if a and b else "-")
        lines.append(f"| {r} | {ref} | " + " | ".join(cells) + " |")

types = None
for (r, sc), b in bench.items():
    if r in RUNS and b.get("type_fall_rates"):
        types = list(b["type_fall_rates"])
        break
if types:
    for sc in SCENARIOS[:2]:
        lines += ["", f"## Falls per terrain type ({sc}): all rows (hardest rows 6-9)", "", "| run | " + " | ".join(types) + " |",
                  "|---|" + "---|" * len(types)]
        for r in RUNS:
            b = bench.get((r, sc))
            if b and b.get("type_fall_rates"):
                def cell(v):  # per difficulty row: all rows (the hardest rows 6-9)
                    if not isinstance(v, list):
                        return f"{100 * v:.1f} %"
                    return f"{100 * sum(v) / len(v):.1f} % ({100 * sum(v[6:]) / max(1, len(v[6:])):.0f} %)"

                lines.append(f"| {r} | " + " | ".join(cell(b["type_fall_rates"][t]) for t in types) + " |")

if (HERE / "map_check" / "map_check.jsonl").exists():
    lines += ["", "## Map quality: share of scan points known (height error of known points)", "",
              "| dog / sensor | memory: ahead | under body | behind | latest frame: ahead | under body | behind |",
              "|---|---|---|---|---|---|---|"]
    for row in map(json.loads, (HERE / "map_check" / "map_check.jsonl").read_text().splitlines()):
        cells = [f"{100 * row[f'{m}_{g}_known']:.0f} % ({row[f'{m}_{g}_err_cm']:.1f} cm)"
                 for m in ("memory", "latest") for g in ("ahead", "under", "behind")]
        if Path(row["run"]).name not in RUNS and row["policy_on_map"]:
            continue
        who = (f"{Path(row['run']).name} / {row['sensor']}" + ("" if row["policy_on_map"] else " (walks on perfect scan)")
               + (", forward" if row.get("forward") else ""))
        lines.append(f"| {who} | " + " | ".join(cells) + " |")

for st in [HERE / r / "result.json" for r in RUNS if r.startswith("student")]:
    if not st.exists():
        continue
    d = json.loads(st.read_text())
    last = d["loss_curve"][-1]
    lines += ["", f"## Memory student {st.parent.name}", "",
              f"- training {d['iterations']} iterations, {d['train_time_s'] / 60:.0f} min; final action loss "
              f"{last['action_loss']:.4f}, rebuild error under the body {last['rebuild_err_under_body_cm']:.1f} cm",
              f"- own evaluation: falls {100 * d['fall_rate']:.1f} %, walk score {d['walk_score']:.3f}; with blackouts "
              f"falls {100 * d['dark']['fall_rate']:.1f} %",
              f"- picture: {st.parent.name}/rebuilt_terrain.png"]
(HERE / args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
