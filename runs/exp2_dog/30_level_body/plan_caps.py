"""Step 50, part 1 (CPU): how level can the body stay going up the stairs? The stair planner (tools/dog/stairs_plan.py)
with a limit on the body's nose-up pitch and two balance margins; plans_p<cap>_m<margin>/ and plan_caps.md.

The planner keeps a paw link rigid to its shin, so it cannot use the real paw's housing (no ankle spring: the base would
hit the floor at any shin angle). genome_plans.json is the same dog with the sphere paw (same sole height), and the gap
to the riser is 13.4 mm instead of 4, so the real toes (27.4 mm ahead, the sphere 18) still fit.

    lab runs/exp2_dog/30_level_body/plan_caps.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GENOME = HERE / "genome_plans.json"
RISER_GAP = 0.0134
CAPS = (5, 10, 15, 20, 45)  # [deg] 45 = the planner's full range
MARGINS = (0.015, 0.03)
RISES, TREADS = (2, 3, 4, 5), (25, 20, 15, 12, 9)
CELLS = [f"up_r{r}_t{t}" for r in RISES for t in TREADS]


def folder(cap: int, margin: float) -> Path:
    return HERE / f"plans_p{cap}_m{round(margin * 1000)}"


def shift(res: dict, rear: bool) -> float | None:
    """Mean body x minus the paws' centre [cm] while a rear / front paw swings."""
    vals = [100 * (f["body"][0] - sum(f["paws_before"]) / 4) for f in res["frames"] if f["swing"][1] == ("1" if rear else "0")]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    for margin in MARGINS:
        for cap in CAPS:
            out = folder(cap, margin)
            for _ in range(5):  # the CAD build crashes at random now and then: go on with the cells still missing
                todo = [c for c in CELLS if not (out / f"{c}.json").exists()]
                if not todo:
                    break
                subprocess.run([sys.executable, "tools/dog/stairs_plan.py", "--genome", str(GENOME), "--out_dir", str(out), "--cells", ",".join(todo),
                                "--max_pitch", str(cap), "--stability", str(margin), "--riser_gap", str(RISER_GAP), "--workers", "20"], cwd=ROOT)
    md = ["# How level can the body stay going up? (step 50)", "",
          "Cells: feasible plan for rise / tread at a nose-up limit: margin [cm] / highest pitch used [deg] / body against the paws while a rear paw swings [cm, + = ahead]. '.' = no plan.", ""]
    for margin in MARGINS:
        md += [f"## Balance margin at least {100 * margin:.1f} cm", "", "| rise / tread | slope | " + " | ".join(f"max {c} deg" if c < 45 else "free" for c in CAPS) + " |",
               "|---|---|" + "---|" * len(CAPS)]
        for r in RISES:
            for t in TREADS:
                row, slope = [], None
                for cap in CAPS:
                    res = json.loads((folder(cap, margin) / f"up_r{r}_t{t}.json").read_text())
                    slope = res["slope_deg"]
                    row.append(f"{100 * res['min_margin_m']:.1f} / {res['max_pitch_deg']:.0f} / {shift(res, True):+.1f}" if res["feasible"] else ".")
                md.append(f"| {r} / {t} | {slope:.0f} | " + " | ".join(row) + " |")
        md += ["", "Plans found: " + ", ".join(
            f"{c if c < 45 else 'free'}: {sum(json.loads((folder(c, margin) / f'up_r{r}_t{t}.json').read_text())['feasible'] for r in RISES for t in TREADS)}" for c in CAPS)
            + f" of {len(RISES) * len(TREADS)}", ""]
    (HERE / "plan_caps.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
