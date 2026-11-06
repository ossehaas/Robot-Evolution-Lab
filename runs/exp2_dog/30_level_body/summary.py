"""Tables of step 50 -> results.md: teach_s1 trained on for a level body on the stairs, against teach_s1 itself."""
import json
import sys
from pathlib import Path

HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent  # steps 51, 52 reuse this
TITLE = sys.argv[2] if len(sys.argv) > 2 else "A level body on the stairs (step 50)"
S49 = HERE.parent / "29_long_sole"
REFS = [Path(p) for p in sys.argv[3:]]  # dogs of other steps to compare with (step 53's down teacher)
DOGS = ["teach_s1"] + [d.name for d in REFS] + sorted(d.name for d in HERE.iterdir() if (d / "result.json").exists() and d.name != "smoke")


def read(name: str) -> list[dict]:
    rows = []
    for folder in (S49, *[d.parent for d in REFS], HERE):
        p = folder / name
        rows += [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return rows


def last(name: str, **match) -> dict:
    return {Path(r["run"]).name: r for r in read(name) if all(r.get(k) == v for k, v in match.items())}


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


lunge = last("lunge_capped.jsonl", mode="stairs")
md = [f"# {TITLE}", "", "teach_s1 = step 49's dog, the others trained on from it. Lunge test against the capped plans, on the stairs that have one.", "",
      "## Stairs up, mean over the stairs", "", "| dog | climbed | nose up deg | rear paw up: body ahead cm | front paw up: body ahead cm | front paw up: body over rear paws cm | 3 paws down | stuck per climb |", "|---|---|---|---|---|---|---|---|"]
for n in DOGS:
    if n in lunge:
        c = lunge[n]["cells"]
        md.append(f"| {n} | {f(mean([x['climbed'] for x in c]))} | {f(mean([x['pitch_deg'] for x in c]), 1)} | {f(mean([x['shift_rear_cm'] for x in c]), 1)} | "
                  f"{f(mean([x['shift_front_cm'] for x in c]), 1)} | {f(mean([x.get('rear_height_cm') for x in c]), 1)} | {f(mean([x['three_down'] for x in c]))} | {f(mean([x['stuck_per_climb'] for x in c]))} |")
md += ["", "## Per stairs: climbed / nose up deg / rear paw up, body ahead cm", "", "| rise / tread | plan's shift | " + " | ".join(n for n in DOGS if n in lunge) + " |", "|---|---|" + "---|" * sum(n in lunge for n in DOGS)]
if lunge:
    first = next(iter(lunge.values()))
    for k, cell in enumerate(first["cells"]):
        row = []
        for n in DOGS:
            if n in lunge:
                c = lunge[n]["cells"][k]
                row.append(f"{f(c['climbed'])} / {f(c['pitch_deg'], 0)} / {f(c['shift_rear_cm'], 1)}")
        ref = cell["plan"].get("shift_rear_cm")
        md.append(f"| {cell['rise_cm']:.0f} / {cell['tread_cm']:.0f} | {f(ref, 1)} | " + " | ".join(row) + " |")
sg = last("stairs_goal.jsonl")
md += ["", "## Stairs goal test (clean = reached without any fall, open sides)", "", "| dog | up | down | limits up (treads 25-9) |", "|---|---|---|---|"]
for n in DOGS:
    if n in sg:
        cells = [c for c in sg[n]["cells"] if c["walls"] is None]
        lim = sg[n]["limits_rise_cm"]
        md.append(f"| {n} | {mean([c['clean'] for c in cells if c['dir'] == 'up']):.3f} | {mean([c['clean'] for c in cells if c['dir'] == 'down']):.3f} | "
                  + " / ".join(str(int(lim.get(f"up_t{t}", 0))) for t in (25, 20, 15, 12, 9)) + " |")
md += ["", "## Going down, per stairs: reached / any fall / steps made of 5 (refusing = few steps, no falls)", "", "| rise / tread | " + " | ".join(n for n in DOGS if n in sg) + " |", "|---|" + "---|" * sum(n in sg for n in DOGS)]
if sg:
    keys = [(c["rise_cm"], c["tread_cm"]) for c in next(iter(sg.values()))["cells"] if c["dir"] == "down" and c["walls"] is None and 2 <= c["rise_cm"] <= 5]
    for rise, tread in keys:
        row = []
        for n in DOGS:
            if n in sg:
                c = next(x for x in sg[n]["cells"] if x["dir"] == "down" and x["walls"] is None and x["rise_cm"] == rise and x["tread_cm"] == tread)
                row.append(f"{c['reached']:.2f} / {c['any_fall']:.2f} / {c['progress']:.1f}")
        md.append(f"| {rise:.0f} / {tread:.0f} | " + " | ".join(row) + " |")
bench = {(Path(r["run"]).name, r["scenario"]): r for r in read("benchmark.jsonl")}
gu = last("getup_test.jsonl")
md += ["", "## Nothing forgotten?", "", "| dog | forward falls (score) | course falls (score) | cost of transport | get-up worst |", "|---|---|---|---|---|"]
for n in DOGS:
    if (n, "forward") in bench:
        cells = [f"{100 * bench[(n, s)]['fall_rate']:.1f} % ({bench[(n, s)]['walk_score']:.2f})" if (n, s) in bench else "-" for s in ("forward", "course")]
        worst = min(gu[n][c]["success"] for c in ("standing", "dropped upright", "on its side", "on its back")) if n in gu else None
        md.append(f"| {n} | {cells[0]} | {cells[1]} | {f(bench[(n, 'forward')].get('cost_of_transport'))} | {f(worst)} |")
(HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
