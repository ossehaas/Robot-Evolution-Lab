"""Tables of step 45b -> results.md: the stairs-up expert on the frozen ring-paw base, quiet loss off, with and
without the copy loss. Proof of concept when (next-steps 0b):
the base's skills are unchanged (walking, shoves, getting up, standing), the expert climbs better than the base alone
with the plans' weight shift, and the gate is open in the zone and shut before it, after it and on the course."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
base_name = (HERE / "base.txt").read_text().split()[0] if (HERE / "base.txt").exists() else "?"
S44 = HERE.parent / "26_ring_base"  # the base's own test results (step 47)
NAMES = (base_name, "copy_s1", "copy_s2", "noquiet_s1", "noquiet_s2")


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def last_by_name(*paths: Path, **match) -> dict:
    out = {}
    for p in paths:
        for r in lines(p):
            if all(r.get(k) == v for k, v in match.items()):
                out[Path(r["run"]).name] = r
    return out


def f(x, nd=1, unit=""):
    return "-" if x is None else f"{x:.{nd}f}{unit}"


md = [f"# One model, many skills: the stairs-up expert taught with the plans, on the frozen base {base_name} (step 45b)", ""]
lunge = last_by_name(HERE / "lunge.jsonl", mode="stairs")
course = last_by_name(HERE / "lunge.jsonl", mode="course")
names = [n for n in NAMES if n in lunge]

# per dog: mean over the cells
md += ["## Weight shift, crawl and gate (lunge test, stairs up, cells with a plan)", "",
       "Shift = body x minus the paws' centre while exactly one paw is up (plan: behind the paws with a front paw up, "
       "ahead with a rear paw up). Gate: before the zone (floor), in it, after it (landing / the way down), and the "
       "share of steps with the gate open on the walking course.", "",
       "| dog | climbed | shift front cm (plan) | shift rear cm (plan) | 3 paws down | nose up | gate before / zone / after "
       "| gate open on course |", "|---|---|---|---|---|---|---|---|"]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


for n in names:
    cells = lunge[n]["cells"]
    gate = " / ".join(f(mean([c[k] for c in cells]), 2) for k in ("gate_before", "gate_zone", "gate_after"))
    co = course.get(n, {}).get("gate_open_share")
    md.append(f"| {n} | {f(mean([c['climbed'] for c in cells]), 2)} | "
              f"{f(mean([c['shift_front_cm'] for c in cells]))} ({f(mean([c['plan'].get('shift_front_cm') for c in cells]))}) | "
              f"{f(mean([c['shift_rear_cm'] for c in cells]))} ({f(mean([c['plan'].get('shift_rear_cm') for c in cells]))}) | "
              f"{f(mean([c['three_down'] for c in cells]), 2)} | {f(mean([c['pitch_deg'] for c in cells]), 0, ' deg')} | "
              f"{gate if lunge[n]['gate'] else '-'} | {'-' if co is None else f'{100 * co:.1f} %'} |")
md.append("")
for n in names:
    md += [f"### {n}: per stairs (rise / tread cm): climbed, shift front / rear cm (plan), 3 paws down, gate in the zone", "",
           "| rise | tread | climbed | shift front | shift rear | 3 down | gate zone |", "|---|---|---|---|---|---|---|"]
    for c in lunge[n]["cells"]:
        pl = c["plan"]
        md.append(f"| {c['rise_cm']:.0f} | {c['tread_cm']:.0f} | {f(c['climbed'], 2)} | {f(c['shift_front_cm'])} "
                  f"({f(pl.get('shift_front_cm'))}) | {f(c['shift_rear_cm'])} ({f(pl.get('shift_rear_cm'))}) | "
                  f"{f(c['three_down'], 2)} | {f(c['gate_zone'], 2)} |")
    md.append("")

# climbing-limit table
sg = last_by_name(HERE / "stairs_goal.jsonl")
sn = [n for n in NAMES if n in sg]
if sn:
    md += ["## Stairs goal test (clean = reached without any fall, open sides)", "",
           "| dog | mean clean | up | down | limits up (treads 25-9) | limits down |", "|---|---|---|---|---|---|"]
    for n in sn:
        cells = [c for c in sg[n]["cells"] if c["walls"] is None]
        up = [c["clean"] for c in cells if c["dir"] == "up"]
        down = [c["clean"] for c in cells if c["dir"] == "down"]
        lim = sg[n]["limits_rise_cm"]
        md.append(f"| {n} | {sum(c['clean'] for c in cells) / len(cells):.3f} | {sum(up) / len(up):.3f} | "
                  f"{sum(down) / len(down):.3f} | " + " / ".join(str(int(lim.get(f"up_t{t}", 0))) for t in (25, 20, 15, 12, 9))
                  + " | " + " / ".join(str(int(lim.get(f"down_t{t}", 0))) for t in (25, 20, 15, 12, 9)) + " |")
    md.append("")

# the base's skills, unchanged?
bench = {(Path(r["run"]).name, r["scenario"]): r for r in lines(S44 / "benchmark.jsonl") + lines(HERE / "benchmark.jsonl")}
bn = [n for n in NAMES if (n, "forward") in bench]
if bn:
    md += ["## The base's skills (must be unchanged)", "", "| dog | forward | course | power |", "|---|---|---|---|"]
    for n in bn:
        cells = [f"{100 * bench[(n, sc)]['fall_rate']:.1f} % ({bench[(n, sc)]['walk_score']:.2f})" if (n, sc) in bench else "-"
                 for sc in ("forward", "course")]
        md.append(f"| {n} | " + " | ".join(cells) + f" | {bench[(n, 'forward')].get('mechanical_power_w', float('nan')):.1f} W |")
rec = last_by_name(S44 / "recover_test.jsonl", HERE / "recover_test.jsonl")
rn = [n for n in NAMES if n in rec]
if rn:
    strengths = list(rec[rn[0]]["by_shove_m_s"])
    md += ["", "One shove while walking (stays up / gets up / stays down):", "",
           "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for n in rn:
        md.append(f"| {n} | " + " | ".join(f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / {100 * v['stayed_down']:.0f} %"
                                          for v in rec[n]["by_shove_m_s"].values()) + " |")
gu = last_by_name(S44 / "getup_test.jsonl", HERE / "getup_test.jsonl")
gn = [n for n in NAMES if n in gu]
if gn:
    cats = ("standing", "dropped upright", "on its side", "on its back")
    md += ["", "Getting up within 8 s:", "", "| dog | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for n in gn:
        md.append(f"| {n} | " + " | ".join(f"{100 * gu[n][c]['success']:.0f} %" for c in cats) + " |")
tilt = {Path(r["run"]).name: r for r in lines(S44 / "tilt.jsonl") + lines(HERE / "tilt.jsonl")
        if not r["hold"] and not r.get("set") and not r.get("shove_m_s")}
tn = [n for n in NAMES if n in tilt]
if tn:
    md += ["", "Standing (tilt test, calm, plates up to 15 deg): share of the time still, falls:", "",
           "| dog | still | fell |", "|---|---|---|"]
    for n in tn:
        c = [x for x in tilt[n]["cells"] if x["angle_deg"] <= 15]
        md.append(f"| {n} | {100 * sum(x['still'] or 0 for x in c) / len(c):.0f} % | "
                  f"{100 * sum(x['fell'] for x in c) / len(c):.0f} % |")
if (HERE / "best.txt").exists():
    md += ["", f"Best per variant (clean share up, treads 15 / 9 cm): {(HERE / 'best.txt').read_text().strip()}"]
vids = sorted((HERE / "videos").glob("*.mp4")) if (HERE / "videos").exists() else []
if vids:
    md += ["", "Videos: " + ", ".join(f"`videos/{v.name}`" for v in vids)]
(HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
