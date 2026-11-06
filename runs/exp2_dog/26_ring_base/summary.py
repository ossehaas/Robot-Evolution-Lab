"""Tables of step 47 -> results.md: hold_s2 retrained on the ring paw, against step 46's hold_s2 with the sphere paw
and with the ring paw untrained."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S46 = HERE.parent / "25_hard_paw"
REFS = ["hold_s2_tpu", "hold_s2_ring"]
DOGS = REFS + ["ring_s1", "ring_s2"]


def lines(path: Path) -> list[dict]:
    """This step's rows of a result file plus step 46's rows of the reference dogs."""
    def read(p):
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return [r for r in read(S46 / path.name) if Path(r["run"]).name in REFS] + read(path)


def last(path: Path, **match) -> dict:
    out = {}
    for r in lines(path):
        if all(r.get(k) == v for k, v in match.items()):
            out[Path(r["run"]).name] = r
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def f(x, nd=2, unit=""):
    return "-" if x is None else f"{x:.{nd}f}{unit}"


md = ["# The base on the ring paw (step 47)", "",
      "hold_s2_tpu = hold_s2 on the old sphere paw, hold_s2_ring = the same policy on the ring paw (step 46); ring_s1 / "
      "ring_s2 = hold_s2 trained on with the ring paw (3000 it). Stumble = share of paw-steps on the way up with a paw pushed "
      "sideways more than twice as hard as it is pressed down (a toe against a riser); stuck = such a stumble lasting "
      "more than 0.3 s, per climb.", ""]
lunge = last(HERE / "lunge.jsonl", mode="stairs")
md += ["## Stairs up (lunge test, rises 1-5 cm, all treads)", "",
       "| dog | climbed | stumble | stuck per climb | shift front cm | shift rear cm | nose up |", "|---|---|---|---|---|---|---|"]
for n in DOGS:
    if n in lunge:
        c = lunge[n]["cells"]
        md.append(f"| {n} | {f(mean([x['climbed'] for x in c]))} | {f(mean([x['stumble'] for x in c]), 4)} | "
                  f"{f(mean([x['stuck_per_climb'] for x in c]))} | {f(mean([x['shift_front_cm'] for x in c]), 1)} | "
                  f"{f(mean([x['shift_rear_cm'] for x in c]), 1)} | {f(mean([x['pitch_deg'] for x in c]), 0, ' deg')} |")
sg = last(HERE / "stairs_goal.jsonl")
md += ["", "## Stairs goal test (clean = reached without any fall, open sides)", "",
       "| dog | mean clean | up | down | limits up (treads 25-9) | limits down |", "|---|---|---|---|---|---|"]
for n in DOGS:
    if n in sg:
        cells = [c for c in sg[n]["cells"] if c["walls"] is None]
        up = [c["clean"] for c in cells if c["dir"] == "up"]
        down = [c["clean"] for c in cells if c["dir"] == "down"]
        lim = sg[n]["limits_rise_cm"]
        md.append(f"| {n} | {sum(c['clean'] for c in cells) / len(cells):.3f} | {sum(up) / len(up):.3f} | "
                  f"{sum(down) / len(down):.3f} | " + " / ".join(str(int(lim.get(f"up_t{t}", 0))) for t in (25, 20, 15, 12, 9))
                  + " | " + " / ".join(str(int(lim.get(f"down_t{t}", 0))) for t in (25, 20, 15, 12, 9)) + " |")
bench = {(Path(r["run"]).name, r["scenario"]): r for r in lines(HERE / "benchmark.jsonl")}
md += ["", "## Walking, getting up, shoves, standing on tilted plates", "",
       "| dog | forward falls (score) | course falls (score) | power |", "|---|---|---|---|"]
for n in DOGS:
    if (n, "forward") in bench:
        cells = [f"{100 * bench[(n, sc)]['fall_rate']:.1f} % ({bench[(n, sc)]['walk_score']:.2f})" if (n, sc) in bench
                 else "-" for sc in ("forward", "course")]
        md.append(f"| {n} | " + " | ".join(cells) + f" | {bench[(n, 'forward')].get('mechanical_power_w', float('nan')):.1f} W |")
gu = last(HERE / "getup_test.jsonl")
cats = ("standing", "dropped upright", "on its side", "on its back")
md += ["", "Getting up within 8 s:", "", "| dog | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
for n in DOGS:
    if n in gu:
        md.append(f"| {n} | " + " | ".join(f"{100 * gu[n][c]['success']:.0f} %" for c in cats) + " |")
rec = last(HERE / "recover_test.jsonl")
rn = [n for n in DOGS if n in rec]
if rn:
    strengths = list(rec[rn[0]]["by_shove_m_s"])
    md += ["", "One shove while walking (stays up / gets up / stays down):", "",
           "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for n in rn:
        md.append(f"| {n} | " + " | ".join(f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / "
                                          f"{100 * v['stayed_down']:.0f} %" for v in rec[n]["by_shove_m_s"].values()) + " |")
tilt = {(Path(r["run"]).name, bool(r.get("shove_m_s"))): r for r in lines(HERE / "tilt.jsonl") if not r["hold"]}
md += ["", "Standing on tilted plates (calm / shoved 0.3 m/s): still % up to 15 deg, falls up to 15 deg, falls at 20 deg:",
       "", "| dog | calm still | calm falls <= 15 | calm falls 20 | shoved falls <= 15 | shoved falls 20 |",
       "|---|---|---|---|---|---|"]
for n in DOGS:
    row = []
    for shoved in (False, True):
        r = tilt.get((n, shoved))
        if r is None:
            row += ["-"] * (3 if not shoved else 2)
            continue
        lo = [x for x in r["cells"] if x["angle_deg"] <= 15]
        at20 = [x for x in r["cells"] if x["angle_deg"] == 20]
        if not shoved:
            row.append(f"{100 * sum(x['still'] or 0 for x in lo) / len(lo):.0f} %")
        row += [f"{100 * sum(x['fell'] for x in lo) / len(lo):.0f} %", f"{100 * sum(x['fell'] for x in at20) / max(len(at20), 1):.0f} %"]
    md.append(f"| {n} | " + " | ".join(row) + " |")
if (HERE / "best.txt").exists():
    md += ["", f"Best and pass mark for step 45b: {(HERE / 'best.txt').read_text().strip()}"]
vids = sorted((HERE / "videos").glob("*.mp4")) if (HERE / "videos").exists() else []
if vids:
    md += ["", "Videos: " + ", ".join(f"`videos/{v.name}`" for v in vids)]
(HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
