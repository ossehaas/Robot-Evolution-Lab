"""Tables of step 42 -> results.md: why descents are refused (reward terms per phase), the tilting plateau (standing
limits per direction, standing still), the stricter stair plans in physics, and the trainings (desc, margin) against
the start dog free_s2 and step 41's dogs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S40, S41 = HERE.parent / "18_stairs_goal", HERE.parent / "19_stairs_proof"
TREADS = (25, 20, 15, 12, 9)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def latest(rows: list[dict], key=lambda r: Path(r["run"]).name) -> dict:
    out = {}
    for r in rows:
        out[key(r)] = r
    return out


md = ["# Balance first, then the descent (step 42)", ""]

# 1 descent terms
dt = latest(lines(HERE / "descent_terms.jsonl"))
if dt:
    md += ["## Why the dogs refuse to go down (reward terms per second, per phase of the first descent)", ""]
    for name, r in dt.items():
        md += [f"{name}: reached the bottom " + ", ".join(f"{c} {100 * x:.0f} %" for c, x in zip(r["cells"], r["reached"])), "",
               "| term | landing | edge | stairs | refusing |", "|---|---|---|---|---|"]
        terms = list(r["phases"]["landing"]["terms"])
        # the terms that differ most between walking on the landing and hesitating at the edge first
        terms.sort(key=lambda k: -abs(r["phases"]["edge"]["terms"][k] - r["phases"]["landing"]["terms"][k]))
        for k in terms:
            md.append(f"| {k} | " + " | ".join(f"{r['phases'][p]['terms'][k]:+.2f}" for p in ("landing", "edge", "stairs", "refusing")) + " |")
        md.append("| seconds | " + " | ".join(f"{r['phases'][p]['seconds']}" for p in ("landing", "edge", "stairs", "refusing")) + " |")
        md.append("")

# 2 tilt
tilt = latest(lines(HERE / "tilt.jsonl"), key=lambda r: Path(r["run"]).name + (" (servos holding the pose)" if r["hold"] else ""))
if tilt:
    names = {0.0: "nose down", 45.0: "diagonal", 90.0: "sideways", 180.0: "nose up"}
    md += ["## The tilting plateau: share standing for 12 s (fell / slid), per tilt and downhill direction", "",
           "On paper (tools/dog/tilt_limits.py): the standing pose tips at 20 deg facing downhill, 28 nose up, 29 sideways; "
           "with the best posture (body back / crouched) beyond 45 deg, where the paws slide (mu 1.0).", ""]
    for name, r in tilt.items():
        dirs = sorted({c["direction_deg"] for c in r["cells"]})
        md += [f"{name}:", "", "| tilt | " + " | ".join(names.get(d, str(d)) for d in dirs) + " |", "|---|" + "---|" * len(dirs)]
        for a in sorted({c["angle_deg"] for c in r["cells"]}):
            row = {c["direction_deg"]: c for c in r["cells"] if c["angle_deg"] == a}
            md.append(f"| {a:.0f} deg | " + " | ".join(f"{100 * row[d]['stood']:.0f} % ({100 * row[d]['fell']:.0f}/{100 * row[d]['slid']:.0f})" for d in dirs) + " |")
        flat = [c for c in r["cells"] if c["angle_deg"] == 0.0 and c["power_w"] is not None]
        if flat:
            md.append(f"\nStanding still on the flat: {sum(c['power_w'] for c in flat) / len(flat):.2f} W, summed joint speed "
                      f"{sum(c['joint_rad_s'] for c in flat) / len(flat):.3f} rad/s")
        md.append("")

# 3 replays of the stricter plans
for fname, title in (("replay.jsonl", "nominal"), ("replay_randomized.jsonl", "randomized (4 dogs per plan)")):
    rows = lines(HERE / fname)
    if rows:
        ok = [r for r in rows if r["climbed"] > 0]
        md += [f"## Stair plans with a 3 cm balance margin, replayed open loop ({title})", "",
               f"{len(ok)} of {len(rows)} plans climbed by at least one dog: " + ", ".join(
                   f"{r['plan'][:-5]} ({100 * r['climbed']:.0f} %)" for r in sorted(ok, key=lambda r: r["plan"])), ""]

# 4 training
sg = latest(lines(S40 / "stairs_goal.jsonl") + lines(S41 / "stairs_goal.jsonl") + lines(HERE / "stairs_goal.jsonl"))
names = [n for n in ("free_s2", "cont_s1", "rsi_s2", "desc_s1", "desc_s2", "margin_s1", "margin_s2") if n in sg]
possible = set()
for folder in ("plans", "plans_retry"):
    for f in (S41 / folder).glob("*_r*_t*.json"):
        p = json.loads(f.read_text())
        if p.get("feasible"):
            possible.add((p["dir"], round(p["rise"] * 100), round(p["tread"] * 100)))
if names:
    md += ["## Stairs goal test (clean = reached without any fall)", "",
           "| dog | up, possible | down, possible | no plan (up / down) | limits up (treads 25-9) | limits down | mean clean |",
           "|---|---|---|---|---|---|---|"]
    for n in names:
        cells = [c for c in sg[n]["cells"] if c["walls"] is None]

        def share(d, want):
            cs = [c for c in cells if c["dir"] == d and ((d, int(c["rise_cm"]), int(c["tread_cm"])) in possible) == want]
            return sum(c["clean"] for c in cs) / max(1, len(cs))

        lim = sg[n]["limits_rise_cm"]
        md.append(f"| {n} | {100 * share('up', True):.0f} % | {100 * share('down', True):.0f} % | "
                  f"{100 * share('up', False):.0f} / {100 * share('down', False):.0f} % | "
                  + " / ".join(str(int(lim.get(f"up_t{t}", 0))) for t in TREADS) + " | "
                  + " / ".join(str(int(lim.get(f"down_t{t}", 0))) for t in TREADS)
                  + f" | {sum(c['clean'] for c in cells) / len(cells):.3f} |")
    md += ["", "Descents per cell (possible ones), clean %:", ""]
    down = sorted((k for k in possible if k[0] == "down"), key=lambda k: (k[2], k[1]))
    md += ["| dog | " + " | ".join(f"r{r} t{t}" for _, r, t in down) + " |", "|---|" + "---|" * len(down)]
    for n in names:
        cs = {(int(c["rise_cm"]), int(c["tread_cm"])): c for c in sg[n]["cells"] if c["walls"] is None and c["dir"] == "down"}
        md.append(f"| {n} | " + " | ".join(f"{100 * cs[(r, t)]['clean']:.0f}" if (r, t) in cs else "-" for _, r, t in down) + " |")
    md.append("")

bench = {(Path(r["run"]).name, r["scenario"]): r for r in lines(S40 / "benchmark.jsonl") + lines(S41 / "benchmark.jsonl")
         + lines(HERE / "benchmark.jsonl")}
bn = [n for n in names if (n, "forward") in bench]
if bn:
    md += ["## Nothing forgotten? Walking (falls, walk score)", "", "| dog | forward | course | stairs straight | power |", "|---|---|---|---|---|"]
    for n in bn:
        cells = [f"{100 * bench[(n, sc)]['fall_rate']:.1f} % ({bench[(n, sc)]['walk_score']:.2f})" if (n, sc) in bench else "-"
                 for sc in ("forward", "course", "stairs_straight")]
        md.append(f"| {n} | " + " | ".join(cells) + f" | {bench[(n, 'forward')].get('mechanical_power_w', float('nan')):.1f} W |")
rec = latest(lines(S40 / "recover_test.jsonl") + lines(S41 / "recover_test.jsonl") + lines(HERE / "recover_test.jsonl"))
gu = latest(lines(S40 / "getup_test.jsonl") + lines(S41 / "getup_test.jsonl") + lines(HERE / "getup_test.jsonl"))
rn = [n for n in names if n in rec]
if rn:
    strengths = list(rec[rn[0]]["by_shove_m_s"])
    md += ["", "One shove while walking (stays up / gets up / stays down):", "",
           "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for n in rn:
        md.append(f"| {n} | " + " | ".join(f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / {100 * v['stayed_down']:.0f} %"
                                          for v in rec[n]["by_shove_m_s"].values()) + " |")
gn = [n for n in names if n in gu]
if gn:
    cats = ("standing", "dropped upright", "on its side", "on its back")
    md += ["", "Getting up within 8 s:", "", "| dog | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for n in gn:
        md.append(f"| {n} | " + " | ".join(f"{100 * gu[n][c]['success']:.0f} %" for c in cats) + " |")
if (HERE / "best.txt").exists():
    md += ["", f"Best per variant: {(HERE / 'best.txt').read_text().strip()}"]
vids = sorted((HERE / "videos").glob("*.mp4")) if (HERE / "videos").exists() else []
if vids:
    md += ["", "Videos: " + ", ".join(f"`videos/{v.name}`" for v in vids)]
(HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
