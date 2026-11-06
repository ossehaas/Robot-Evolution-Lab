"""Tables of step 43 -> results.md: base skills B1 (stand still) and B2 (tilt, shove) per dog with pass / fail against
.documentation/base-skills.md, the tilt tables, and what else the dogs can still do (stairs, walking, shoves, getting
up)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S40, S42 = HERE.parent / "18_stairs_goal", HERE.parent / "20_balance_and_style"
NAMES = ("free_s2", "amp_w6m_slow_s1", "still_s1", "still_s2", "tilt_s1", "tilt_s2")
DIRS = {0.0: "nose down", 45.0: "diagonal", 90.0: "sideways", 180.0: "nose up"}
# B2: up to 80 % of the standing pose's tilt limit (A2: 20 deg nose down, 28 nose up / sideways), in 5 deg steps
B2_MAX = {0.0: 15.0, 45.0: 15.0, 90.0: 20.0, 180.0: 20.0}


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def tilt_rows() -> dict:
    out = {}
    for r in lines(HERE / "tilt.jsonl"):
        key = Path(r["run"]).name + (" (servos holding the pose)" if r["hold"] else "")
        out[(key, r.get("shove_m_s", 0.0))] = r
    return out


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f} %"


md = ["# Base skills B1 + B2: stand still, stay balanced on a tilt (step 43)", ""]
tilt = tilt_rows()
dogs = [k for k in dict.fromkeys(k for k, _ in tilt)]

# pass / fail
md += ["## Pass / fail (`.documentation/base-skills.md`)", "",
       "B1 on the flat: still >= 90 % of the time after 2 s, still within 2 s, moves < 2 cm in 10 s. "
       "B2 on plates up to 15 deg (nose down, diagonal) / 20 deg (sideways, nose up): never falls, still >= 80 %; "
       "after a 0.3 m/s shove no fall and still again within 3 s.", "",
       "| dog | B1 still (flat) | settle | moved | power | B1 | B2 worst still | B2 falls | after shove: falls / resettle | B2 |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for name in dogs:
    calm, shoved = tilt.get((name, 0.0)), tilt.get((name, 0.3))
    if calm is None:
        continue
    flat = [c for c in calm["cells"] if c["angle_deg"] == 0.0]
    still = sum(c["still"] or 0.0 for c in flat) / len(flat)
    settle = max((c["settle_s"] if c["settle_s"] is not None else 99.0) for c in flat)
    moved = max(c["moved_cm"] or 0.0 for c in flat)
    power = sum(c["power_w"] or 0.0 for c in flat) / len(flat)
    b1 = still >= 0.9 and settle <= 2.0 and moved < 2.0 and not any(c["fell"] for c in flat)
    b2_cells = [c for c in calm["cells"] if c["angle_deg"] <= B2_MAX[c["direction_deg"]]]
    worst_still = min((c["still"] or 0.0) for c in b2_cells)
    falls = max(c["fell"] for c in b2_cells)
    b2 = falls == 0 and worst_still >= 0.8
    shove_txt = "-"
    if shoved is not None:
        s_cells = [c for c in shoved["cells"] if c["angle_deg"] <= B2_MAX[c["direction_deg"]]]
        s_fall = max((c.get("fell_after") or 0.0) for c in s_cells)
        s_settle = max((c.get("resettle_s") if c.get("resettle_s") is not None else 99.0) for c in s_cells)
        shove_txt = f"{pct(s_fall)} / {'never' if s_settle >= 99 else f'{s_settle:.1f} s'}"
        b2 = b2 and s_fall == 0 and s_settle <= 3.0
    md.append(f"| {name} | {pct(still)} | {'never' if settle >= 99 else f'{settle:.1f} s'} | {moved:.1f} cm | "
              f"{power:.2f} W | {'pass' if b1 else 'FAIL'} | {pct(worst_still)} | {pct(falls)} | {shove_txt} | "
              f"{'pass' if b2 else 'FAIL'} |")
md.append("")

# tilt tables
for (name, shove), r in tilt.items():
    dirs = sorted({c["direction_deg"] for c in r["cells"]})
    title = f"{name}, {'shoved 0.3 m/s at 6 s' if shove else 'standing 12 s'}"
    md += [f"### {title}: fell % / still % (settle s)" + (" / after the shove: fell %, resettle s" if shove else ""), "",
           "| tilt | " + " | ".join(DIRS.get(d, str(d)) for d in dirs) + " |", "|---|" + "---|" * len(dirs)]
    for a in sorted({c["angle_deg"] for c in r["cells"]}):
        row = {c["direction_deg"]: c for c in r["cells"] if c["angle_deg"] == a}

        def cell(c):
            txt = f"{100 * c['fell']:.0f} / {pct(c['still'])} ({'-' if c['settle_s'] is None else c['settle_s']})"
            if shove:
                txt += f" / {pct(c.get('fell_after'))}, {'-' if c.get('resettle_s') is None else c['resettle_s']}"
            return txt

        md.append(f"| {a:.0f} deg | " + " | ".join(cell(row[d]) for d in dirs) + " |")
    md.append("")

# nothing forgotten
sg = {Path(r["run"]).name: r for r in lines(S40 / "stairs_goal.jsonl") + lines(HERE / "stairs_goal.jsonl")}
trained = [n for n in NAMES if n in sg]
if trained:
    md += ["## Stairs goal test (clean = reached without any fall, open sides)", "",
           "| dog | mean clean | up | down | limits up (treads 25-9) | limits down |", "|---|---|---|---|---|---|"]
    for n in trained:
        cells = [c for c in sg[n]["cells"] if c["walls"] is None]
        up = [c["clean"] for c in cells if c["dir"] == "up"]
        down = [c["clean"] for c in cells if c["dir"] == "down"]
        lim = sg[n]["limits_rise_cm"]
        md.append(f"| {n} | {sum(c['clean'] for c in cells) / len(cells):.3f} | {sum(up) / len(up):.3f} | "
                  f"{sum(down) / len(down):.3f} | " + " / ".join(str(int(lim.get(f"up_t{t}", 0))) for t in (25, 20, 15, 12, 9))
                  + " | " + " / ".join(str(int(lim.get(f"down_t{t}", 0))) for t in (25, 20, 15, 12, 9)) + " |")
    md.append("")
bench = {(Path(r["run"]).name, r["scenario"]): r for r in lines(S40 / "benchmark.jsonl") + lines(HERE / "benchmark.jsonl")}
bn = [n for n in NAMES if (n, "forward") in bench]
if bn:
    md += ["## Walking (falls, walk score)", "", "| dog | forward | course | stairs straight | power |", "|---|---|---|---|---|"]
    for n in bn:
        cells = [f"{100 * bench[(n, sc)]['fall_rate']:.1f} % ({bench[(n, sc)]['walk_score']:.2f})" if (n, sc) in bench else "-"
                 for sc in ("forward", "course", "stairs_straight")]
        md.append(f"| {n} | " + " | ".join(cells) + f" | {bench[(n, 'forward')].get('mechanical_power_w', float('nan')):.1f} W |")
rec = {Path(r["run"]).name: r for r in lines(S40 / "recover_test.jsonl") + lines(HERE / "recover_test.jsonl")}
gu = {Path(r["run"]).name: r for r in lines(S40 / "getup_test.jsonl") + lines(HERE / "getup_test.jsonl")}
rn = [n for n in NAMES if n in rec]
if rn:
    strengths = list(rec[rn[0]]["by_shove_m_s"])
    md += ["", "One shove while walking (stays up / gets up / stays down):", "",
           "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for n in rn:
        md.append(f"| {n} | " + " | ".join(f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / {100 * v['stayed_down']:.0f} %"
                                          for v in rec[n]["by_shove_m_s"].values()) + " |")
gn = [n for n in NAMES if n in gu]
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
