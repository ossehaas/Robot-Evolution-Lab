"""Tables of step 40 -> results.md: the climbing-limit tables (stairs goal test) of dog v1, the original walker and the
trained dogs, the limits and sweet spots, and the checks that nothing was forgotten (walking, shoves, getting up),
with dog v1's step 39 results for comparison."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "17_recover_bc"
DOG_V1 = "bc_s2"
TREAD = 15  # the sweet spot is given for this tread


def rows(name: str, dirs=(PREV, HERE)) -> list[dict]:
    out = []
    for d in dirs:
        p = d / name
        if p.exists():
            out += [json.loads(line) for line in p.read_text().splitlines()]
    return out


def latest(lines: list[dict], key=lambda r: Path(r["run"]).name) -> dict:
    out = {}
    for r in lines:
        out[key(r)] = r
    return out


def sweet_spot(line: dict, d: str, tread: int) -> int:
    col = {int(c["rise_cm"]): c for c in line["cells"] if c["dir"] == d and c["walls"] is None and int(c["tread_cm"]) == tread}
    base, best = (col.get(1) or {}).get("time_s"), 0
    for r in sorted(col):
        c = col[r]
        if c["clean"] < 0.98 or c["time_s"] is None or base is None or c["time_s"] > 1.3 * base:
            break
        best = r
    return best


def label(name: str) -> str:
    return {DOG_V1: "dog v1 (zero-shot)", "amp_w6m_slow_s1": "original walker (zero-shot, cannot get up)"}.get(name, name)


lines = ["# Stairs as a goal task: the climbing limit (step 40)", "",
         "Every dog walks a straight staircase of 5 risers up (floor -> landing) or down (landing -> floor), driven by",
         "the navigation layer at 20 cm/s. clean = reached the goal without any fall. Falls on the stairs are soft (get",
         "up and go on); falling down the stairs (> 1 step lower while fallen, or off an open side) is a hard fail.", ""]
sg = latest(rows("stairs_goal.jsonl", (HERE,)))
if sg:
    some = next(iter(sg.values()))
    treads = sorted({int(c["tread_cm"]) for c in some["cells"]}, reverse=True)
    rises = sorted({int(c["rise_cm"]) for c in some["cells"]})
    z = some["max_problems"]
    lines += [f"## Limits: highest rise [cm] done clean by >= {100 * (1 - z):.0f} % (every lower rise too)", "",
              "| dog | " + " | ".join(f"up t{t}" for t in treads) + " | " + " | ".join(f"down t{t}" for t in treads)
              + f" | sweet spot t{TREAD} up / down |", "|---|" + "---|" * (2 * len(treads) + 1)]
    for name, line in sg.items():
        lim = line["limits_rise_cm"]
        lines.append(f"| {label(name)} | " + " | ".join(str(lim.get(f"up_t{t}", "-")) for t in treads) + " | "
                     + " | ".join(str(lim.get(f"down_t{t}", "-")) for t in treads)
                     + f" | {sweet_spot(line, 'up', TREAD)} / {sweet_spot(line, 'down', TREAD)} cm |")
    lines += ["", "Slope of each cell: atan(rise / tread), e.g. 6 cm on 15 cm = 22 deg, 8 cm on 12 cm = 34 deg, "
              "10 cm on 9 cm = 48 deg.", ""]
    for name, line in sg.items():
        lines += [f"## {label(name)}", ""]
        for d in ("up", "down"):
            cells = {(int(c["rise_cm"]), int(c["tread_cm"])): c for c in line["cells"] if c["dir"] == d and c["walls"] is None}
            lines += [f"{d}: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]", "",
                      "| rise | " + " | ".join(f"{t} cm" for t in treads) + " |", "|---|" + "---|" * len(treads)]
            for r in rises:
                vals = []
                for t in treads:
                    c = cells.get((r, t))
                    vals.append("-" if c is None else f"{100 * c['clean']:.0f}" + (f" ({c['time_s']})" if c["time_s"] else ""))
                lines.append(f"| {r} cm | " + " | ".join(vals) + " |")
            # what went wrong, per rise over all treads
            lines += ["", f"{d}, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps "
                      "made good (of 5); open sides vs walls: clean", "",
                      "| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |",
                      "|---|---|---|---|---|---|---|"]
            for r in rises:
                sel = [c for c in line["cells"] if c["dir"] == d and int(c["rise_cm"]) == r]
                both = [c for c in sel if c["walls"] is None]
                n = sum(c["n"] for c in both)

                def mean(key, cs=both):
                    m = sum(c["n"] for c in cs)
                    return sum(c[key] * c["n"] for c in cs) / max(1, m)

                open_, walls = [c for c in sel if c["walls"] is False], [c for c in sel if c["walls"] is True]
                lines.append(f"| {r} cm | {100 * mean('reached'):.0f} % | {100 * mean('any_fall'):.0f} % | "
                             f"{100 * mean('hard'):.0f} % | {100 * mean('stayed_down'):.0f} % | {mean('progress'):.1f} | "
                             f"{100 * mean('clean', open_):.0f} / {100 * mean('clean', walls):.0f} % |")
            lines.append("")

bench = {(Path(r["run"]).name, r["scenario"]): r for r in rows("benchmark.jsonl")}
names = [n for n in dict.fromkeys(k[0] for k in bench) if n in (DOG_V1,) or (HERE / n).exists()]
if names:
    lines += ["## Nothing forgotten? Walking (falls, walk score; steps climbed on the hardest pyramid stairs)", "",
              "| dog | forward | course | stairs straight | steps up (rows 6-9) | power forward |", "|---|---|---|---|---|---|"]
    for name in names:
        cells = []
        for sc in ("forward", "course", "stairs_straight"):
            b = bench.get((name, sc))
            cells.append(f"{100 * b['fall_rate']:.1f} % ({b['walk_score']:.2f})" if b else "-")
        climb = (bench.get((name, "stairs_straight")) or {}).get("type_level_climb_m", {}).get("stairs_climb", [])
        steps = [climb[lv] / (0.005 + 0.055 * lv / 9) for lv in range(6, len(climb))]
        cells.append(f"{sum(steps) / len(steps):.1f}" if steps else "-")
        fw = bench.get((name, "forward"))
        cells.append(f"{fw['mechanical_power_w']:.1f} W" if fw and "mechanical_power_w" in fw else "-")
        lines.append(f"| {label(name) if name == DOG_V1 else name} | " + " | ".join(cells) + " |")
rec = [r for r in rows("recover_test.jsonl") if Path(r["run"]).name == DOG_V1 or Path(r["run"]).parent == HERE]
if rec:
    strengths = list(rec[0]["by_shove_m_s"])
    lines += ["", "## One shove while walking at 20 cm/s: stays up / falls and gets up / stays down", "",
              "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for r in latest(rec).values():
        lines.append(f"| {Path(r['run']).name} | " + " | ".join(
            f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / {100 * v['stayed_down']:.0f} %"
            for v in r["by_shove_m_s"].values()) + " |")
gu = [r for r in rows("getup_test.jsonl") if Path(r["run"]).name == DOG_V1 or Path(r["run"]).parent == HERE]
if gu:
    cats = ("standing", "dropped upright", "on its side", "on its back")
    lines += ["", "## Getting up: share standing within 8 s (median time)", "", "| dog | " + " | ".join(cats) + " |",
              "|---|" + "---|" * len(cats)]
    for r in latest(gu).values():
        lines.append(f"| {Path(r['run']).name} | " + " | ".join(
            f"{100 * r[c]['success']:.0f} % ({r[c]['median_s']} s)" if r[c]["median_s"] is not None
            else f"{100 * r[c]['success']:.0f} %" for c in cats) + " |")
if (HERE / "best.txt").exists():
    lines += ["", f"Best dog on the stairs: {(HERE / 'best.txt').read_text().strip()}"]
vids = sorted((HERE / "videos").glob("*.mp4")) if (HERE / "videos").exists() else []
if vids:
    lines += ["", "Videos (dog v1 / best anchor / best free, tread 15 cm, open sides): "
              + ", ".join(f"`videos/{v.name}`" for v in vids)]
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
