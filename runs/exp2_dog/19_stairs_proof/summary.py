"""Tables of step 41 -> results.md: the proof (which stairs are physically possible, quasi-static plans), the proof in
physics (plan replays), and the training with the plans as trainer (mid-climb starts), against the start dog free_s2
(step 40) and the control without mid-climb starts."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "18_stairs_goal"
TREADS = (25, 20, 15, 12, 9)
RISES = range(1, 11)


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def plans(folder: str) -> dict:
    out = {}
    for f in (HERE / folder).glob("*_r*_t*.json"):
        p = json.loads(f.read_text())
        out[(p["dir"], round(p["rise"] * 100), round(p["tread"] * 100))] = p
    return out


main, retry, full = plans("plans"), plans("plans_retry"), plans("plans_fullrange")
md = ["# Physically possible, then learned (step 41)", "",
      "## The proof: is there a slow, statically stable crawl? (tools/dog/stairs_plan.py)", "",
      "x = a plan exists (proven possible: every pose checked for reach within 95 % of the joint range, collisions, "
      "balance >= 1.5 cm inside the support, torque <= 80 %); w = found only by the wider search; F = only with the full "
      "joint range and 5 mm swing clearance; . = no plan; - = not tried (a lower rise had no plan)", ""]
for d in ("up", "down"):
    md += [f"{d}:", "", "| rise | " + " | ".join(f"tread {t} cm" for t in TREADS) + " |", "|---|" + "---|" * len(TREADS)]
    for r in RISES:
        cells = []
        for t in TREADS:
            k = (d, r, t)
            if main.get(k, {}).get("feasible"):
                cells.append("x")
            elif retry.get(k, {}).get("feasible"):
                cells.append("w")
            elif full.get(k, {}).get("feasible"):
                cells.append("F")
            elif k in main or k in retry or k in full:
                cells.append(".")
            else:
                cells.append("-")
        if any(c != "-" for c in cells):
            md.append(f"| {r} cm | " + " | ".join(cells) + " |")
    md.append("")

for name, title in (("replay.jsonl", "nominal"), ("replay_randomized.jsonl", "randomized (4 dogs per plan)"),
                    ("replay_feedback.jsonl", "with feedback (body follows the standing paws)")):
    rows = lines(HERE / name)
    if not rows:
        continue
    md += [f"## The proof in physics: plan replay, {title}", "",
           "| plan | climbed | fell | largest height / pitch error | seconds |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["dir"], r["tread"], r["rise"])):
        md.append(f"| {r['plan'][:-5]} | {100 * r['climbed']:.0f} % | {100 * r['fell']:.0f} % | "
                  f"{100 * r['max_height_error_m']:.1f} cm / {r['max_pitch_error_deg']} deg | {r['seconds']} |")
    md.append("")

sg = {Path(r["run"]).name: r for r in lines(PREV / "stairs_goal.jsonl") + lines(HERE / "stairs_goal.jsonl")}
names = [n for n in ("bc_s2", "free_s2", "cont_s1", "rsi_s1", "rsi_s2", "rsi_terms_s1", "rsi_terms_s2", "desc_s1") if n in sg]
if names:
    md += ["## What the dogs do: stairs goal test (clean = reached without any fall, 40 runs per cell)", "",
           "Limits: highest rise [cm] up to which every rise is done clean by >= 90 %; mean clean share over all cells.", "",
           "| dog | up, treads " + " / ".join(map(str, TREADS)) + " | down, treads " + " / ".join(map(str, TREADS)) +
           " | mean clean |", "|---|---|---|---|"]
    for n in names:
        lim = sg[n]["limits_rise_cm"]
        cells = [c for c in sg[n]["cells"] if c["walls"] is None]
        md.append(f"| {n} | " + " / ".join(str(int(lim.get(f'up_t{t}', 0))) for t in TREADS) + " | "
                  + " / ".join(str(int(lim.get(f'down_t{t}', 0))) for t in TREADS)
                  + f" | {sum(c['clean'] for c in cells) / len(cells):.3f} |")
    md += ["", "Clean share where a plan exists (physically possible) vs where none was found:", "",
           "| dog | up, possible | up, no plan | down, possible | down, no plan |", "|---|---|---|---|---|"]
    possible = {k for k, p in {**full, **retry, **main}.items() if p.get("feasible")}
    for k, p in retry.items():
        if p.get("feasible"):
            possible.add(k)
    for n in names:
        row = []
        for d in ("up", "down"):
            for want in (True, False):
                cs = [c for c in sg[n]["cells"] if c["walls"] is None and c["dir"] == d
                      and ((d, int(c["rise_cm"]), int(c["tread_cm"])) in possible) == want]
                row.append(f"{100 * sum(c['clean'] for c in cs) / max(1, len(cs)):.0f} % ({len(cs)})")
        md.append(f"| {n} | " + " | ".join(row) + " |")
    md += ["", "Per cell, clean % (up / down), for the possible cells:", ""]
    for d in ("up", "down"):
        md += [f"{d}:", "", "| dog | " + " | ".join(f"r{r} t{t}" for (dd, r, t) in sorted(possible, key=lambda k: (k[2], k[1]))
                                                       if dd == d) + " |",
               "|---|" + "---|" * sum(1 for k in possible if k[0] == d)]
        for n in names:
            cells = {(int(c["rise_cm"]), int(c["tread_cm"])): c for c in sg[n]["cells"] if c["walls"] is None and c["dir"] == d}
            md.append(f"| {n} | " + " | ".join(f"{100 * cells[(r, t)]['clean']:.0f}" if (r, t) in cells else "-"
                                              for (dd, r, t) in sorted(possible, key=lambda k: (k[2], k[1])) if dd == d) + " |")
        md.append("")

bench = {(Path(r["run"]).name, r["scenario"]): r for r in lines(PREV / "benchmark.jsonl") + lines(HERE / "benchmark.jsonl")}
bnames = [n for n in ("free_s2", "cont_s1", "rsi_s1", "rsi_s2", "rsi_terms_s1", "rsi_terms_s2", "desc_s1") if (n, "forward") in bench]
if bnames:
    md += ["## Nothing forgotten? Walking (falls, walk score), shoves, getting up", "",
           "| dog | forward | course | stairs straight | power |", "|---|---|---|---|---|"]
    for n in bnames:
        cells = [f"{100 * bench[(n, sc)]['fall_rate']:.1f} % ({bench[(n, sc)]['walk_score']:.2f})" if (n, sc) in bench else "-"
                 for sc in ("forward", "course", "stairs_straight")]
        fw = bench.get((n, "forward"), {})
        md.append(f"| {n} | " + " | ".join(cells) + f" | {fw.get('mechanical_power_w', float('nan')):.1f} W |")
rec = {Path(r["run"]).name: r for r in lines(PREV / "recover_test.jsonl") + lines(HERE / "recover_test.jsonl")}
if any(n in rec for n in bnames):
    strengths = list(next(iter(rec.values()))["by_shove_m_s"])
    md += ["", "One shove while walking (stays up / gets up / stays down):", "",
           "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for n in bnames:
        if n in rec:
            md.append(f"| {n} | " + " | ".join(f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / "
                                              f"{100 * v['stayed_down']:.0f} %" for v in rec[n]["by_shove_m_s"].values()) + " |")
gu = {Path(r["run"]).name: r for r in lines(PREV / "getup_test.jsonl") + lines(HERE / "getup_test.jsonl")}
if any(n in gu for n in bnames):
    cats = ("standing", "dropped upright", "on its side", "on its back")
    md += ["", "Getting up within 8 s:", "", "| dog | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for n in bnames:
        if n in gu:
            md.append(f"| {n} | " + " | ".join(f"{100 * gu[n][c]['success']:.0f} %" for c in cats) + " |")
if (HERE / "best.txt").exists():
    md += ["", f"Best per variant (mean clean share): {(HERE / 'best.txt').read_text().strip()}"]
vids = sorted((HERE / "videos").glob("*.mp4")) if (HERE / "videos").exists() else []
if vids:
    md += ["", "Videos: " + ", ".join(f"`videos/{v.name}`" for v in vids)]
(HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
