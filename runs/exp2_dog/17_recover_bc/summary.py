"""Tables of step 39 -> results.md, with step 38's dogs (original walker, distilled, RL polish) for comparison."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "16_recover"


def rows(name: str) -> list[dict]:
    out = []
    for d in (PREV, HERE):
        p = d / name
        if p.exists():
            out += [json.loads(line) for line in p.read_text().splitlines()]
    return out


lines = ["# Balance training with the teachers as an anchor (step 39)", ""]
rec = rows("recover_test.jsonl")
if rec:
    strengths = list(rec[0]["by_shove_m_s"])
    lines += ["## One shove while walking at 20 cm/s: stays up / falls and gets up / stays down", "",
              "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for r in rec:
        lines.append(f"| {Path(r['run']).name} | " + " | ".join(
            f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / {100 * v['stayed_down']:.0f} %"
            for v in r["by_shove_m_s"].values()) + " |")
gu = rows("getup_test.jsonl")
if gu:
    cats = ("standing", "dropped upright", "on its side", "on its back")
    lines += ["", "## Getting up: share standing within 8 s (median time)", "", "| dog | " + " | ".join(cats) + " |",
              "|---|" + "---|" * len(cats)]
    for r in gu:
        lines.append(f"| {Path(r['run']).name} | " + " | ".join(
            f"{100 * r[c]['success']:.0f} % ({r[c]['median_s']} s)" if r[c]["median_s"] is not None
            else f"{100 * r[c]['success']:.0f} %" for c in cats) + " |")
bench = {}
for r in rows("benchmark.jsonl"):
    bench[(Path(r["run"]).name, r["scenario"])] = r
if bench:
    lines += ["", "## Walking (falls, walk score; steps climbed up on the hardest stairs rows)", "",
              "| dog | forward | course | stairs straight | steps up (rows 6-9) | power forward |", "|---|---|---|---|---|---|"]
    for name in dict.fromkeys(k[0] for k in bench):
        cells = []
        for sc in ("forward", "course", "stairs_straight"):
            b = bench.get((name, sc))
            cells.append(f"{100 * b['fall_rate']:.1f} % ({b['walk_score']:.2f})" if b else "-")
        climb = (bench.get((name, "stairs_straight")) or {}).get("type_level_climb_m", {}).get("stairs_climb", [])
        steps = [climb[lv] / (0.005 + 0.055 * lv / 9) for lv in range(6, len(climb))]
        cells.append(f"{sum(steps) / len(steps):.1f}" if steps else "-")
        fw = bench.get((name, "forward"))
        cells.append(f"{fw['mechanical_power_w']:.1f} W" if fw and "mechanical_power_w" in fw else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
if (HERE / "best.txt").exists():
    lines += ["", f"Best seed: {(HERE / 'best.txt').read_text().strip()}"]
for v in ("shoves_walking", "flips_walking"):
    p = HERE / "videos" / f"{v}.json"
    if p.exists():
        d = json.loads(p.read_text())
        lines += [f"Video {v}.mp4: went down {d.get('went_down')}x, stayed on its feet {d.get('stayed_on_its_feet')}x, "
                  f"got up {d['stood_up']} (mean {d['mean_s']} s)"]
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
