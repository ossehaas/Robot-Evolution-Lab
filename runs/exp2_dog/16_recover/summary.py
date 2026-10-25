"""Tables of step 38 -> results.md: shove and recovery, getting up, walking quality."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
lines = ["# Keep balance and get up (step 38)", ""]


def rows(name: str) -> list[dict]:
    path = HERE / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


rec = rows("recover_test.jsonl")
if rec:
    strengths = list(rec[0]["by_shove_m_s"])
    lines += ["## One shove while walking at 20 cm/s: stays up / falls and gets up / stays down", "",
              "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for r in rec:
        cells = [f"{100 * v['stayed_up']:.0f} / {100 * v['fell_got_up']:.0f} / {100 * v['stayed_down']:.0f} %"
                 for v in r["by_shove_m_s"].values()]
        lines.append(f"| {Path(r['run']).name} | " + " | ".join(cells) + " |")

gu = rows("getup_test.jsonl")
if gu:
    cats = ("standing", "dropped upright", "on its side", "on its back")
    lines += ["", "## Getting up: share standing within 8 s (median time)", "", "| dog | " + " | ".join(cats) + " |",
              "|---|" + "---|" * len(cats)]
    for r in gu:
        cells = [f"{100 * r[c]['success']:.0f} % ({r[c]['median_s']} s)" if r[c]["median_s"] is not None
                 else f"{100 * r[c]['success']:.0f} %" for c in cats]
        lines.append(f"| {Path(r['run']).name} | " + " | ".join(cells) + " |")

bench = {}
for r in rows("benchmark.jsonl"):
    bench[(Path(r["run"]).name, r["scenario"])] = r
if bench:
    lines += ["", "## Walking (falls, walk score; stairs: steps climbed up on the hardest rows)", "",
              "| dog | forward | course | stairs straight | steps up (rows 6-9) | power forward |", "|---|---|---|---|---|---|"]
    for name in dict.fromkeys(k[0] for k in bench):
        cells = []
        for sc in ("forward", "course", "stairs_straight"):
            b = bench.get((name, sc))
            cells.append(f"{100 * b['fall_rate']:.1f} % ({b['walk_score']:.2f})" if b else "-")
        st = bench.get((name, "stairs_straight"))
        climb = (st or {}).get("type_level_climb_m", {}).get("stairs_climb", [])
        steps = [climb[lv] / (0.005 + 0.055 * lv / 9) for lv in range(6, len(climb))]
        cells.append(f"{sum(steps) / len(steps):.1f}" if steps else "-")
        fw = bench.get((name, "forward"))
        cells.append(f"{fw['mechanical_power_w']:.1f} W" if fw and "mechanical_power_w" in fw else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

for v in ("shoves_walking", "flips_walking"):
    p = HERE / "videos" / f"{v}.json"
    if p.exists():
        d = json.loads(p.read_text())
        lines += ["", f"Video {v}.mp4: went down {d.get('went_down')}x, stayed on its feet {d.get('stayed_on_its_feet')}x, "
                      f"got up {d['stood_up']} (mean {d['mean_s']} s)"]
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
