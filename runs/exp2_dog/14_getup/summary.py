"""Tables of step 36 -> results.md: stairs actually climbed, balance under a shove, getting up."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
lines = ["# Getting up, being pushed, climbing (step 36)", ""]


def rows(name: str) -> list[dict]:
    path = HERE / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def step_height(level: int) -> float:
    return 0.005 + (0.06 - 0.005) * level / 9  # terrain.DOG_ROUGH stairs: 0.5 cm (row 0) to 6 cm (row 9)


climb = rows("climb.jsonl")
if climb:
    lines += ["## Stairs, walking straight on: falls and steps actually climbed (rows 6-9 = the hardest)", "",
              "| dog | falls | up: steps climbed (rows 6-9) | up: falls (rows 6-9) | down: steps (rows 6-9) | down: falls (rows 6-9) |",
              "|---|---|---|---|---|---|"]
    for r in climb:
        c, f = r.get("type_level_climb_m", {}), r.get("type_fall_rates", {})
        cells = []
        for t in ("stairs_climb", "stairs_descend"):
            h = c.get(t, [])
            steps = [h[lv] / step_height(lv) for lv in range(6, len(h))]
            cells.append(f"{sum(steps) / len(steps):.1f}" if steps else "-")
            fr = f.get(t, [])[6:]
            cells.append(f"{100 * sum(fr) / len(fr):.0f} %" if fr else "-")
        lines.append(f"| {Path(r['run']).name} | {100 * r['fall_rate']:.1f} % | " + " | ".join(cells) + " |")

push = rows("push_test.jsonl")
if push:
    strengths = list(push[0]["fall_rate_by_push_m_s"])
    lines += ["", "## Balance: falls after one shove while walking at 20 cm/s", "",
              "| dog | " + " | ".join(f"{s} m/s" for s in strengths) + " |", "|---|" + "---|" * len(strengths)]
    for r in push:
        lines.append(f"| {Path(r['run']).name} | "
                     + " | ".join(f"{100 * r['fall_rate_by_push_m_s'][s]:.0f} %" for s in strengths) + " |")

getup = rows("getup_test.jsonl")
if getup:
    cats = ("standing", "dropped upright", "on its side", "on its back")
    lines += ["", "## Getting up: share standing within 8 s (median time)", "",
              "| dog | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for r in getup:
        cells = [f"{100 * r[c]['success']:.0f} % ({r[c]['median_s']} s)" if r[c]["median_s"] is not None
                 else f"{100 * r[c]['success']:.0f} %" for c in cats]
        lines.append(f"| {Path(r['run']).name} | " + " | ".join(cells) + " |")
    lines += ["", "Video: videos/getup_<best run>.mp4"]
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
