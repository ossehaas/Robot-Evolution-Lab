"""Get-up table of the design variants (step 37) -> results.md, with step 36's dog (abduction 29 deg, head random,
no self-collision) for reference."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = ("standing", "dropped upright", "on its side", "on its back")
rows = []
for path in (HERE.parent / "14_getup" / "getup_test.jsonl", HERE / "getup_test.jsonl"):
    if path.exists():
        rows += [json.loads(line) for line in path.read_text().splitlines()]
rows = [r for r in rows if Path(r["run"]).name.startswith("getup")]
lines = ["# Getting up: design variants (step 37)", "",
         "Share standing within 8 s (median time). Abduction = sideways leg range; head = neck/head/tail driven by the "
         "policy. Step 37 runs have self-collision on.", "",
         "| run | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
for r in rows:
    cells = [f"{100 * r[c]['success']:.0f} % ({r[c]['median_s']} s)" if r[c]["median_s"] is not None
             else f"{100 * r[c]['success']:.0f} %" for c in CATS]
    lines.append(f"| {Path(r['run']).name} | " + " | ".join(cells) + " |")
lines += ["", "Videos: videos/<run>.mp4 (the four starts side by side)."]
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
