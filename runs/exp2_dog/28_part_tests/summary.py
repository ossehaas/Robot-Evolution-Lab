"""Step 48: tables of the paw test stand from stand.jsonl; written next to it as stand.md and printed."""

import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "stand.jsonl"
rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def pick(row, test, **kw):
    return [c for c in row["cases"] if c["test"] == test and all(c.get(k) == v for k, v in kw.items())]


lines = ["Stance (the shin sweeps +11 -> -18 deg under load), at 6 / 12 / 18 N: work through the ankle [mJ], ankle vault [mm], "
         "share of the normal force on the housing, x travel [mm]", "",
         "| design | work | vault | on the housing | x |", "|---|---|---|---|---|"]
for r in rows:
    c = pick(r, "roll")
    lines.append(f"| {r['name']} | " + " / ".join(f"{x['work_mJ']:.0f}" for x in c) + " | " + " / ".join(f"{x['vault_mm']:.1f}" for x in c) + " | "
                 + " / ".join(f"{x['on_house']:.0%}" for x in c) + " | " + " / ".join(f"{x['x_mm']:.0f}" for x in c) + " |")
lines += ["", "Pulled over the floor at 12 N, shin tilt -20 / -10 / 0 / 10 / 20 deg: peak pull (force / load), friction while sliding, share on the housing", "",
          "| design | peak pull | friction | on the housing |", "|---|---|---|---|"]
for r in rows:
    c = sorted(pick(r, "drag", load=12.0), key=lambda x: x["tilt"])
    lines.append(f"| {r['name']} | " + " / ".join(f"{x['grip']:.2f}" for x in c) + " | " + " / ".join(f"{x['friction']:.2f}" for x in c) + " | "
                 + " / ".join(f"{x['on_house']:.0%}" for x in c) + " |")
lines += ["", "Peak pull at shin tilt 0, load 6 / 12 / 18 N (above the friction: the paw tips and its edge digs in):", "", "| design | peak pull |", "|---|---|"]
for r in rows:
    c = sorted(pick(r, "drag", tilt=0.0), key=lambda x: x["load"])
    lines.append(f"| {r['name']} | " + " / ".join(f"{x['grip']:.2f}" for x in c) + " |")
lines += ["", "Step edge (4 cm riser, 18 cases: 5 / 15 / 30 mm below the edge, shin -10 / 0 / 10 deg, pushed with 6 / 12 N, lift at most 30 N):", "",
          "| design | got over | lift per N of push: mean over the climb, median case | peak, median case | peak, worst case | not over |", "|---|---|---|---|---|---|"]
for r in rows:
    c = pick(r, "edge")
    ok = [x for x in c if x["cleared"]]
    lp = sorted(x["lift_per_push"] for x in c)
    bad = [f"{x['below'] * 1000:.0f}mm/{x['tilt']:.0f}deg/{x['push']:.0f}N" for x in c if not x["cleared"]]
    lines.append(f"| {r['name']} | {len(ok)} of {len(c)} | {sorted(x['mean_per_push'] for x in c)[len(c) // 2]:.2f} | {lp[len(lp) // 2]:.2f} | {lp[-1]:.2f} | {', '.join(bad[:4]) or '-'} |")
lines += ["", "The same climb in two phases, sole 30 mm below the edge, shin 0, pushed with 6 / 12 N: lift per N of push while the paw is against the "
          "vertical face (sole 4-10 mm above its start), and the peak over the last 6 mm (what passes the edge last)", "",
          "| design | against the face | last 6 mm |", "|---|---|---|"]
for r in rows:
    tr = np.load(path.parent / f"trace_{r['name'].replace(' ', '_').replace('+', 'plus').replace('.', '_')}.npz")
    cases = json.loads(str(tr["cases"]))
    face, last = [], []
    for push in (6.0, 12.0):
        i = next(k for k, c in enumerate(cases) if c["test"] == "edge" and c["below"] == 0.03 and c["tilt"] == 0.0 and c["push"] == push)
        z, f = tr["z"][:, i], tr["fz"][:, i]
        face.append(np.median(f[(z >= 0.014) & (z <= 0.020)]) / push)
        last.append(f[(z >= 0.034) & (z < 0.040)].max() / push)
    lines.append(f"| {r['name']} | " + " / ".join(f"{x:.2f}" for x in face) + " | " + " / ".join(f"{x:.2f}" for x in last) + " |")
lines += ["", "Drop 3 / 6 cm with 0.6 kg: peak force over 15 ms [N], peak on the housing, lowest point of the z slider (0 = sole just touching) [mm], rebound [mm]", "",
          "| design | peak | on the housing | lowest | rebound |", "|---|---|---|---|---|"]
for r in rows:
    c = pick(r, "drop")
    lines.append(f"| {r['name']} | " + " / ".join(f"{x['peak_N']:.0f}" for x in c) + " | " + " / ".join(f"{x['peak_house_N']:.0f}" for x in c) + " | "
                 + " / ".join(f"{x['lowest_mm']:.1f}" for x in c) + " | " + " / ".join(f"{x['rebound_mm']:.1f}" for x in c) + " |")
text = "\n".join(lines)
print(text)
path.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
