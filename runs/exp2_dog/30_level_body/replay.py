"""Step 50, part 2 (no training): does a flatter body tip less? The plans of plan_caps.py (nose up at most 5 / 10 / 15 deg and
free, balance margin 1.5 cm) played in physics without a policy (tools/dog/stairs_replay.py), one nominal dog per plan and
16 with the training randomization. Replays drift and are chaotic (step 41): read the table as a trend. The dog is the
planner's own, with the sphere paw (genome_plans.json = bc_s2's genome), not the real paw.

    lab runs/exp2_dog/30_level_body/replay.py
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("still_clock_batch", HERE.parent / "22_still_clock" / "batch.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)
LAB, run, lines = s44.LAB, s44.run, s44.lines
CAPS = {"max 5": "p5", "max 10": "p10", "max 15": "p15", "free": "p45"}
VIDEOS = {"p15": ("up_r4_t9", "up_r4_t15", "up_r5_t20"), "p45": ("up_r4_t9", "up_r4_t15", "up_r5_t20"), "p10": ("up_r4_t15", "up_r5_t20")}
OUT = HERE / "replay"


def replay(plans: Path, out: Path, *extra: str) -> None:
    run([LAB, "tools/dog/stairs_replay.py", str(plans), "--out", str(out), *extra], OUT / "replay.log")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    (HERE / "replay.failed").unlink(missing_ok=True)
    smoke = OUT / "smoke.jsonl"
    if not lines(smoke):
        replay(HERE / "plans_p15_m15" / "up_r4_t9.json", smoke)
    if not lines(smoke):
        (HERE / "replay.failed").write_text("smoke: no result, see replay/replay.log\n")
        return 1
    for cap in CAPS.values():
        plans = HERE / f"plans_{cap}_m15"
        for name, extra in (("nominal", ()), ("random", ("--copies", "16"))):
            out = OUT / f"{cap}_{name}.jsonl"
            if not lines(out):
                replay(plans, out, *extra)
    for cap, stems in VIDEOS.items():
        for stem in stems:
            if not (OUT / cap / "videos" / f"replay_{stem}.mp4").exists():
                replay(HERE / f"plans_{cap}_m15", OUT / cap / "video.jsonl", "--video", f"{stem}.json")
    cells = sorted({(r["rise"], -r["tread"]) for cap in CAPS.values() for r in lines(OUT / f"{cap}_nominal.jsonl")})
    md = ["# Replays of the capped plans (step 50)", "",
          "Share of dogs that climbed: nominal dog (1) / randomized dogs (16). pitch err = largest pitch error of the randomized dogs [deg]. '.' = no plan.", "",
          "| rise / tread | " + " | ".join(CAPS) + " |", "|---|" + "---|" * len(CAPS)]
    table = {c: {name: {(r["rise"], -r["tread"]): r for r in lines(OUT / f"{c}_{name}.jsonl")} for name in ("nominal", "random")} for c in CAPS.values()}
    for cell in cells:
        row = []
        for c in CAPS.values():
            nom, rnd = table[c]["nominal"].get(cell), table[c]["random"].get(cell)
            row.append(f"{nom['climbed']:.0%} / {rnd['climbed']:.0%} (err {rnd['max_pitch_error_deg']:.0f})" if nom and rnd else ".")
        md.append(f"| {100 * cell[0]:.0f} / {-100 * cell[1]:.0f} | " + " | ".join(row) + " |")
    md += ["", "Mean over the stairs each has a plan for, randomized: " + ", ".join(
        f"{k}: {sum(r['climbed'] for r in table[c]['random'].values()) / max(len(table[c]['random']), 1):.0%}" for k, c in CAPS.items()), ""]
    (HERE / "replay.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    (HERE / "replay.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
