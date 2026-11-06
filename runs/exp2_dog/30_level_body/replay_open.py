"""Step 50, part 2 (no training): does a flatter body tip less? The capped plans played in physics without a policy
(tools/dog/stairs_replay.py), 16 randomized dogs per plan and one nominal, open loop as step 41's replays that climbed.

  open loop, margin 1.5 cm and 3 cm, nose up at most 5 / 10 / 15 deg and free
  feedback, margin 3 cm: step 41 said feedback falls because paws land 1-2 cm off and the margin is about that (the
  feedback runs at margin 1.5 cm are replay/*.jsonl: nothing climbed)

Replays drift and are chaotic (step 41): read the table as a trend. The dog is the planner's own, with the sphere paw
(genome_plans.json = bc_s2's genome), not the real paw. A run that ends without results is tried again (random crashes).

    lab runs/exp2_dog/30_level_body/replay_open.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "replay_open"
_spec = importlib.util.spec_from_file_location("still_clock_batch", HERE.parent / "22_still_clock" / "batch.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)
LAB, run, lines = s44.LAB, s44.run, s44.lines
CAPS = {"max 5": "p5", "max 10": "p10", "max 15": "p15", "free": "p45"}
RUNS = [("open", "m15", ("--open_loop",)), ("open", "m30", ("--open_loop",)), ("feedback", "m30", ())]
VIDEOS = {"p10": ("up_r4_t15", "up_r5_t20"), "p15": ("up_r4_t9", "up_r4_t15", "up_r5_t20"), "p45": ("up_r4_t9", "up_r4_t15", "up_r5_t20")}


def replay(plans: Path, out: Path, *extra: str) -> None:
    for _ in range(3):
        if lines(out):
            return
        code = run([LAB, "tools/dog/stairs_replay.py", str(plans), "--out", str(out), *extra], OUT / "replay.log")
        print(f"{out.name}: exit code {code}, {len(lines(out))} results", flush=True)


def table(mode: str, margin: str) -> list[str]:
    data = {c: {kind: {(r["rise"], -r["tread"]): r for r in lines(OUT / f"{mode}_{margin}_{c}_{kind}.jsonl")} for kind in ("nominal", "random")} for c in CAPS.values()}
    cells = sorted({cell for d in data.values() for cell in d["random"]})
    md = [f"## {mode}, balance margin {int(margin[1:]) / 10:.1f} cm", "", "| rise / tread | " + " | ".join(CAPS) + " |", "|---|" + "---|" * len(CAPS)]
    for cell in cells:
        row = []
        for c in CAPS.values():
            nom, rnd = data[c]["nominal"].get(cell), data[c]["random"].get(cell)
            row.append(f"{nom['climbed']:.0%} / {rnd['climbed']:.0%}" if nom and rnd else ".")
        md.append(f"| {100 * cell[0]:.0f} / {-100 * cell[1]:.0f} | " + " | ".join(row) + " |")
    both = [cell for cell in cells if all(cell in data[c]["random"] for c in CAPS.values())]
    md += ["", f"Randomized dogs that climbed, mean over the {len(both)} stairs all four have a plan for: " + ", ".join(
        f"{k} {sum(data[c]['random'][cell]['climbed'] for cell in both) / max(len(both), 1):.0%}" for k, c in CAPS.items()), ""]
    return md


def main() -> int:
    OUT.mkdir(exist_ok=True)
    (HERE / "replay_open.failed").unlink(missing_ok=True)
    smoke = OUT / "smoke.jsonl"
    replay(HERE / "plans_p45_m15" / "up_r2_t15.json", smoke, "--open_loop", "--copies", "4")
    if not lines(smoke):
        (HERE / "replay_open.failed").write_text("smoke: no result, see replay_open/replay.log\n")
        return 1
    for mode, margin, extra in RUNS:
        for cap in CAPS.values():
            plans = HERE / f"plans_{cap}_{margin}"
            replay(plans, OUT / f"{mode}_{margin}_{cap}_nominal.jsonl", *extra)
            replay(plans, OUT / f"{mode}_{margin}_{cap}_random.jsonl", *extra, "--copies", "16")
    for cap, stems in VIDEOS.items():
        for stem in stems:
            if not (OUT / cap / "videos" / f"replay_{stem}.mp4").exists():
                code = run([LAB, "tools/dog/stairs_replay.py", str(HERE / f"plans_{cap}_m15"), "--open_loop", "--video", f"{stem}.json", "--out", str(OUT / cap / "video.jsonl")], OUT / "replay.log")
                print(f"video {cap} {stem}: exit code {code}", flush=True)
    md = ["# Replays of the capped plans (step 50)", "", "Share of dogs that climbed without a fall: nominal dog (1) / randomized dogs (16). '.' = no plan.", ""]
    for mode, margin, _ in RUNS:
        md += table(mode, margin)
    (HERE / "replay_open.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    (HERE / "replay_open.done").write_text("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
