"""One-off: sort runs/ into per-experiment folders, keep the best runs, reduce comparisons and failed experiments to
their results, delete superseded versions, and write runs/INDEX.md (with a one-line result for every removed run).

    python tools/reorganize_runs.py            # dry run: print the plan
    python tools/reorganize_runs.py --apply    # do it
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# top-level folder -> new location (contents keep their names)
GROUPS = {
    "crawler_q4": "exp1_crawler/1_walking/crawler_q4",
    "crawler_q4_steer": "exp1_crawler/1_walking/crawler_q4_steer",
    "crawler_q4_rough": "exp1_crawler/1_walking/crawler_q4_rough",
    "crawler_q4_steer_v1": "exp1_crawler/1_walking/crawler_q4_steer_v1",
    "evo_rough_001": "exp1_crawler/2_evolution/evo_rough_001",
    "evo_rough_002": "exp1_crawler/2_evolution/evo_rough_002",
    "compare": "exp1_crawler/3_comparisons/compare",
    "compare2": "exp1_crawler/3_comparisons/compare2",
    "skills": "exp1_crawler/4_skills",
    "stairs": "exp1_crawler/5_stairs",
    "nav": "exp1_crawler/6_navigation/nav",
    "poc": "exp1_crawler/6_navigation/poc",
    "closest": "exp1_crawler/7_closest_point",
    "detour": "exp1_crawler/8_detour",
    "dog_designs": "exp2_dog/1_design",
    "dog_capability": "exp2_dog/1_design/capability",
    "dog_walk": "exp2_dog/2_walking",
    "dog_spine": "exp2_dog/3_flexible_back",
    "dog_stairs": "exp2_dog/4_stairs_seeing",
    "dog_jump": "exp2_dog/5_jumping",
}
LOOSE_LOGS = {"crawler_q4_rough.log": "exp1_crawler/1_walking", "crawler_q4_steer.log": "exp1_crawler/1_walking",
              "evo_rough_001.log": "exp1_crawler/2_evolution", "evo_rough_002.log": "exp1_crawler/2_evolution"}
DELETE_TOP = ["_check_crawler", "random_robots", "distill_check"]

# per run (old path relative to runs/): "keep" (as is), "results" (keep result.json/genome/videos/notes only),
# anything not listed inside a group with an explicit list is deleted. The reason goes into INDEX.md.
KEEP = {
    "crawler_q4": "first walk (forward only)",
    "crawler_q4_steer": "steerable walking (M1)",
    "crawler_q4_rough": "crawler on rough terrain (baseline for evolution)",
    "evo_rough_002/g05_025": "best evolved body (used for skills and navigation)",
    "compare2/evolved_imu_h8": "evolved body with realistic IMU + memory (the sensor setup used afterwards)",
    "skills/student": "distilled student: beats all teachers (0.278)",
    "poc/student_v2": "walker used by the hierarchical navigators",
    "poc/nav_camera": "best navigator: 9.3 goals/min, camera-like goal signal",
    "poc/e2e_rough": "best navigator on a bumpy floor: 7.4 goals/min",
    "poc/hier_flat": "best hierarchical navigator: 7.6 goals/min",
    "closest/ring_v2": "unreachable goal in a ring: 70 % end within 5 cm of the closest point",
    "detour/cup_v1": "detour round a half circle: 73 % reach the goal",
    "dog_walk/calm_v2": "best dog walker (calm, 10 % falls on rough, CoT 2.57, 0 W standing)",
    "dog_stairs/seeing_v3": "best dog on the dog course: seeing (height scan), 7.1 % falls",
    "dog_jump/sts3215": "jump with the real servos: lowest paw 8 cm",
    "dog_jump/qdd": "jump with hypothetical fast motors: lowest paw 19 cm",
}
RESULTS_ONLY = {
    "evo_rough_001": "failed evolution run: the winner was a stander exploiting the fitness (journal step 11)",
    "compare/*": "body/sensor comparison runs (journal steps 11-14)",
    "compare2/*": "sensor comparison runs (journal step 14)",
    "skills/generalist": "RL generalist (0.256), beaten by the distilled student",
    "skills/spec_stairs": "stairs specialist teacher",
    "skills/spec_blocks": "blocks specialist teacher",
    "skills/student_polished": "RL polish after distillation fell back to 0.235",
    "stairs/*": "stairs tests A1/A2/B/C; note: the height scan of test C saw nothing (bug found 2026-10-09)",
    "nav/goal_v1": "first navigator (8.9 goals/min), superseded by poc/nav_camera",
    "poc/nav_control": "navigation control run (8.8 goals/min)",
    "poc/nav_strict": "stricter collision penalty (9.0 goals/min); nav_camera continues from it",
    "poc/hier_rough": "hierarchical navigator on a bumpy floor (6.0 goals/min)",
    "poc/student_v2_polished": "RL polish of the student (worse than the student)",
    "closest/*": "ring v1 (parked at the wall) and the pocket experiment (not solved)",
    "detour/zeroshot_nav_camera": "detour zero-shot baseline (1 % at 180 deg)",
    "dog_spine/*": "flexible back: every variant less stable than the stiff body (journal step 23)",
    "dog_stairs/blind": "blind control for seeing_v3: 16.0 % falls",
    "dog_stairs/range75": "stairs with 75 % joint range (blind): 18 % falls; start of blind/seeing",
    "dog_stairs/range50": "stairs with 50 % joint range: 21 % falls",
    "dog_stairs/eval_range75": "per-type falls of range75 (stairs/blocks are the problem)",
    "dog_stairs/zeroshot_calm_v2": "calm_v2 on the dog course without training: 28 % falls",
}
DELETE = {  # superseded versions and scratch (results are in the journal; one line each goes into INDEX.md)
    "crawler_q4_steer_v1": "superseded by crawler_q4_steer (symmetric command ranges)",
    "evo_rough_002/*": "other individuals of evolution run 2 (leaderboard.md keeps their scores)",
    "dog_walk/*": "intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md",
    "dog_stairs/seeing": "seeing v1: scan in every history frame, curriculum collapsed",
    "dog_stairs/seeing_v2": "seeing v2: the scan read -1 everywhere (bug)",
    "dog_stairs/_scan": "scratch",
    "dog_jump/v1": "jump v1: kneeling and somersault exploits",
}
NOTE_FILES = (".md", ".png", ".mp4", ".txt", ".json", ".py", ".cmd")  # kept in group folders
RESULT_FILES = {"result.json", "genome.json", "leaderboard.md", "fitness.png", "gallery.png"}


def summary(run: Path) -> str:
    f = run / "result.json"
    if not f.exists():
        return ""
    r = json.loads(f.read_text())
    parts = []
    for key, fmt in (("walk_score", "walk {:.2f}"), ("tracking_score", "tracking {:.2f}"), ("goals_per_minute", "{:.1f} goals/min"),
                     ("fall_rate", "falls {:.0%}"), ("cost_of_transport", "CoT {:.2f}"), ("within_5cm_rate", "within 5 cm {:.0%}"),
                     ("inside_success_rate", "inside goal {:.0%}"), ("jump_height_m_mean", "jump {:.3f} m"),
                     ("speed_m_s", "speed {:.2f} m/s"), ("train_time_s", "{:.0f} s training")):
        if key in r and isinstance(r[key], (int, float)):
            parts.append(fmt.format(r[key]))
    return ", ".join(parts)


def match(table: dict, rel: str) -> str | None:
    if rel in table:
        return table[rel]
    parent = rel.rsplit("/", 1)[0] if "/" in rel else None
    return table.get(f"{parent}/*") if parent else None


def plan():
    actions = []  # (kind, old_rel, new_rel, reason)
    for top in DELETE_TOP:
        if (RUNS / top).exists():
            actions.append(("delete", top, None, "scratch / temporary check"))
    for top, dest in GROUPS.items():
        src = RUNS / top
        if not src.exists():
            continue
        top_kind = "keep" if top in KEEP else ("results" if top in RESULTS_ONLY else ("delete" if top in DELETE else None))
        if top_kind:  # the whole top-level folder is one run (or one failed experiment)
            reason = KEEP.get(top) or RESULTS_ONLY.get(top) or DELETE.get(top)
            if top == "evo_rough_001":
                for child in sorted(src.iterdir()):
                    if child.is_dir() and child.name.startswith("g"):
                        actions.append(("delete", f"{top}/{child.name}", None, "individual of the failed evolution run 1"))
                actions.append(("move", top, dest, reason))
            elif top_kind == "delete":
                actions.append(("delete", top, None, reason))
            else:
                actions.append(("move" if top_kind == "keep" else "results", top, dest, reason))
            continue
        # a group folder: decide per child
        for child in sorted(src.iterdir()):
            rel = f"{top}/{child.name}"
            if child.is_dir() and child.name == "videos":
                actions.append(("move", rel, f"{dest}/videos", "videos"))
            elif child.is_dir():
                if rel in KEEP:
                    actions.append(("move", rel, f"{dest}/{child.name}", KEEP[rel]))
                elif match(RESULTS_ONLY, rel):
                    actions.append(("results", rel, f"{dest}/{child.name}", match(RESULTS_ONLY, rel)))
                elif match(DELETE, rel) or child.name.startswith("_") or child.name in ("eval",):
                    actions.append(("delete", rel, None, match(DELETE, rel) or "scratch / temporary evaluation"))
                elif top == "evo_rough_002":
                    actions.append(("delete", rel, None, "other individuals of evolution run 2"))
                else:
                    actions.append(("move", rel, f"{dest}/{child.name}", "kept (not classified)"))
            elif child.suffix in NOTE_FILES:
                actions.append(("move", rel, f"{dest}/{child.name}", "notes / scripts / summary"))
            else:  # training logs of runs, .done markers, play/probe logs
                keep_log = child.suffix == ".log" and (src / child.stem).is_dir() and (
                    f"{top}/{child.stem}" in KEEP or match(RESULTS_ONLY, f"{top}/{child.stem}"))
                if keep_log:
                    actions.append(("move", rel, f"{dest}/{child.name}", "training log"))
                else:
                    actions.append(("delete", rel, None, "log / marker"))
    for name, dest in LOOSE_LOGS.items():
        if (RUNS / name).exists():
            actions.append(("move", name, f"{dest}/{name}", "training log"))
    return actions


def rewrite_json_paths(path: Path, mapping: list[tuple[str, str]]) -> None:
    text = path.read_text()
    new = text
    for old, rep in mapping:
        for sep_old, sep_new in ((f"runs/{old}", f"runs/{rep}"), (f"runs\\\\{old}", "runs\\\\" + rep.replace("/", "\\\\"))):
            new = new.replace(sep_old + "/", sep_new + "/").replace(sep_old + "\\\\", sep_new + "\\\\").replace(sep_old + '"', sep_new + '"')
    if new != text:
        path.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    actions = plan()
    removed, kept, reduced = [], [], []
    for kind, old, new, reason in actions:
        line = f"{kind:8s} {old:45s} -> {new or '-':55s} {reason}"
        print(line)
        if kind == "delete" and (RUNS / old).is_dir():
            removed.append((old, reason, summary(RUNS / old)))
        elif kind == "move" and (RUNS / old).is_dir() and (RUNS / old / "result.json").exists():
            kept.append((new, reason, summary(RUNS / old)))
        elif kind == "results":
            reduced.append((new, reason, summary(RUNS / old)))
    if not args.apply:
        print(f"\n{len(actions)} actions; dry run (use --apply)")
        return

    def group_videos(old_rel: str) -> Path | None:
        top = old_rel.split("/")[0]
        return RUNS / GROUPS[top].split("/")[0] / GROUPS[top].split("/")[1] / "videos" if top in GROUPS else None

    for kind, old, new, reason in actions:
        src = RUNS / old
        if not src.exists():
            continue
        if kind == "delete" and src.is_dir():
            # keep the videos of a removed run, named after it
            dest = group_videos(old)
            for clip in src.glob("**/videos/*.mp4"):
                if dest is not None and "play" not in clip.parts:
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(clip, dest / f"{old.split('/')[-1]}__{clip.name}")
        if kind == "delete":
            shutil.rmtree(src) if src.is_dir() else src.unlink()
        elif kind in ("move", "results"):
            dst = RUNS / new
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            if kind == "results":  # strip models, robot files, tensorboard logs
                for item in list(dst.rglob("*")):
                    if item.is_file() and item.name not in RESULT_FILES and item.suffix not in (".mp4", ".md", ".txt", ".png"):
                        item.unlink()
                for item in sorted(dst.rglob("*"), key=lambda p: -len(p.parts)):
                    if item.is_dir() and not any(item.iterdir()):
                        item.rmdir()
    # empty old top-level folders
    for top in GROUPS:
        p = RUNS / top
        if p.exists() and p.is_dir() and not any(p.rglob("*.*")):
            shutil.rmtree(p)
    # absolute paths stored in result.json files (the hierarchical navigators load their walker from one)
    mapping = sorted(GROUPS.items(), key=lambda kv: -len(kv[0]))
    for f in RUNS.rglob("result.json"):
        rewrite_json_paths(f, mapping)

    lines = ["# Runs index", "", "Training output, sorted per experiment (reorganized 2026-10-10). Only the best run of each",
             "successful line is kept in full; comparisons and failed experiments keep their results (result.json, videos,",
             "summaries) without models; superseded versions were removed. Details and numbers: `.documentation/project-journal.md`.",
             "", "## Kept in full", "", "| Run | What | Result |", "|---|---|---|"]
    lines += [f"| `{n}` | {r} | {s} |" for n, r, s in kept]
    lines += ["", "## Results only (models removed)", "", "| Run | What | Result |", "|---|---|---|"]
    lines += [f"| `{n}` | {r} | {s} |" for n, r, s in reduced]
    lines += ["", "## Removed", "", "| Old path | Why | Result |", "|---|---|---|"]
    lines += [f"| `runs/{o}` | {r} | {s} |" for o, r, s in removed]
    lines += ["", "## Old -> new paths", "", "| Old | New |", "|---|---|"]
    lines += [f"| `runs/{o}` | `runs/{n}` |" for o, n in GROUPS.items()]
    (RUNS / "INDEX.md").write_text("\n".join(lines) + "\n")
    print("done; wrote runs/INDEX.md")


if __name__ == "__main__":
    main()
