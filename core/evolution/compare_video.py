"""Record several trained robots on the same terrain and put the videos side by side with labels.

    python -m core.evolution.compare_video --out runs/skills/videos/stairs_climb.mp4 \\
        --terrain_type stairs_climb --level 5 \\
        --run "Stairs specialist=runs/skills/spec_stairs" --run "Generalist=runs/skills/generalist" \\
        --run "Combined student=runs/skills/student_polished"

Every robot gets the same scripted tour (forward, curve, turn, sidestep, backward, full speed); one robot per video
so the view is uncluttered. Each recording runs in its own simulator process.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def record(run_dir: Path, terrain_type: str | None, level: int | None, name: str, eye=None,
           heading_hold: float | None = None, terrain: str | None = None, drive: str = "tour") -> Path | None:
    cmd = [sys.executable, "-m", "core.evolution.play", "--run_dir", str(run_dir), "--video", "--num_envs", "1",
           "--drive", drive, "--out_name", name]  # fmt: skip
    if eye:
        cmd += ["--eye", *map(str, eye)]
    if heading_hold is not None:
        cmd += ["--heading_hold", str(heading_hold)]
    if terrain:
        cmd += ["--terrain", terrain]  # e.g. flat: a plane instead of the run's own course
    if terrain_type:
        cmd += ["--terrain_type", terrain_type]  # the terrain kind itself comes from the run's result.json
    if level is not None:
        cmd += ["--level", str(level)]
    log = run_dir / "compare_video.log"
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    out = run_dir / "videos" / name
    if not out.exists():
        print(f"recording failed for {run_dir}, see {log}")
        return None
    return out


def compose(clips: list[tuple[str, Path]], subtitle: str, out: Path) -> None:
    import cv2
    from moviepy.editor import VideoFileClip, clips_array

    def label(text: str):
        def f(get_frame, t):
            fr = get_frame(t).copy()
            h = fr.shape[0]
            cv2.rectangle(fr, (0, h - 92), (fr.shape[1], h - 40), (0, 0, 0), -1)
            cv2.putText(fr, text, (24, h - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(fr, subtitle, (24, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            return fr

        return f

    from moviepy.editor import ColorClip

    videos = [VideoFileClip(str(path)).fl(label(text)) for text, path in clips]
    duration = min(v.duration for v in videos)
    scale = 0.6 if len(videos) <= 2 else 0.5
    tiles = [v.subclip(0, duration).resize(scale) for v in videos]
    per_row = 3 if len(tiles) > 4 else (2 if len(tiles) == 4 else len(tiles))
    while len(tiles) % per_row:
        tiles.append(ColorClip(tiles[0].size, color=(255, 255, 255), duration=duration))
    grid = [tiles[i : i + per_row] for i in range(0, len(tiles), per_row)]
    out.parent.mkdir(parents=True, exist_ok=True)
    clips_array(grid).write_videofile(str(out), fps=50, codec="libx264", audio=False, logger=None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Side-by-side videos of trained robots on the same terrain.")
    parser.add_argument("--run", action="append", required=True, help='"Label=run_dir" or "Label=run_dir@level" (repeatable, 1-6 runs; 4 = 2x2 grid, 5-6 = 2x3)')
    parser.add_argument("--terrain_type", default=None, help="rough terrain type, e.g. stairs_climb (omit for flat)")
    parser.add_argument("--level", type=int, default=5, help="difficulty row 0-9 on rough terrain")
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eye", type=float, nargs=3, default=None, help="camera position relative to the robot [m]")
    parser.add_argument("--heading_hold", type=float, default=None, help="keep the heading while walking straight (gain)")
    parser.add_argument("--terrain", default=None, help="terrain kind instead of the run's own (e.g. flat)")
    parser.add_argument("--drive", choices=("tour", "straight"), default="tour",
                        help="tour = forward, curves, turns, sidesteps, backwards; straight = only straight on")
    args = parser.parse_args(argv)

    clips = []
    for k, spec in enumerate(args.run):
        text, run_dir = spec.split("=", 1)
        level = args.level
        if "@" in run_dir:  # "Label=run_dir@7": this panel on difficulty level 7
            run_dir, level = run_dir.rsplit("@", 1)
            level = int(level)
        name = f"cmp_{args.terrain_type or 'terrain'}_l{level}_{k}.mp4"
        path = record(Path(run_dir), args.terrain_type, level, name, args.eye, args.heading_hold, args.terrain, args.drive)
        if path:
            clips.append((text, path))
            print(f"recorded {text}")
    if not clips:
        return 1
    subtitle = args.subtitle or (
        f"{args.terrain_type.replace('_', ' ')}, difficulty {args.level} of 9" if args.terrain_type else "flat ground"
    )
    compose(clips, subtitle, args.out)
    print(f"video: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
