"""Side check (not a numbered step): Isaac Lab 3.0.0 GA next to the 3.0.0-EA we run, and PhysX next to Newton.

NVIDIA released Isaac Lab 3.0.0 GA on 2026-10-19 (Isaac Sim 6.2, Newton 1.6.1, rsl-rl 5.5.1; 367 commits after EA). It
lives in IsaacLab-3.0.0/ (a git worktree of IsaacLab/, own environment, launcher lab3.cmd / lab3.sh), so the EA
environment and every running batch stay untouched. Two questions:

  1  upgrade: does a dog trained on EA walk the same on GA (same engine, PhysX)? If yes, GA can become the default.
  2  engine: does a dog trained on PhysX still walk on Newton (never trained there)? A policy that survives a different
     physics engine is a good sign for the real robot. Plus: how fast does each engine train here (iteration time,
     start-up), since an early test (journal, step 7) found Newton slower and with different friction.

Cells: EA PhysX (baseline), GA PhysX, EA Newton, GA Newton. Dogs: dog v1 (17_recover_bc/bc_s2) and free_s2
(18_stairs_goal, the start dog of steps 43-45). Per cell and dog: tools/dog/benchmark.py flat, course, forward,
stairs_straight, pushes; per cell: 60 training iterations of step 44's recipe (2048 robots, timing only); videos
PhysX | Newton side by side per environment (the course tour, standing with a shove). results.md at the end.

Smoke results (2026-10-30, 64 robots, dog v1, flat): EA PhysX walk 0.606 / CoT 2.47, GA PhysX 0.606 / 2.53 (the
upgrade keeps the dog; GA needed one port: the scene is declared on cfg.scene, core/sim/locomotion_env.py
DECLARED_SCENE); EA Newton does not start (its ray caster fails on the height scan, IndexError in
newton_raycast_sensor, fixed in GA); GA Newton walks upright but slowly: falls 6 %, walk 0.13 (wider contact
buffers njmax 600 / nconmax 200 change nothing: a model difference, e.g. joint friction or contact friction, not a
buffer overflow). Video: smoke/videos/ga_flat_straight.mp4.

Smoke test first: a 64-robot flat benchmark per cell. EA PhysX must pass; another cell that fails is written to
results.md as a finding and skipped (the GA or Newton code path needs work), the rest runs on. ~2-3 h, one simulator
at a time (RAM). Rerun to continue: finished results are kept.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "runs" / "exp2_dog" / "24_isaaclab_ga"
_spec = importlib.util.spec_from_file_location("still_clock_batch", ROOT / "runs" / "exp2_dog" / "22_still_clock" / "batch.py")
s44 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s44)
WIN = os.name == "nt"
ENVS = {"ea": (str(ROOT / ("lab.cmd" if WIN else "lab.sh")), ROOT / "IsaacLab"),
        "ga": (str(ROOT / ("lab3.cmd" if WIN else "lab3.sh")), ROOT / "IsaacLab-3.0.0")}
CELLS = [("ea", "physx"), ("ga", "physx"), ("ea", "newton"), ("ga", "newton")]
DOGS = [ROOT / "runs" / "exp2_dog" / "17_recover_bc" / "bc_s2", ROOT / "runs" / "exp2_dog" / "18_stairs_goal" / "free_s2"]
SCENARIOS = ("flat", "course", "forward", "stairs_straight", "pushes")
SPEED_ITERATIONS = 60
SIDE_EYE = s44.SIDE_EYE
# (name, terrain, terrain type, level, drive, subtitle)
VIDEOS = [("tour_course", None, None, 5, "tour", "the dog course, difficulty 5 of 9: PhysX (trained) vs Newton (never seen)"),
          ("stand_shove", "flat", None, 0, "stand_shove", "standing on the flat, shoved 0.3 m/s sideways at 6 s")]
lines = s44.lines


def py(env: str) -> str:
    return str(ENVS[env][1] / ".venv" / ("Scripts/python.exe" if WIN else "bin/python"))


def run(cmd: list[str], log: Path) -> int:
    return s44.run(cmd, log)


def has(path: Path, name: str, **match) -> bool:
    return s44.has(path, name, **match)


def bench_file(env: str) -> Path:
    return HERE / f"benchmark_{env}.jsonl"


def benchmark(env: str, physics: str, dog: Path, sc: str, log: Path, out: Path | None = None, num_envs: int = 1024) -> None:
    out = out or bench_file(env)
    if not has(out, dog.name, scenario=sc, physics=physics):
        run([ENVS[env][0], "tools/dog/benchmark.py", str(dog), sc, "--physics", physics, "--num_envs", str(num_envs),
             "--out", str(out)], log)


def smoke() -> list[tuple[str, str]]:
    """The cells that start and finish a small benchmark; [] when the EA PhysX baseline itself fails."""
    out = HERE / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    ok = []
    for env, physics in CELLS:
        if not Path(py(env)).exists():
            continue
        f = out / f"benchmark_{env}.jsonl"
        benchmark(env, physics, DOGS[0], "flat", out / f"smoke_{env}_{physics}.log", out=f, num_envs=64)
        if has(f, DOGS[0].name, scenario="flat", physics=physics):
            ok.append((env, physics))
        elif (env, physics) == ("ea", "physx"):
            (HERE / "batch.failed").write_text("smoke (EA PhysX baseline): see smoke/smoke_ea_physx.log\n")
            return []
    (out / "ok").write_text(json.dumps(ok) + "\n")
    return ok


def speed(env: str, physics: str) -> None:
    """60 iterations of step 44's training (hold, seed 1) in this cell: only the timing in result.json counts."""
    out = HERE / "speed" / f"{env}_{physics}"
    if (out / "result.json").exists():
        return
    cmd = s44.train_cmd(out, "hold", 1, 2048, SPEED_ITERATIONS)
    cmd[0] = ENVS[env][0]
    # the GPU is shared with other programs: their load before the run, to judge the timing
    run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv"], HERE / "speed.log")
    run(cmd + ["--physics", physics], HERE / "speed.log")


def videos(env: str, cells: list[tuple[str, str]]) -> None:
    phys = [p for e, p in cells if e == env]
    if len(phys) < 2:
        return
    for name, terrain, ttype, level, drive, subtitle in VIDEOS:
        for dog in DOGS:
            video = HERE / "videos" / f"{env}_{dog.name}_{name}.mp4"
            if video.exists():
                continue
            cmd = [py(env), "-m", "core.evolution.compare_video", "--out", str(video), "--level", str(level),
                   "--drive", drive, "--heading_hold", "2.0", "--subtitle", f"Isaac Lab {env.upper()}: {subtitle}",
                   "--run", f"PhysX={dog}", "--run", f"Newton={dog}+newton"]
            if terrain:
                cmd += ["--terrain", terrain]
            if ttype:
                cmd += ["--terrain_type", ttype]
            if drive == "stand_shove":
                cmd += ["--eye", *SIDE_EYE]
            run(cmd, HERE / "videos.log")


def report(cells: list[tuple[str, str]]) -> None:
    names = {"ea": "EA", "ga": "GA", "physx": "PhysX", "newton": "Newton"}
    head = [f"{names[e]} {names[p]}" for e, p in CELLS]
    md = ["# Isaac Lab 3.0.0 GA and Newton: side check", "",
          "Falls / walk score per dog and scenario (tools/dog/benchmark.py, 1024 robots, heading hold 2). "
          "Every dog was trained on EA PhysX.", ""]
    missing = [h for h, c in zip(head, CELLS) if c not in cells]
    if missing:
        md += [f"Did not start or finish the smoke test (see smoke/*.log): {', '.join(missing)}.", ""]
    for dog in DOGS:
        md += [f"## {dog.parent.name}/{dog.name}", "", "| scenario | " + " | ".join(head) + " |",
               "|---" * (len(head) + 1) + "|"]
        for sc in SCENARIOS:
            row = []
            for env, physics in CELLS:
                r = next((r for r in lines(bench_file(env)) if Path(r["run"]).name == dog.name and r["scenario"] == sc
                          and r.get("physics") == physics), None)
                row.append(f"{r['fall_rate']:.1%} / {r['walk_score']:.3f}" if r else "-")
            md.append(f"| {sc} | " + " | ".join(row) + " |")
        md.append("")
    md += ["## Training speed", "", f"{SPEED_ITERATIONS} iterations of step 44's recipe, 2048 robots.", "",
           "| cell | start-up [s] | per iteration [s] |", "|---|---|---|"]
    for h, (env, physics) in zip(head, CELLS):
        f = HERE / "speed" / f"{env}_{physics}" / "result.json"
        if f.exists():
            r = json.loads(f.read_text())
            md.append(f"| {h} | {r['setup_time_s']:.0f} | {r['train_time_s'] / r['iterations']:.2f} |")
        else:
            md.append(f"| {h} | - | - |")
    md += ["", "Videos: `videos/<env>_<dog>_<scene>.mp4`, PhysX left, Newton right."]
    (HERE / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "batch.failed").unlink(missing_ok=True)
    if not Path(py("ga")).exists():
        print("Isaac Lab GA environment missing: IsaacLab-3.0.0/.venv (uv sync, see .env_init/readme.txt)")
        return 1
    cells = smoke()
    if not cells:
        return 1
    log = HERE / "tests.log"
    for env, physics in cells:
        for dog in DOGS:
            for sc in SCENARIOS:
                benchmark(env, physics, dog, sc, log)
        report(cells)
    for env, physics in cells:
        speed(env, physics)
    report(cells)
    for env in ENVS:
        videos(env, cells)
    report(cells)
    (HERE / "batch.done").write_text("done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
