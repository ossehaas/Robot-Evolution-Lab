"""Why do the dogs refuse to go down stairs they could physically descend? (journal step 42)

Dogs start on the landing of chosen stairs (down 2 cm / 25 cm treads as an easy reference, and the hard but possible
descents 3 cm / 9, 3 cm / 12, 4 cm / 12 and 4 cm / 15 cm) and are driven to the bottom by the navigation layer, as in
the stairs goal test. Every 20 ms the training reward terms are read (from the environment's per-term sums) and
assigned to a phase of the first descent:

    landing   body and front paws still on the landing, more than 12 cm from the top edge
    edge      a front paw within 12 cm of the edge or already over a lower step, the body still on the landing
    stairs    the body over the steps
    refusing  in 'edge', commanded forward (> 5 cm/s) but moving slower than 2 cm/s

Per phase and term: the mean reward per second (positive = paid, negative = cost). The terms that change most from
'landing' to 'edge' / 'refusing' are what the dog weighs when it hesitates. Appends one JSON line to --out.

    lab tools/dog/descent_terms.py runs/exp2_dog/18_stairs_goal/free_s2 --out runs/exp2_dog/20_balance_and_style/descent_terms.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.evolution.train import build_robot_files  # noqa: E402
from core.morphology import Genome, build_robot  # noqa: E402
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg  # noqa: E402
from core.sim.terrain import STAIRS_RISES  # noqa: E402

CELLS = [(0.02, 25), (0.03, 9), (0.03, 12), (0.04, 12), (0.04, 15)]  # (rise [m], tread [cm]), open sides
PHASES = ("landing", "edge", "stairs", "refusing")
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--per_cell", type=int, default=60)
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), len(CELLS) * args.per_cell, False, task="command",
                   terrain="dog_stairs", sensors=res["sensors"], obs_history=res["obs_history"],
                   height_scan=res.get("height_scan", False), terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.recover, cfg.topple_push, cfg.getup = True, (), False
cfg.episode_length_s = 20.0
cfg.stairs_goal_dir = -1
cfg.curriculum = False

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    dev, n = base.device, base.num_envs
    actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()
    names = list(cfg.terrain.terrain_generator.sub_terrains)
    cell = torch.arange(n, device=dev) // args.per_cell
    rows = torch.tensor([STAIRS_RISES.index(r) for r, _ in CELLS], device=dev)[cell]
    cols = torch.tensor([names.index(f"goal_stairs_t{t}") for _, t in CELLS], device=dev)[cell]
    base.eval_mode = True
    base.curriculum_enabled = False
    t = base.terrain
    t.terrain_levels[:], t.terrain_types[:] = rows, cols
    t.env_origins[:] = t.terrain_origins[rows, cols]
    obs, _ = env.reset()
    g = base._sg_geo
    keys = [k for k in base._episode_sums if not k.startswith("_")]
    prev = {k: base._episode_sums[k].clone() for k in keys}
    sums = {p: {k: torch.zeros(n, device=dev) for k in keys} for p in PHASES}
    steps = {p: torch.zeros(n, device=dev) for p in PHASES}
    active = torch.ones(n, dtype=torch.bool, device=dev)  # until the first episode of this dog ends or it arrives
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    dt = base.step_dt
    with torch.inference_mode():
        for _ in range(int(cfg.episode_length_s / dt) - 2):
            obs, _, dones, _ = env.step(actor(obs["policy"]))
            d = base.robot.data
            body_steps = base._sg_steps_under(d.root_pos_w.torch[:, :2])
            tips = base._paw_tips()
            front = torch.stack([base._sg_steps_under(tips[:, i, :2]) for i in (0, 1)], -1)
            local_x = base._sg_tile_xy(tips[:, :2, :2].mean(1))[:, 0]
            top_riser = g.first_riser + (g.num_steps - 1) * base._sg_tread_col[t.terrain_types]
            near_edge = (local_x - top_riser) < 0.12  # front paws (facing -x) within 12 cm of the top edge
            on_landing = body_steps == g.num_steps
            phase_edge = on_landing & (near_edge | (front < g.num_steps).any(-1))
            phase_landing = on_landing & ~phase_edge
            phase_stairs = ~on_landing & (body_steps > 0)
            speed = torch.linalg.norm(d.root_lin_vel_b.torch[:, :2], dim=-1)
            refusing = phase_edge & (base.commands[:, 0] > 0.05) & (speed < 0.02)
            reached |= base.sg_leg >= 1
            live = active & ~reached & ~dones.bool()
            for p, m in zip(PHASES, (phase_landing, phase_edge, phase_stairs, refusing)):
                sel = (m & live).float()
                steps[p] += sel
                for k in keys:
                    sums[p][k] += (base._episode_sums[k] - prev[k]) / dt * sel
            prev = {k: base._episode_sums[k].clone() for k in keys}
            active &= ~dones.bool()
    env.close()
    out = {"run": str(run).replace("\\", "/"), "cells": [f"r{round(r * 100)} t{tr}" for r, tr in CELLS],
           "reached": [round(float(reached[cell == i].float().mean()), 3) for i in range(len(CELLS))], "phases": {}}
    for p in PHASES:
        total = float(steps[p].sum())
        out["phases"][p] = {"seconds": round(total * dt, 1),
                            "terms": {k: round(float(sums[p][k].sum()) / max(total, 1.0), 3) for k in keys}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    print(f"[descent_terms] {run.name}: reached the bottom " + ", ".join(f"{c} {100 * r:.0f} %" for c, r in zip(out["cells"], out["reached"])))
    print("  term (per second)    " + "".join(f"{p:>11}" for p in PHASES))
    for k in keys:
        vals = [out["phases"][p]["terms"][k] for p in PHASES]
        print(f"  {k:<20} " + "".join(f"{v:11.3f}" for v in vals))
    print("  seconds              " + "".join(f"{out['phases'][p]['seconds']:11.1f}" for p in PHASES))
sys.exit(0)
