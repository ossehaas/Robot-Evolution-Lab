"""Stairs goal test (journal step 40): the climbing-limit table. Every dog walks one straight staircase, up from the
floor to the landing or down from the landing to the floor, driven by the navigation layer (turn towards the goal, then
walk at it at --speed). The course (terrain dog_stairs) has one tile per rise (rows, 1-10 cm) and tread x open sides /
walls (columns, treads 25-9 cm); robots are spread evenly over all tiles and both directions, each runs --rounds
episodes of 30 s. Falls are soft (recover mode: the dog may get up and go on); falling down the stairs ends the episode
(hard). Per (direction, rise, tread, sides), the first leg of each episode:

    reached       share that reached the goal (within 30 s)
    clean         reached without any fall: no problems at all
    any_fall      share with at least one fall on the way (soft or hard)
    hard          fell down the stairs (more than one step down while fallen, or off an open side)
    stayed_down   fell and did not get up within 6 s
    time_s        median time to the goal of those that reached it
    progress      mean steps made good (of 5)

Limits: per direction and tread, the highest rise up to which every rise is done clean by at least 1 - --max_problems of
the dogs. Appends one JSON line {run, speed, cells, limits} to --out and prints the tables.

    lab tools/dog/stairs_goal_test.py runs/exp2_dog/17_recover_bc/bc_s2 --out runs/exp2_dog/18_stairs_goal/stairs_goal.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=2000)  # 100 tiles x 2 directions x 10
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--speed", type=float, default=0.20, help="walking speed the navigation layer asks for [m/s]")
ap.add_argument("--max_problems", type=float, default=0.10, help="share of dogs with a problem allowed at the limit")
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, False, task="command", terrain="dog_stairs",
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False),
                   terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.recover, cfg.topple_push, cfg.getup = True, (), False
cfg.episode_length_s = 30.0
cfg.map_dropout_rate = 0.0
cfg.stairs_goal_speed, cfg.stairs_goal_dir = args.speed, 0

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    dev = base.device
    if (run / "student.pt").exists():
        from core.evolution.distill_memory import RecurrentPolicy

        policy = RecurrentPolicy(torch.jit.load(str(run / "student.pt"), map_location=dev).eval(), base)
    else:
        actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()

        class Policy:
            def __call__(self, obs):
                return actor(obs["policy"])

            def reset(self, dones):
                pass

        policy = Policy()
    base.prepare_evaluation()
    obs, _ = env.reset()
    base.finished_episodes.zero_()
    counted = torch.zeros(base.num_envs, dtype=torch.long, device=dev)
    records = []
    with torch.inference_mode():
        for _ in range(args.rounds * (int(base.max_episode_length) + 2)):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            newly = (base.finished_episodes > counted) & (counted < args.rounds)
            if bool(newly.any()):
                records.append({k: v[newly].clone() for k, v in base.sg_last.items()})
            counted = torch.minimum(base.finished_episodes, torch.full_like(counted, args.rounds))
            if bool((counted >= args.rounds).all()):
                break
    env.close()
    rec = {k: torch.cat([r[k] for r in records]).cpu() for k in records[0]}
    rise_cm, tread_cm = (rec["rise"] * 100).round(), (rec["tread"] * 100).round()
    clean = (rec["reached"] > 0) & (rec["falls"] == 0) & (rec["hard"] == 0)

    def cell(sel: torch.Tensor) -> dict:
        n = int(sel.sum())
        times = rec["time"][sel & (rec["reached"] > 0)]
        return {"n": n, "reached": round(float(rec["reached"][sel].mean()), 3), "clean": round(float(clean[sel].float().mean()), 3),
                "any_fall": round(float((rec["falls"][sel] > 0).float().mean()), 3),
                "falls_per_episode": round(float(rec["falls"][sel].mean()), 2),
                "hard": round(float(rec["hard"][sel].mean()), 3), "stayed_down": round(float(rec["stayed_down"][sel].mean()), 3),
                "time_s": round(float(times.median()), 1) if len(times) else None,
                "progress": round(float(rec["progress"][sel].mean()), 2)}

    cells, limits = [], {}
    for d, dname in ((1.0, "up"), (-1.0, "down")):
        for t in sorted(set(tread_cm.tolist()), reverse=True):
            limit = 0
            for r in sorted(set(rise_cm.tolist())):
                both = (rec["dir"] == d) & (tread_cm == t) & (rise_cm == r)
                if not bool(both.any()):
                    continue
                for walls in (0.0, 1.0):
                    sel = both & (rec["walls"] == walls)
                    if bool(sel.any()):
                        cells.append({"dir": dname, "rise_cm": r, "tread_cm": t, "walls": bool(walls),
                                      "slope_deg": round(math.degrees(math.atan2(r, t)), 1), **cell(sel)})
                cells.append({"dir": dname, "rise_cm": r, "tread_cm": t, "walls": None,
                              "slope_deg": round(math.degrees(math.atan2(r, t)), 1), **cell(both)})
                if limit == r - 1 and float(clean[both].float().mean()) >= 1.0 - args.max_problems:
                    limit = int(r)
            limits[f"{dname}_t{int(t)}"] = limit
    line = {"run": str(run).replace("\\", "/"), "speed_m_s": args.speed, "rounds": args.rounds,
            "max_problems": args.max_problems, "cells": cells, "limits_rise_cm": limits}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    for dname in ("up", "down"):
        rows = [c for c in cells if c["dir"] == dname and c["walls"] is None]
        treads = sorted({c["tread_cm"] for c in rows}, reverse=True)
        print(f"[stairs_goal_test] {run.name} {dname}: clean share per rise (rows) and tread (columns, cm)")
        print("  rise " + " ".join(f"{t:>6.0f}" for t in treads))
        for r in sorted({c["rise_cm"] for c in rows}):
            vals = {c["tread_cm"]: c["clean"] for c in rows if c["rise_cm"] == r}
            print(f"  {r:4.0f} " + " ".join(f"{100 * vals.get(t, float('nan')):5.0f}%" for t in treads))
    print("[stairs_goal_test] limits (rise in cm, clean >= "
          f"{100 * (1 - args.max_problems):.0f} %):", limits)
sys.exit(0)
