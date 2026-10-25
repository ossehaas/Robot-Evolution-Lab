"""Can the dog get up? Every robot one episode from a fixed start: a quarter each standing, dropped upright, on its side,
on its back (legs anywhere in their range when fallen). Success = standing (upright, at standing height, at least
three paws carrying weight, nothing else touching) held for 0.5 s. Appends one JSON line per run to --out.

    lab tools/dog/getup_test.py runs/exp2_dog/14_getup/getup_s1 --out runs/exp2_dog/14_getup/getup_test.jsonl

Works for any dog run: a walking policy (trained without get-up) shows what it does when it finds itself on the floor.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation

from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

CATEGORIES = ("standing", "dropped upright", "on its side", "on its back")
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--seconds", type=float, default=8.0)
ap.add_argument("--set", action="append", default=[], help="extra environment setting, e.g. self_collisions=True")
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, False, task="command", terrain="flat",
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False),
                   terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
for item in args.set:
    import ast

    key, value = item.split("=", 1)
    setattr(cfg, key, ast.literal_eval(value))
cfg.getup, cfg.getup_category = True, -2  # robot i starts in category i % 4
cfg.command_standing_prob = 1.0
cfg.episode_length_s = args.seconds + 1.0

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    actor = torch.jit.load(str(run / "policy.pt"), map_location=base.device).eval()
    obs, _ = env.reset()
    n, dt = base.num_envs, base.step_dt
    held = torch.zeros(n, device=base.device)
    stood_at = torch.full((n,), float("nan"), device=base.device)
    upright_end = torch.zeros(n, dtype=torch.bool, device=base.device)
    with torch.inference_mode():
        for step in range(int(args.seconds / dt)):
            obs, _, _, _ = env.step(actor(obs["policy"]))
            held = torch.where(base.getup_standing, held + dt, torch.zeros_like(held))
            new = (held >= 0.5) & torch.isnan(stood_at)
            stood_at[new] = (step + 1) * dt - 0.5
        upright_end = base.getup_standing.clone()
    cat = base.getup_category
    line = {"run": str(run).replace("\\", "/"), "seconds": args.seconds, "settings": args.set}
    for k, name in enumerate(CATEGORIES):
        sel = cat == k
        ok = ~torch.isnan(stood_at[sel])
        t = stood_at[sel][ok]
        line[name] = {"success": round(float(ok.float().mean()), 3),
                      "median_s": round(float(t.median()), 2) if len(t) else None,
                      "standing_at_end": round(float(upright_end[sel].float().mean()), 3)}
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print("[getup_test]", run.name, " | ".join(f"{k}: {v['success']:.0%} ({v['median_s']} s)" for k, v in line.items()
                                            if isinstance(v, dict)))
    env.close()
sys.exit(0)
