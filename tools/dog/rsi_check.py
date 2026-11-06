"""Check of the mid-climb starts (stairs_rsi_*, journal step 41): dogs on the stairs goal course start in random poses
of the stair plans and hold them (servo targets = the start pose) for 1 s. Reports how far the paws are from the step
surface under them right after the start (should be ~0: on the treads, not floating, not inside) and the share still
upright after holding the pose (the plan poses are statically stable, so nearly all). Appends one JSON line to --out;
exit code 1 if the check fails.

    lab tools/dog/rsi_check.py --plans runs/exp2_dog/19_stairs_proof/plans --out runs/exp2_dog/19_stairs_proof/rsi_check.jsonl
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

DOG = Path("runs/exp2_dog/17_recover_bc/bc_s2")
ap = argparse.ArgumentParser()
ap.add_argument("--plans", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=300)
ap.add_argument("--dog", type=Path, default=DOG, help="the dog to start in the plans' poses (its paw may differ from the plans')")
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True
DOG = args.dog

res = json.loads((DOG / "result.json").read_text())
usd = build_robot_files((DOG / "genome.json").resolve(), DOG / "robot")
spec = build_robot(Genome.load(DOG / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, False, task="command", terrain="dog_stairs",
                   sensors=res["sensors"], obs_history=res["obs_history"], height_scan=res.get("height_scan", False))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.recover, cfg.topple_push, cfg.getup = True, (), False
cfg.recover_max_down_s = 1e9
cfg.stairs_rsi_plans, cfg.stairs_rsi_prob = args.plans, 1.0
cfg.curriculum = False

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    base.spread_over_terrain_levels()  # rows (rises) spread; training mode, so the mid-climb starts apply
    t = base.terrain
    # only tiles that have a plan (rows up to the highest planned rise)
    idx = base._sg_rsi["index"]
    planned_rows = int((idx >= 0).any(0).any(-1).nonzero().max()) + 1
    t.terrain_levels[:] = torch.arange(base.num_envs, device=base.device) % planned_rows
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
    obs, _ = env.reset()
    hold = base.actions[:, base._act_ids].clone()
    mid = base.sg_mid_climb.clone()
    fell = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    with torch.inference_mode():
        for k in range(int(1.0 / base.step_dt)):
            env.step(hold)
            fell |= base.fallen
            if k == 0:  # after one step: the link poses are updated from the new state
                tips = base._paw_tips()
                rise = base._sg_rises[t.terrain_levels]
                ground = torch.stack([base._sg_steps_under(tips[:, i, :2]) for i in range(4)], -1) * rise.unsqueeze(-1)
                gap = (tips[..., 2] - base._foot_radius - t.env_origins[:, 2:3] - ground)[mid]  # (mid-climb dogs, 4)
    dogs = base.num_envs
    env.close()
    line = {"plans": args.plans, "dogs": dogs, "mid_climb": int(mid.sum()),
            "upright_after_1s_normal_start": round(1 - float(fell[~mid].float().mean()), 3) if bool((~mid).any()) else None,
            "paw_gap_mm_mean": round(float(gap.abs().mean()) * 1000, 1),
            "paw_gap_mm_p95": round(float(torch.quantile(gap.abs().flatten(), 0.95)) * 1000, 1),
            "paw_gap_mm_min": round(float(gap.min()) * 1000, 1), "upright_after_1s": round(1 - float(fell[mid].float().mean()), 3)}
    ok = line["paw_gap_mm_p95"] < 8.0 and line["upright_after_1s"] >= 0.8
    line["ok"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print("[rsi_check]", line)
sys.exit(0 if ok else 1)
