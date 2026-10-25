"""How good is the terrain map? A trained dog walks the course (real walking commands, all difficulty rows) while the
map is built from its chest sensor. Per scan region (ahead of the front paws, under the body, behind it) it reports
the share of scan points the map knows and its height error there, with memory and with only the latest frame.
Saves a picture of a few moments (true scan / map / latest frame only) for dogs on the stairs.

    lab tools/dog/map_check.py runs/exp2_dog/11_imitation/amp_w6m_slow_s1 --terrain_map depth --out runs/scratch/map_check

A dog trained with the perfect scan keeps walking on it (the map is only built alongside); a dog trained on the map
walks on its map. Also the quick end-to-end test of the map code (few dogs, short).
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

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--terrain_map", choices=("depth", "tof"), default=None, help="default: the run's own")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--dark", type=float, default=0.0, help="sensor blackouts per second")
ap.add_argument("--forward", action="store_true", help="walking commands within +-30 deg of straight ahead")
ap.add_argument("--set", action="append", default=[], help="extra environment setting, e.g. map_odom_scale_error=0.1")
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True
args.out.mkdir(parents=True, exist_ok=True)

run = args.run
res = json.loads((run / "result.json").read_text())
kind = args.terrain_map or res.get("terrain_map", "") or "depth"
own_map = bool(res.get("terrain_map", ""))
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, True, task="command",
                   terrain=res.get("terrain_arg", "dog_rough"), sensors=res["sensors"], obs_history=res["obs_history"],
                   height_scan=True, terrain_map=kind)
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.map_policy_input = own_map and kind == res.get("terrain_map")
cfg.heading_hold_gain = 2.0
cfg.map_dropout_rate = args.dark
if args.forward:
    cfg.eval_heading_range = (-0.52, 0.52)
for item in args.set:
    import ast

    key, value = item.split("=", 1)
    setattr(cfg, key, ast.literal_eval(value))

with launch_simulation(cfg, args):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    actor = torch.jit.load(str(run / "policy.pt"), map_location=base.device).eval()
    base.prepare_evaluation()
    gen = cfg.terrain.terrain_generator
    names = list(gen.sub_terrains)
    cum = torch.cumsum(torch.tensor([gen.sub_terrains[n].proportion for n in names]), 0)
    cum = cum / cum[-1]
    col_type = [names[int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
    q = base._map_query  # scan points in the heading frame (x forward)
    half = spec.genome.body.length / 2
    regions = {"ahead": q[:, 0] > half, "under": q[:, 0].abs() <= half, "behind": q[:, 0] < -half}
    stats = {f"{m}_{r}_{k}": 0.0 for m in ("memory", "latest") for r in regions for k in ("known", "err_cm", "n")}
    clip, p = cfg.height_scan_clip, cfg.height_scan_points
    obs, _ = env.reset()
    shots = []
    with torch.inference_mode():
        for step in range(args.steps):
            obs, _, _, _ = env.step(actor(obs["policy"]))
            if step < 25:
                continue  # the map is still empty at the start
            truth = base.scan_truth * clip
            for m, view in (("memory", base.map_obs), ("latest", base.map_view_obs)):
                h, known = view[:, :p] * clip, view[:, p:] > 0.5
                for r, sel in regions.items():
                    k = known[:, sel]
                    stats[f"{m}_{r}_known"] += float(k.float().mean())
                    err = (h[:, sel] - truth[:, sel]).abs()
                    stats[f"{m}_{r}_err_cm"] += float((err * k).sum() / k.sum().clamp(min=1)) * 100
                    stats[f"{m}_{r}_n"] += 1
            if step in (100, 250, 400):
                stair = [i for i in range(base.num_envs) if col_type[int(base.terrain.terrain_types[i])].startswith("stairs")
                         and int(base.terrain.terrain_levels[i]) >= 5][:2] or [0]
                for i in stair:
                    shots.append((step, i, base.scan_truth[i].cpu(), base.map_obs[i].cpu(), base.map_view_obs[i].cpu()))
    summary = {}
    for m in ("memory", "latest"):
        for r in regions:
            n = max(1.0, stats[f"{m}_{r}_n"])
            summary[f"{m}_{r}_known"] = round(stats[f"{m}_{r}_known"] / n, 3)
            summary[f"{m}_{r}_err_cm"] = round(stats[f"{m}_{r}_err_cm"] / n, 2)
    line = {"run": str(run).replace("\\", "/"), "sensor": kind, "policy_on_map": cfg.map_policy_input, "dark": args.dark,
            "forward": args.forward,
            "settings": args.set, **summary}
    with open(args.out / "map_check.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print("[map_check]", json.dumps(line))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pat = cfg.height_scanner.pattern_cfg
    nx, ny = round(pat.size[0] / pat.resolution) + 1, round(pat.size[1] / pat.resolution) + 1
    fig, axes = plt.subplots(len(shots), 3, figsize=(9, 2.2 * len(shots)), squeeze=False)
    for row, (step, i, truth, mem, view) in enumerate(shots):
        for col, (title, v) in enumerate((("true ground", truth), ("map (memory)", mem), ("latest frame only", view))):
            h = v[:p].clone() * clip * 100
            if col > 0:
                h[v[p:] < 0.5] = float("nan")
            # the grid is ordered x fastest (pattern "xy"): reshape to (y, x); x forward = right, y left = up
            img = (-h).reshape(ny, nx).flip(0).numpy()  # ground height relative to standing (up = positive)
            ax = axes[row, col]
            im = ax.imshow(img, cmap="viridis", vmin=-8, vmax=8, extent=(q[:, 0].min().item(), q[:, 0].max().item(),
                                                                          q[:, 1].min().item(), q[:, 1].max().item()))
            ax.axvline(half, color="w", lw=0.5)
            ax.axvline(-half, color="w", lw=0.5)
            ax.set_title(f"{title} (dog {i}, t={step * base.step_dt:.1f}s)", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=axes, shrink=0.6, label="ground height [cm], grey = unknown")
    png = args.out / f"map_{run.name}_{kind}{'_forward' if args.forward else ''}.png"
    fig.savefig(png, dpi=110)
    print(f"[map_check] picture: {png}")
    env.close()
sys.exit(0)
