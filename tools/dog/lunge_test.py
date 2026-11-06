"""Lunge and gate test (journal step 45): does the dog climb with the stair plans' weight shift, and does the gate switch?

Stairs mode (default, terrain dog_stairs): every dog climbs one straight staircase up (navigation layer, one episode of
--seconds), spread over all rises and treads that have a stairs-up plan. In the plan zone (the body between the plan's
first and last keyframe, not fallen) per step:

    shift_front_cm  body x minus the paws' centre while exactly one front paw is up (plan: behind the paws, < 0)
    shift_rear_cm   the same while exactly one rear paw is up (plan: ahead of the paws, > 0)
    three_down      share of the zone time with at least 3 paws down (the plan's crawl; a trot has 2)
    pitch_deg       body pitch, nose up (plan: 10-25 deg on the steep stairs)
    rear_height_cm  body above the standing rear paws' soles while exactly one front paw is up: do the legs on the lower
                    step stretch to level the body? (owner, 2026-11-03; the plan's value per cell for reference)
    gate_zone       mean gate in the zone; gate_before / gate_after: on the floor before it, on the landing after it
                    (only for an expert model: policy.pt with gate_value)

and over the whole climb (going up, not fallen): stumble, the share of paw-steps with a paw pushed sideways more than
twice as hard as it is pressed down (a toe against a riser, as the env's stumble term), and stuck_per_climb, how often a
paw stays like that for more than 0.3 s in a row (owner, 2026-10-31: the soft paw front sticks under the step edge).

plus climbed (the landing reached without a fall) and, per cell, the plan's own shift_front / shift_rear for reference.
Course mode (--course, terrain dog_rough, walking commands): the share of steps with the gate open (> 0.5) on the
course, where it should stay shut. Appends one JSON line per run and mode to --out.

    lab tools/dog/lunge_test.py runs/exp2_dog/23_stairs_expert/plan_s1 --out runs/exp2_dog/23_stairs_expert/lunge.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.evolution.train import build_robot_files  # noqa: E402
from core.morphology import Genome, build_robot  # noqa: E402
from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg  # noqa: E402
from tools.dog.stairs_plan import Terrain  # noqa: E402

PLANS = "runs/exp2_dog/19_stairs_proof/plans_all"
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=1000)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--plans", default=PLANS)
ap.add_argument("--course", action="store_true", help="gate on the walking course instead of the stairs")
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

run = args.run
res = json.loads((run / "result.json").read_text())
usd = build_robot_files((run / "genome.json").resolve(), run / "robot")
spec = build_robot(Genome.load(run / "genome.json"))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, False, task="command",
                   terrain="dog_rough" if args.course else "dog_stairs", sensors=res["sensors"],
                   obs_history=res["obs_history"], height_scan=res.get("height_scan", False),
                   terrain_map=res.get("terrain_map", ""))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.recover, cfg.topple_push, cfg.getup = True, (), False
cfg.episode_length_s = args.seconds
cfg.map_dropout_rate = 0.0
cfg.stairs_goal_speed, cfg.stairs_goal_dir = args.speed, 1
cfg.plan_track_plans, cfg.expert_label_obs = args.plans, not args.course

with launch_simulation(cfg, args):
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg), clip_actions=None)
    base = env.unwrapped
    dev, n = base.device, base.num_envs
    actor = torch.jit.load(str(run / "policy.pt"), map_location=dev).eval()
    has_gate = hasattr(actor, "gate_value")
    base.prepare_evaluation()
    obs, _ = env.reset()
    steps = int(args.seconds / base.step_dt) - 2
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    acc = {k: z() for k in ("front_sum", "front_n", "rear_sum", "rear_n", "zone_n", "three_n", "pitch_sum",
                            "gz_sum", "gb_sum", "gb_n", "ga_sum", "ga_n", "open_n", "all_n", "stumble_n", "climb_n",
                            "stuck_n", "rear_h_sum")}
    stumble_run = torch.zeros(n, 4, device=dev)  # seconds each paw has been stumbling in a row
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        for _ in range(steps):
            policy_obs = obs["policy"]
            gate = actor.gate_value(policy_obs) if has_gate else None
            obs, _, _, _ = env.step(actor(policy_obs))
            d = base.robot.data
            if args.course:
                if gate is not None:
                    acc["open_n"] += (gate > 0.5).float()
                acc["all_n"] += 1.0
                continue
            fell |= base.fallen
            reached |= base.sg_leg >= 1
            paw_f = base.contact.data.net_forces_w.torch[:, base._foot_sensor_ids]
            in_contact = torch.linalg.norm(paw_f, dim=-1) > 1.0
            side = torch.linalg.norm(paw_f[..., :2], dim=-1)
            climbing = (base.stairs_goal_on() & (base.sg_dir > 0) & ~base.fallen).unsqueeze(-1)
            stumble = (side > 2.0 * paw_f[..., 2].abs()) & (side > 1.0) & climbing
            acc["stumble_n"] += stumble.float().sum(dim=-1)
            acc["climb_n"] += 4.0 * climbing.squeeze(-1).float()
            before_run = stumble_run.clone()
            stumble_run = torch.where(stumble, stumble_run + base.step_dt, torch.zeros_like(stumble_run))
            acc["stuck_n"] += ((stumble_run > 0.3) & (before_run <= 0.3)).float().sum(dim=-1)
            zone, pid, x = base.plan_zone()
            g = base._sg_geo
            x0 = base.terrain.env_origins[:, 0] - g.size[0] / 2 + g.first_riser
            shift = x - (d.body_pos_w.torch[:, base._foot_body_ids, 0] - x0.unsqueeze(-1)).mean(dim=-1)
            one_up = (~in_contact).sum(dim=-1) == 1
            rear_up = (~in_contact[:, base._rear_feet]).any(dim=-1)
            front1, rear1 = (zone & one_up & ~rear_up).float(), (zone & one_up & rear_up).float()
            acc["front_sum"] += shift * front1
            acc["front_n"] += front1
            radius = torch.tensor(cfg.foot_radius, device=dev)[base._rear_feet]
            soles = (base._paw_tips()[:, base._rear_feet, 2] - radius).mean(dim=-1)
            acc["rear_h_sum"] += (d.root_pos_w.torch[:, 2] - soles) * front1
            acc["rear_sum"] += shift * rear1
            acc["rear_n"] += rear1
            acc["zone_n"] += zone.float()
            acc["three_n"] += (zone & (in_contact.sum(dim=-1) >= 3)).float()
            _, pitch, _ = euler_xyz_from_quat(d.root_quat_w.torch)
            acc["pitch_sum"] -= torch.atan2(torch.sin(pitch), torch.cos(pitch)) * zone.float()  # nose up
            if gate is not None:
                pt = base._pt
                on_up = base.stairs_goal_on() & (base.sg_dir > 0) & ~base.fallen
                before = on_up & ~zone & (x < pt["x_min"][pid])
                after = (on_up & ~zone & (x > pt["x_max"][pid])) | (base.stairs_goal_on() & (base.sg_dir < 0))
                acc["gz_sum"] += gate * zone.float()
                acc["gb_sum"] += gate * before.float()
                acc["gb_n"] += before.float()
                acc["ga_sum"] += gate * after.float()
                acc["ga_n"] += after.float()
        if not args.course:
            types, levels = base.terrain.terrain_types, base.terrain.terrain_levels
            rise = base._sg_rises[levels]
            tread = base._sg_tread_col[types]
            walls = base._sg_walls_col[types]
            has_plan = base._pt["index"][levels, base._pt["tread_col"][types]] >= 0
    env.close()

    line = {"run": str(run).replace("\\", "/"), "mode": "course" if args.course else "stairs", "gate": has_gate}
    if args.course:
        line["gate_open_share"] = round(float(acc["open_n"].sum() / acc["all_n"].sum()), 4) if has_gate else None
        print(f"[lunge_test] {run.name} course: gate open {line['gate_open_share']}")
    else:
        def ratio(a, b, sel, scale=1.0, nd=2):
            den = float(acc[b][sel].sum())
            return round(scale * float(acc[a][sel].sum()) / den, nd) if den > 0 else None

        plan_ref = {}
        for f in (f for folder in args.plans.split(",") for f in sorted(Path(folder).glob("up_r*_t*.json"))):
            p = json.loads(f.read_text())
            if p.get("feasible") and (round(100 * p["rise"]), round(100 * p["tread"])) not in plan_ref:  # the first folder wins
                fr = p["frames"]
                ref = {}
                for name, rear in (("shift_front_cm", False), ("shift_rear_cm", True)):
                    vals = [x["body"][0] - sum(x["paws_before"]) / 4 for x in fr if (x["swing"][1] == "1") == rear]
                    ref[name] = round(100 * sum(vals) / len(vals), 2) if vals else None
                ground = Terrain.stairs(p["rise"], p["tread"], 5, True)
                vals = [x["body"][1] - float(ground.height(np.array(x["paws_before"][2:])).mean()) for x in fr if x["swing"][1] == "0"]
                ref["rear_height_cm"] = round(100 * sum(vals) / len(vals), 2) if vals else None
                plan_ref[(round(100 * p["rise"]), round(100 * p["tread"]))] = ref
        cells = []
        rise_cm, tread_cm = (rise * 100).round().cpu(), (tread * 100).round().cpu()
        for r in sorted(set(rise_cm.tolist())):
            for t in sorted(set(tread_cm.tolist()), reverse=True):
                sel = ((rise_cm == r) & (tread_cm == t)).to(dev) & has_plan & ~walls
                if not bool(sel.any()):
                    continue
                cells.append({"rise_cm": r, "tread_cm": t, "n": int(sel.sum()),
                              "climbed": round(float((reached & ~fell)[sel].float().mean()), 3),
                              "stumble": ratio("stumble_n", "climb_n", sel, 1.0, 4),
                              "stuck_per_climb": round(float(acc["stuck_n"][sel].mean()), 2),
                              "shift_front_cm": ratio("front_sum", "front_n", sel, 100.0),
                              "shift_rear_cm": ratio("rear_sum", "rear_n", sel, 100.0),
                              "rear_height_cm": ratio("rear_h_sum", "front_n", sel, 100.0),
                              "three_down": ratio("three_n", "zone_n", sel, 1.0, 3),
                              "pitch_deg": ratio("pitch_sum", "zone_n", sel, 180.0 / math.pi, 1),
                              "gate_zone": ratio("gz_sum", "zone_n", sel, 1.0, 3) if has_gate else None,
                              "gate_before": ratio("gb_sum", "gb_n", sel, 1.0, 3) if has_gate else None,
                              "gate_after": ratio("ga_sum", "ga_n", sel, 1.0, 3) if has_gate else None,
                              "plan": plan_ref.get((int(r), int(t)), {})})
        line["cells"] = cells
        print(f"[lunge_test] {run.name} stairs up: rise tread climbed | shift front / rear cm (plan) | 3 down | gate")
        for c in cells:
            pl = c["plan"]
            print(f"  {c['rise_cm']:2.0f} {c['tread_cm']:3.0f}  {c['climbed']:5.2f} | {c['shift_front_cm']} / "
                  f"{c['shift_rear_cm']} ({pl.get('shift_front_cm')} / {pl.get('shift_rear_cm')}) | {c['three_down']} | "
                  f"{c['gate_before']} {c['gate_zone']} {c['gate_after']} | stumble {c['stumble']} "
                  f"stuck {c['stuck_per_climb']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
