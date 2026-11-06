"""Replay the quasi-static stair plans (tools/dog/stairs_plan.py) in the simulator (journal step 41): the proof in
physics. No policy: every dog gets servo targets from the plan at 50 Hz. Body shifts are
interpolated in body pose with the standing paws held by inverse kinematics; a swing moves its paw along the planned
lift / over / down path; both slow down where a joint would turn faster than --max_joint_speed. Feedback (default):
a standing paw is taken where it really touched down (x measured at first contact, height = the tread), and every step
the joint angles put the body in its planned pose relative to the standing paws (shifted with them), so small slips do
not add up (--open_loop: the planned joint angles as they are). Each plan runs on its own stairs tile (row = rise, column = tread with open sides) from the
planned start pose. --copies 0: one dog per plan, nominal (no randomization); --copies k: k dogs per plan with the
training randomization (masses +-15 %, servo gains +-25 %, paw grip 0.7-1.0) to see how robust the plan is.

Climbed = at the end all four paws are on the last segment (landing up / floor down) and the dog has never fallen
(tilted > 35 deg or the body touching the ground). Also reported: the largest deviation of the body from the plan
(height, pitch). Appends one JSON line per plan to --out; with --video one plan is filmed from the side.

    lab tools/dog/stairs_replay.py runs/exp2_dog/19_stairs_proof/plans --out runs/exp2_dog/19_stairs_proof/replay.jsonl
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
from tools.dog.stairs_plan import LEGS, Dog2D, Planner, Terrain  # noqa: E402

DOG = Path("runs/exp2_dog/17_recover_bc/bc_s2")  # its config: servo model, joint set, action scale

ap = argparse.ArgumentParser()
ap.add_argument("plans", type=Path, help="directory with the plan JSON files (or one file)")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--copies", type=int, default=0, help="0 = one nominal dog per plan; k = k randomized dogs per plan")
ap.add_argument("--shift_s", type=float, default=0.6)
ap.add_argument("--swing_s", type=float, default=0.8)
ap.add_argument("--settle_s", type=float, default=2.0, help="standing still after the last step")
ap.add_argument("--max_joint_speed", type=float, default=3.0, help="[rad/s] slower where needed (servo: 4.7 rad/s)")
ap.add_argument("--open_loop", action="store_true",
                help="play the planned joint angles as they are (default: feedback, the standing paws are held where "
                     "they really landed and the body is put in its planned pose above them)")
ap.add_argument("--trace", action="store_true", help="print dog 0's body against the plan every 0.25 s")
ap.add_argument("--video", default=None, help="film this plan (file name, e.g. up_r6_t9.json) from the side")
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

files = [args.plans] if args.plans.is_file() else sorted(args.plans.glob("*_r*_t*.json"))
plans = [(f.name, json.loads(f.read_text())) for f in files]
plans = [(n, p) for n, p in plans if p.get("feasible")]
import time  # noqa: E402

STARTED = time.time()
VIDEO_PLAN = args.video  # the plan to film (args.video becomes the recorder switch below)
if args.video:
    plans = [(n, p) for n, p in plans if n == args.video]
if not plans:
    raise SystemExit("no feasible plans to replay")
dog2d = Dog2D(str(DOG / "genome.json"))
DT = 0.02


def trajectory(plan: dict):
    """Per 20 ms step: joint targets (dict), planned body pose, planned paw centres (4 x (x, z)) and the swinging leg
    (-1 = none), in plan coordinates."""
    up = plan["dir"] == "up"
    terrain = Terrain.stairs(plan["rise"], plan["tread"], 5, up)
    planner = Planner(dog2d, terrain, 0.0)
    paws = list(plan["start_paws"])
    body = tuple(plan["start_body"])
    qs, bodies, targets, swings = [], [], [], []

    def paw_targets(paws_now, swing=None, xz=None):
        return [xz if i == swing else (paws_now[i], planner.paw_z(paws_now[i])) for i in range(4)]

    def solve(paws_now, body_now, swing=None, xz=None):
        pose = [np.array([v]) for v in body_now]
        q = {}
        for i, leg in enumerate(LEGS):
            px, pz = xz if i == swing else (paws_now[i], planner.paw_z(paws_now[i]))
            h, k, _ = dog2d.ik(leg, pose[2], pose[0], pose[1], np.array([px]), np.array([pz]))
            q[f"{leg}_j1"], q[f"{leg}_j2"] = float(h[0]), float(k[0])
        q["neck_j0"] = float(body_now[3])
        return q

    def timed(samples: list[tuple[dict, tuple]], seconds: float, tgts: list, swing: int) -> None:
        """Append fine samples (joint angles, body) at 20 ms steps: the motion takes at least `seconds`, and longer
        where a joint would otherwise turn faster than --max_joint_speed (near a stretched leg)."""
        q_prev = qs[-1]
        times = []
        for k, (q, _) in enumerate(samples):
            ref = q_prev if k == 0 else samples[k - 1][0]
            dq = max(abs(q[j] - ref[j]) for j in q)
            times.append(max(seconds / len(samples), dq / args.max_joint_speed))
        cum = np.concatenate(([0.0], np.cumsum(times)))
        pts = [(q_prev, bodies[-1])] + samples
        keys = list(q_prev)
        arr = np.array([[q[j] for j in keys] for q, _ in pts])
        barr = np.array([list(b) for _, b in pts])
        tarr = np.array([np.ravel(targets[-1])] + [np.ravel(t_) for t_ in tgts])
        for t in np.arange(DT, cum[-1] + 1e-9, DT):
            qs.append(dict(zip(keys, (float(np.interp(t, cum, arr[:, i])) for i in range(len(keys))))))
            bodies.append(tuple(float(np.interp(t, cum, barr[:, i])) for i in range(4)))
            targets.append(np.array([np.interp(t, cum, tarr[:, i]) for i in range(8)]).reshape(4, 2))
            swings.append(swing)

    qs.append(solve(paws, body))
    bodies.append(body)
    targets.append(np.array(paw_targets(paws)))
    swings.append(-1)
    for f in plan["frames"]:
        target = tuple(f["body"])
        fine = []
        for s_ in np.linspace(0, 1, 41)[1:]:
            b = tuple(body[i] + s_ * (target[i] - body[i]) for i in range(4))
            fine.append((solve(paws, b), b))
        timed(fine, args.shift_s, [paw_targets(paws)] * len(fine), -1)
        body = target
        i = LEGS.index(f["swing"])
        path = [tuple(p) for p in f["path"]] + [(f["to"], planner.paw_z(f["to"]))]
        start = (paws[i], planner.paw_z(paws[i]))
        pts = [start] + path
        seg = np.cumsum([0.0] + [math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])])
        fine, tg = [], []
        for s_ in np.linspace(0, seg[-1], 161)[1:]:
            k = min(int(np.searchsorted(seg, s_, side="right")) - 1, len(pts) - 2)
            u = (s_ - seg[k]) / max(seg[k + 1] - seg[k], 1e-9)
            xz = (pts[k][0] + u * (pts[k + 1][0] - pts[k][0]), pts[k][1] + u * (pts[k + 1][1] - pts[k][1]))
            fine.append((solve(paws, body, swing=i, xz=xz), body))
            tg.append(paw_targets(paws, swing=i, xz=xz))
        timed(fine, args.swing_s, tg, i)
        paws[i] = f["to"]
    for _ in range(round(args.settle_s / DT)):
        qs.append(qs[-1])
        bodies.append(body)
        targets.append(targets[-1])
        swings.append(-1)
    return qs, bodies, targets, swings


res = json.loads((DOG / "result.json").read_text())
usd = build_robot_files((DOG / "genome.json").resolve(), DOG / "robot")
spec = build_robot(Genome.load(DOG / "genome.json"))
randomized = args.copies > 0 and not args.video
per_plan = args.copies if randomized else 1
num_envs = len(plans) * per_plan
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), num_envs, randomized,
                   task="command", terrain="dog_stairs", sensors=res["sensors"], obs_history=res["obs_history"],
                   height_scan=res.get("height_scan", False))
apply_overrides(cfg, {k: v for k, v in res.get("env_overrides", {}).items() if not k.startswith("amp_")})
cfg.recover, cfg.topple_push, cfg.getup = True, (), False
cfg.recover_max_down_s = 1e9
cfg.curriculum = False
if cfg.events is not None:
    cfg.events.push = None  # no shoves: the plan alone
    cfg.paw_friction_range = (0.7, 1.0)
trajs = [trajectory(p) for _, p in plans]
steps = max(len(t[0]) for t in trajs)
cfg.episode_length_s = steps * DT + 5.0
if args.video:
    cfg.viewer.eye = (0.15, -0.95, 0.45)
    from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video

    args.video, args.video_length, args.video_interval = True, steps, None
    enable_cameras_for_video(args)
    apply_video_recording(cfg, str(Path(args.out).parent), args, subdir="replay_video")

with launch_simulation(cfg, args):
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from core.sim.locomotion_env import EvoLocomotionEnv
    from core.sim.terrain import STAIRS_RISES, STAIRS_TREADS

    env = RslRlVecEnvWrapper(EvoLocomotionEnv(cfg, render_mode="rgb_array" if args.video else None), clip_actions=None)
    base = env.unwrapped
    dev, n = base.device, base.num_envs
    g = base._sg_geo
    # tiles: row = rise, column = the open-sided staircase of the tread
    names = list(cfg.terrain.terrain_generator.sub_terrains)
    plan_of = torch.arange(n, device=dev) // per_plan
    rows = torch.tensor([STAIRS_RISES.index(round(p["rise"], 3)) for _, p in plans], device=dev)[plan_of]
    cols = torch.tensor([names.index(f"goal_stairs_t{round(p['tread'] * 100)}") for _, p in plans], device=dev)[plan_of]
    t = base.terrain
    base.curriculum_enabled = False
    t.terrain_levels[:], t.terrain_types[:] = rows, cols
    t.env_origins[:] = t.terrain_origins[rows, cols]
    obs, _ = env.reset()
    # plan coordinates -> world: up = +x from the first riser, down = -x from the top riser (the dog faces -x)
    ups = torch.tensor([p["dir"] == "up" for _, p in plans], device=dev)[plan_of]
    treads = torch.tensor([p["tread"] for _, p in plans], device=dev)[plan_of]
    edge = torch.where(ups, torch.full_like(treads, g.first_riser), g.first_riser + (g.num_steps - 1) * treads)
    sign = torch.where(ups, 1.0, -1.0)
    origin = t.env_origins

    def world_pose(body):  # body (n, 4): x, z, pitch, neck in plan coordinates
        x = origin[:, 0] - g.size[0] / 2 + edge + sign * body[:, 0]
        z = origin[:, 2] + body[:, 1]
        yaw = math.pi * (~ups).float()
        q = quat_from_euler_xyz(torch.zeros_like(x), -body[:, 2], yaw)
        return torch.cat((torch.stack((x, origin[:, 1], z), -1), q), -1)

    def pad(xs):
        return xs + [xs[-1]] * (steps - len(xs))

    bodies = torch.tensor([[list(b) for b in pad(t[1])] for t in trajs], device=dev)[plan_of]
    rows_np = plan_of.cpu().numpy()
    paw_plan = np.stack([np.stack(pad(t[2])) for t in trajs])[rows_np]  # (n, steps, 4, 2)
    swing_plan = np.array([pad(t[3]) for t in trajs])[rows_np]  # (n, steps)
    act_names = [j for j in base.obs_joint_names if j not in cfg.expression_joints]
    jr = dict(zip(base.robot.joint_names, base.joint_range.tolist()))
    scale = cfg.action_scale

    # all servo targets up front, as actions (P, steps, joints) in the policy's joint order
    table = torch.tensor([[[q.get(j, 0.0) / (jr[j] * scale) for j in act_names] for q in pad(t[0])] for t in trajs],
                         device=dev)

    edge_np, sign_np = edge.cpu().numpy(), sign.cpu().numpy()
    ox, oz = (origin[:, 0] - g.size[0] / 2).cpu().numpy(), origin[:, 2].cpu().numpy()
    act_col = {j: i for i, j in enumerate(act_names)}
    # feedback state: x of every standing paw where it really touched down (plan coordinates); a paw whose swing has
    # ended but has not touched yet keeps pushing to its planned foothold
    state = {"anchor_x": None, "pending": np.zeros((n, 4), bool)}
    terrains = [Terrain.stairs(p["rise"], p["tread"], 5, p["dir"] == "up") for _, p in plans]
    plan_np = plan_of.cpu().numpy()

    def ground_z(x):  # (n, 4) paw centre height on the tread under x
        z = np.zeros_like(x)
        for pi, t in enumerate(terrains):
            sel = plan_np == pi
            z[sel] = t.height(x[sel]) + dog2d.paw_r
        return z

    def measured_paws():
        tips = base._paw_tips().cpu().numpy()  # (n, 4, 3) sphere centres, in LEGS order (the foot links' order)
        return sign_np[:, None] * (tips[..., 0] - ox[:, None] - edge_np[:, None])

    def actions_at(k):
        open_loop = table[plan_of, k]
        if args.open_loop:
            return open_loop
        mx = measured_paws()
        forces = base.contact.data.net_forces_w.torch[:, base._foot_sensor_ids]
        contact = (torch.linalg.norm(forces, dim=-1) > 1.0).cpu().numpy()
        if state["anchor_x"] is None:
            state["anchor_x"] = mx.copy()
        swing_now = swing_plan[:, k]
        if k > 0:
            prev = swing_plan[:, k - 1]
            ended = (prev >= 0) & (swing_now != prev)
            state["pending"][np.nonzero(ended)[0], prev[ended]] = True
        touched = state["pending"] & contact
        state["anchor_x"] = np.where(touched, mx, state["anchor_x"])
        state["pending"] &= ~contact
        ax = np.where(state["pending"], paw_plan[:, k, :, 0], state["anchor_x"])
        if args.trace and k % 12 == 0:
            print(f"[paws] t {k * DT:5.2f}s swing {swing_now[0]}  measured " + " ".join(f"{v:+.3f}" for v in mx[0])
                  + "  used " + " ".join(f"{v:+.3f}" for v in ax[0]) + "  plan " + " ".join(f"{v:+.3f}" for v in paw_plan[0, k, :, 0])
                  + f"  contact {contact[0].astype(int).tolist()} pending {state['pending'][0].astype(int).tolist()}", flush=True)
        az = np.where(state["pending"], paw_plan[:, k, :, 1], ground_z(ax))
        body = bodies[:, k].cpu().numpy().copy()
        # the body follows the paws: shifted by how far the standing paws are from their planned footholds, so the
        # centre of gravity stays where the plan put it relative to the support (height and pitch as planned)
        standing = (swing_now[:, None] != np.arange(4)[None, :]) & ~state["pending"]
        offset = np.where(standing, ax - paw_plan[:, k, :, 0], 0.0).sum(1) / np.maximum(standing.sum(1), 1)
        body[:, 0] += offset
        out = open_loop.clone()
        for i, leg in enumerate(LEGS):
            swing = swing_now == i
            px = np.where(swing, paw_plan[:, k, i, 0], ax[:, i])
            pz = np.where(swing, paw_plan[:, k, i, 1], az[:, i])
            h, kn, ok = dog2d.ik(leg, body[:, 2], body[:, 0], body[:, 1], px, pz)
            okt = torch.tensor(ok, device=dev)
            for name, v in ((f"{leg}_j1", h), (f"{leg}_j2", kn)):
                col = act_col[name]
                fb = torch.tensor(v, dtype=torch.float32, device=dev) / (jr[name] * scale)
                out[:, col] = torch.where(okt, fb, out[:, col])
        return out

    start = world_pose(bodies[:, 0])
    ids = torch.arange(n, device=dev)
    base.robot.write_root_pose_to_sim_index(root_pose=start, env_ids=ids)
    base.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    a0 = actions_at(0)
    full = torch.zeros(n, base.robot.num_joints, device=dev)
    full[:, base._act_ids] = a0 * base.joint_range[base._act_ids] * scale
    base.robot.write_joint_state_to_sim_index(position=full, velocity=torch.zeros_like(full), env_ids=ids)
    base.robot.actuators.target_command.set_position_index(value=full, env_ids=ids)
    base.actions[:] = 0.0
    base.actions[:, base._act_ids] = a0
    base.prev_actions[:] = base.actions
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    dz = torch.zeros(n, device=dev)
    dpitch = torch.zeros(n, device=dev)
    with torch.inference_mode():
        for k in range(steps):
            obs, _, _, _ = env.step(actions_at(k))
            d = base.robot.data
            fell |= base.fallen
            fwd = quat_apply(d.root_quat_w.torch, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
            nose_up = torch.asin(fwd[:, 2].clamp(-1.0, 1.0))
            dz = torch.maximum(dz, (d.root_pos_w.torch[:, 2] - origin[:, 2] - bodies[:, k, 1]).abs())
            dpitch = torch.maximum(dpitch, (nose_up - bodies[:, k, 2]).abs())
            if args.trace and k % 12 == 0:
                plan_x = float(world_pose(bodies[:, k])[0, 0])
                print(f"[trace] t {k * DT:5.2f}s  x {float(d.root_pos_w.torch[0, 0]) - plan_x:+.3f} (vs plan)  "
                      f"z {float(d.root_pos_w.torch[0, 2] - origin[0, 2]):.3f} / {float(bodies[0, k, 1]):.3f}  "
                      f"nose up {math.degrees(float(nose_up[0])):+5.1f} / {math.degrees(float(bodies[0, k, 2])):+5.1f} deg  "
                      f"fallen {bool(base.fallen[0])}", flush=True)
        tips = base._paw_tips()
        paw_steps = torch.stack([base._sg_steps_under(tips[:, i, :2]) for i in range(4)], -1)
        target = torch.where(ups, float(g.num_steps), 0.0).unsqueeze(-1)
        arrived = (paw_steps == target).all(-1)
    env.close()
    ok = arrived & ~fell
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for p_i, (name, plan) in enumerate(plans):
            sel = plan_of == p_i
            line = {"plan": name, "dir": plan["dir"], "rise": plan["rise"], "tread": plan["tread"],
                    "randomized": randomized, "climbed": float(ok[sel].float().mean()),
                    "fell": float(fell[sel].float().mean()), "arrived": float(arrived[sel].float().mean()),
                    "max_height_error_m": round(float(dz[sel].max()), 3), "max_pitch_error_deg": round(math.degrees(float(dpitch[sel].max())), 1),
                    "seconds": round(len(trajs[p_i][0]) * DT, 1)}
            f.write(json.dumps(line) + "\n")
            print(f"[stairs_replay] {name} ({'randomized' if randomized else 'nominal'}): climbed {line['climbed']:.0%}, "
                  f"fell {line['fell']:.0%}, height error {line['max_height_error_m']} m, "
                  f"pitch error {line['max_pitch_error_deg']} deg")
    if args.video:
        clips = sorted((p for p in (Path(args.out).parent / "videos" / "replay_video").glob("*.mp4")
                        if p.stat().st_mtime >= STARTED), key=lambda p: p.stat().st_mtime)
        if clips:
            final = Path(args.out).parent / "videos" / f"replay_{Path(VIDEO_PLAN).stem}.mp4"
            final.parent.mkdir(parents=True, exist_ok=True)
            if len(clips) == 1:
                clips[0].replace(final)
            else:  # the recorder splits long recordings: join them
                from moviepy.editor import VideoFileClip, concatenate_videoclips

                parts = [VideoFileClip(str(c)) for c in clips]
                concatenate_videoclips(parts).write_videofile(str(final), fps=50, codec="libx264", audio=False, logger=None)
                for part, c in zip(parts, clips):
                    part.close()
                    c.unlink()
            print(f"video: {final}")
sys.exit(0)
