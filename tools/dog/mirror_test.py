"""Physical check of core.sim.symmetry: dog B receives the mirrored actions of dog A; its observations must equal
the mirrored observations of A."""
import argparse, math, torch
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import make_env_cfg

parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args([])
g = Path("prototypes/puppy/genome.json")
usd = build_robot_files(g.resolve(), Path("runs/scratch/mirror_test/robot"))
spec = build_robot(Genome.load(g))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 2, False, task="command", sensors="imu", obs_history=1)
cfg.reset_yaw_range = 0.0; cfg.reset_joint_fraction = 0.0
for k in ("gravity_noise", "gyro_noise", "accel_noise", "gyro_bias", "accel_bias"):
    setattr(cfg, k, 0.0)
cfg.scene.env_spacing = 3.0
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv
    from core.sim.symmetry import _layout
    env = EvoLocomotionEnv(cfg)
    env.expression_override = torch.zeros(1, len(env._expr_ids), device=env.device)
    env.command_override = torch.tensor([[0.2, 0.1, 0.5], [0.2, -0.1, -0.5]], device=env.device)
    obs, _ = env.reset()
    env._phase[1] = math.pi
    lay = _layout(env)
    pp, ps = lay["policy"]; ap, asg = lay["actions"]
    # segment names of one frame, in order
    n = env.robot.num_joints
    segs = [("gravity", 3), ("gyro", 3), ("accel", 3), ("joint_pos", n), ("last_action", n), ("clock", 2), ("command", 3)]
    worst = {s: 0.0 for s, _ in segs}
    for step in range(150):
        t = step * env.step_dt
        a = torch.stack([0.6 * torch.sin(torch.arange(12, device=env.device) * 0.7 + 2 * math.pi * 1.3 * t)])
        acts = torch.cat((a, a[:, ap] * asg), 0)
        obs, _, _, _, _ = env.step(acts)
        o = obs["policy"]
        diff = (o[1] - o[0][pp] * ps).abs()
        off = 0
        for s, k in segs:
            worst[s] = max(worst[s], diff[off:off + k].max().item())
            off += k
        if step in (2, 5, 10, 20, 30):
            A = o[0][pp] * ps
            print("step", step, "gyro A(mirrored)", [round(v, 3) for v in A[3:6].tolist()], "B", [round(v, 3) for v in o[1][3:6].tolist()])
            print("        accel A(mirrored)", [round(v, 3) for v in A[6:9].tolist()], "B", [round(v, 3) for v in o[1][6:9].tolist()])
            print("        grav  A(mirrored)", [round(v, 3) for v in A[0:3].tolist()], "B", [round(v, 3) for v in o[1][0:3].tolist()])
            print("        root y A/B", [round(v, 4) for v in (env.robot.data.root_pos_w.torch[:, 1] - env.scene.env_origins[:, 1]).tolist()], "reset?", env.episode_length_buf.tolist())
    seg_detail = (o[1] - o[0][pp] * ps)
    names = env.robot.joint_names
    jp = seg_detail[9:9 + n]
    print("joint_pos diff per joint:", {nm: round(v, 3) for nm, v in zip(names, jp.tolist())})
    env.close()
