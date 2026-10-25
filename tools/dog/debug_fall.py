import argparse, torch
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import make_env_cfg

parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args([])
g = Path("prototypes/puppy/genome.json")
usd = build_robot_files(g.resolve(), Path("runs/scratch/debug_fall/robot"))
spec = build_robot(Genome.load(g))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 4, True, task="command", sensors="imu", obs_history=8)
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv
    env = EvoLocomotionEnv(cfg)
    env.reset()
    print("body names:", env.robot.body_names[:4], "... root:", env.robot.body_names[0])
    print("joint order:", env.robot.joint_names)
    for step in range(40):
        a = torch.zeros(4, cfg.action_space, device=env.device)
        _, _, term, trunc, _ = env.step(a)
        d = env.robot.data
        if step % 5 == 0 or term.any():
            print(step, "z", [round(v, 3) for v in d.root_pos_w.torch[:, 2].tolist()],
                  "up", [round(v, 3) for v in (-d.projected_gravity_b.torch[:, 2]).tolist()], "term", term.tolist())
    env.close()
