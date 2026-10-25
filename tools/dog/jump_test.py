"""Scripted maximum jump (no learning): stand, crouch, then all hips and knees to full extension at once.

    lab tools/dog/jump_test.py STS3215_12V      (or FAST2, FAST4: hypothetical servos 2x / 4x faster)
"""
import argparse, math, sys
from dataclasses import replace
import torch
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.morphology import components as C
from core.sim.locomotion_env_cfg import make_env_cfg

variant = sys.argv[1] if len(sys.argv) > 1 else "STS3215_12V"
base = C.SERVOS["STS3215_12V"]
# same motor, lower gear ratio: k x faster, k x weaker, reflected inertia / k^2 (it scales with the ratio squared)
for k in (2, 4):
    C.SERVOS[f"GEAR{k}"] = replace(base, name=f"same motor, gear ratio / {k}", max_speed=k * base.max_speed,
                                   stall_torque=base.stall_torque / k, armature=base.armature / k**2,
                                   kp=base.kp / k, kd=base.kd / k**1.5)
# quasi-direct-drive leg motor (small BLDC + ~6:1 planetary, as in research robot dogs): fast, strong, low inertia
C.SERVOS["QDD"] = replace(base, name="quasi-direct-drive BLDC", max_speed=30.0, stall_torque=3.0, armature=3e-4,
                          kp=20.0, kd=0.3)
# sensitivity: the STS3215 if its reflected inertia were 3x lower than my estimate
C.SERVOS["STS_LOWARM"] = replace(base, name="STS3215, armature / 3", armature=base.armature / 3)
parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args([])
g = Path("prototypes/puppy/genome.json")
usd = build_robot_files(g.resolve(), Path("runs/scratch/jump_test/robot"))
spec = build_robot(Genome.load(g))
spec.servo = variant
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 16, False, task="command", sensors="imu", obs_history=1)
cfg.action_scale = 1.0
cfg.episode_length_s = 20.0
cfg.reset_yaw_range = 0.0
cfg.fall_up_threshold = -1.0  # do not reset whatever happens
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv
    env = EvoLocomotionEnv(cfg)
    env.expression_override = torch.zeros(1, len(env._expr_ids), device=env.device)
    env.command_override = torch.zeros(3, device=env.device)
    env.reset()
    names = [env.robot.joint_names[i] for i in env._act_ids.tolist()]
    rng = env.joint_range[env._act_ids]

    def pose(knee_fold: float):
        """Paw under the hip; knee folded by knee_fold rad from the rest bend (negative = towards straight)."""
        t = torch.zeros(len(names), device=env.device)
        for k, n in enumerate(names):
            front = n[1] == "0"
            if n.endswith("_j2"):
                t[k] = -knee_fold if front else knee_fold
            elif n.endswith("_j1"):
                t[k] = knee_fold / 2 if front else -knee_fold / 2
        return (t / rng).clamp(-1, 1).expand(env.num_envs, -1)

    stand_z = None
    peak, airborne_steps, z0 = 0.0, 0, 0.0
    for step in range(int(3.0 / env.step_dt)):
        t = step * env.step_dt
        if t < 1.0:
            a = pose(0.0)
        elif t < 1.8:
            a = pose(0.75)  # crouch
        else:
            a = pose(-0.95)  # explode to (almost) straight legs
        env.step(a)
        z = env.robot.data.root_pos_w.torch[:, 2].mean().item()
        if 0.9 < t < 1.0:
            stand_z = z
        if 1.7 < t < 1.8:
            z0 = z
        if t >= 1.8:
            peak = max(peak, z)
            f = env.contact.data.net_forces_w.torch[:, env._foot_sensor_ids]
            if bool((torch.linalg.norm(f, dim=-1) < 1.0).all(dim=-1).float().mean() > 0.5):
                airborne_steps += 1
    print(f"{variant}: standing {1000 * stand_z:.0f} mm, crouched {1000 * z0:.0f} mm, peak {1000 * peak:.0f} mm "
          f"-> body rises {1000 * (peak - stand_z):.0f} mm above standing; all paws off the ground for "
          f"{1000 * airborne_steps * env.step_dt:.0f} ms")
    env.close()
