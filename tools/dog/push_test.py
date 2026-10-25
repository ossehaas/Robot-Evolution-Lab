"""How hard can the dog be pushed? Robots walk straight on at 20 cm/s on a flat floor (heading hold on); after 3 s each
gets one shove (a sudden sideways/forwards/backwards velocity change in a random direction, no other randomization).
The robots are split over shove strengths; falls in the 4 s after the shove are counted. Appends one JSON line.

    lab tools/dog/push_test.py runs/exp2_dog/13_stairs/base_s1 --out runs/exp2_dog/14_getup/push_test.jsonl
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

STRENGTHS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)  # [m/s]; for scale: walking speed 0.2, the robust recipe used 0.3-0.4
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--num_envs", type=int, default=1022)  # a multiple of len(STRENGTHS)
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
cfg.heading_hold_gain = 2.0
cfg.map_dropout_rate = 0.0
cfg.episode_length_s = 30.0  # no time-outs during the test

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
    base.command_override = torch.tensor([0.2, 0.0, 0.0], device=dev)
    n = base.num_envs
    strength = torch.tensor(STRENGTHS, device=dev)[torch.arange(n, device=dev) % len(STRENGTHS)]
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    fell_before = torch.zeros(n, dtype=torch.bool, device=dev)
    obs, _ = env.reset()
    push_step, end_step = int(3.0 / base.step_dt), int(7.0 / base.step_dt)
    with torch.inference_mode():
        for step in range(end_step):
            if step == push_step:
                angle = torch.rand(n, device=dev) * 2 * math.pi
                vel = torch.cat((base.robot.data.root_lin_vel_w.torch, base.robot.data.root_ang_vel_w.torch), dim=-1).clone()
                vel[:, 0] += strength * torch.cos(angle)
                vel[:, 1] += strength * torch.sin(angle)
                base.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=torch.arange(n, device=dev))
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            down = dones.bool() & base.reset_terminated
            if step < push_step:
                fell_before |= down
            else:
                fell |= down & ~fell_before
    line = {"run": str(run).replace("\\", "/"), "walking_m_s": 0.2,
            "fell_before_push": round(float(fell_before.float().mean()), 3),
            "fall_rate_by_push_m_s": {f"{s:.1f}": round(float(fell[(strength == s) & ~fell_before].float().mean()), 3)
                                      for s in STRENGTHS}}
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print("[push_test]", run.name, line["fall_rate_by_push_m_s"])
    env.close()
sys.exit(0)
