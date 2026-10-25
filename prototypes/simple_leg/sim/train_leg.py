"""Smallest complete RL example: the bolted 2-servo leg learns to put its foot on a random target point.

    python train_leg.py                         # train headless (~1-2 min), then report the reach error
    python train_leg.py --viz kit --num_envs 64 # watch training in the viewport (targets are red spheres)
    python train_leg.py --play --viz kit        # replay the saved policy

Checkpoints go to prototypes/simple_leg/train/. The walking robots use the same building blocks at a larger
scale: see core/sim/locomotion_env.py and core/evolution/train.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the simple leg to reach target points.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=150)
parser.add_argument("--play", action="store_true", help="skip training and replay the saved policy")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab import cloner  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply, sample_uniform  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

HERE = Path(__file__).resolve().parent
USD_PATH = str(HERE / "simple_leg" / "simple_leg.usda").replace("\\", "/")
LOG_DIR = HERE.parent / "train"
CALF_TIP = (0.0, 0.0, -0.10)  # foot tip in the calf link frame (URDF: 0.1 m cylinder below the knee)
BASE_HEIGHT = 0.30


@configclass
class LegReachEnvCfg(DirectRLEnvCfg):
    episode_length_s = 3.0
    decimation = 4
    action_space = 2
    observation_space = 2 + 2 + 3 + 3 + 2
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=0.4, replicate_physics=True)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Leg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, fix_root_link=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, BASE_HEIGHT), joint_pos={".*": 0.0}),
        actuators={
            "servos": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=2.0,
                damping=0.05,
                armature=3.0e-4,
                friction=0.01,
                joint_effort_limit=0.14,
                joint_velocity_limit=10.0,
            )
        },
    )
    joint_range = 1.4  # [rad] commanded range, slightly inside the +-1.57 URDF limits


class LegReachEnv(DirectRLEnv):
    cfg: LegReachEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.calf = self.leg.find_bodies("calf_link")[0][0]
        self.base = self.leg.find_bodies("base_link")[0][0]
        self.tip = torch.tensor(CALF_TIP, device=self.device).repeat(self.num_envs, 1)
        self.target = torch.zeros(self.num_envs, 3, device=self.device)
        self.actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

    def _setup_scene(self):
        self.leg = Articulation(self.cfg.robot)
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0("/World/envs/env_0", "/World/envs/env_{}", self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        self.scene.articulations["leg"] = self.leg
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/targets",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.01, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1))
                    )
                },
            )
        )

    def _foot(self) -> torch.Tensor:
        d = self.leg.data
        return d.body_pos_w.torch[:, self.calf] + quat_apply(d.body_quat_w.torch[:, self.calf], self.tip)

    def _pre_physics_step(self, actions):
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1.0, 1.0))
        self.leg.actuators.target_command.set_position_index(value=self.actions * self.cfg.joint_range)

    def _apply_action(self):
        pass

    def _get_observations(self):
        d = self.leg.data
        base = d.body_pos_w.torch[:, self.base]
        obs = torch.cat(
            (
                d.joint_pos.torch / self.cfg.joint_range,
                d.joint_vel.torch * 0.1,
                (self._foot() - base) * 10.0,
                (self.target - base) * 10.0,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        dist = torch.linalg.norm(self._foot() - self.target, dim=-1)
        rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
        return (-dist * 10.0 + (dist < 0.01).float() - 0.05 * rate) * self.step_dt

    def _get_dones(self):
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(timeout), timeout

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        pose = self.leg.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.leg.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        q = sample_uniform(-0.5, 0.5, (len(env_ids), 2), self.device)
        self.leg.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=env_ids)
        # target: forward kinematics of a random reachable pose, so every target can actually be reached
        hip = sample_uniform(-1.2, 1.2, (len(env_ids),), self.device)
        knee = sample_uniform(-1.2, 1.2, (len(env_ids),), self.device)
        hip_pos = pose[:, :3] + torch.tensor([0.0, 0.0, -0.025], device=self.device)
        # hip and knee rotate about +y: a link hanging along -z swings towards -x for positive angles
        knee_pos = hip_pos + 0.1 * torch.stack((-torch.sin(hip), torch.zeros_like(hip), -torch.cos(hip)), dim=-1)
        a = hip + knee
        self.target[env_ids] = knee_pos + 0.1 * torch.stack((-torch.sin(a), torch.zeros_like(a), -torch.cos(a)), dim=-1)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.markers.visualize(translations=self.target)


@configclass
class LegPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 150
    save_interval = 50
    experiment_name = "simple_leg_reach"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[64, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def main():
    env_cfg = LegReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if not args_cli.play else min(args_cli.num_envs, 16)
    env_cfg.sim.device = args_cli.device or "cuda:0"
    env = RslRlVecEnvWrapper(LegReachEnv(env_cfg), clip_actions=1.0)
    agent_cfg = handle_deprecated_rsl_rl_cfg(LegPPOCfg(), metadata.version("rsl-rl-lib"))
    agent_cfg.max_iterations = args_cli.iterations
    agent_cfg.device = env.unwrapped.device
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(LOG_DIR), device=agent_cfg.device)
    checkpoint = LOG_DIR / "leg_reach.pt"
    if args_cli.play:
        runner.load(str(checkpoint))
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        runner.save(str(checkpoint))
        print(f"[simple_leg] saved {checkpoint}")

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    obs, _ = env.reset()
    steps = int(base.max_episode_length) - 2
    while simulation_app.is_running():
        with torch.inference_mode():
            for _ in range(steps):
                obs, _, _, _ = env.step(policy(obs))
            err = torch.linalg.norm(base._foot() - base.target, dim=-1)
        print(f"[simple_leg] reach error at episode end: mean {1000 * err.mean():.1f} mm, worst {1000 * err.max():.1f} mm")
        if not args_cli.play or args_cli.visualizer is None:
            break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
