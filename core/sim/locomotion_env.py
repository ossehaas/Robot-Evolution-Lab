"""Locomotion tasks for any generated robot.

Import this module only after the simulator has been launched (it loads Isaac Sim's USD libraries);
the configuration lives in locomotion_env_cfg.py.

The policy commands servo angles at 50 Hz (the SG90 PWM rate). Two tasks (``cfg.task``):

    forward  walk along +x as fast as possible while staying upright and not wasting energy
    command  follow a velocity command (forward, sideways, turn rate) in the robot's own frame, so the
             robot can be steered; commands change every few seconds and include standing still

Observations are split so the trained policy can run on the real robot:

    policy (what the ESP32 can measure): IMU gravity vector + gyro, servo angles, last action, gait clock
                                         (+ the velocity command in the command task)
    critic (privileged, training only):  policy obs + body velocity, chassis height, joint velocities
"""

from __future__ import annotations

import inspect
import math
from collections import deque
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu, RayCaster
from isaaclab.utils.math import (euler_xyz_from_quat, quat_apply, quat_from_angle_axis, quat_from_euler_xyz, quat_mul,
                               sample_uniform)

from .locomotion_env_cfg import EvoLocomotionEnvCfg

# Isaac Lab 3.0.0 GA builds the scene from its config (assets and sensors declared on cfg.scene, cloned by the scene
# itself); 3.0.0-EA leaves the cloning to the environment (its grid_transforms still takes a device)
DECLARED_SCENE = "device" not in inspect.signature(cloner.grid_transforms).parameters
SCENE_ENTITIES = ("terrain", "robot", "height_scanner", "tof", "map_camera", "imu", "contact_sensor")

DOG_GAITS = ("trot", "walk_trot", "free")  # dog-style training; the gait pattern term differs (see _get_rewards)


class EvoLocomotionEnv(DirectRLEnv):
    cfg: EvoLocomotionEnvCfg

    def __init__(self, cfg: EvoLocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        events = cfg.events
        if events is not None and getattr(events, "push", None) is not None:
            v = cfg.push_velocity
            events.push.interval_range_s = tuple(cfg.push_interval)
            events.push.params = {"velocity_range": {"x": (-v, v), "y": (-v, v), "yaw": (-2 * v, 2 * v)}}
        if events is not None and cfg.paw_friction_range:
            # floors from slippery tiles to grippy rubber: the paws' friction (the floor multiplies with 1.0)
            from isaaclab.envs import mdp
            from isaaclab.managers import EventTermCfg, SceneEntityCfg

            lo, hi = cfg.paw_friction_range
            events.paw_friction = EventTermCfg(
                func=mdp.randomize_rigid_body_material, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names=list(cfg.foot_body_names)),
                        "static_friction_range": (lo, hi), "dynamic_friction_range": (0.85 * lo, 0.85 * hi),
                        "restitution_range": (0.0, 0.0), "num_buckets": 64, "make_consistent": True})
        if cfg.self_collisions:
            cfg.robot.spawn.articulation_props.enabled_self_collisions = True
        if cfg.policy_drives_expression and cfg.expression_joints:
            cfg.action_space += len(cfg.expression_joints)  # neck, head and tail become policy actions
            cfg.expression_joints = []
        gen = getattr(cfg.terrain, "terrain_generator", None)
        if cfg.stair_tread_range and gen is not None:
            # stairs with treads shortening over the difficulty rows (a copy: the generator config is shared)
            from isaaclab.terrains.trimesh import mesh_terrains

            from .terrain import tread_stairs

            subs = dict(gen.sub_terrains)
            for name, sub in subs.items():
                if name.startswith("stairs"):
                    base = (mesh_terrains.inverted_pyramid_stairs_terrain if "Inverted" in type(sub).__name__
                            else mesh_terrains.pyramid_stairs_terrain)
                    subs[name] = sub.replace(function=tread_stairs(base, *cfg.stair_tread_range))
            cfg.terrain.terrain_generator = gen.replace(sub_terrains=subs)
        if cfg.terrain_map and not cfg.map_policy_input and not getattr(cfg, "_map_space_fixed", False):
            cfg.observation_space -= cfg.height_scan_points  # the policy gets the perfect scan, without known flags
            cfg._map_space_fixed = True
        if cfg.skill_signal_obs and not getattr(cfg, "_skill_space_fixed", False):
            cfg.observation_space += 3  # walk, stairs up, stairs down: after everything else
            cfg._skill_space_fixed = True
        if DECLARED_SCENE:
            from isaaclab.assets import AssetBaseCfg

            for name in SCENE_ENTITIES:
                setattr(cfg.scene, name, getattr(cfg, name))
            cfg.scene.light = AssetBaseCfg(
                prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)))
        super().__init__(cfg, render_mode, **kwargs)
        n = self.robot.num_joints
        # joints are ordered by the physics backend; look ranges up by name
        name_to_range = dict(zip(self.cfg.joint_names, self.cfg.joint_ranges))
        self.joint_range = torch.tensor([name_to_range[j] for j in self.robot.joint_names], device=self.device)
        self.actions = torch.zeros(self.num_envs, n, device=self.device)  # all joints, incl. expression joints
        self.prev_actions = torch.zeros_like(self.actions)
        expr = set(cfg.expression_joints)
        # passive joints (spring hinges of a flexible back) are neither driven nor observed by the policy. The policy
        # sees the other joints in a fixed order (by depth in the joint tree, then design order), so a policy trained
        # on the stiff body fits the flexible one although the physics engine orders the joints differently.
        passive = set(cfg.passive_joints)
        names = list(self.robot.joint_names)
        if passive:
            design = {j: k for k, j in enumerate(cfg.joint_names)}
            order = sorted((j for j in names if j not in passive), key=lambda j: (cfg.joint_depth[j], design[j]))
        else:
            order = names
        self.obs_joint_names = order
        self._obs_ids = torch.tensor([names.index(j) for j in order], device=self.device)
        self._act_ids = torch.tensor([names.index(j) for j in order if j not in expr], device=self.device)
        self._expr_ids = torch.tensor([i for i, j in enumerate(self.robot.joint_names) if j in expr], dtype=torch.long,
                                      device=self.device)
        # policy actions are in the observation's joint order: map them onto the robot's joints unless that is identical
        self._remap_actions = not torch.equal(self._act_ids, torch.arange(n, device=self.device))
        # per expression joint: offset, amplitude, frequency [Hz], phase; and when the next motion starts [s]
        self._expr_params = torch.zeros(self.num_envs, len(self._expr_ids), 4, device=self.device)
        self._expr_next = torch.zeros(self.num_envs, device=self.device)
        self.expression_override: torch.Tensor | None = None  # (num_envs, n_expression) in -1..1 of each range
        # servo target = action * joint range * scale (policy joints) or * 1 (expression joints)
        self._target_scale = torch.full((n,), cfg.action_scale, device=self.device)
        if len(self._expr_ids) > 0:
            self._target_scale[self._expr_ids] = 1.0
        self.start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # bookkeeping for fitness evaluation: distance covered in the last finished episode of each env
        self.last_episode_distance = torch.zeros(self.num_envs, device=self.device)
        self.last_episode_fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.finished_episodes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # per-term reward sums for logging; keys starting with "_" are diagnostics (logged as episode means)
        keys = ["upright", "energy", "action_rate", "joint_limit"] + (["action_limit"] if cfg.action_limit_weight > 0 else [])
        self._action_excess = torch.zeros(self.num_envs, device=self.device)
        if self.contact is not None:
            self._feet_ids, _ = self.contact.find_sensors(self.cfg.foot_body_names)
            keys.append("feet_air_time")
        if cfg.foot_clearance_weight > 0:
            self._foot_body_ids = [self.robot.find_bodies(n)[0][0] for n in cfg.foot_body_names]
            self._foot_offsets = torch.tensor(cfg.foot_offsets, device=self.device)
            self._foot_radius = torch.tensor(cfg.foot_radius, device=self.device)
            self._foot_ground_z = torch.zeros(self.num_envs, len(cfg.foot_body_names), device=self.device)
            keys.append("foot_clearance")
        if cfg.getup:
            keys += ["gu_upright", "gu_height", "gu_rise", "gu_stand", "gu_pose", "gu_stall", "_standing"]
            self._gu_height = torch.zeros(self.num_envs, device=self.device)  # last step's body height (rise term)
            self.getup_category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.getup_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if cfg.recover:
            keys += ["gu_upright", "gu_height", "gu_rise", "fall", "_fallen"]
            self._gu_height = torch.zeros(self.num_envs, device=self.device)
            self.getup_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # read by tests and videos
            self._was_fallen = torch.zeros_like(self.fallen)
            self._down_time = torch.zeros(self.num_envs, device=self.device)
        if cfg.topple_push:
            self._topple_next = torch.zeros(self.num_envs, device=self.device)  # episode time of the next shove [s]
        if cfg.plan_track_weight > 0 or cfg.plan_support_weight > 0 or cfg.expert_label_obs:
            keys += ["plan_track", "plan_support", "_plan_zone", "_plan_shift_front_cm", "_plan_shift_rear_cm"]
        keys += [k for k, w in (("stumble", cfg.stumble_weight), ("foothold", cfg.foothold_weight),
                                ("stairs_support", cfg.stairs_support_weight), ("lift_first", cfg.lift_first_weight))
                 if w > 0]
        # terrain type of every column of the course (stairs tiles, tiles without style reward)
        self._col_stairs = self._col_amp_skip = self._col_stairs_terms = None
        gen = getattr(cfg.terrain, "terrain_generator", None)
        if gen is not None and cfg.terrain_kind == "rough":
            names = list(gen.sub_terrains)
            cum = torch.cumsum(torch.tensor([gen.sub_terrains[k].proportion for k in names]), 0)
            cum = cum / cum[-1]
            col = [names[int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
            self._col_stairs = torch.tensor([c.startswith("stairs") for c in col], device=self.device)
            self._col_amp_skip = torch.tensor([c in cfg.amp_skip_types for c in col], device=self.device)
            if cfg.stairs_terms_types:
                self._col_stairs_terms = torch.tensor([c in cfg.stairs_terms_types for c in col], device=self.device)
            self._init_stairs_goal([gen.sub_terrains[c] for c in col], gen.num_rows, tuple(gen.size))
            self._init_tilt([gen.sub_terrains[c] for c in col], gen.num_rows)
        else:
            self._sg_col = self._tilt_col = None
        if self._sg_col is not None:
            keys += ["sg_goal", "sg_hard", "_sg_goals"]
            if cfg.stairs_margin_weight > 0:
                keys += ["sg_margin", "_sg_margin_cm"]
            if cfg.stairs_stall_weight > 0:
                keys += ["sg_stall"]
        if cfg.gait in DOG_GAITS:
            keys += ["trot", "base_height", "undesired_contact", "abduction_pose", "leg_clash", "stand_still", "flight",
                     "_paws_down"]
            jn = self.robot.joint_names
            # hip joints per side: positive angle swings a leg backwards, so front - rear = how far they close in
            self._clash_pairs = [(jn.index(f"{s}0_j1"), jn.index(f"{s}1_j1")) for s in "LR"]
            names = self.contact.body_names
            self._foot_sensor_ids = [names.index(f) for f in cfg.foot_body_names]
            # diagonal pairs: front-left + rear-right step together, then front-right + rear-left
            self._trot_sign = torch.tensor([1.0 if f[0] + f[1] in ("L0", "R1") else -1.0 for f in cfg.foot_body_names],
                                           device=self.device)
            walk_offset = {"L1": 0.0, "L0": 0.25, "R1": 0.5, "R0": 0.75}  # lateral-sequence walk: 3 paws always down
            trot_offset = {"L0": 0.0, "R1": 0.0, "R0": 0.5, "L1": 0.5}
            self._walk_offset = torch.tensor([walk_offset[f[:2]] for f in cfg.foot_body_names], device=self.device)
            self._trot_offset = torch.tensor([trot_offset[f[:2]] for f in cfg.foot_body_names], device=self.device)
            self._other_sensor_ids = [i for i, nm in enumerate(names) if nm not in cfg.foot_body_names]
            self._chassis_sensor_id = [names.index(b) for b in ("chassis", "rear") if b in names]
            self._abd_ids = [i for i, j in enumerate(self.robot.joint_names) if j.endswith("_j0") and j[0] in "LR"]
            # gait phase, advanced every step at a rate that grows with the commanded speed (slow = calm steps)
            self._phase = torch.zeros(self.num_envs, device=self.device)
        if cfg.task == "jump":
            keys += ["jump_height", "jump_rise", "jump_air", "_best_clearance", "_airborne"]
            names = self.contact.body_names
            self._foot_sensor_ids = [names.index(f) for f in cfg.foot_body_names]
            self._chassis_sensor_id = [names.index(b) for b in ("chassis", "rear") if b in names]
            self._foot_body_ids = [self.robot.find_bodies(f)[0][0] for f in cfg.foot_body_names]
            self._foot_offsets = torch.tensor(cfg.foot_offsets, device=self.device)
            self._foot_radius = torch.tensor(cfg.foot_radius, device=self.device)
            self._jump_best = torch.zeros(self.num_envs, device=self.device)  # best lowest-paw height this episode
            self._rise_best = torch.zeros(self.num_envs, device=self.device)  # best body rise this episode
            self.last_episode_jump = torch.zeros(self.num_envs, device=self.device)
            self.last_episode_rise = torch.zeros(self.num_envs, device=self.device)
            self.last_episode_air = torch.zeros(self.num_envs, device=self.device)
            self._air_steps = torch.zeros(self.num_envs, device=self.device)
        self.amp = None
        if cfg.amp_motion_file:
            from .amp import AmpDiscriminator

            # the style judge compares the legs only (the recording has no neck, head or tail, which the policy may
            # drive when policy_drives_expression is set)
            self._amp_ids = torch.tensor([i for i in self._act_ids.tolist()
                                          if self.robot.joint_names[i] not in ("neck_j0", "head_j1", "tail_j0")],
                                         device=self.device)
            act_names = [self.robot.joint_names[i] for i in self._amp_ids.tolist()]
            self.amp = AmpDiscriminator(cfg.amp_motion_file, act_names, self.device, mirror=cfg.amp_mirror)
            self._amp_prev = torch.zeros(self.num_envs, self.amp.dim // 2, device=self.device)
            self._amp_steps = 0
            self.amp_style = torch.zeros(self.num_envs, device=self.device)
            if cfg.amp_disc_file:
                self.amp.load(cfg.amp_disc_file)
                self.amp.frozen = True
            keys += ["style", "_style"]
        if cfg.task == "command":
            keys += ["track_lin", "track_ang", "progress", "vertical", "roll_pitch_rate", "_lin_vel_error", "_yaw_rate_error"]
        elif cfg.task == "goal":
            keys += ["goal_progress", "goal_reached", "collision", "_goals", "_collision"]
            if cfg.terrain_kind == "enclosure":
                keys.append("closeness")
        else:
            keys += ["forward", "lateral", "heading"]
        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in keys}
        # command task: (vx, vy, yaw rate) in the robot frame; set command_override to drive the robot yourself
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_override: torch.Tensor | None = None
        self._command_target = torch.zeros(self.num_envs, 3, device=self.device)  # commands move towards this
        # heading hold: the turn rate asked for (before the correction), the held heading and whether one is held
        self._yaw_asked = torch.zeros(self.num_envs, device=self.device)
        self._heading_target = torch.zeros(self.num_envs, device=self.device)
        self._heading_held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ranges = (cfg.command_vx_range, cfg.command_vy_range, cfg.command_yaw_range)
        self._command_scale = torch.tensor([1.0 / max(abs(v) for v in r) for r in ranges], device=self.device)
        self._resample_steps = max(1, int(round(cfg.command_resample_s / self.step_dt)))
        self.curriculum_enabled = cfg.curriculum
        self.eval_mode = False
        # sensor model state: per-robot biases, held optical-flow reading, observation history
        self._gyro_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._accel_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._flow_held = torch.zeros(self.num_envs, 2, device=self.device)
        self._obs_hist = None
        if cfg.obs_history > 1:
            extra = 0 if cfg.height_scan_in_history else cfg.height_scan_points
            if cfg.terrain_map and cfg.map_policy_input:
                extra *= 2  # map heights + known flags after the history
            extra += 3 if cfg.skill_signal_obs else 0
            frame = (cfg.observation_space - extra) // cfg.obs_history
            self._obs_hist = torch.zeros(self.num_envs, cfg.obs_history, frame, device=self.device)
        self._hist_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # goal task: goal positions, progress bookkeeping, per-episode counters (also read by the evaluation)
        if cfg.task == "goal":
            from .terrain import GOAL_SPOTS

            self._goal_spots = torch.tensor(GOAL_SPOTS, device=self.device)
            self.goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self._goal_prev_dist = torch.zeros(self.num_envs, device=self.device)
            self.episode_goals = torch.zeros(self.num_envs, device=self.device)
            self.episode_collision_steps = torch.zeros(self.num_envs, device=self.device)
            self.last_episode_goals = torch.zeros(self.num_envs, device=self.device)
            self.last_episode_collision = torch.zeros(self.num_envs, device=self.device)
            self.last_episode_seconds = torch.zeros(self.num_envs, device=self.device)
            self._tof_held = torch.ones(self.num_envs, 64, device=self.device)
            # enclosure: is the current goal inside the ring; where episodes ended (read by the evaluation)
            self.goal_unreachable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.goal_ideal_dist = torch.zeros(self.num_envs, device=self.device)
            self.episode_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
            self.last_episode_end_xy = torch.zeros(self.num_envs, 2, device=self.device)
            self.last_episode_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
            self.last_episode_goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
            self.last_episode_unreachable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._tof_tick = 0
        # hierarchical goal task: frozen walking policy + its own observation history
        self.low_level = None
        if cfg.low_level_policy:
            self.low_level = torch.jit.load(cfg.low_level_policy, map_location=self.device).eval()
            n_joints = self.robot.num_joints
            ll_frame = 3 + 3 + 3 + n_joints + n_joints + 2 + 3  # imu senses, joints, last action, clock, command
            self._ll_hist = torch.zeros(self.num_envs, cfg.low_level_history, ll_frame, device=self.device)
            self._ll_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._steer_scale = torch.tensor(
                [max(abs(v) for v in cfg.command_vx_range), max(abs(v) for v in cfg.command_vy_range),
                 max(abs(v) for v in cfg.command_yaw_range)], device=self.device)
        # terrain map: grid, drifting pose estimate, frames on their way into the map, sensor blackouts. Also read by
        # the memory student (core/evolution/distill_memory.py): map_obs (with memory), map_view_obs (latest frame
        # only), scan_truth (the perfect scan), obs_frame (this step's body senses, last action, clock, command)
        self.terrain_map = None
        self.map_obs = self.map_view_obs = self.scan_truth = self.obs_frame = None
        if cfg.terrain_map:
            from .terrain_map import OdometryError, TerrainMap

            self.terrain_map = TerrainMap(self.num_envs, cfg.map_cells, cfg.map_resolution, self.device)
            self.odometry = OdometryError(self.num_envs, self.device, cfg.map_odom_drift_xy, cfg.map_odom_drift_yaw,
                                          cfg.map_odom_drift_z, cfg.map_odom_scale_error)
            # the scan grid in the robot's heading frame, exactly as the perfect scanner casts it
            pat = cfg.height_scanner.pattern_cfg
            starts, _ = pat.func(pat, self.device)
            self._map_query = starts[:, :2] + torch.tensor(cfg.height_scanner.offset.pos[:2], device=self.device)
            self._map_cam_offset = torch.tensor(cfg.map_camera_offset, device=self.device)
            self._map_pending: deque = deque()
            self._map_tick = 0
            self._map_step = -1
            self._map_frame_t = -math.inf
            self._dark_until = torch.zeros(self.num_envs, device=self.device)

    def _init_tilt(self, col_cfgs: list, num_rows: int) -> None:
        """Tilted plates and pads (terrain.MeshTiltedPlateCfg): tilt angle per column and row, downhill direction per
        column. On these tiles the dog starts standing on the plate (perpendicular to it, facing a random way) and
        is told to stand still (base skill B2)."""
        from .terrain import MeshTiltedPlateCfg

        tilt = [isinstance(c, MeshTiltedPlateCfg) for c in col_cfgs]
        self._tilt_col = None
        if not any(tilt):
            return
        dev = self.device
        self._tilt_col = torch.tensor(tilt, device=dev)
        self._tilt_dir_col = torch.tensor([math.radians(getattr(c, "direction", 0.0)) for c in col_cfgs], device=dev)
        self._tilt_angle = torch.tensor(
            [[math.radians(c.angles[min(r, len(c.angles) - 1)]) if t else 0.0 for r in range(num_rows)]
             for c, t in zip(col_cfgs, tilt)], device=dev)  # (columns, rows)

    def tilt_on(self) -> torch.Tensor:
        """(E,) robots on a tilted plate or pad."""
        if self._tilt_col is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._tilt_col[self.terrain.terrain_types]

    def tilt_normal(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """(N, 3) up direction of the plate under each robot (straight up off the tilt tiles)."""
        types, levels = self.terrain.terrain_types, self.terrain.terrain_levels
        if env_ids is not None:
            types, levels = types[env_ids], levels[env_ids]
        a = torch.where(self._tilt_col[types], self._tilt_angle[types, levels], torch.zeros_like(levels, dtype=torch.float))
        phi = self._tilt_dir_col[types]
        return torch.stack((torch.sin(a) * torch.cos(phi), torch.sin(a) * torch.sin(phi), torch.cos(a)), dim=-1)

    def _init_stairs_goal(self, col_cfgs: list, num_rows: int, tile: tuple) -> None:
        """Stairs goal course (cfg.stairs_goal): geometry per column (tread, walls) and per row (rise) of the straight
        staircase tiles, and the per-robot state of the goal walk."""
        from .terrain import MeshStraightStairsCfg

        goal = [c for c in col_cfgs if isinstance(c, MeshStraightStairsCfg)]
        self._sg_col = None
        if not (self.cfg.stairs_goal and goal):
            return
        g = goal[0].replace(size=tile)
        if len(g.rises) != num_rows:
            raise ValueError(f"the stairs goal course needs one row per rise ({len(g.rises)}), the generator has {num_rows}")
        dev, n = self.device, self.num_envs
        self._sg_col = torch.tensor([isinstance(c, MeshStraightStairsCfg) for c in col_cfgs], device=dev)
        self._sg_tread_col = torch.tensor([getattr(c, "tread", 1.0) for c in col_cfgs], device=dev)
        self._sg_walls_col = torch.tensor([bool(getattr(c, "walls", False)) for c in col_cfgs], device=dev)
        self._sg_rises = torch.tensor(g.rises, device=dev)
        self._sg_geo = g  # num_steps, first_riser, landing_end, width, size (shared by all stairs goal tiles)
        self.goal_pos = torch.zeros(n, 3, device=dev)  # current goal (world), also drawn in videos
        self.sg_dir = torch.ones(n, device=dev)  # +1 = this leg goes up, -1 = down
        self.sg_first_dir = torch.ones(n, device=dev)
        self.sg_leg = torch.zeros(n, dtype=torch.long, device=dev)  # goals reached this episode
        self._sg_speed = torch.full((n,), self.cfg.stairs_goal_speed, device=dev)
        # first leg (what the tests measure): time to the goal, falls, steps made good, fell down the stairs
        self.sg_first_time = torch.full((n,), float("nan"), device=dev)
        self.sg_falls = torch.zeros(n, device=dev)
        self.sg_progress = torch.zeros(n, device=dev)
        self.sg_hard_leg1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sg_hard_ep = torch.zeros(n, dtype=torch.bool, device=dev)
        self.sg_reached_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.sg_hard_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sg_fall_k = torch.zeros(n, device=dev)
        self._sg_was_fallen = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sg_step = -1
        self.sg_mid_climb = torch.zeros(n, dtype=torch.bool, device=dev)  # this episode started mid-climb
        # curriculum per direction (stairs_dir_curriculum): the row (rise) of the next up / down episode of each robot
        self._sg_level = self.terrain.terrain_levels.clone().unsqueeze(-1).repeat(1, 2)  # [:, 0] down, [:, 1] up
        self._sg_rsi = None
        if self.cfg.stairs_rsi_plans and self.cfg.stairs_rsi_prob > 0:
            self._load_rsi_plans()
        if (self.cfg.plan_track_weight > 0 or self.cfg.plan_support_weight > 0 or self.cfg.expert_label_obs
                or self.cfg.expert_target_obs):
            self._load_plan_track()
        self.sg_stalling = torch.zeros(n, dtype=torch.bool, device=dev)
        self._sg_stall_t = torch.zeros(n, device=dev)  # [s] since the body last came closer to the goal
        self._sg_stall_best = torch.full((n,), 1e9, device=dev)  # closest to the goal on this leg [m]
        self._sg_stall_cells = None
        if self.cfg.stairs_stall_weight > 0:
            self._load_stall_cells()
        # the finished episode's first leg (read by tools/dog/stairs_goal_test.py)
        self.sg_last = {k: torch.zeros(n, device=dev) for k in
                        ("dir", "reached", "time", "falls", "progress", "hard", "stayed_down", "rise", "tread", "walls")}

    def _load_rsi_plans(self) -> None:
        """Mid-climb starts (stairs_rsi_*): the four-paw poses of the quasi-static stair plans
        (tools/dog/stairs_plan.py) per (direction, rise row, tread), as body pose (x, z, pitch, neck in plan
        coordinates: from the first edge along the walking direction) and joint angles."""
        import json
        from pathlib import Path

        from .terrain import STAIRS_TREADS

        g, dev = self._sg_geo, self.device
        names = list(self.robot.joint_names)
        rises = [round(float(r), 3) for r in g.rises]
        index = torch.full((2, len(rises), len(STAIRS_TREADS)), -1, dtype=torch.long, device=dev)
        frames = []
        for folder in str(self.cfg.stairs_rsi_plans).split(","):
            for f in sorted(Path(folder).glob("*_r*_t*.json")):
                plan = json.loads(f.read_text())
                if not plan.get("feasible") or round(plan["rise"], 3) not in rises or plan["tread"] not in STAIRS_TREADS:
                    continue
                d, row, col = int(plan["dir"] == "up"), rises.index(round(plan["rise"], 3)), STAIRS_TREADS.index(plan["tread"])
                if index[d, row, col] >= 0:
                    continue  # the first folder wins
                poses = []
                for fr in plan["frames"]:
                    q = torch.zeros(len(names))
                    for j, v in fr["q"].items():
                        if j in names:
                            q[names.index(j)] = v
                    poses.append(torch.cat((torch.tensor(fr["body"], dtype=torch.float32), q)))
                index[d, row, col] = len(frames)
                frames.append(torch.stack(poses))
        if not frames:
            raise ValueError(f"no feasible stair plans in {self.cfg.stairs_rsi_plans}")
        longest = max(len(f) for f in frames)
        table = torch.stack([torch.cat((f, f[-1:].expand(longest - len(f), -1))) for f in frames]).to(dev)
        self._sg_rsi = {"index": index, "frames": table, "count": torch.tensor([len(f) for f in frames], device=dev),
                        # nearest tread (the column treads are float32: 0.15 is stored as 0.15000000596)
                        "tread_col": torch.tensor([min(range(len(STAIRS_TREADS)), key=lambda i: abs(STAIRS_TREADS[i] - t))
                                                   for t in self._sg_tread_col.tolist()], device=dev)}
        print(f"[stairs goal] mid-climb starts from {len(frames)} plans ({int((index[1] >= 0).sum())} up, "
              f"{int((index[0] >= 0).sum())} down), share {self.cfg.stairs_rsi_prob}")

    def _load_plan_track(self) -> None:
        """The stairs-up plans as a reference (plan_track_*, expert_label_obs): per keyframe the body's x (from the first
        edge), height and pitch, which leg swings (front / rear) and the weight shift (body x minus the paws' centre)."""
        import json
        from pathlib import Path

        from .terrain import STAIRS_TREADS

        g, dev = self._sg_geo, self.device
        rises = [round(float(r), 3) for r in g.rises]
        index = torch.full((len(rises), len(STAIRS_TREADS)), -1, dtype=torch.long, device=dev)
        plans, poses = [], []
        names = list(self.robot.joint_names)
        feet = list(self.cfg.foot_body_names)
        for folder in str(self.cfg.plan_track_plans or self.cfg.stairs_rsi_plans).split(","):
            for f in sorted(Path(folder).glob("up_r*_t*.json")):
                plan = json.loads(f.read_text())
                if not plan.get("feasible") or round(plan["rise"], 3) not in rises or plan["tread"] not in STAIRS_TREADS:
                    continue
                row, col = rises.index(round(plan["rise"], 3)), STAIRS_TREADS.index(plan["tread"])
                if index[row, col] >= 0:
                    continue  # the first folder wins
                index[row, col] = len(plans)
                plans.append(torch.tensor([[fr["body"][0], fr["body"][1], fr["body"][2], float(fr["swing"][1] == "1"),
                                            fr["body"][0] - sum(fr["paws_before"]) / 4.0] for fr in plan["frames"]]))
                # per keyframe the joint pose (NaN = not in the plan: abduction, head, tail; and the swinging leg,
                # whose pose holds only before it lifts)
                pose = torch.full((len(plan["frames"]), len(names)), float("nan"))
                for k, fr in enumerate(plan["frames"]):
                    swing = next((f[:2] for f in feet if f.startswith(fr["swing"])), None)
                    for j, v in fr["q"].items():
                        if j in names and not (swing and j.startswith(swing + "_")):
                            pose[k, names.index(j)] = v
                poses.append(pose)
        if not plans:
            raise ValueError(f"no feasible stairs-up plans in {self.cfg.plan_track_plans or self.cfg.stairs_rsi_plans}")
        longest = max(len(p) for p in plans)
        pad = torch.tensor([1e3, 0.0, 0.0, -1.0, 0.0])  # padding: x far away, a swing class no robot matches
        table = torch.stack([torch.cat((p, pad.expand(longest - len(p), -1))) for p in plans]).to(dev)
        q_table = torch.stack([torch.cat((q, q[-1:].expand(longest - len(q), -1))) for q in poses]).to(dev)
        foot_of = [next((i for i, f in enumerate(feet) if j.startswith(f[:2] + "_")), -1) for j in names]
        self._pt = {"index": index, "frames": table, "q": q_table, "joint_foot": torch.tensor(foot_of, device=dev),
                    "x_min": torch.tensor([float(p[:, 0].min()) for p in plans], device=dev),
                    "x_max": torch.tensor([float(p[:, 0].max()) for p in plans], device=dev),
                    "tread_col": torch.tensor([min(range(len(STAIRS_TREADS)), key=lambda i: abs(STAIRS_TREADS[i] - t))
                                               for t in self._sg_tread_col.tolist()], device=dev)}
        self._rear_feet = torch.tensor([i for i, f in enumerate(self.cfg.foot_body_names) if f[1] == "1"], device=dev)
        print(f"[plan track] {len(plans)} stairs-up plans as the reference / gate label")

    def _load_stall_cells(self) -> None:
        """Stalling on a descent is paid for only on stairs with a descent plan: (rise row, tread column) -> bool."""
        import json
        from pathlib import Path

        from .terrain import STAIRS_TREADS

        rises = [round(float(r), 3) for r in self._sg_geo.rises]
        cells = torch.zeros(len(rises), len(STAIRS_TREADS), dtype=torch.bool, device=self.device)
        for folder in str(self.cfg.stairs_stall_plans or self.cfg.stairs_rsi_plans).split(","):
            for f in Path(folder).glob("down_r*_t*.json"):
                plan = json.loads(f.read_text())
                if plan.get("feasible") and round(plan["rise"], 3) in rises and plan["tread"] in STAIRS_TREADS:
                    cells[rises.index(round(plan["rise"], 3)), STAIRS_TREADS.index(plan["tread"])] = True
        if not bool(cells.any()):
            raise ValueError(f"no feasible descent plans in {self.cfg.stairs_stall_plans or self.cfg.stairs_rsi_plans}")
        self._sg_stall_cells = cells
        self._sg_stall_col = torch.tensor([min(range(len(STAIRS_TREADS)), key=lambda i: abs(STAIRS_TREADS[i] - t))
                                           for t in self._sg_tread_col.tolist()], device=self.device)
        print(f"[stairs goal] stalling costs on {int(cells.sum())} descents with a plan")

    def _sg_stall_update(self, dist: torch.Tensor, reached: torch.Tensor, fallen: torch.Tensor) -> None:
        """Once per step: how long the body has not come closer to the goal, and whether that counts as stalling."""
        cfg = self.cfg
        types, levels = self.terrain.terrain_types, self.terrain.terrain_levels
        active = self.stairs_goal_on() & (self.sg_dir < 0) & self._sg_stall_cells[levels, self._sg_stall_col[types]] & ~fallen
        closer = dist < self._sg_stall_best - cfg.stairs_stall_gain
        self._sg_stall_best.copy_(torch.where(closer, dist, self._sg_stall_best))
        self._sg_stall_t.copy_(torch.where(active & ~closer, self._sg_stall_t + self.step_dt, torch.zeros_like(dist)))
        self.sg_stalling.copy_(active & (self._sg_stall_t > cfg.stairs_stall_s))
        self._sg_stall_best[reached] = 1e9  # the next leg starts afresh

    def plan_zone(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(zone (E,) bool, plan id (E,), body x in plan coordinates (E,)): the robot climbs a stairs goal tile with a
        plan for it, its body between the plan's first and last keyframe, not fallen."""
        pt = getattr(self, "_pt", None)
        n = self.num_envs
        if pt is None:
            none = torch.zeros(n, dtype=torch.bool, device=self.device)
            return none, torch.zeros(n, dtype=torch.long, device=self.device), torch.zeros(n, device=self.device)
        g = self._sg_geo
        types, levels = self.terrain.terrain_types, self.terrain.terrain_levels
        plan = pt["index"][levels, pt["tread_col"][types]]
        pid = plan.clamp(min=0)
        x0 = self.terrain.env_origins[:, 0] - g.size[0] / 2 + g.first_riser
        x = self.robot.data.root_pos_w.torch[:, 0] - x0
        zone = self.stairs_goal_on() & (self.sg_dir > 0) & (plan >= 0) & (x >= pt["x_min"][pid]) & (x <= pt["x_max"][pid])
        if self.cfg.recover:
            zone = zone & ~self.fallen
        return zone, pid, x

    def _plan_ref_index(self, x: torch.Tensor, pid: torch.Tensor, in_contact: torch.Tensor) -> torch.Tensor:
        """(E,) the nearest keyframe (by x) with the same leg (front / rear) up; the body sways back and forth, so x alone
        is ambiguous (up to 5 cm of height), x and the leg are not (<= 1.3 cm, checked on all plans)."""
        frames = self._pt["frames"][pid]
        rear_up = (~in_contact[:, self._rear_feet]).any(dim=-1)
        near = (frames[..., 0] - x.unsqueeze(-1)).abs()
        near = torch.where(frames[..., 3] == rear_up.float().unsqueeze(-1), near, torch.full_like(near, 1e9))
        return near.argmin(dim=-1)

    def plan_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The expert's copy target (expert_target_obs): the plan's joint pose at the nearest keyframe as policy actions
        (servo target = action x joint range x action scale) and a 0/1 mask (in the plan zone, joints in the plan, not of
        a leg whose paw is in the air: the plan's pose is for a standing leg, and the dog may swing another leg than it)."""
        if getattr(self, "_pt", None) is None:  # a test without plans: nothing to copy
            none = torch.zeros(self.num_envs, len(self._act_ids), device=self.device)
            return none, none.clone()
        zone, pid, x = self.plan_zone()
        forces = self.contact.data.net_forces_w.torch[:, self._foot_sensor_ids]
        in_contact = torch.linalg.norm(forces, dim=-1) > 1.0
        k = self._plan_ref_index(x, pid, in_contact)
        q = self._pt["q"][pid, k]  # (E, joints), NaN where not in the plan
        foot = self._pt["joint_foot"]
        airborne = (foot >= 0) & ~in_contact[:, foot.clamp(min=0)]
        known = ~torch.isnan(q) & zone.unsqueeze(-1) & ~airborne
        act = torch.nan_to_num(q) / (self.joint_range * self._target_scale)
        return act[:, self._act_ids], known[:, self._act_ids].float()

    def _plan_terms(self, in_contact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """plan_track and plan_support (per second, before weights) and diagnostics, in the plan zone."""
        pt, g, d = self._pt, self._sg_geo, self.robot.data
        zone, pid, x = self.plan_zone()
        rows = torch.arange(self.num_envs, device=self.device)
        frames = pt["frames"][pid]  # (E, F, 5): x, height, pitch, rear leg swings, weight shift
        one_up = (~in_contact).sum(dim=-1) == 1
        rear_up = (~in_contact[:, self._rear_feet]).any(dim=-1)
        ref = frames[rows, self._plan_ref_index(x, pid, in_contact)]
        height = d.root_pos_w.torch[:, 2] - self.terrain.env_origins[:, 2]
        _, pitch, _ = euler_xyz_from_quat(d.root_quat_w.torch)
        nose_up = -torch.atan2(torch.sin(pitch), torch.cos(pitch))  # the plans' pitch is nose up (-rotation about y)
        r_height = torch.exp(-torch.square((height - ref[:, 1]) / 0.02))
        r_pitch = torch.exp(-torch.square((nose_up - ref[:, 2]) / 0.1))
        # weight shift (body x minus the paws' centre) while exactly one paw is up
        x0 = self.terrain.env_origins[:, 0] - g.size[0] / 2 + g.first_riser
        shift = x - (d.body_pos_w.torch[:, self._foot_body_ids, 0] - x0.unsqueeze(-1)).mean(dim=-1)
        r_shift = torch.exp(-torch.square((shift - ref[:, 4]) / self.cfg.plan_shift_sigma)) * one_up.float()
        track = (r_height + r_pitch + r_shift) / (2.0 + one_up.float()) * zone.float()
        support = ((in_contact.sum(dim=-1) >= 3) & zone).float()
        on1 = (zone & one_up).float()
        diag = {"plan_zone": zone.float(), "plan_shift_front_cm": 100.0 * shift * on1 * (~rear_up).float(),
                "plan_shift_rear_cm": 100.0 * shift * on1 * rear_up.float()}
        return track, support, diag

    def stairs_goal_on(self) -> torch.Tensor:
        """(E,) bool: the robot is on a stairs goal tile."""
        return self._sg_col[self.terrain.terrain_types]

    def skill_signal(self) -> torch.Tensor:
        """(E, 3) one-hot walk / stairs up / stairs down (skill_signal_obs, step 53): a perfect recognizer, from the
        simulator's truth about the tile and the goal's direction."""
        signal = torch.zeros(self.num_envs, 3, device=self.device)
        stairs = self.stairs_goal_on() if self._sg_col is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        signal[:, 0] = (~stairs).float()
        if self._sg_col is not None:
            signal[:, 1] = (stairs & (self.sg_dir > 0)).float()
            signal[:, 2] = (stairs & (self.sg_dir < 0)).float()
        return signal

    def _sg_tile_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """World xy -> tile-local xy (0..size, the layout of terrain.straight_stairs)."""
        return xy - self.terrain.env_origins[:, :2] + torch.tensor(self._sg_geo.size, device=self.device) / 2

    def _sg_steps_under(self, xy: torch.Tensor) -> torch.Tensor:
        """(E,) float: number of steps up of the ground under xy on the robot's staircase (0 = floor, num_steps =
        landing; 0 beside an open staircase)."""
        from .terrain import straight_stairs_steps

        return straight_stairs_steps(self._sg_tile_xy(xy), self._sg_tread_col[self.terrain.terrain_types], self._sg_geo)

    def support_margin(self) -> torch.Tensor:
        """(E,) how far the centre of gravity (all bodies, vertical projection) lies inside the support polygon of the
        paws on the ground [m]; negative outside. Polygon = the touching paws in the order front left, front right,
        rear right, rear left (clockwise seen from above for any heading); with two paws down the distance to the line
        between them counts negative, with fewer -5 cm."""
        d = self.robot.data
        mass = d.body_mass.torch
        com = (d.body_com_pos_w.torch[..., :2] * mass.unsqueeze(-1)).sum(1) / mass.sum(1, keepdim=True)
        feet = self._paw_tips()[..., :2]  # foot_body_names order: L0, R0, L1, R1
        cyc = [0, 1, 3, 2]  # front left, front right, rear right, rear left
        pts = feet[:, cyc]
        forces = self.contact.data.net_forces_w.torch[:, self._foot_sensor_ids]
        down = (torch.linalg.norm(forces, dim=-1) > 1.0)[:, cyc]
        k = down.sum(-1)
        big = torch.full_like(k, 0, dtype=torch.float32) + 1.0
        margin = big.clone()
        for i in range(4):
            # the next touching paw after paw i in the cyclic order
            nxt = torch.full_like(k, -1)
            for off in (3, 2, 1):
                j = (i + off) % 4
                nxt = torch.where(down[:, j], torch.full_like(nxt, j), nxt)
            a = pts[:, i]
            b = torch.gather(pts, 1, nxt.clamp(min=0).view(-1, 1, 1).expand(-1, 1, 2))[:, 0]
            e = b - a
            length = torch.linalg.norm(e, dim=-1).clamp(min=1e-6)
            cross = e[:, 0] * (com[:, 1] - a[:, 1]) - e[:, 1] * (com[:, 0] - a[:, 0])
            dist = -cross / length  # clockwise polygon: inside is to the right of every edge
            use = down[:, i] & (nxt >= 0) & (k >= 3)
            margin = torch.where(use, torch.minimum(margin, dist), margin)
        # two paws: minus the distance to the segment between them; fewer: -5 cm
        two = k == 2
        if bool(two.any()):
            idx = torch.argsort(down.float(), dim=-1, descending=True)[:, :2]
            a = torch.gather(pts, 1, idx[:, :1, None].expand(-1, 1, 2))[:, 0]
            b = torch.gather(pts, 1, idx[:, 1:, None].expand(-1, 1, 2))[:, 0]
            e = b - a
            t = (((com - a) * e).sum(-1) / (e * e).sum(-1).clamp(min=1e-9)).clamp(0, 1)
            seg = torch.linalg.norm(a + t.unsqueeze(-1) * e - com, dim=-1)
            margin = torch.where(two, -seg, margin)
        return torch.where(k < 2, torch.full_like(margin, -0.05), margin)

    def ground_height(self) -> torch.Tensor:
        """(E,) body height above the ground under it: above the tile origin; on stairs goal tiles above the mean of
        the steps under the front and the rear hips (8 cm ahead of / behind the centre), so the standing height
        changes smoothly while the dog spans a riser."""
        d = self.robot.data
        z = d.root_pos_w.torch[:, 2] - self.terrain.env_origins[:, 2]
        if self._tilt_col is not None:
            # tilt tiles: distance from the plate's plane (through the tile origin), measured along its normal
            n = self.tilt_normal()
            rel = d.root_pos_w.torch - self.terrain.env_origins
            z = torch.where(self.tilt_on(), torch.sum(rel * n, dim=-1), z)
        if self._sg_col is None:
            return z
        _, _, yaw = euler_xyz_from_quat(d.root_quat_w.torch)
        half = 0.08 * torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        xy = d.root_pos_w.torch[:, :2]
        steps = 0.5 * (self._sg_steps_under(xy + half) + self._sg_steps_under(xy - half))
        rise = self._sg_rises[self.terrain.terrain_levels]
        return z - torch.where(self.stairs_goal_on(), rise * steps, torch.zeros_like(z))

    def _sg_set_goal(self, env_ids: torch.Tensor) -> None:
        """Goal for the current leg: on the landing past the top riser (up) or on the floor before the first (down)."""
        g = self._sg_geo
        types, levels = self.terrain.terrain_types[env_ids], self.terrain.terrain_levels[env_ids]
        top_riser = g.first_riser + (g.num_steps - 1) * self._sg_tread_col[types]
        up = self.sg_dir[env_ids] > 0
        x = torch.where(up, top_riser + self.cfg.stairs_goal_past, torch.full_like(top_riser, g.first_riser - self.cfg.stairs_goal_past))
        origin = self.terrain.env_origins[env_ids]
        self.goal_pos[env_ids, 0] = origin[:, 0] + x - g.size[0] / 2
        self.goal_pos[env_ids, 1] = origin[:, 1]
        self.goal_pos[env_ids, 2] = origin[:, 2] + up.float() * g.num_steps * self._sg_rises[levels]

    def _sg_reset(self, env_ids: torch.Tensor, pose: torch.Tensor, yaw: torch.Tensor) -> None:
        """Start the robots on stairs goal tiles in front of the staircase (up) or on the landing (down), facing the
        goal (pose and yaw of env_ids are changed in place). With mid-climb starts a share of them starts instead in
        a random four-paw pose of the stair plan for their tile; returns (mask over env_ids, pitch, joint angles) of
        those, or None."""
        cfg, g = self.cfg, self._sg_geo
        sel = self.stairs_goal_on()[env_ids]
        if not bool(sel.any()):
            return None
        ids, n = env_ids[sel], int(sel.sum())
        if cfg.stairs_goal_dir != 0:
            up = torch.full((n,), cfg.stairs_goal_dir > 0, dtype=torch.bool, device=self.device)
        elif self.eval_mode:
            up = (ids // self.terrain.terrain_origins.shape[0]) % 2 == 0  # every row gets both directions
        else:
            up = torch.rand(n, device=self.device) < 0.5
        self.sg_dir[ids] = up.float() * 2 - 1
        self.sg_first_dir[ids] = self.sg_dir[ids]
        if cfg.stairs_dir_curriculum and self.curriculum_enabled and not self.eval_mode:
            t = self.terrain
            t.terrain_levels[ids] = self._sg_level[ids, up.long()]
            t.env_origins[ids] = t.terrain_origins[t.terrain_levels[ids], t.terrain_types[ids]]
        types, levels = self.terrain.terrain_types[ids], self.terrain.terrain_levels[ids]
        top_riser = g.first_riser + (g.num_steps - 1) * self._sg_tread_col[types]
        if self.eval_mode:
            approach = torch.full((n,), cfg.stairs_goal_approach[1], device=self.device)
            spread, self._sg_speed[ids] = cfg.stairs_goal_eval_yaw_spread, cfg.stairs_goal_speed
        else:
            approach = sample_uniform(*cfg.stairs_goal_approach, (n,), self.device)
            spread = cfg.stairs_goal_yaw_spread
            self._sg_speed[ids] = sample_uniform(*cfg.stairs_goal_speed_range, (n,), self.device)
        x = torch.where(up, g.first_riser - approach, top_riser + cfg.stairs_goal_past + 0.1)
        origin = self.terrain.env_origins[ids]
        pose[sel, 0] = origin[:, 0] + x - g.size[0] / 2
        pose[sel, 1] = origin[:, 1] + sample_uniform(-0.05, 0.05, (n,), self.device)
        pose[sel, 2] += (~up).float() * g.num_steps * self._sg_rises[levels]
        yaw[sel] = math.pi * (~up).float() + sample_uniform(-spread, spread, (n,), self.device)
        self._sg_set_goal(ids)
        self.sg_leg[ids] = 0
        self.sg_first_time[ids] = float("nan")
        self.sg_falls[ids] = 0.0
        self.sg_progress[ids] = 0.0
        self.sg_hard_leg1[ids] = False
        self._sg_hard_ep[ids] = False
        self._sg_was_fallen[ids] = False
        self._sg_stall_t[ids] = 0.0
        self._sg_stall_best[ids] = 1e9
        if self._sg_rsi is None or self.eval_mode:
            return None
        r = self._sg_rsi
        plan = r["index"][up.long(), levels, r["tread_col"][types]]
        mid = (plan >= 0) & (torch.rand(n, device=self.device) < cfg.stairs_rsi_prob)
        if not bool(mid.any()):
            return None
        span = torch.ones(n, device=self.device)
        if cfg.stairs_rsi_down_early > 0:
            # a share of the descents starts within the first 40 % of the plan: the first steps off the landing
            early = ~up & (torch.rand(n, device=self.device) < cfg.stairs_rsi_down_early)
            span = torch.where(early, 0.4, 1.0)
        k = (torch.rand(n, device=self.device) * span * r["count"][plan.clamp(min=0)]).long()
        pose_k = r["frames"][plan.clamp(min=0), k]  # (n, 4 + joints)
        body = pose_k[:, :4]
        edge = torch.where(up, torch.full_like(top_riser, g.first_riser), top_riser)
        x = origin[:, 0] - g.size[0] / 2 + edge + (up.float() * 2 - 1) * body[:, 0]
        rows = sel.nonzero(as_tuple=False).squeeze(-1)[mid]
        m = int(mid.sum())
        pose[rows, 0] = x[mid]
        pose[rows, 1] = origin[mid, 1] + sample_uniform(-0.02, 0.02, (m,), self.device)
        pose[rows, 2] = origin[mid, 2] + body[mid, 1] + 0.003  # a hair above: never start inside a step
        yaw[rows] = math.pi * (~up[mid]).float() + sample_uniform(-0.05, 0.05, (m,), self.device)
        out = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        out[rows] = True
        pitch = torch.zeros(len(env_ids), device=self.device)
        pitch[rows] = body[mid, 2]
        joints = torch.zeros(len(env_ids), pose_k.shape[1] - 4, device=self.device)
        joints[rows] = pose_k[mid, 4:]
        return out, pitch, joints

    def _sg_navigate(self) -> None:
        """Navigation layer: walking commands towards the goal for the robots on stairs goal tiles (turn in place when
        the goal is far off to the side, else walk at it, slowing down for the last 25 cm)."""
        cfg, d = self.cfg, self.robot.data
        on = self.stairs_goal_on()
        if not bool(on.any()):
            return
        rel = self.goal_pos[:, :2] - d.root_pos_w.torch[:, :2]
        _, _, yaw = euler_xyz_from_quat(d.root_quat_w.torch)
        error = torch.remainder(torch.atan2(rel[:, 1], rel[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
        dist = torch.linalg.norm(rel, dim=-1)
        turn = torch.clamp(cfg.stairs_goal_yaw_gain * error, -cfg.stairs_goal_yaw_max, cfg.stairs_goal_yaw_max)
        walk = (error.abs() < cfg.stairs_goal_turn_first).float()
        vx = self._sg_speed * torch.cos(error).clamp(min=0.0) * walk * (dist / 0.25).clamp(0.4, 1.0)
        cmd = torch.stack((vx, torch.zeros_like(vx), turn), dim=-1)
        self._command_target[on] = cmd[on]

    def _sg_update(self) -> None:
        """Once per step, before the dones: goals reached (the goal flips to the other end), first-leg bookkeeping,
        and falling down the stairs (while fallen, the ground under the body more than one step below where the
        fall began)."""
        if self._sg_step == int(self.common_step_counter):
            return
        self._sg_step = int(self.common_step_counter)
        cfg, g = self.cfg, self._sg_geo
        on = self.stairs_goal_on()
        pos = self.robot.data.root_pos_w.torch
        steps = self._sg_steps_under(pos[:, :2])
        fallen = self.fallen if cfg.recover else torch.zeros_like(on)
        fresh = self.episode_length_buf > 1  # just placed: last step's fallen state belongs to the old episode
        fallen = fallen & fresh
        new_fall = fallen & ~self._sg_was_fallen
        self._sg_fall_k.copy_(torch.where(new_fall, steps, self._sg_fall_k))
        self._sg_was_fallen.copy_(fallen)
        hard = on & fallen & (steps < self._sg_fall_k - 1)
        leg1 = on & (self.sg_leg == 0)
        self.sg_falls += (new_fall & leg1).float()
        made = torch.where(self.sg_first_dir > 0, steps, g.num_steps - steps)
        self.sg_progress.copy_(torch.where(leg1, torch.maximum(self.sg_progress, made), self.sg_progress))
        dist = torch.linalg.norm(self.goal_pos[:, :2] - pos[:, :2], dim=-1)
        reached = on & fresh & ~fallen & ~hard & (dist < cfg.stairs_goal_reach_radius)
        t = self.episode_length_buf.float() * self.step_dt
        self.sg_first_time.copy_(torch.where(reached & leg1, t, self.sg_first_time))
        self.sg_hard_leg1 |= hard & leg1
        self._sg_hard_ep |= hard
        self.sg_reached_now.copy_(reached)
        self.sg_hard_now.copy_(hard)
        if self._sg_stall_cells is not None:
            self._sg_stall_update(dist, reached, fallen)
        if bool(reached.any()):
            ids = reached.nonzero(as_tuple=False).squeeze(-1)
            self.sg_leg[ids] += 1
            self.sg_dir[ids] = -self.sg_dir[ids]
            self._sg_set_goal(ids)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        if DECLARED_SCENE:  # GA: the scene has built everything from cfg.scene
            for name in SCENE_ENTITIES:
                setattr(self, "contact" if name == "contact_sensor" else name,
                        self.scene[name] if getattr(self.cfg, name) is not None else None)
            return
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0(
            src, dest, self.scene.num_envs, self.device, pos, global_paths=(self.cfg.terrain.prim_path,)
        )
        cloner.replicate(plan, stage=self.scene.stage)
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        self.height_scanner = None
        if self.cfg.height_scanner is not None:
            self.height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self.height_scanner
        self.tof = None
        if self.cfg.tof is not None:
            self.tof = RayCaster(self.cfg.tof)
            self.scene.sensors["tof"] = self.tof
        self.map_camera = None
        if self.cfg.map_camera is not None:
            self.map_camera = RayCaster(self.cfg.map_camera)
            self.scene.sensors["map_camera"] = self.map_camera
        self.imu = None
        if self.cfg.imu is not None:
            self.imu = Imu(self.cfg.imu)
            self.scene.sensors["imu"] = self.imu
        self.contact = None
        if self.cfg.contact_sensor is not None:
            self.contact = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact"] = self.contact
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ control

    def _low_level_action(self) -> torch.Tensor:
        """Servo actions from the frozen walking policy for the current steering command."""
        d = self.robot.data
        frame = torch.cat(
            self._sense_body() + [d.joint_pos.torch / self.joint_range, self.actions, self._clock(),
                                  self.commands * self._command_scale], dim=-1)
        self._ll_hist[:, :-1] = self._ll_hist[:, 1:].clone()
        self._ll_hist[:, -1] = frame
        fill = self._ll_fill.nonzero(as_tuple=False).squeeze(-1)
        if fill.numel() > 0:
            self._ll_hist[fill] = frame[fill].unsqueeze(1)
            self._ll_fill[fill] = False
        with torch.no_grad():
            return self.low_level(self._ll_hist.reshape(self.num_envs, -1)).clamp(-1.0, 1.0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.topple_push:
            self._topple()
        if self.low_level is not None:
            # the trained policy steers; the frozen walker turns the steering command into servo actions
            self.commands.copy_(torch.clamp(actions, -1.0, 1.0) * self._steer_scale)
            actions = self._low_level_action()
        if self.cfg.action_limit_weight > 0:
            self._action_excess = torch.sum(torch.clamp(actions.abs() - 1.0, min=0.0), dim=-1)
        actions = torch.clamp(actions, -1.0, 1.0)
        if actions.shape[1] != self.actions.shape[1] or self._remap_actions:
            full = torch.zeros_like(self.actions)
            full[:, self._act_ids] = actions
            full[:, self._expr_ids] = self._expression_targets()
            actions = full
        # update in place: the buffers must stay normal tensors even when stepped under inference mode
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions)
        # with one step of delay the servos receive the previous command, like the real control loop
        command = self.prev_actions if self.cfg.action_delay_steps > 0 else self.actions
        self.robot.actuators.target_command.set_position_index(value=command * self.joint_range * self._target_scale)
        if self.cfg.gait in DOG_GAITS:
            lo, hi = self.cfg.gait_freq_range
            speed = torch.linalg.norm(self.commands[:, :2], dim=-1) + 0.1 * self.commands[:, 2].abs()
            freq = lo + (hi - lo) * torch.clamp(speed / self.cfg.gait_freq_speed, max=1.0)
            if self.cfg.gait_clock_idle != "run":
                freq = freq * self._clock_running().float()  # the clock stops while standing still
            self._phase.add_(2.0 * math.pi * freq * self.step_dt).remainder_(2.0 * math.pi)
        if self.cfg.task == "command":
            if self.cfg.heading_hold_gain > 0:
                self.commands[:, 2] = self._yaw_asked  # undo last step's correction before ramping
            if self.command_override is not None:
                self.commands[:] = self.command_override
            else:
                due = (self.episode_length_buf % self._resample_steps == 0) & (self.episode_length_buf > 0)
                if torch.any(due):
                    self._resample_commands(due.nonzero(as_tuple=False).squeeze(-1))
                if self._sg_col is not None:
                    self._sg_navigate()  # stairs goal tiles: commands towards the goal
                if self._tilt_col is not None:
                    self._command_target[self.tilt_on()] = 0.0  # tilt tiles: stand still
                if self.cfg.command_accel > 0:
                    # speed changes gradually (accelerating and slowing down), not as on/off steps
                    limit = torch.tensor([self.cfg.command_accel, self.cfg.command_accel, self.cfg.command_yaw_accel],
                                         device=self.device) * self.step_dt
                    delta = torch.maximum(torch.minimum(self._command_target - self.commands, limit), -limit)
                    self.commands.add_(delta)
                else:
                    self.commands.copy_(self._command_target)
            if self.cfg.heading_hold_gain > 0:
                self._hold_heading()

    def _topple(self) -> None:
        """Topple shoves (topple_push): a sudden horizontal velocity in a random direction plus a little spin, each robot
        every topple_interval seconds."""
        cfg = self.cfg
        t = self.episode_length_buf.float() * self.step_dt
        due = t >= self._topple_next
        if self._sg_col is not None:
            due &= ~self.stairs_goal_on()  # no topple shoves on the stairs goal tiles
        due = due.nonzero(as_tuple=False).squeeze(-1)
        if due.numel() == 0:
            return
        n = len(due)
        speed = sample_uniform(*cfg.topple_push, (n,), self.device)
        angle = sample_uniform(-math.pi, math.pi, (n,), self.device)
        d = self.robot.data
        vel = torch.cat((d.root_lin_vel_w.torch[due], d.root_ang_vel_w.torch[due]), dim=-1).clone()
        vel[:, 0] += speed * torch.cos(angle)
        vel[:, 1] += speed * torch.sin(angle)
        vel[:, 3:5] += sample_uniform(-cfg.topple_spin, cfg.topple_spin, (n, 2), self.device)
        self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=due)
        self._topple_next[due] = t[due] + sample_uniform(*cfg.topple_interval, (n,), self.device)

    def _recover_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Up (cos of the tilt), body height and fallen (tilted > recover_fallen_up or the body on the ground); also
        updates getup_standing (upright, at height, 3+ paws carrying, nothing else touching)."""
        cfg, d = self.cfg, self.robot.data
        up = -d.projected_gravity_b.torch[:, 2]
        height = self.ground_height()
        forces = self.contact.data.net_forces_w.torch
        body = (torch.linalg.norm(forces[:, self._chassis_sensor_id], dim=-1) > 1.0).any(dim=-1)
        fallen = (up < cfg.recover_fallen_up) | body
        paws_down = torch.linalg.norm(forces[:, self._foot_sensor_ids], dim=-1) > 0.5
        other = (torch.linalg.norm(forces[:, self._other_sensor_ids], dim=-1) > 1.0).any(dim=-1)
        standing = ((up > 0.9) & ((height - cfg.base_height_target).abs() < 0.025) & (paws_down.sum(dim=-1) >= 3)
                    & ~other)
        self.getup_standing.copy_(standing)
        self.fallen.copy_(fallen)
        return up, height, fallen

    def _hold_heading(self) -> None:
        """Add a corrective turn rate while walking is asked for without turning (see heading_hold_gain)."""
        from isaaclab.utils.math import quat_apply

        q = self.robot.data.root_quat_w.torch
        fwd = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3))
        heading = torch.atan2(fwd[:, 1], fwd[:, 0])
        self._yaw_asked.copy_(self.commands[:, 2])
        hold = (self._yaw_asked.abs() < 0.05) & (torch.linalg.norm(self.commands[:, :2], dim=-1) > 0.05)
        if self._sg_col is not None and self.command_override is None:
            hold &= ~self.stairs_goal_on()  # the navigation layer steers there
        new = hold & ~self._heading_held
        self._heading_target[new] = heading[new]
        self._heading_held.copy_(hold)
        error = torch.remainder(self._heading_target - heading + math.pi, 2.0 * math.pi) - math.pi
        correction = torch.clamp(self.cfg.heading_hold_gain * error, -self.cfg.heading_hold_max, self.cfg.heading_hold_max)
        self.commands[:, 2] = self._yaw_asked + correction * hold.float()

    def _expression_targets(self) -> torch.Tensor:
        """Targets of the expression joints (-1..1 of their range): the override if set, else random motions."""
        if self.expression_override is not None:
            return self.expression_override.expand(self.num_envs, -1)
        cfg = self.cfg
        t = self.episode_length_buf.float() * self.step_dt
        due = (t >= self._expr_next).nonzero(as_tuple=False).squeeze(-1)
        if due.numel() > 0:
            k, m = len(due), len(self._expr_ids)
            p = self._expr_params
            p[due, :, 0] = sample_uniform(-0.8, 0.8, (k, m), self.device)
            still = torch.rand(k, m, device=self.device) < cfg.expression_still_prob
            p[due, :, 1] = torch.where(still, 0.0, sample_uniform(0.2, 0.8, (k, m), self.device))
            p[due, :, 2] = sample_uniform(*cfg.expression_freq_hz, (k, m), self.device)
            p[due, :, 3] = sample_uniform(0.0, 2 * math.pi, (k, m), self.device)
            self._expr_next[due] = t[due] + sample_uniform(*cfg.expression_hold_s, (k,), self.device)
        p = self._expr_params
        wave = p[..., 0] + p[..., 1] * torch.sin(2 * math.pi * p[..., 2] * t.unsqueeze(-1) + p[..., 3])
        return wave.clamp(-1.0, 1.0)

    def _update_curriculum(self, ids: torch.Tensor, seconds: torch.Tensor) -> None:
        """Move robots that tracked their commands well to a harder terrain row, struggling ones to an easier row."""
        error = self._episode_sums["_lin_vel_error"][ids] / seconds
        fell = self.reset_terminated[ids]
        completed = self.reset_time_outs[ids]
        move_up = completed & ~fell & (error < self.cfg.curriculum_up_error)
        move_down = fell | (error > self.cfg.curriculum_down_error)
        if self._sg_col is not None:
            # stairs goal tiles (row = rise): up after reaching a goal without falling down the stairs, down otherwise
            on = self.stairs_goal_on()[ids]
            reached = self.sg_leg[ids] >= 1
            move_up = torch.where(on, reached & ~self._sg_hard_ep[ids], move_up)
            move_down = torch.where(on, ~reached | self._sg_hard_ep[ids], move_down)
            if self.cfg.stairs_dir_curriculum:
                # stairs tiles: only the level of the direction just walked moves (climbing and descending have their
                # own limits; one shared level kept high descents untrained where climbing up failed)
                sg = ids[on]
                d = (self.sg_first_dir[sg] > 0).long()
                lv = self._sg_level[sg, d] + move_up[on].long() - (move_down[on] & ~move_up[on]).long()
                top = self.terrain.max_terrain_level
                lv = torch.where(lv >= top, torch.randint_like(lv, top), lv.clamp(min=0))
                self._sg_level[sg, d] = lv
                ids, move_up, move_down = ids[~on], move_up[~on], move_down[~on]
                log = self.extras.setdefault("log", {})
                log["Curriculum/stairs_up_level"] = self._sg_level[self.stairs_goal_on(), 1].float().mean().item()
                log["Curriculum/stairs_down_level"] = self._sg_level[self.stairs_goal_on(), 0].float().mean().item()
        if self._tilt_col is not None and "fall" in self._episode_sums:
            # tilt tiles (row = tilt): steeper after standing a whole episode without a fall, flatter after a fall
            on = self.tilt_on()[ids]
            ended_fallen = self.reset_terminated[ids]
            fell_once = (self._episode_sums["fall"][ids] < 0.0) | ended_fallen
            move_up = torch.where(on, self.reset_time_outs[ids] & ~fell_once, move_up)
            move_down = torch.where(on, fell_once, move_down)
            log = self.extras.setdefault("log", {})
            log["Curriculum/tilt_level"] = self.terrain.terrain_levels[self.tilt_on()].float().mean().item()
        self.terrain.update_env_origins(ids, move_up, move_down & ~move_up)

    def prepare_evaluation(self) -> None:
        """Evaluation mode: moving commands only, curriculum frozen, robots spread over all terrain rows."""
        self.eval_mode = True
        self.spread_over_terrain_levels()

    def spread_over_terrain_levels(self) -> None:
        """Freeze the curriculum and spread robots evenly over all difficulty rows (rough terrain)."""
        self.curriculum_enabled = False
        if self.cfg.terrain_kind not in ("rough", "obstacles") and not self.cfg.cup:
            return
        t = self.terrain
        levels = torch.arange(self.num_envs, device=self.device) % t.terrain_origins.shape[0]
        t.terrain_levels[:] = levels
        t.env_origins[:] = t.terrain_origins[levels, t.terrain_types]

    def _paw_tips(self) -> torch.Tensor:
        """World position of every paw's sphere centre (E, paws, 3)."""
        d = self.robot.data
        quat = d.body_quat_w.torch[:, self._foot_body_ids]
        pos = d.body_pos_w.torch[:, self._foot_body_ids]
        offsets = self._foot_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        return pos + quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(pos.shape)

    def _ground_near_paws(self, tip: torch.Tensor, x_range: tuple[float, float], half_y: float
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """True ground (perfect scanner, 5 cm grid) in a box around each paw, in the robot's heading frame (x forward):
        highest and lowest point (E, paws) and whether the box holds any scan point."""
        hits = self.height_scanner.data.ray_hits_w.torch  # (E, R, 3)
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w.torch)
        rel = hits[:, None, :, :2] - tip[:, :, None, :2]  # (E, paws, R, 2)
        c, s = torch.cos(yaw).view(-1, 1, 1), torch.sin(yaw).view(-1, 1, 1)
        fx = c * rel[..., 0] + s * rel[..., 1]
        fy = -s * rel[..., 0] + c * rel[..., 1]
        z = torch.nan_to_num(hits[:, None, :, 2].expand_as(fx), nan=-1e3, posinf=-1e3, neginf=-1e3)
        inside = (fx >= x_range[0]) & (fx <= x_range[1]) & (fy.abs() <= half_y) & (z > -1e2)
        top = torch.where(inside, z, torch.full_like(z, -1e3)).amax(dim=-1)
        low = torch.where(inside, z, torch.full_like(z, 1e3)).amin(dim=-1)
        return top, low, inside.any(dim=-1)

    def _stairs_terms_on(self) -> torch.Tensor:
        """(E,) float: 1 where the stairs terms apply (stairs_terms_types; everywhere if unset)."""
        if self._col_stairs_terms is None:
            return torch.ones(self.num_envs, device=self.device)
        return self._col_stairs_terms[self.terrain.terrain_types].float()

    def _foot_clearance(self, moving: torch.Tensor) -> torch.Tensor:
        """Mean over swinging feet of (lift above the foot's last stance height) / target, capped at 1. The target is
        fixed, or (foot_clearance_terrain) the height of the ground just ahead of the paw + a margin."""
        cfg = self.cfg
        tip = self._paw_tips()
        sole = tip[..., 2] - self._foot_radius
        in_contact = self.contact.data.current_contact_time.torch[:, self._feet_ids] > 0.0
        self._foot_ground_z.copy_(torch.where(in_contact, sole, self._foot_ground_z))
        target = torch.full_like(sole, cfg.foot_clearance_target)
        if cfg.foot_clearance_terrain and self.height_scanner is not None:
            top, _, seen = self._ground_near_paws(tip, (-0.03, 0.10), 0.04)
            need = (top - self._foot_ground_z + cfg.foot_clearance_margin).clamp(cfg.foot_clearance_target,
                                                                                 cfg.foot_clearance_max)
            target = torch.where(seen & (self._stairs_terms_on() > 0).unsqueeze(-1), need, target)
        lift = ((sole - self._foot_ground_z) / target).clamp(0.0, 1.0)
        swing = (~in_contact).float()
        return torch.sum(lift * swing, dim=-1) / swing.sum(dim=-1).clamp(min=1.0) * moving

    def _new_goal(self, env_ids: torch.Tensor, robot_xy: torch.Tensor | None = None) -> None:
        """Pick a different clear spot on the ring around the robot's tile centre as the next goal."""
        if self.cfg.terrain_kind == "enclosure":
            self._new_enclosure_goal(env_ids, robot_xy)
            return
        step = torch.randint(1, len(self._goal_spots), (len(env_ids),), device=self.device)
        self.goal_idx[env_ids] = (self.goal_idx[env_ids] + step) % len(self._goal_spots)
        self.goal_pos[env_ids, :2] = self.terrain.env_origins[env_ids, :2] + self._goal_spots[self.goal_idx[env_ids]]
        self.goal_pos[env_ids, 2] = self.terrain.env_origins[env_ids, 2]
        self._goal_prev_dist[env_ids] = self.goal_distance()[env_ids]

    def _new_enclosure_goal(self, env_ids: torch.Tensor, robot_xy: torch.Tensor | None = None) -> None:
        """Goal inside the ring wall (unreachable, off centre in a random direction) or outside it (reachable).
        In evaluation even envs always get an unreachable goal and odd envs a reachable one."""
        cfg, n = self.cfg, len(env_ids)
        if self.eval_mode:
            # alternate per block of terrain rows, so every difficulty row gets both kinds
            inside = (env_ids // self.terrain.terrain_origins.shape[0]) % 2 == 0
        else:
            inside = torch.rand(n, device=self.device) < cfg.unreachable_prob
        radius = torch.where(inside, sample_uniform(*cfg.enclosure_goal_offset, (n,), self.device),
                             sample_uniform(*cfg.enclosure_outside_goal, (n,), self.device))
        angle = sample_uniform(-math.pi, math.pi, (n,), self.device)
        centre = self.terrain.env_origins[env_ids, :2]
        if cfg.enclosure_far_start and robot_xy is not None:
            # goal on the far side of the ring, seen from the robot (within +-20 degrees)
            away = robot_xy - centre
            angle = torch.atan2(away[:, 1], away[:, 0]) + math.pi + sample_uniform(-0.35, 0.35, (n,), self.device)
        if cfg.cup:
            # inside goals left of the cup centre, near its closed side
            angle = torch.where(inside, math.pi + sample_uniform(-0.4, 0.4, (n,), self.device), angle)
        rel = radius.unsqueeze(-1) * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        if cfg.pocket is not None:
            # keep goals inside the ring out of the slot's walls: mirror them to the other half of the ring
            from .terrain import ENCLOSURE_WALL

            r_out, width, depth = cfg.pocket
            front = r_out - depth - ENCLOSURE_WALL - 0.05
            in_block = inside & (rel[:, 0] > front) & (rel[:, 1].abs() < width / 2 + ENCLOSURE_WALL + 0.04)
            rel[in_block, 0] = -rel[in_block, 0]
        goal = centre + rel
        # closest distance the robot's centre can reach to a goal inside: the outer wall in the goal's direction,
        # or on the pocket course the slot bottom (if that is closer)
        offset = torch.linalg.norm(rel, dim=-1)
        ring = self.terrain.cfg.terrain_generator.sub_terrains["ring"].radius
        ideal = ring + cfg.enclosure_robot_gap - offset
        if cfg.pocket is not None:
            r_out, width, depth = cfg.pocket
            cut = (rel[:, 0] > 0) & (rel[:, 1].abs() / offset.clamp(min=1e-6) * r_out < width / 2 + 0.06)
            ideal = torch.where(cut, torch.full_like(ideal, 1e3), ideal)
            bottom = torch.stack((torch.full_like(offset, r_out - depth + cfg.enclosure_robot_gap), torch.zeros_like(offset)), -1)
            ideal = torch.minimum(ideal, torch.linalg.norm(bottom - rel, dim=-1))
        self.goal_ideal_dist[env_ids] = torch.where(inside, ideal, torch.zeros_like(ideal))
        # a reachable goal right next to the robot would be a free reach bonus: put it on the other side
        robot = self.robot.data.root_pos_w.torch[env_ids, :2] if robot_xy is None else robot_xy
        near = ~inside & (torch.linalg.norm(goal - robot, dim=-1) < 0.5)
        goal[near] = 2 * centre[near] - goal[near]
        self.goal_unreachable[env_ids] = inside  # cup: "inside the cup", reachable through the open side
        self.goal_pos[env_ids, :2] = goal
        self.goal_pos[env_ids, 2] = self.terrain.env_origins[env_ids, 2]
        self._goal_prev_dist[env_ids] = torch.linalg.norm(goal - robot, dim=-1)

    def goal_distance(self) -> torch.Tensor:
        return torch.linalg.norm(self.goal_pos[:, :2] - self.robot.data.root_pos_w.torch[:, :2], dim=-1)

    def goal_in_body_frame(self) -> torch.Tensor:
        """Goal position relative to the robot, rotated into its heading (x forward, y left) [m].

        With ``goal_signal="camera"`` the robot's pose comes from a simulated overhead camera: delayed, noisy and
        refreshed at the camera rate."""
        d = self.robot.data
        _, _, true_yaw = euler_xyz_from_quat(d.root_quat_w.torch)
        pose = torch.cat((d.root_pos_w.torch[:, :2], true_yaw.unsqueeze(-1)), dim=-1)
        if self.cfg.goal_signal == "camera":
            if not hasattr(self, "_cam_buffer"):
                self._cam_buffer = pose.unsqueeze(1).repeat(1, self.cfg.camera_delay_steps + 1, 1)
                self._cam_pose = pose.clone()
                self._cam_tick = 0
            self._cam_buffer[:, :-1] = self._cam_buffer[:, 1:].clone()
            self._cam_buffer[:, -1] = pose
            if self._cam_tick % self.cfg.camera_update_every == 0:
                seen = self._cam_buffer[:, 0].clone()  # the oldest pose = what the camera saw delay steps ago
                seen[:, :2] += torch.randn_like(seen[:, :2]) * self.cfg.camera_pos_noise
                seen[:, 2] += torch.randn_like(seen[:, 2]) * self.cfg.camera_yaw_noise
                self._cam_pose.copy_(seen)
            self._cam_tick += 1
            pose = self._cam_pose
        rel = self.goal_pos[:, :2] - pose[:, :2]
        yaw = pose[:, 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]), dim=-1)

    def obstacle_contact(self) -> torch.Tensor:
        """True where any body pushes sideways against something (a pillar): horizontal force clearly above the
        vertical one. Walking on the floor never does that, since friction stays below the normal force."""
        force = self.contact.data.net_forces_w.torch
        horizontal = torch.linalg.norm(force[..., :2], dim=-1)
        return torch.any(horizontal > 1.2 * force[..., 2].abs() + 0.2, dim=-1)

    def _sense_tof(self) -> torch.Tensor:
        """8 x 8 distances, scaled to 0..1 of the range, refreshed at the sensor rate with noise."""
        cfg = self.cfg
        if self._tof_tick % cfg.tof_update_every == 0:
            data = self.tof.data
            dist = torch.linalg.norm(data.ray_hits_w.torch - data.pos_w.torch.unsqueeze(1), dim=-1)
            dist = torch.nan_to_num(dist, nan=cfg.tof_max_range, posinf=cfg.tof_max_range, neginf=cfg.tof_max_range)
            dist = dist * (1.0 + torch.randn_like(dist) * cfg.tof_noise_rel) + torch.randn_like(dist) * cfg.tof_noise_abs
            self._tof_held.copy_(dist.clamp(0.0, cfg.tof_max_range) / cfg.tof_max_range)
        self._tof_tick += 1
        return self._tof_held

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        cfg, n = self.cfg, len(env_ids)
        if self.eval_mode:
            # a real walking command: speed 10-30 cm/s in a random direction plus a moderate turn rate
            heading = sample_uniform(*cfg.eval_heading_range, (n,), self.device)
            speed = sample_uniform(*cfg.eval_speed_range, (n,), self.device)
            yaw = self._turn_on_stairs(env_ids, sample_uniform(*cfg.eval_yaw_range, (n,), self.device))
            self._command_target[env_ids] = torch.stack((speed * torch.cos(heading), speed * torch.sin(heading), yaw), dim=-1)
            return
        cmd = torch.stack(
            (
                sample_uniform(*cfg.command_vx_range, (n,), self.device),
                sample_uniform(*cfg.command_vy_range, (n,), self.device),
                sample_uniform(*cfg.command_yaw_range, (n,), self.device),
            ),
            dim=-1,
        )
        cmd[:, 2] = self._turn_on_stairs(env_ids, cmd[:, 2])
        roll = torch.rand(n, device=self.device)
        cmd[roll < cfg.command_standing_prob] = 0.0
        turn_only = (roll >= cfg.command_standing_prob) & (roll < cfg.command_standing_prob + cfg.command_turn_in_place_prob)
        cmd[turn_only, :2] = 0.0
        self._command_target[env_ids] = cmd

    def _getup_starts(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Start category per robot (0 standing, 1 dropped upright, 2 on its side, 3 on its back): the mix moves from
        getup_mix_start to getup_mix_end over the curriculum, or fixed (getup_category) for tests and videos."""
        cfg, n = self.cfg, len(env_ids)
        if cfg.getup_category >= 0:
            cat = torch.full((n,), cfg.getup_category, dtype=torch.long, device=self.device)
        elif cfg.getup_category == -2:
            cat = env_ids.long() % 4
        else:
            f = min(1.0, float(self.common_step_counter) / max(1, cfg.getup_curriculum_steps))
            mix = torch.tensor([a + (b - a) * f for a, b in zip(cfg.getup_mix_start, cfg.getup_mix_end)], device=self.device)
            cat = torch.multinomial(mix / mix.sum(), n, replacement=True)
        self.getup_category[env_ids] = cat
        return cat

    def _turn_on_stairs(self, env_ids: torch.Tensor, yaw_rate: torch.Tensor) -> torch.Tensor:
        """Turn-rate commands with those of robots on stair tiles scaled by stairs_turn_scale."""
        if self.cfg.stairs_turn_scale == 1.0 or self._col_stairs is None:
            return yaw_rate
        on = self._col_stairs[self.terrain.terrain_types[env_ids]]
        return torch.where(on, yaw_rate * self.cfg.stairs_turn_scale, yaw_rate)

    def _apply_action(self) -> None:
        pass  # position targets persist across the decimation substeps; the servo model runs every physics step

    def show_command_arrows(self) -> None:
        """Draw the commanded (green) and actual (blue) planar velocity above every robot. For play/videos."""
        if not hasattr(self, "_goal_arrows"):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

            self._goal_arrows = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/command/goal"))
            self._vel_arrows = VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/command/actual"))
        d = self.robot.data
        root_quat = d.root_quat_w.torch
        for markers, xy, lift in (
            (self._goal_arrows, self.commands[:, :2], 0.10),
            (self._vel_arrows, d.root_lin_vel_b.torch[:, :2], 0.115),
        ):
            pos = d.root_pos_w.torch.clone()
            pos[:, 2] += lift
            yaw = torch.atan2(xy[:, 1], xy[:, 0])
            zeros = torch.zeros_like(yaw)
            quat = quat_mul(root_quat, quat_from_euler_xyz(zeros, zeros, yaw))
            scale = torch.tensor([1.0, 0.035, 0.035], device=self.device).repeat(self.num_envs, 1)
            scale[:, 0] = torch.clamp(torch.linalg.norm(xy, dim=-1) * 1.2, min=0.005)
            markers.visualize(translations=pos, orientations=quat, scales=scale)

    # ------------------------------------------------------------------ observations

    def _gait_phase(self) -> torch.Tensor:
        if self.cfg.gait in DOG_GAITS:
            return self._phase
        return 2.0 * math.pi * self.cfg.gait_clock_hz * self.episode_length_buf.float() * self.step_dt

    def _clock(self) -> torch.Tensor:
        phase = self._gait_phase()
        clock = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
        if self.cfg.gait in DOG_GAITS and self.cfg.gait_clock_idle == "zero":
            clock = clock * self._clock_running().float().unsqueeze(-1)
        return clock

    def _clock_running(self) -> torch.Tensor:
        """gait_clock_idle "hold" / "zero": the clock runs unless upright and told to stand still (while fallen it runs as
        before, so getting up and the get-up teacher see what they were trained with)."""
        running = self._command_moving()
        if self.cfg.recover:
            running = running | self.fallen
        return running

    def _command_moving(self) -> torch.Tensor:
        """Robots whose command asks for motion (not "stand still")."""
        return (torch.linalg.norm(self.commands[:, :2], dim=-1) > 0.05) | (self.commands[:, 2].abs() > 0.2)

    def _sense_body(self) -> list[torch.Tensor]:
        """Body sensing as the policy receives it: ideal, or with the noise of real sensors."""
        cfg, d = self.cfg, self.robot.data
        if cfg.sensor_suite == "ideal":
            return [d.projected_gravity_b.torch, d.root_ang_vel_b.torch * cfg.ang_vel_scale]

        def noise(like: torch.Tensor, std: float) -> torch.Tensor:
            return torch.randn_like(like) * std

        gravity = d.projected_gravity_b.torch
        gravity = gravity + noise(gravity, cfg.gravity_noise)
        gyro = self.imu.data.ang_vel_b.torch
        gyro = gyro + self._gyro_bias + noise(gyro, cfg.gyro_noise)
        accel = self.imu.data.lin_acc_b.torch
        accel = accel + self._accel_bias + noise(accel, cfg.accel_noise)
        sensed = [gravity, gyro * cfg.ang_vel_scale, accel / 9.81]
        if cfg.sensor_suite == "imu_flow":
            vel = d.root_lin_vel_b.torch[:, :2]
            reading = vel * (1.0 + noise(vel, cfg.flow_rel_noise)) + noise(vel, cfg.flow_abs_noise)
            dropped = torch.rand(self.num_envs, 1, device=self.device) < cfg.flow_dropout
            self._flow_held.copy_(torch.where(dropped, self._flow_held, reading))
            sensed.append(self._flow_held * 5.0)
        return sensed

    def _get_observations(self) -> dict:
        d = self.robot.data
        obs_ids = self._obs_ids
        joint_pos = d.joint_pos.torch[:, obs_ids] / self.joint_range[obs_ids]
        joint_vel = d.joint_vel.torch[:, obs_ids] * self.cfg.joint_vel_scale
        frame = self._sense_body() + [joint_pos, self.actions[:, obs_ids], self._clock()]
        scan_obs = None
        if self.height_scanner is not None:
            scan = self.height_scanner.data
            # ground height under each ray relative to the chassis (measured from the chassis itself: the sensor's
            # reported position does not include its mounting offset), minus the standing height: flat floor ~ 0
            heights = d.root_pos_w.torch[:, 2:3] - scan.ray_hits_w.torch[..., 2] - self.cfg.height_scan_offset
            clip = self.cfg.height_scan_clip
            scan_obs = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0).clamp(-clip, clip) / clip
            if self.cfg.height_scan_in_history:
                frame.append(scan_obs)
                scan_obs = None
        if self.cfg.policy_sees_joint_velocity:
            frame.append(joint_vel)
        if self.cfg.task == "command":
            frame.append(self.commands * self._command_scale)
        if self.cfg.task == "goal":
            goal = self.goal_in_body_frame()
            frame += [(goal / 2.0).clamp(-1.0, 1.0), (torch.linalg.norm(goal, dim=-1, keepdim=True) / 2.0).clamp(max=1.0),
                      self._sense_tof()]
            if self.low_level is not None:
                frame.append(self.commands * self._command_scale)
        frame = torch.cat(frame, dim=-1)
        self.obs_frame = frame
        policy = frame
        if self._obs_hist is not None:
            # shift in the newest frame; robots that were just reset start with a history of identical frames
            self._obs_hist[:, :-1] = self._obs_hist[:, 1:].clone()
            self._obs_hist[:, -1] = frame
            fill = self._hist_fill.nonzero(as_tuple=False).squeeze(-1)
            if fill.numel() > 0:
                self._obs_hist[fill] = frame[fill].unsqueeze(1)
                self._hist_fill[fill] = False
            # a copy: the trainer keeps a reference to this tensor, the buffer itself changes in place next step
            policy = self._obs_hist.reshape(self.num_envs, -1).clone()
        height = self.ground_height().unsqueeze(-1)
        critic = torch.cat((frame, d.root_lin_vel_b.torch * 5.0, height * 10.0, joint_vel), dim=-1)
        if scan_obs is not None:  # the current scan, once, after the history
            self.scan_truth = scan_obs
            seen = scan_obs
            if self.terrain_map is not None:
                self._update_map()
                if self.cfg.map_policy_input:
                    seen = self.map_obs
            policy = torch.cat((policy, seen), dim=-1)
            critic = torch.cat((critic, scan_obs), dim=-1)
        if self.cfg.skill_signal_obs:
            policy = torch.cat((policy, self.skill_signal()), dim=-1)
        out = {"policy": policy, "critic": critic}
        if self.cfg.expert_label_obs:
            out["expert_label"] = self.plan_zone()[0].float().unsqueeze(-1)
        if self.cfg.expert_target_obs:
            out["expert_target"] = torch.cat(self.plan_target(), dim=-1)
        return out

    def _update_map(self) -> None:
        """One control step of the terrain map: advance the drifting pose estimate, capture a sensor frame on frame
        steps, store frames whose delay has passed and the paws on the ground, then read the scan grid from the map."""
        if self._map_step == int(self.common_step_counter):
            return  # already updated this step (an extra observation call, e.g. for the final observation)
        self._map_step = int(self.common_step_counter)
        cfg, d, n = self.cfg, self.robot.data, self.num_envs
        self._map_tick += 1
        now = self._map_tick * self.step_dt
        pos, quat = d.root_pos_w.torch, d.root_quat_w.torch
        just_reset = self.episode_length_buf == 0  # the pose may not be updated yet after a reset
        self.odometry.step(pos, skip=just_reset)
        est = self.odometry.to_map(pos.unsqueeze(1))[:, 0]
        if cfg.map_dropout_rate > 0:
            start = (now >= self._dark_until) & (torch.rand(n, device=self.device) < cfg.map_dropout_rate * self.step_dt)
            dark_for = sample_uniform(*cfg.map_dropout_s, (n,), self.device)
            self._dark_until.copy_(torch.where(start, now + dark_for, self._dark_until))
        if self._map_tick % cfg.map_update_every == 0:
            hits = self.map_camera.data.ray_hits_w.torch
            origin = pos + quat_apply(quat, self._map_cam_offset.expand(n, 3))
            v = hits - origin.unsqueeze(1)
            dist = torch.linalg.norm(v, dim=-1)
            valid = torch.isfinite(dist) & (dist >= cfg.map_min_range) & (dist <= cfg.map_max_range)
            valid &= (~just_reset & (now >= self._dark_until)).unsqueeze(-1)
            v = torch.where(valid.unsqueeze(-1), v, torch.zeros_like(v))
            dist = torch.where(valid, dist, torch.ones_like(dist))
            noisy = dist * (1.0 + torch.randn_like(dist) * cfg.map_noise_rel) + torch.randn_like(dist) * cfg.map_noise_abs
            v = v * (noisy / dist).unsqueeze(-1)
            # orientation-estimate error: the whole frame tilted by a small rotation w = (wx, wy, 0): v + w x v
            w = torch.randn(n, 1, 2, device=self.device) * cfg.map_tilt_noise
            v = v + torch.stack((w[..., 1] * v[..., 2], -w[..., 0] * v[..., 2],
                                 w[..., 0] * v[..., 1] - w[..., 1] * v[..., 0]), dim=-1)
            points = self.odometry.to_map(origin.unsqueeze(1) + v)  # placed with the estimate at capture time
            self._map_pending.append((self._map_tick + cfg.map_delay_steps, points, valid))
        while self._map_pending and self._map_pending[0][0] <= self._map_tick:
            _, points, valid = self._map_pending.popleft()
            self.terrain_map.write(points, valid, est[:, :2], now)
            self._map_frame_t = now
        if cfg.map_feet and hasattr(self, "_foot_body_ids") and hasattr(self, "_foot_sensor_ids"):
            fq = d.body_quat_w.torch[:, self._foot_body_ids]
            fp = d.body_pos_w.torch[:, self._foot_body_ids]
            offsets = self._foot_offsets.unsqueeze(0).expand(n, -1, -1)
            tip = fp + quat_apply(fq.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(fp.shape)
            sole = torch.cat((tip[..., :2], (tip[..., 2] - self._foot_radius).unsqueeze(-1)), dim=-1)
            down = torch.linalg.norm(self.contact.data.net_forces_w.torch[:, self._foot_sensor_ids], dim=-1) > 1.0
            self.terrain_map.write(self.odometry.to_map(sole), down & ~just_reset.unsqueeze(-1), est[:, :2], now)
        # read the scan grid at the estimated pose: heights as the perfect scan reports them, plus known flags
        from .terrain_map import rotate_z

        _, _, yaw = euler_xyz_from_quat(quat)
        q = est[:, :2].unsqueeze(1) + rotate_z(self._map_query.expand(n, -1, -1), yaw + self.odometry.yaw)
        clip = cfg.height_scan_clip

        def scan(h: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
            rel = (est[:, 2:3] - h - cfg.height_scan_offset).clamp(-clip, clip) / clip
            return torch.cat((torch.where(known, rel, torch.zeros_like(rel)), known.float()), dim=-1)

        self.map_view_obs = scan(*self.terrain_map.read(q, now, 0.0, since=self._map_frame_t))
        self.map_obs = (self.map_view_obs if cfg.map_memory_s <= 0
                        else scan(*self.terrain_map.read(q, now, cfg.map_memory_s)))

    # ------------------------------------------------------------------ rewards

    def tracking_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Planar velocity error [m/s] and yaw-rate error [rad/s] with respect to the current command."""
        d = self.robot.data
        lin = torch.linalg.norm(self.commands[:, :2] - d.root_lin_vel_b.torch[:, :2], dim=-1)
        ang = torch.abs(self.commands[:, 2] - d.root_ang_vel_b.torch[:, 2])
        return lin, ang

    def _get_rewards(self) -> torch.Tensor:
        cfg, d = self.cfg, self.robot.data
        if cfg.getup:
            return self._getup_rewards()
        if cfg.recover:
            up_r, height_r, fallen = self._recover_state()
        tilt = torch.sum(torch.square(d.projected_gravity_b.torch[:, :2]), dim=-1)
        motors = self._obs_ids  # springs do no work for the battery
        power = torch.sum(torch.abs(d.applied_torque.torch[:, motors] * d.joint_vel.torch[:, motors]), dim=-1)
        # the policy's own actions only (expression joints move on their own)
        action_rate = torch.sum(torch.square(self.actions[:, self._act_ids] - self.prev_actions[:, self._act_ids]), dim=-1)
        over = torch.clamp(torch.abs(d.joint_pos.torch[:, motors]) / self.joint_range[motors] - 0.95, min=0.0)
        terms = {
            "upright": -cfg.upright_weight * tilt,
            "energy": -cfg.energy_weight * power,
            "action_rate": -cfg.action_rate_weight * action_rate,
            "joint_limit": -cfg.joint_limit_weight * torch.sum(over, dim=-1),
        }
        if cfg.action_limit_weight > 0:
            terms["action_limit"] = -cfg.action_limit_weight * self._action_excess
        diagnostics = {}
        if cfg.task == "command":
            lin_err, ang_err = self.tracking_errors()
            ang_vel = d.root_ang_vel_b.torch
            terms["track_lin"] = cfg.tracking_lin_weight * torch.exp(-torch.square(lin_err / cfg.tracking_lin_sigma))
            cmd_xy = self.commands[:, :2]
            cmd_speed = torch.linalg.norm(cmd_xy, dim=-1)
            along = torch.sum(cmd_xy * d.root_lin_vel_b.torch[:, :2], dim=-1) / cmd_speed.clamp(min=1e-3)
            moving_cmd = (cmd_speed > 0.05).float()
            terms["progress"] = cfg.progress_weight * moving_cmd * torch.minimum(along, cmd_speed)
            terms["track_ang"] = cfg.tracking_ang_weight * torch.exp(-torch.square(ang_err / cfg.tracking_ang_sigma))
            terms["vertical"] = -cfg.vertical_vel_weight * torch.square(d.root_lin_vel_b.torch[:, 2])
            terms["roll_pitch_rate"] = -cfg.roll_pitch_rate_weight * torch.sum(torch.square(ang_vel[:, :2]), dim=-1)
            diagnostics = {"lin_vel_error": lin_err, "yaw_rate_error": ang_err}
            if self._sg_col is not None:
                reached = self.sg_reached_now.float() / self.step_dt  # paid once per goal
                terms["sg_goal"] = cfg.stairs_goal_bonus * reached
                terms["sg_hard"] = -cfg.stairs_hard_fall_penalty * self.sg_hard_now.float() / self.step_dt
                diagnostics["sg_goals"] = reached
                if cfg.stairs_stall_weight > 0:
                    terms["sg_stall"] = -cfg.stairs_stall_weight * self.sg_stalling.float()
                if cfg.stairs_margin_weight > 0:
                    # the plans' rule: keep the centre of gravity well inside the paws on the ground (stairs tiles,
                    # while not fallen)
                    margin = self.support_margin()
                    on = self.stairs_goal_on() & ~(self.fallen if cfg.recover else torch.zeros_like(self.stairs_goal_on()))
                    share = (margin / cfg.stairs_margin_target).clamp(-1.0, 1.0)
                    terms["sg_margin"] = cfg.stairs_margin_weight * share * on.float()
                    diagnostics["sg_margin_cm"] = 100.0 * margin * on.float()
            moving = self._command_moving()
        elif cfg.task == "goal":
            dist = self.goal_distance()
            approach = (self._goal_prev_dist - dist) / self.step_dt
            self._goal_prev_dist.copy_(dist)
            reached = dist < cfg.goal_reach_radius
            hit = self.obstacle_contact()
            terms["goal_progress"] = cfg.goal_progress_weight * approach.clamp(-cfg.goal_progress_cap, cfg.goal_progress_cap)
            terms["goal_reached"] = cfg.goal_reach_bonus * reached.float() / self.step_dt
            terms["collision"] = -cfg.collision_weight * hit.float()
            if cfg.terrain_kind == "enclosure":
                excess = (dist - self.goal_ideal_dist).clamp(min=0.0)
                terms["closeness"] = cfg.closeness_weight * torch.where(
                    self.goal_unreachable, torch.exp(-excess / cfg.closeness_excess_scale), torch.exp(-dist / cfg.closeness_scale))
            diagnostics = {"goals": reached.float() / self.step_dt, "collision": hit.float()}
            self.episode_goals += reached.float()
            self.episode_collision_steps += hit.float()
            if torch.any(reached):
                self._new_goal(reached.nonzero(as_tuple=False).squeeze(-1))
            moving = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        elif cfg.task == "jump":
            # height of every paw sole above the floor; the lowest one is the jump height (0 while any paw touches)
            quat = d.body_quat_w.torch[:, self._foot_body_ids]
            pos = d.body_pos_w.torch[:, self._foot_body_ids]
            offsets = self._foot_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
            tip = pos + quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(pos.shape)
            sole = tip[..., 2] - self._foot_radius - self.terrain.env_origins[:, 2:3]
            clearance = sole.min(dim=-1).values.clamp(min=0.0)
            # airborne: no sensed body touches anything (paws, shins, thighs, shoulders, body), so kneeling on the
            # thighs with the paws up does not count
            forces = self.contact.data.net_forces_w.torch
            airborne = (torch.linalg.norm(forces, dim=-1) < 1.0).all(dim=-1) & (clearance > 0.003)
            # only an upright jump counts (tilt < ~35 deg): upside down the paws are on top (a somersault exploit)
            upright = -d.projected_gravity_b.torch[:, 2] > 0.82
            clearance = torch.where(airborne & upright, clearance, torch.zeros_like(clearance))
            gain = (clearance - self._jump_best).clamp(min=0.0)
            self._jump_best.copy_(torch.maximum(self._jump_best, clearance))  # in place: buffers must stay normal tensors
            rise = (d.root_pos_w.torch[:, 2] - self.terrain.env_origins[:, 2] - cfg.base_height_target).clamp(min=0.0)
            self._rise_best.copy_(torch.maximum(self._rise_best, rise))
            self._air_steps += airborne.float()
            terms["jump_height"] = cfg.jump_weight * gain / self.step_dt  # summed x dt: jump_weight x best height
            terms["jump_rise"] = cfg.jump_rise_weight * rise
            terms["jump_air"] = cfg.jump_air_weight * airborne.float()
            diagnostics = {"best_clearance": gain / self.step_dt, "airborne": airborne.float()}
            moving = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            vel_w = d.root_lin_vel_w.torch
            heading = quat_apply(d.root_quat_w.torch, d.FORWARD_VEC_B.torch)
            terms["forward"] = cfg.forward_weight * torch.clamp(vel_w[:, 0], max=cfg.forward_speed_cap)
            terms["lateral"] = -cfg.lateral_weight * torch.abs(vel_w[:, 1])
            terms["heading"] = -cfg.heading_weight * (1.0 - heading[:, 0])
            moving = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if cfg.gait in DOG_GAITS:
            forces = self.contact.data.net_forces_w.torch
            in_contact = torch.linalg.norm(forces[:, self._foot_sensor_ids], dim=-1) > 1.0
            idle = ~moving.unsqueeze(-1)  # standing still: all four paws down
            # walking and trotting at these speeds have no flight phase: penalize moments with no paw on the ground
            terms["flight"] = -cfg.flight_weight * (~in_contact.any(dim=-1) & moving).float()
            diagnostics["paws_down"] = in_contact.float().sum(dim=-1)
            if cfg.stumble_weight > 0:
                # a paw pushed sideways much harder than it is pressed down: a toe against a step edge
                paw_f = forces[:, self._foot_sensor_ids]
                side = torch.linalg.norm(paw_f[..., :2], dim=-1)
                stumble = (side > 2.0 * paw_f[..., 2].abs()) & (side > 1.0)
                terms["stumble"] = -cfg.stumble_weight * stumble.float().sum(dim=-1) * self._stairs_terms_on()
            if cfg.foothold_weight > 0 and self.height_scanner is not None:
                # touching down with a step edge under the paw (paid once per touchdown)
                landed = self.contact.compute_first_contact(self.step_dt).torch[:, self._feet_ids].bool()
                top, low, seen = self._ground_near_paws(self._paw_tips(), (-0.045, 0.045), 0.035)
                edge = seen & (top - low > 0.02)
                terms["foothold"] = (-cfg.foothold_weight * (landed & edge).float().sum(dim=-1) / self.step_dt
                                     * self._stairs_terms_on())
            if cfg.lift_first_weight > 0 and self.height_scanner is not None:
                # a swinging paw still below a step up within 6 cm ahead of it should rise, not push forwards into it
                tip = self._paw_tips()
                sole = tip[..., 2] - self._foot_radius
                top, _, seen = self._ground_near_paws(tip, (0.0, 0.06), 0.04)
                riser = seen & (top - self._foot_ground_z > 0.01) & (sole < top + 0.005) & ~in_contact
                _, _, yaw = euler_xyz_from_quat(d.root_quat_w.torch)
                paw_v = d.body_lin_vel_w.torch[:, self._foot_body_ids]
                ahead = torch.cos(yaw).unsqueeze(-1) * paw_v[..., 0] + torch.sin(yaw).unsqueeze(-1) * paw_v[..., 1]
                terms["lift_first"] = (-cfg.lift_first_weight * (ahead.clamp(min=0.0) * riser.float()).sum(dim=-1)
                                       * self._stairs_terms_on())
            if cfg.stairs_support_weight > 0 and self._col_stairs is not None:
                on_stairs = self._col_stairs[self.terrain.terrain_types]
                steady = on_stairs & moving & (in_contact.sum(dim=-1) >= 3)
                terms["stairs_support"] = cfg.stairs_support_weight * steady.float()
            if cfg.gait == "trot":
                want = (torch.sin(self._phase).unsqueeze(-1) * self._trot_sign) > 0.0  # this foot should be in stance
                terms["trot"] = cfg.gait_trot_weight * (in_contact == (want | idle)).float().mean(dim=-1)
            elif cfg.gait == "walk_trot":
                # a four-beat walk when slow (3 paws down), a trot when fast, blended over a speed band
                cycle = (self._phase / (2.0 * math.pi)).unsqueeze(-1)
                walk = torch.remainder(cycle - self._walk_offset, 1.0) < 0.75
                trot = torch.remainder(cycle - self._trot_offset, 1.0) < 0.5
                lo, hi = cfg.gait_walk_trot_speed
                speed = torch.linalg.norm(self.commands[:, :2], dim=-1)
                w = torch.clamp((speed - lo) / (hi - lo), 0.0, 1.0)
                match_walk = (in_contact == (walk | idle)).float().mean(dim=-1)
                match_trot = (in_contact == (trot | idle)).float().mean(dim=-1)
                terms["trot"] = cfg.gait_trot_weight * ((1.0 - w) * match_walk + w * match_trot)
            else:  # free: no prescribed rhythm, only standing still is asked for
                terms["trot"] = cfg.gait_trot_weight * (in_contact | ~idle).float().mean(dim=-1)
            if getattr(self, "_pt", None) is not None:
                track, support, plan_diag = self._plan_terms(in_contact)
                terms["plan_track"] = cfg.plan_track_weight * track
                terms["plan_support"] = cfg.plan_support_weight * support
                diagnostics.update(plan_diag)
            height = self.ground_height()
            # linear: also a clear signal when far too low (a crawl), plus a peak for standing at the right height
            error = (height - cfg.base_height_target).abs()
            terms["base_height"] = cfg.base_height_weight * (torch.exp(-torch.square(error / 0.02)) - error / 0.05)
            if self._sg_col is not None and cfg.stairs_goal_height_scale != 1.0:
                terms["base_height"] = terms["base_height"] * (
                    1.0 + (cfg.stairs_goal_height_scale - 1.0) * self.stairs_goal_on().float())
            touching = torch.linalg.norm(forces[:, self._other_sensor_ids], dim=-1) > 1.0
            terms["undesired_contact"] = -cfg.undesired_contact_weight * touching.any(dim=-1).float()
            q = d.joint_pos.torch
            clash = sum(torch.square(torch.clamp(q[:, f] - q[:, r] - cfg.leg_clash_margin, min=0.0)) for f, r in self._clash_pairs)
            terms["leg_clash"] = -cfg.leg_clash_weight * clash
            # standing still means standing still: no marching or jitter when there is nothing to do
            idle = (~moving).float()
            if cfg.recover and cfg.stand_still_upright_only:
                idle = idle * (~fallen).float()  # getting up is not "standing still"
            terms["stand_still"] = -cfg.stand_still_weight * idle * torch.sum(d.joint_vel.torch[:, self._act_ids].abs(), dim=-1)
            terms["abduction_pose"] = -cfg.abduction_pose_weight * torch.sum(torch.square(d.joint_pos.torch[:, self._abd_ids]), dim=-1)
        if self.amp is not None:
            # style: how much the last transition looks like the real dog's motion
            ids = self._amp_ids
            cur = torch.cat((d.joint_pos.torch[:, ids], d.joint_vel.torch[:, ids], d.root_lin_vel_b.torch,
                             d.root_ang_vel_b.torch, d.projected_gravity_b.torch), dim=-1)
            trans = torch.cat((self._amp_prev, cur), dim=-1)
            # the previous state must be from the same episode (and exist: random start lengths at the first step)
            valid = (self.episode_length_buf > 1) & (self._amp_steps > 0)
            style = self.amp.reward(trans) * valid.float()
            if cfg.recover:  # lying on the floor is not the recorded dog's style: not scored, not shown to the judge
                style = style * (~fallen).float()
                valid = valid & ~fallen
            if cfg.amp_skip_types and self._col_amp_skip is not None:
                # no style on these tiles (stairs, rubble: the recorded dog only walked on a flat floor), nor are
                # their motions shown to the discriminator
                skip = self._col_amp_skip[self.terrain.terrain_types]
                style = style * (~skip).float()
                valid = valid & ~skip
            terms["style"] = cfg.amp_weight * style
            diagnostics["style"] = style
            self.amp_style.copy_(style)
            if bool(valid.any()):
                self.amp.add(trans[valid])
            self._amp_prev.copy_(cur)
            self._amp_steps += 1
            if self._amp_steps % cfg.amp_train_every == 0:
                self.amp.train(cfg.amp_train_steps)
        if self.contact is not None and cfg.feet_air_time_weight > 0:
            first_contact = self.contact.compute_first_contact(self.step_dt).torch[:, self._feet_ids]
            last_air = self.contact.data.last_air_time.torch[:, self._feet_ids]
            air = torch.sum((last_air - cfg.feet_air_time_target) * first_contact, dim=-1) * moving
            # first_contact fires once per touchdown; divide by step_dt so the per-second weighting holds
            terms["feet_air_time"] = cfg.feet_air_time_weight * air / self.step_dt
        if cfg.foot_clearance_weight > 0:
            terms["foot_clearance"] = cfg.foot_clearance_weight * self._foot_clearance(moving)
        if cfg.recover:
            # while fallen: the get-up rewards (rising potential-based); a fall costs once, when it happens
            f = fallen.float()
            target = cfg.base_height_target
            below = (height_r < target).float() * (self.episode_length_buf > 1).float()
            rise = ((height_r - self._gu_height) / self.step_dt).clamp(-0.5, 0.5) * below
            self._gu_height.copy_(height_r)
            terms["gu_upright"] = cfg.getup_upright_weight * 0.5 * (1.0 + up_r) * f
            terms["gu_height"] = cfg.getup_height_weight * (height_r / target).clamp(0.0, 1.0) * up_r.clamp(min=0.0) * f
            terms["gu_rise"] = cfg.getup_rise_weight * rise * f
            terms["fall"] = -cfg.recover_fall_penalty * (fallen & ~self._was_fallen).float() / self.step_dt
            self._was_fallen.copy_(fallen)
            diagnostics["fallen"] = f

        reward = sum(terms.values()) * self.step_dt
        for k, v in list(terms.items()) + [(f"_{k}", v) for k, v in diagnostics.items()]:
            self._episode_sums[k] += v * self.step_dt
        return reward - cfg.fall_penalty * self.reset_terminated.float()

    def _getup_rewards(self) -> torch.Tensor:
        """Get up: upright, height, rising (potential-based), standing still on four paws near the rest pose; energy,
        smoothness, joint limits and servo protection as costs."""
        cfg, d = self.cfg, self.robot.data
        up = -d.projected_gravity_b.torch[:, 2]
        height = d.root_pos_w.torch[:, 2] - self.terrain.env_origins[:, 2]
        target = cfg.base_height_target
        motors = self._obs_ids
        forces = self.contact.data.net_forces_w.torch
        # at least three paws carry weight: a dog standing still often rests one paw lightly (a walker standing still
        # had one paw at ~0.7 N against ~6 N on each of the others)
        paws_down = torch.linalg.norm(forces[:, self._foot_sensor_ids], dim=-1) > 0.5
        other = (torch.linalg.norm(forces[:, self._other_sensor_ids], dim=-1) > 1.0).any(dim=-1)
        standing = (up > 0.9) & ((height - target).abs() < 0.025) & (paws_down.sum(dim=-1) >= 3) & ~other
        self.getup_standing.copy_(standing)
        # height change below standing height; a step after a reset counts nothing (the robot was just moved)
        below = (height < target).float() * (self.episode_length_buf > 1).float()
        rise = ((height - self._gu_height) / self.step_dt).clamp(-0.5, 0.5) * below
        self._gu_height.copy_(height)
        q = d.joint_pos.torch[:, motors] / self.joint_range[motors]
        torque, speed = d.applied_torque.torch[:, motors], d.joint_vel.torch[:, motors]
        stall = ((torque.abs() > cfg.getup_stall_torque) & (speed.abs() < 0.5)).float().sum(dim=-1)
        power = torch.sum(torch.abs(torque * speed), dim=-1)
        action_rate = torch.sum(torch.square(self.actions[:, self._act_ids] - self.prev_actions[:, self._act_ids]), dim=-1)
        over = torch.clamp(q.abs() - 0.95, min=0.0)
        terms = {
            "gu_upright": cfg.getup_upright_weight * 0.5 * (1.0 + up),
            "gu_height": cfg.getup_height_weight * (height / target).clamp(0.0, 1.0) * up.clamp(min=0.0),
            "gu_rise": cfg.getup_rise_weight * rise,
            "gu_stand": cfg.getup_stand_weight * standing.float(),
            "gu_pose": cfg.getup_pose_weight * torch.exp(-torch.mean(torch.square(q), dim=-1) / 0.1) * (up > 0.8).float(),
            "gu_stall": -cfg.getup_stall_weight * stall,
            "energy": -cfg.energy_weight * power,
            "action_rate": -cfg.action_rate_weight * action_rate,
            "joint_limit": -cfg.joint_limit_weight * torch.sum(over, dim=-1),
        }
        reward = sum(terms.values()) * self.step_dt
        for k, v in list(terms.items()) + [("_standing", standing.float())]:
            self._episode_sums[k] += v * self.step_dt
        return reward

    # ------------------------------------------------------------------ episode handling

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        up = -self.robot.data.projected_gravity_b.torch[:, 2]
        fell = up < self.cfg.fall_up_threshold
        if self.cfg.gait in DOG_GAITS or self.cfg.task == "jump":
            # the body touching the ground also counts as a fall (no lying down and kicking)
            force = self.contact.data.net_forces_w.torch[:, self._chassis_sensor_id]
            fell = fell | (torch.linalg.norm(force, dim=-1) > 1.0).any(dim=-1)
        if self.cfg.getup:
            fell = torch.zeros_like(fell)  # lying on the ground is where getting up starts
        if self.cfg.recover:
            # a fall is allowed (get up and go on); staying down too long ends the episode (fallen: last step's state)
            self._down_time.copy_(torch.where(self.fallen, self._down_time + self.step_dt, torch.zeros_like(self._down_time)))
            fell = self._down_time > self.cfg.recover_max_down_s
        if self._sg_col is not None:
            self._sg_update()
            fell = fell | self.sg_hard_now  # fell down the stairs
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        # record the finished episode before the state is reset (skip the very first reset at startup)
        started = self.episode_length_buf[env_ids] > 0
        if torch.any(started):
            ids = env_ids[started]
            disp = self.robot.data.root_pos_w.torch[ids] - self.start_pos[ids]
            self.last_episode_distance[ids] = disp[:, 0]
            self.last_episode_fell[ids] = self.reset_terminated[ids]
            if self.cfg.task == "jump":
                self.last_episode_jump[ids] = self._jump_best[ids]
                self.last_episode_rise[ids] = self._rise_best[ids]
                self.last_episode_air[ids] = self._air_steps[ids] * self.step_dt
                self._jump_best[ids] = 0.0
                self._rise_best[ids] = 0.0
                self._air_steps[ids] = 0.0
            self.finished_episodes[ids] += 1
            log = self.extras.setdefault("log", {})
            log["Episode/forward_distance"] = disp[:, 0].mean().item()
            log["Episode/fall_rate"] = self.reset_terminated[ids].float().mean().item()
            seconds = (self.episode_length_buf[ids].float() * self.step_dt).clamp(min=self.step_dt)
            if self.cfg.task == "goal":
                self.last_episode_goals[ids] = self.episode_goals[ids]
                self.last_episode_collision[ids] = self.episode_collision_steps[ids] * self.step_dt / seconds
                self.last_episode_seconds[ids] = seconds
                origin = self.terrain.env_origins[ids, :2]
                self.last_episode_end_xy[ids] = self.robot.data.root_pos_w.torch[ids, :2] - origin
                self.last_episode_start_xy[ids] = self.episode_start_xy[ids]
                self.last_episode_goal_xy[ids] = self.goal_pos[ids, :2] - origin
                self.last_episode_unreachable[ids] = self.goal_unreachable[ids]
                log["Episode/goals_per_minute"] = (self.episode_goals[ids] / seconds * 60.0).mean().item()
                if (self.cfg.terrain_kind == "obstacles" or self.cfg.cup) and self.curriculum_enabled:
                    fell = self.reset_terminated[ids]
                    up = (self.episode_goals[ids] >= (1 if self.cfg.cup else 2)) & ~fell
                    down = (self.episode_goals[ids] == 0) | fell
                    self.terrain.update_env_origins(ids, up, down & ~up)
                    log["Curriculum/terrain_level"] = self.terrain.terrain_levels.float().mean().item()
            if self._sg_col is not None:
                last, types = self.sg_last, self.terrain.terrain_types[ids]
                last["dir"][ids] = self.sg_first_dir[ids]
                last["reached"][ids] = (self.sg_leg[ids] >= 1).float()
                last["time"][ids] = self.sg_first_time[ids]
                last["falls"][ids] = self.sg_falls[ids]
                last["progress"][ids] = self.sg_progress[ids]
                last["hard"][ids] = self.sg_hard_leg1[ids].float()
                # the episode ended because the dog stayed down (recover_max_down_s) before reaching its first goal
                last["stayed_down"][ids] = (self.reset_terminated[ids] & ~self._sg_hard_ep[ids] & (self.sg_leg[ids] == 0)).float()
                last["rise"][ids] = self._sg_rises[self.terrain.terrain_levels[ids]]
                last["tread"][ids] = self._sg_tread_col[types]
                last["walls"][ids] = self._sg_walls_col[types].float()
            if self.cfg.task != "goal" and self.cfg.terrain_kind == "rough" and self.curriculum_enabled:
                self._update_curriculum(ids, seconds)
                log["Curriculum/terrain_level"] = self.terrain.terrain_levels.float().mean().item()
            if self.amp is not None:
                for k, v in self.amp.last_loss.items():
                    log[f"AMP/{k}"] = v
            for k, v in self._episode_sums.items():
                if k.startswith("_"):
                    log[f"Episode/{k[1:]}"] = (v[ids] / seconds).mean().item()
                else:
                    log[f"Episode_Reward/{k}"] = (v[ids] / self.max_episode_length_s).mean().item()
        for v in self._episode_sums.values():
            v[env_ids] = 0.0

        super()._reset_idx(env_ids)
        self.actions[env_ids] = 0.0
        self._expr_next[env_ids] = 0.0
        if self.cfg.gait in DOG_GAITS:
            self._phase[env_ids] = 0.0
        self._expr_params[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        if self.cfg.sensor_suite != "ideal":
            n = len(env_ids)
            self._gyro_bias[env_ids] = sample_uniform(-self.cfg.gyro_bias, self.cfg.gyro_bias, (n, 3), self.device)
            self._accel_bias[env_ids] = sample_uniform(-self.cfg.accel_bias, self.cfg.accel_bias, (n, 3), self.device)
            self._flow_held[env_ids] = 0.0
        self._hist_fill[env_ids] = True
        if self.cfg.recover:
            self.fallen[env_ids] = False
            self._was_fallen[env_ids] = False
            self._down_time[env_ids] = 0.0
            self._gu_height[env_ids] = self.cfg.base_height_target
        if self.cfg.topple_push:
            self._topple_next[env_ids] = sample_uniform(*self.cfg.topple_interval, (len(env_ids),), self.device)
        if self.terrain_map is not None:
            self.terrain_map.clear(env_ids)
            self.odometry.reset(env_ids)
            self._map_step = -1  # the next observation updates the map, also within the same step (env.reset())
            self._dark_until[env_ids] = 0.0
            # frames captured before the reset never reach the new map (copies: they may be inference-mode tensors)
            self._map_pending = deque((t, p, v.clone().index_fill_(0, env_ids.long(), False)) for t, p, v in self._map_pending)
        if self.low_level is not None:
            self._ll_fill[env_ids] = True
            self.commands[env_ids] = 0.0

        # root: default pose, random heading
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.terrain.env_origins[env_ids]
        if self.cfg.terrain_kind == "enclosure":
            # start somewhere outside the ring wall
            r = sample_uniform(*self.cfg.enclosure_start, (len(env_ids),), self.device)
            a = sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device)
            if self.cfg.cup:
                spread = self.cfg.cup_eval_start_spread if self.eval_mode else self.cfg.cup_start_spread
                a = math.pi + sample_uniform(-spread, spread, (len(env_ids),), self.device)
            pose[:, 0] += r * torch.cos(a)
            pose[:, 1] += r * torch.sin(a)
            self.episode_start_xy[env_ids] = pose[:, :2] - self.terrain.env_origins[env_ids, :2]
        yaw = sample_uniform(-self.cfg.reset_yaw_range, self.cfg.reset_yaw_range, (len(env_ids),), self.device)
        if self.cfg.stairs_square_yaw > 0 and self._col_stairs is not None:
            # on stair tiles face the steps (the pyramid's sides run along the world axes), roughly straight on
            sq = self.cfg.stairs_square_yaw
            axis = torch.randint(0, 4, (len(env_ids),), device=self.device).float() * (math.pi / 2)
            square = axis + sample_uniform(-sq, sq, (len(env_ids),), self.device)
            yaw = torch.where(self._col_stairs[self.terrain.terrain_types[env_ids]], square, yaw)
        mid_climb = None
        if self._sg_col is not None:
            # in front of the staircase or on the landing, facing the goal (or mid-climb, from a stair plan)
            mid_climb = self._sg_reset(env_ids, pose, yaw)
        zeros = torch.zeros_like(yaw)
        roll, pitch = zeros.clone(), zeros.clone()
        if mid_climb is not None:
            pitch = torch.where(mid_climb[0], -mid_climb[1], pitch)  # nose up = negative rotation about y
        fallen = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if self.cfg.getup:
            cat = self._getup_starts(env_ids)
            fallen = cat > 0
            n = len(env_ids)
            # dropped upright (1), on its side (2, either side), on its back (3); a little tilt either way
            roll = torch.where(cat == 2, (math.pi / 2) * torch.sign(torch.rand(n, device=self.device) - 0.5), roll)
            roll = torch.where(cat == 3, torch.full_like(roll, math.pi), roll)
            roll = roll + fallen.float() * sample_uniform(-0.3, 0.3, (n,), self.device)
            pitch = fallen.float() * sample_uniform(-0.3, 0.3, (n,), self.device)
            pose[:, 2] += fallen.float() * 0.08  # dropped from a little higher: the legs never start in the ground
        if self._tilt_col is not None:
            tilt = self.tilt_on()[env_ids]
            if self.cfg.tilt_reset_yaw_random:
                yaw = torch.where(tilt, sample_uniform(-math.pi, math.pi, (len(env_ids),), self.device), yaw)
        pose[:, 3:7] = quat_mul(quat_from_euler_xyz(roll, pitch, yaw), pose[:, 3:7])
        if self._tilt_col is not None and bool(tilt.any()):
            # tilt tiles: placed standing on the plate, perpendicular to it (rotated about the axis that lowers the
            # downhill side), at the standing height above it
            ids = env_ids[tilt]
            up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(ids), 3)
            n = self.tilt_normal(ids)
            axis = torch.linalg.cross(up, n)
            sin = torch.linalg.norm(axis, dim=-1)
            axis = torch.where(sin.unsqueeze(-1) > 1e-6, axis / sin.clamp(min=1e-9).unsqueeze(-1),
                               torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(len(ids), 3))
            q_tilt = quat_from_angle_axis(torch.atan2(sin, n[:, 2]), axis)
            height = self.robot.data.default_root_pose.torch[ids, 2]
            pose[tilt, :3] = self.terrain.env_origins[ids] + n * height.unsqueeze(-1)
            pose[tilt, 3:7] = quat_mul(q_tilt, pose[tilt, 3:7])
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        self.start_pos[env_ids] = pose[:, :3]

        # joints: small random offsets around the rest pose
        noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.robot.num_joints), self.device)
        joint_pos = noise * self.cfg.reset_joint_fraction * self.joint_range
        if self.cfg.getup:  # fallen: the legs anywhere in their range
            joint_pos = torch.where(fallen.unsqueeze(-1), noise * 0.9 * self.joint_range, joint_pos)
            self._gu_height[env_ids] = pose[:, 2] - self.terrain.env_origins[env_ids, 2]
        if self._sg_col is not None:
            self.sg_mid_climb[env_ids] = False if mid_climb is None else mid_climb[0]
        if mid_climb is not None:
            m = mid_climb[0]
            joint_pos = torch.where(m.unsqueeze(-1), mid_climb[2], joint_pos)
            # the last (and delayed) command holds the pose, so the first step does not pull the legs to rest
            held = mid_climb[2][m] / (self.joint_range * self._target_scale)
            self.actions[env_ids[m]] = held
            self.prev_actions[env_ids[m]] = held
        self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.actuators.target_command.set_position_index(value=joint_pos, env_ids=env_ids)
        self._heading_held[env_ids] = False
        self._yaw_asked[env_ids] = 0.0
        if self.cfg.task == "command" and self.command_override is None:
            self._resample_commands(env_ids)
            # a new episode starts from standstill when commands ramp, else at the new command at once
            self.commands[env_ids] = 0.0 if self.cfg.command_accel > 0 else self._command_target[env_ids]
        if self.cfg.task == "goal":
            self.episode_goals[env_ids] = 0.0
            self.episode_collision_steps[env_ids] = 0.0
            self._new_goal(env_ids, pose[:, :2])
            # the robot was just moved: measure the first distance from its new start position
            self._goal_prev_dist[env_ids] = torch.linalg.norm(self.goal_pos[env_ids, :2] - pose[:, :2], dim=-1)
            if hasattr(self, "_cam_buffer"):  # the overhead camera sees the robot at its new start pose
                start = torch.cat((pose[:, :2], yaw.unsqueeze(-1)), dim=-1)
                self._cam_buffer[env_ids] = start.unsqueeze(1)
                self._cam_pose[env_ids] = start
