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

import math
from collections import deque
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform

from .locomotion_env_cfg import EvoLocomotionEnvCfg

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

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
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
        due = (t >= self._topple_next).nonzero(as_tuple=False).squeeze(-1)
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
        height = d.root_pos_w.torch[:, 2] - self.terrain.env_origins[:, 2]
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
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)

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
        height = d.root_pos_w.torch[:, 2:3] - self.terrain.env_origins[:, 2:3]
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
        return {"policy": policy, "critic": critic}

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
            moving = (torch.linalg.norm(self.commands[:, :2], dim=-1) > 0.05) | (self.commands[:, 2].abs() > 0.2)
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
            height = d.root_pos_w.torch[:, 2] - self.terrain.env_origins[:, 2]
            # linear: also a clear signal when far too low (a crawl), plus a peak for standing at the right height
            error = (height - cfg.base_height_target).abs()
            terms["base_height"] = cfg.base_height_weight * (torch.exp(-torch.square(error / 0.02)) - error / 0.05)
            touching = torch.linalg.norm(forces[:, self._other_sensor_ids], dim=-1) > 1.0
            terms["undesired_contact"] = -cfg.undesired_contact_weight * touching.any(dim=-1).float()
            q = d.joint_pos.torch
            clash = sum(torch.square(torch.clamp(q[:, f] - q[:, r] - cfg.leg_clash_margin, min=0.0)) for f, r in self._clash_pairs)
            terms["leg_clash"] = -cfg.leg_clash_weight * clash
            # standing still means standing still: no marching or jitter when there is nothing to do
            idle = (~moving).float()
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
            elif self.cfg.terrain_kind == "rough" and self.curriculum_enabled:
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
        zeros = torch.zeros_like(yaw)
        roll, pitch = zeros.clone(), zeros.clone()
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
        pose[:, 3:7] = quat_mul(quat_from_euler_xyz(roll, pitch, yaw), pose[:, 3:7])
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        self.start_pos[env_ids] = pose[:, :3]

        # joints: small random offsets around the rest pose
        noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.robot.num_joints), self.device)
        joint_pos = noise * self.cfg.reset_joint_fraction * self.joint_range
        if self.cfg.getup:  # fallen: the legs anywhere in their range
            joint_pos = torch.where(fallen.unsqueeze(-1), noise * 0.9 * self.joint_range, joint_pos)
            self._gu_height[env_ids] = pose[:, 2] - self.terrain.env_origins[env_ids, 2]
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
