"""Configuration of the forward-locomotion task (safe to import before the simulator is launched).

See locomotion_env.py for the task itself.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class DomainRandomizationCfg:
    """Startup randomization so policies survive the gap between simulation and a printed robot."""

    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    servo_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
        },
    )


@configclass
class EvoLocomotionEnvCfg(DirectRLEnvCfg):
    # timing: physics at 200 Hz, policy at 50 Hz
    episode_length_s: float = 12.0
    decimation: int = 4
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4)

    # spaces are filled in by make_env_cfg() once the robot is known
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=0.6, replicate_physics=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # the robot's own materials (TPU feet vs PLA body) decide the friction: combine by multiplying with 1.0
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = None
    events: DomainRandomizationCfg | None = DomainRandomizationCfg()

    # robot description (filled by make_env_cfg)
    joint_ranges: list[float] = []  # symmetric range of every joint [rad], in articulation order
    joint_names: list[str] = []

    # control
    servo_model: str = "dc_motor"  # see core/sim/robot_cfg.py
    action_delay_steps: int = 1  # 0 or 1 policy step (20 ms): IMU read + inference + PWM update on the ESP32

    # senses (see roadmap M2b):
    #   "ideal"     perfect gravity direction + gyro (the original setup)
    #   "imu"       MPU6050-like: noisy gravity estimate, gyro with per-robot bias, raw accelerometer readings
    #   "imu_flow"  "imu" + a downward optical-flow sensor measuring speed over ground (noisy, with dropouts)
    sensor_suite: str = "ideal"
    obs_history: int = 1  # policy sees this many most recent sensor frames (memory to infer speed / being stuck)
    imu: ImuCfg | None = None
    gravity_noise: float = 0.02  # orientation-filter error on the gravity direction (unit vector components)
    gyro_noise: float = 0.03  # [rad/s] per reading
    gyro_bias: float = 0.02  # [rad/s] per-robot constant offset, resampled at reset
    accel_noise: float = 0.08  # [m/s^2] per reading (MPU6050 noise + vibration)
    accel_bias: float = 0.15  # [m/s^2] per-robot constant offset
    flow_rel_noise: float = 0.05  # optical flow: relative scale error per reading
    flow_abs_noise: float = 0.01  # [m/s]
    flow_dropout: float = 0.05  # chance per reading that the sensor reports nothing (last value is held)

    # observations
    gait_clock_hz: float = 1.5
    ang_vel_scale: float = 0.25
    joint_vel_scale: float = 0.05
    policy_sees_joint_velocity: bool = False  # SG90s report no position/velocity; angle = last command

    # reset randomization
    reset_yaw_range: float = 0.3  # [rad]
    reset_joint_fraction: float = 0.1  # of each joint's range

    # task: "forward" = walk along +x as fast as possible (the original task),
    #       "command" = follow a velocity command (forward, sideways, turn rate) given in the robot's frame,
    #       "goal"    = walk to a goal point (given relative to the robot) without hitting obstacles (roadmap M3)
    task: str = "forward"

    # goal task
    goal_reach_radius: float = 0.12  # [m]
    goal_progress_weight: float = 10.0  # per m/s of approach speed
    goal_progress_cap: float = 0.4  # [m/s]
    goal_reach_bonus: float = 5.0  # one-off per goal reached
    collision_weight: float = 2.0  # per second in contact with an obstacle
    # enclosure terrain (unreachable-goal test): the goal is either inside the closed ring (off centre, cannot be
    # reached: get as close as possible) or a normal goal outside it; robots start outside the ring
    unreachable_prob: float = 0.5
    enclosure_goal_offset: tuple[float, float] = (0.10, 0.30)  # [m] goal inside the ring, from its centre
    enclosure_outside_goal: tuple[float, float] = (0.65, 1.20)  # [m] reachable goals, from the ring centre
    enclosure_start: tuple[float, float] = (0.75, 1.10)  # [m] start position, from the ring centre
    # per second: reachable goal exp(-distance / closeness_scale); unreachable goal exp(-excess / closeness_excess_scale),
    # excess = distance beyond the closest reachable one (known to the simulator only: it shapes the reward, the
    # policy never sees it)
    closeness_weight: float = 4.0
    closeness_scale: float = 0.25  # [m]
    closeness_excess_scale: float = 0.06  # [m]
    enclosure_far_start: bool = False  # start on the opposite side of the ring from the goal (videos)
    enclosure_robot_gap: float = 0.08  # [m] how close the robot's centre gets to the outer wall surface
    pocket: tuple[float, float, float] | None = None  # (ring radius, slot width, slot depth) on the pocket course
    # cup course (detour test): goals inside the open ring are reachable through its open side; robots start on the
    # closed side within +-cup_start_spread of it (pi = anywhere round the cup)
    cup: bool = False
    cup_start_spread: float = 3.14159  # [rad]
    cup_eval_start_spread: float = 0.45  # [rad] evaluation: always on the closed side
    # distance sensor (VL53L5CX-like 8x8 time-of-flight, at the chassis front, tilted down)
    tof: RayCasterCfg | None = None
    tof_max_range: float = 2.0  # [m]
    tof_noise_rel: float = 0.03
    tof_noise_abs: float = 0.005  # [m]
    tof_update_every: int = 3  # policy steps between readings: 50 Hz / 3 ~ 16.7 Hz (sensor: 15 Hz in 8x8 mode)
    # hierarchical goal task: the trained policy only outputs a steering command (vx, vy, yaw rate); a frozen walking
    # policy (TorchScript, trained on the command task with "imu" senses) turns it into servo angles
    low_level_policy: str = ""
    low_level_history: int = 8
    # goal signal: "perfect", or "camera" = an overhead camera tracking the robot (as in a real test setup):
    # position updates at ~25 Hz, 1 cm / 2 degree noise, 80 ms delay
    goal_signal: str = "perfect"
    camera_update_every: int = 2  # policy steps (50 Hz / 2 = 25 Hz)
    camera_delay_steps: int = 4  # 80 ms
    camera_pos_noise: float = 0.01  # [m]
    camera_yaw_noise: float = 0.035  # [rad]

    # natural gait (dog): "trot" = diagonal legs together in time with the gait clock, body at its standing height,
    # only the paws touch the ground, legs kept under the body. "" = no style terms (the spider-like robots).
    gait: str = ""  # "trot" | "walk_trot" (4-beat walk when slow, trot when fast) | "free" (no rhythm prescribed)
    gait_freq_range: tuple[float, float] = (1.4, 2.6)  # [Hz] trot frequency from slow to fast commands
    gait_freq_speed: float = 0.35  # [m/s] command speed at which the highest frequency is reached
    gait_walk_trot_speed: tuple[float, float] = (0.25, 0.40)  # [m/s] walk below, trot above, blended in between
    gait_trot_weight: float = 1.0
    flight_weight: float = 0.0  # per second with no paw on the ground while walking (walk and trot have no flight)  # per second, share of feet whose contact matches the trot pattern
    base_height_target: float = 0.0  # [m] chassis height above the ground
    base_height_weight: float = 1.0  # per second: exp(-(error / 2 cm)^2) - error / 5 cm
    undesired_contact_weight: float = 2.0  # per second with chassis, shoulder or thigh touching anything
    abduction_pose_weight: float = 1.0  # per rad^2 of sideways leg angle (no sprawling)
    stand_still_weight: float = 0.0  # per second, per rad/s of summed joint speed while the command is "stand still"
    # front and rear leg on one side collide when their hips swing towards each other by more than ~0.8 rad in total
    # (prototypes/puppy interference check); the simulation has no self-collision, so penalize getting close
    leg_clash_margin: float = 0.55  # [rad] closing angle where the penalty starts
    leg_clash_weight: float = 20.0  # per rad^2 beyond the margin

    # expression joints (dog: neck nod, head tilt, tail wag): not driven by the policy. During training they follow
    # random motions (still poses, slow looks, fast wagging) so the walker learns to keep its balance whatever the
    # head and tail do; afterwards they can be scripted freely (set env.expression_override).
    expression_joints: list[str] = []
    passive_joints: list[str] = []  # spring hinges (flexible back): not driven, not observed by the policy
    joint_depth: dict = {}  # joint -> depth in the joint tree, passive joints not counted (set by make_env_cfg)
    # penalty per second on the policy's raw actions beyond -1..1 (summed over joints). Without it, clipped actions
    # drift far beyond the limit (the clip hides it) and the policy turns into bang-bang control. Needs the trainer to
    # pass unclipped actions (train.py does when this is > 0).
    action_limit_weight: float = 0.0
    robot_mass: float = 0.0  # [kg] set by make_env_cfg (for the cost of transport)
    # policy action -1..1 maps to this share of each joint's range around the rest pose (1 = the full range)
    action_scale: float = 1.0
    expression_hold_s: tuple[float, float] = (1.0, 4.0)  # how long one motion lasts
    expression_freq_hz: tuple[float, float] = (0.3, 3.0)
    expression_still_prob: float = 0.4  # share of motions that are a fixed pose

    # jump task (dog): the reward is the highest point of the episode, measured at the LOWEST paw (only a real jump,
    # with all four paws off the ground, counts). Paid as it is reached: each step pays the amount by which the
    # episode's record is beaten, so an episode's jump reward = jump_weight x its best height.
    jump_weight: float = 50.0  # per metre of the best paw clearance
    jump_rise_weight: float = 10.0  # per second, per metre the body is above its standing height (shapes the push-off)
    jump_air_weight: float = 2.0  # per second with all four paws off the ground

    # style reward from real dog motion (AMP, core/sim/amp.py): motion file from core/mocap/retarget.py; "" = off
    amp_motion_file: str = ""
    amp_weight: float = 2.0  # per second, times the 0..1 style reward
    amp_train_every: int = 4  # policy steps between discriminator updates
    amp_train_steps: int = 1
    amp_mirror: bool = True  # add left-right mirrored recordings (removes the recorded dog's turning bias)
    amp_disc_file: str = ""  # a trained discriminator (amp_disc.pt of a run), loaded frozen: scores, does not learn

    # heading hold: while the command asks to walk without turning, the heading at that moment is kept: a drift becomes
    # a corrective turn command (gain * heading error, capped), which the policy sees and tracks like any turn command
    # (as Isaac Lab's heading commands; on the robot the heading comes from the IMU). 0 = off
    heading_hold_gain: float = 0.0
    # robustness (dog, with domain randomization; applied when the environment is created, so overrides work):
    # shove strength [m/s, yaw rad/s = 2x] and interval [s]; paw grip range (static friction, () = always TPU 1.0)
    push_velocity: float = 0.2
    push_interval: tuple = (4.0, 8.0)
    paw_friction_range: tuple = ()
    heading_hold_max: float = 0.5  # rad/s

    # command task: commands are resampled every few seconds, including standing still and turning in place
    command_resample_s: float = 4.0
    command_accel: float = 0.0  # [m/s^2] commands ramp towards their new value at this rate (0 = jump at once)
    command_yaw_accel: float = 0.0  # [rad/s^2]
    command_vx_range: tuple[float, float] = (-0.30, 0.35)  # [m/s]
    command_vy_range: tuple[float, float] = (-0.30, 0.30)  # [m/s]
    command_yaw_range: tuple[float, float] = (-1.2, 1.2)  # [rad/s]
    command_standing_prob: float = 0.1
    command_turn_in_place_prob: float = 0.15
    # evaluation commands always ask for real movement, so standing still can never score well
    eval_speed_range: tuple[float, float] = (0.10, 0.30)  # [m/s] planar speed, in a random direction
    eval_yaw_range: tuple[float, float] = (-0.6, 0.6)  # [rad/s]
    eval_heading_range: tuple[float, float] = (-3.14159, 3.14159)  # [rad] walking direction relative to the body
    tracking_lin_weight: float = 3.0
    tracking_lin_sigma: float = 0.12  # [m/s] velocity error at which the tracking reward drops to 1/e
    # dense shaping: speed made good along the commanded direction, capped at the commanded speed. Without it a
    # body that cannot walk yet gets almost no signal from the narrow tracking kernel (exp(-(0.2/0.12)^2) = 0.06)
    # and settles for turning on the spot.
    progress_weight: float = 15.0  # per m/s
    tracking_ang_weight: float = 1.5
    tracking_ang_sigma: float = 0.4  # [rad/s]
    vertical_vel_weight: float = 2.0
    roll_pitch_rate_weight: float = 0.05

    # terrain: "flat" plane or "rough" (core/sim/terrain.py) with a difficulty curriculum
    terrain_kind: str = "flat"
    curriculum: bool = True
    curriculum_up_error: float = 0.06  # mean velocity error [m/s] below which a robot moves to a harder row
    curriculum_down_error: float = 0.12  # above this (or after a fall) it moves to an easier row

    # foot contacts (enabled on rough terrain): reward proper swing phases so the feet clear edges
    contact_sensor: ContactSensorCfg | None = None
    foot_body_names: list[str] = []
    feet_air_time_weight: float = 0.0
    feet_air_time_target: float = 0.12  # [s] steps shorter than this are penalized, longer ones rewarded
    # foot clearance (stairs test B): reward a swinging foot for rising up to the target above where it last stood
    foot_clearance_weight: float = 0.0
    foot_clearance_target: float = 0.025  # [m]
    foot_offsets: list[tuple[float, float, float]] = []  # foot sphere centre in each foot link's frame
    foot_radius: list[float] = []
    # stairs training (journal step 35; rewards use the true terrain from the perfect scanner, the policy never sees it):
    # terrain-aware clearance: the lift target is the highest ground just ahead of the swinging paw (3 cm behind to
    # 10 cm ahead) + margin, at least foot_clearance_target, at most foot_clearance_max, above where it last stood
    foot_clearance_terrain: bool = False
    foot_clearance_margin: float = 0.015  # [m]
    foot_clearance_max: float = 0.10  # [m]
    stumble_weight: float = 0.0  # per second per paw pushing sideways against something (a toe against a step edge)
    foothold_weight: float = 0.0  # per touchdown with a step edge under the paw (ground within +-4.5 cm along, +-3.5 cm across differs > 2 cm)
    stairs_support_weight: float = 0.0  # per second on stair tiles while moving with at least 3 paws down
    amp_skip_types: tuple = ()  # terrain types where no style reward is paid (the recorded dog walked on a flat floor)
    # terrain types where the stairs terms above (terrain-aware clearance, stumble, foothold, lift first) apply; () =
    # everywhere. Set to amp_skip_types: stair technique on the stairs, the recorded dog's style elsewhere (otherwise
    # the terms change the gait on every tile, the style discriminator wins and the style reward drops to 0)
    stairs_terms_types: tuple = ()
    # lift first: per second, per m/s a swinging paw moves forwards while still below a step up just ahead of it (a
    # paw right in front of a riser must go up, not forwards into it)
    lift_first_weight: float = 0.0
    # square up (training only): dogs on stair tiles start facing the steps (an axis +- this [rad], 0 = off) and the
    # turn-rate commands there are scaled (0 = no turning on stairs: heading hold keeps them straight)
    stairs_square_yaw: float = 0.0
    stairs_turn_scale: float = 1.0
    # get up (journal step 36, dog command task with zero commands): episodes start fallen or standing, a fall does not
    # end the episode, and the rewards pay for getting upright, rising to standing height and standing on four paws
    getup: bool = False
    getup_mix_start: tuple = (0.4, 0.4, 0.2, 0.0)  # share of starts standing / dropped upright / on its side / on its back
    getup_mix_end: tuple = (0.15, 0.2, 0.3, 0.35)  # ... after getup_curriculum_steps policy steps
    getup_curriculum_steps: int = 60000  # policy steps of each robot (= iterations x 24)
    getup_category: int = -1  # tests and videos: -1 = the mix, 0-3 = always that start, -2 = robot index % 4
    getup_upright_weight: float = 2.0  # per second, (1 + up) / 2
    getup_height_weight: float = 2.0  # per second, share of standing height reached (only counted upright)
    getup_rise_weight: float = 5.0  # per m/s of height change below standing height (potential-based: no farming)
    getup_stand_weight: float = 4.0  # per second standing: upright, at height, 3+ paws carrying, nothing else touching
    getup_pose_weight: float = 1.0  # per second near the rest pose while upright
    getup_stall_torque: float = 2.5  # [N*m] servo protection: a servo pushing harder than this while hardly moving...
    getup_stall_weight: float = 0.5  # ... costs this per second
    # recover (journal step 38): walking with falls allowed. A fall no longer ends the episode (unless the dog stays
    # down too long); while fallen the get-up rewards apply and the walking style is not scored; each fall costs
    # recover_fall_penalty once; occasional topple shoves knock it over
    recover: bool = False
    recover_fall_penalty: float = 2.0  # once per fall
    recover_fallen_up: float = 0.82  # fallen: "up" below this (tilt > 35 deg) or the body touching the ground
    recover_max_down_s: float = 6.0  # still down after this long: the episode ends (as a fall)
    topple_push: tuple = ()  # (min, max) [m/s] horizontal shove in a random direction; () = off
    topple_interval: tuple = (8.0, 15.0)  # [s] between topple shoves of one robot
    topple_spin: float = 2.0  # [rad/s] up to this much random spin about a horizontal axis comes with it
    # collisions between the robot's own parts (legs with legs and body). Off for walking (cheaper); on for getting up,
    # where a leg passing through the body would be a solution the real robot cannot use
    self_collisions: bool = False
    # the policy also drives neck, head and tail (instead of their random motions): levers for rolling over (step 37)
    policy_drives_expression: bool = False
    # training only (not re-applied for tests and videos): stair treads from the easiest to the hardest row [m]
    stair_tread_range: tuple = ()
    # terrain height scan (stairs test C, simulation-only "perfect sensor"): heights of a grid around the robot
    height_scanner: RayCasterCfg | None = None
    height_scan_offset: float = 0.0  # chassis standing height, subtracted so a flat floor reads ~0
    height_scan_clip: float = 0.1  # [m] heights beyond +- this are clipped; the policy sees height / clip
    # False: only the current scan is observed, appended after the observation history (dog); True: the scan is part
    # of every history frame (experiment 1 runs)
    height_scan_in_history: bool = True
    height_scan_points: int = 0
    # terrain map (core/sim/terrain_map.py), the real robot's version of the height scan: a chest sensor sees the ground
    # ahead, its points stay in a map placed with the robot's drifting motion estimate, and the policy reads the same
    # scan grid from that map plus a known flag per point. The critic keeps the perfect scan.
    terrain_map: str = ""  # "" = perfect scan | "depth" (depth camera) | "tof" (VL53L5CX 8 x 8)
    map_camera: RayCasterCfg | None = None  # set by make_env_cfg
    map_camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # sensor position in the chassis frame [m]
    map_policy_input: bool = True  # False: keep the map up to date but give the policy the perfect scan (map checks)
    map_resolution: float = 0.025  # [m] cell size
    map_cells: int = 64  # per side: 1.6 m at 2.5 cm
    map_memory_s: float = 1e9  # cells not seen for this long are forgotten; 0 = no memory (only the latest frame)
    map_update_every: int = 2  # policy steps between sensor frames (depth camera ~30 Hz -> 25 Hz)
    map_delay_steps: int = 2  # policy steps from capture until a frame is in the map (exposure, transfer, fusion)
    map_min_range: float = 0.15  # [m] closer surfaces give no reading
    map_max_range: float = 2.0  # [m]
    map_noise_rel: float = 0.01  # range noise, share of the distance
    map_noise_abs: float = 0.003  # [m]
    map_tilt_noise: float = 0.017  # [rad] orientation-estimate error per frame (tilts the whole frame)
    map_odom_drift_xy: float = 0.03  # [m] position error std after 1 m walked
    map_odom_drift_yaw: float = 0.035  # [rad] heading error std after 1 m
    map_odom_drift_z: float = 0.01  # [m] height error std after 1 m
    map_odom_scale_error: float = 0.05  # per episode the walked distance is misjudged by up to this share
    map_feet: bool = True  # paws on the ground write the height under them (leg kinematics)
    map_dropout_rate: float = 0.0  # per second: chance the sensor goes dark (glare, a hand in front of it)
    map_dropout_s: tuple[float, float] = (0.3, 1.5)  # how long it stays dark [s]

    # rewards (per second of simulated time)
    forward_weight: float = 25.0
    forward_speed_cap: float = 0.5  # [m/s]
    lateral_weight: float = 8.0
    heading_weight: float = 1.0
    upright_weight: float = 2.0
    energy_weight: float = 0.3  # per watt of mechanical power
    action_rate_weight: float = 0.05
    joint_limit_weight: float = 1.0
    fall_penalty: float = 3.0
    fall_up_threshold: float = 0.35  # chassis z-axis projected on world up below this = fallen over


def make_env_cfg(
    spec,
    usd_path: str,
    num_envs: int = 2048,
    domain_randomization: bool = True,
    physics: str = "physx",
    servo_model: str = "dc_motor",
    task: str = "forward",
    terrain: str = "flat",
    terrain_types: list[str] | None = None,
    sensors: str = "ideal",
    obs_history: int = 1,
    height_scan: bool = False,
    foot_clearance: bool = False,
    low_level_policy: str | None = None,
    low_level_history: int = 8,
    terrain_map: str = "",
) -> EvoLocomotionEnvCfg:
    """Build the task config for one robot."""
    from .physics import physics_preset
    from .robot_cfg import make_robot_cfg

    cfg = EvoLocomotionEnvCfg()
    dog = getattr(spec.genome, "kind", "") == "dog"
    cfg.robot = make_robot_cfg(spec, usd_path, servo_model=servo_model,
                               contact_sensors=terrain == "rough" or task == "goal" or dog)
    cfg.servo_model = servo_model
    cfg.scene.num_envs = num_envs
    extent = max(0.3, 2.5 * max(float(abs(v)) for link in spec.links for v in link.origin[:2]) + 0.25)
    cfg.scene.env_spacing = extent
    cfg.sim.physics = physics_preset(physics)
    if physics == "physx" and num_envs < 2048:
        # smaller runs (tests, videos): PhysX's default GPU buffers are sized for thousands of robots and also reserve
        # pinned host memory; shrink them with the robot count so a run still starts when RAM is short
        ph = cfg.sim.physics
        for key in ("gpu_max_rigid_contact_count", "gpu_found_lost_pairs_capacity", "gpu_found_lost_aggregate_pairs_capacity",
                    "gpu_total_aggregate_pairs_capacity", "gpu_collision_stack_size", "gpu_heap_capacity",
                    "gpu_temp_buffer_capacity"):
            if hasattr(ph, key):
                setattr(ph, key, getattr(ph, key) // min(16, 2048 // max(num_envs, 1)))
    cfg.joint_names = [j.name for j in spec.joints]
    # the energy weight was tuned for palm-sized robots (~0.1-0.15 kg); walking power grows with weight, so for heavier
    # robots penalize power per kilogram instead, or falling over at once becomes cheaper than staying up
    cfg.energy_weight *= min(1.0, 0.15 / spec.total_mass)
    cfg.robot_mass = spec.total_mass
    cfg.joint_ranges = [float(j.upper) for j in spec.joints]
    if task not in ("forward", "command", "goal", "jump"):
        raise ValueError(f"unknown task {task!r}, choose 'forward', 'command' or 'goal'")
    cfg.task = task
    dog_terrain = terrain == "dog_rough"
    if dog_terrain:
        terrain = "rough"  # same curriculum and evaluation, dog-sized course (terrain.DOG_ROUGH)
    if terrain not in ("flat", "rough", "obstacles", "obstacles_rough", "enclosure", "enclosure_pocket", "cup"):
        raise ValueError(f"unknown terrain {terrain!r}, choose 'flat', 'rough', 'obstacles', 'obstacles_rough', "
                         "'enclosure', 'enclosure_pocket' or 'cup'")
    rough_floor = terrain == "obstacles_rough"
    if rough_floor:
        terrain = "obstacles"  # same course, bumpy floor
    pocket = terrain == "enclosure_pocket"
    cup = terrain == "cup"
    if cup:
        terrain = "enclosure"  # open ring, goal inside reachable round the back
    if pocket:
        terrain = "enclosure"  # ring with a slot (a local minimum of the goal distance)
    cfg.terrain_kind = terrain
    if terrain in ("obstacles", "enclosure") and task != "goal":
        raise ValueError(f"the {terrain} course is made for the goal task")
    if task == "goal":
        cfg.episode_length_s = 20.0
        cfg.reset_yaw_range = 3.14159
        # time-of-flight sensor: 8 x 8 rays over 45 x 45 degrees, centre tilted 20 degrees down
        body = spec.genome.body
        cfg.tof = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis",
            offset=RayCasterCfg.OffsetCfg(pos=(body.length / 2 - 0.005, 0.0, body.height / 2 - 0.005)),
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=8, vertical_fov_range=(-42.5, 2.5), horizontal_fov_range=(-22.5, 22.5), horizontal_res=6.43
            ),
            max_distance=cfg.tof_max_range,
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )
        if terrain == "obstacles":
            from .terrain import obstacle_terrain_importer

            cfg.terrain = obstacle_terrain_importer(rough_floor=rough_floor)
        elif terrain == "enclosure":
            from .terrain import enclosure_terrain_importer

            cfg.terrain = enclosure_terrain_importer(pocket=pocket, cup=cup)
            if cup:
                from .terrain import CUP_RADIUS

                cfg.cup = True
                cfg.enclosure_goal_offset = (0.10, 0.25)
                cfg.enclosure_start = (CUP_RADIUS + 0.30, CUP_RADIUS + 0.65)
                cfg.enclosure_outside_goal = (CUP_RADIUS + 0.25, CUP_RADIUS + 0.80)
                cfg.closeness_weight = 0.0  # a reachable goal: no reward for parking close to it
                cfg.episode_length_s = 30.0
            if pocket:
                from .terrain import POCKET_DEPTH, POCKET_RADIUS, POCKET_WIDTH

                cfg.pocket = (POCKET_RADIUS, POCKET_WIDTH, POCKET_DEPTH)
                cfg.enclosure_start = (POCKET_RADIUS + 0.35, POCKET_RADIUS + 0.70)
                cfg.enclosure_outside_goal = (POCKET_RADIUS + 0.25, POCKET_RADIUS + 0.80)
                cfg.episode_length_s = 30.0  # time to look in the pocket and still go around
        # contacts are needed to detect collisions (and for the air-time reward)
        cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(chassis|rear|[LR]\\d+_s\\d+)", history_length=3, update_period=0.0, track_air_time=True
        )
        cfg.foot_body_names = list(spec.foot_links)
        cfg.feet_air_time_weight = 1.0
    if terrain == "rough":
        from .terrain import rough_terrain_importer

        if task != "command":
            raise ValueError("rough terrain uses the command task (the curriculum is based on command tracking)")
        cfg.terrain = rough_terrain_importer(types=terrain_types, dog=dog_terrain)
        # only the link prims of the robot (the USD also holds the Looks and joints scopes)
        cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(chassis|rear|[LR]\\d+_s\\d+)", history_length=3, update_period=0.0, track_air_time=True
        )
        cfg.foot_body_names = list(spec.foot_links)
        cfg.feet_air_time_weight = 1.0
        if foot_clearance:
            links = spec.link_map
            spheres = [next(c for c in links[name].collisions if c.kind == "sphere") for name in cfg.foot_body_names]
            cfg.foot_offsets = [tuple(float(v) for v in c.transform[:3, 3]) for c in spheres]
            cfg.foot_radius = [float(c.size[0]) for c in spheres]
            cfg.foot_clearance_weight = 2.0
    if task == "command":
        cfg.reset_yaw_range = 3.14159  # commands are relative to the robot, so any heading is fine
    if sensors not in ("ideal", "imu", "imu_flow"):
        raise ValueError(f"unknown sensor suite {sensors!r}, choose 'ideal', 'imu' or 'imu_flow'")
    cfg.sensor_suite = sensors
    cfg.obs_history = max(1, int(obs_history))
    if sensors != "ideal":
        # the IMU board sits on top of the chassis, behind its centre (see core/morphology/builder.py)
        cfg.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis",
            offset=ImuCfg.OffsetCfg(pos=(-0.018, 0.0, 0.033 if dog else spec.genome.body.height / 2 + 0.01)),
        )
    if dog and task == "jump":
        # jumping: the full joint range, paws located for the height measurement, landing on the belly = a fall
        cfg.episode_length_s = 4.0
        cfg.reset_yaw_range = 3.14159
        cfg.fall_up_threshold = 0.5
        cfg.action_scale = 1.0
        cfg.action_limit_weight = 1.0
        cfg.action_rate_weight = 0.01  # explosive moves must be allowed
        cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(chassis|rear|[LR]\\d+_s\\d+)", history_length=3, update_period=0.0,
            track_air_time=True,
        )
        cfg.foot_body_names = list(spec.foot_links)
        cfg.feet_air_time_weight = 0.0
        links = spec.link_map
        spheres = [next(c for c in links[name].collisions if c.kind == "sphere") for name in cfg.foot_body_names]
        cfg.foot_offsets = [tuple(float(v) for v in c.transform[:3, 3]) for c in spheres]
        cfg.foot_radius = [float(c.size[0]) for c in spheres]
        cfg.base_height_target = spec.spawn_height - 0.004
        cfg.fall_penalty = 20.0  # a jump must be landed
    if dog and task == "command":
        # natural gait: trot rhythm, standing height, paws only on the ground
        cfg.gait = "trot"
        cfg.tracking_lin_sigma = 0.08  # walking too fast or too slow both cost (a dog must also walk slowly)
        cfg.curriculum_up_error = 0.15  # the spider's thresholds (6 / 12 cm/s) are out of reach with pushes
        cfg.curriculum_down_error = 0.25
        cfg.fall_up_threshold = 0.5  # tilted more than 60 degrees = fallen
        cfg.action_limit_weight = 1.0
        cfg.action_scale = 0.5  # small steps around the standing pose (as in ANYmal-style training), not wild flailing
        cfg.base_height_target = 0.97 * (spec.spawn_height - 0.004)
        if cfg.contact_sensor is None:
            cfg.contact_sensor = ContactSensorCfg(
                prim_path=r"{ENV_REGEX_NS}/Robot/(chassis|rear|[LR]\d+_s\d+)", history_length=3, update_period=0.0,
                track_air_time=True,
            )
            cfg.foot_body_names = list(spec.foot_links)
            cfg.feet_air_time_weight = 1.0
        cfg.feet_air_time_target = 0.20
        links = spec.link_map
        spheres = [next(c for c in links[name].collisions if c.kind == "sphere") for name in cfg.foot_body_names]
        cfg.foot_offsets = [tuple(float(v) for v in c.transform[:3, 3]) for c in spheres]
        cfg.foot_radius = [float(c.size[0]) for c in spheres]
        cfg.foot_clearance_weight = 2.0
        cfg.foot_clearance_target = 0.04
    from core.morphology.builder import is_passive_joint

    cfg.passive_joints = [j.name for j in spec.joints if is_passive_joint(j.name)]
    if cfg.passive_joints:
        by_child = {j.child: j for j in spec.joints}

        def depth(j) -> int:
            parent = by_child.get(j.parent)
            return (0 if j.name in cfg.passive_joints else 1) + (depth(parent) if parent is not None else 0)

        cfg.joint_depth = {j.name: depth(j) for j in spec.joints}
    n = len(spec.joints) - len(cfg.passive_joints)  # joints the policy observes
    cfg.expression_joints = [j.name for j in spec.joints if j.name in ("neck_j0", "head_j1", "tail_j0")]
    cfg.action_space = n - len(cfg.expression_joints)  # the policy drives the legs only
    frame = 3 + 3 + n + n + 2 + (n if cfg.policy_sees_joint_velocity else 0) + (3 if task == "command" else 0)
    frame += (3 + 64) if task == "goal" else 0  # goal (direction + distance) and the 8 x 8 distance readings
    if low_level_policy:
        if task != "goal" or sensors != "imu":
            raise ValueError("the hierarchical setup is for the goal task with --sensors imu (as the walker was trained)")
        cfg.low_level_policy = str(low_level_policy)
        cfg.low_level_history = low_level_history
        cfg.action_space = 3  # steering command
        frame += 3  # the policy sees the command it gave last
    frame += (3 if sensors != "ideal" else 0) + (2 if sensors == "imu_flow" else 0)
    if height_scan:
        scan_size, scan_res = (0.24, 0.16), 0.04  # 7 x 5 rays around the chassis, aligned with its heading
        scan_x = 0.0
        if dog:
            # 80 x 40 cm, centred 5 cm ahead of the body centre: covers all four paws and two steps ahead
            scan_size, scan_res, scan_x = (0.8, 0.4), 0.05, 0.05
            cfg.height_scan_clip = 0.2  # several 6 cm steps fit in the window
            cfg.height_scan_in_history = False  # 153 inputs once instead of in all 8 history frames
        cfg.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis",
            offset=RayCasterCfg.OffsetCfg(pos=(scan_x, 0.0, 0.5)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=scan_res, size=list(scan_size)),
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
            update_period=cfg.sim.dt * cfg.decimation if dog else 0.0,  # once per control step (50 Hz)
        )
        cfg.height_scan_offset = spec.spawn_height  # chassis height when standing on flat ground
        cfg.height_scan_points = (round(scan_size[0] / scan_res) + 1) * (round(scan_size[1] / scan_res) + 1)
        if cfg.height_scan_in_history:
            frame += cfg.height_scan_points
    if terrain_map:
        if not (dog and height_scan and task == "command"):
            raise ValueError("the terrain map is made for the dog's command task with --height_scan")
        _map_sensor(cfg, spec, terrain_map)
    extra = 0 if cfg.height_scan_in_history else cfg.height_scan_points
    # with the map the policy gets the map's heights and a known flag per scan point; the critic the perfect scan
    cfg.observation_space = frame * cfg.obs_history + extra * (2 if terrain_map else 1)
    cfg.state_space = frame + 3 + 1 + n + extra
    if not domain_randomization:
        cfg.events = None
    elif dog:
        # robustness (a dog among people gets bumped): a random shove every few seconds
        cfg.events.push = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.4, 0.4)}},
        )
        if any(j.name.endswith("_ja") for j in spec.joints):
            # printed TPU damps much less than the trained value: randomize the ankle damping per robot (the servo-gain
            # term above already varies the ankle stiffness by +-25 %, about the uncertainty of the printed flexure)
            cfg.events.ankle_damping = EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ja"),
                        "damping_distribution_params": (0.005, 0.03), "operation": "abs"},
            )

    # viewer: follow the robot of env 0, looking at it from the front-right
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (0.35, -0.40, 0.22)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    return cfg


MAP_SENSOR_PITCH = 40.0  # [deg] the chest sensor looks this far down: the ground from ~20 cm to ~1 m ahead


def _map_sensor(cfg: EvoLocomotionEnvCfg, spec, kind: str) -> None:
    """The terrain map's sensor at the centre of the chassis front, below the neck, tilted down (fixed to the body:
    a camera in the head would swing with the neck)."""
    if kind == "depth":
        # depth camera (RealSense D405 / OAK-D-Lite class): ~87 x 58 deg, sampled as 32 x 24 rays (~2.8 deg apart:
        # ~2.5 cm on the ground 50 cm ahead, the map's cell size), 25 Hz, 40 ms until the frame is in the map
        half_v, half_h, channels, h_res = 29.0, 43.0, 24, 86.0 / 31.0 + 0.003
        cfg.map_update_every, cfg.map_delay_steps = 2, 2
        cfg.map_min_range, cfg.map_max_range = 0.15, 2.0
        cfg.map_noise_rel, cfg.map_noise_abs = 0.01, 0.003
    elif kind == "tof":
        # VL53L5CX: 8 x 8 zones over 45 x 45 deg (zone centres +-19.7 deg), 15 Hz in 8 x 8 mode
        half_v, half_h, channels, h_res = 19.7, 19.7, 8, 5.65
        cfg.map_update_every, cfg.map_delay_steps = 3, 2
        cfg.map_min_range, cfg.map_max_range = 0.02, 2.0
        cfg.map_noise_rel, cfg.map_noise_abs = cfg.tof_noise_rel, cfg.tof_noise_abs
    else:
        raise ValueError(f"unknown terrain map sensor {kind!r}, choose 'depth' or 'tof'")
    body = spec.genome.body
    cfg.terrain_map = kind
    cfg.map_camera_offset = (body.length / 2, 0.0, 0.0)
    cfg.map_camera = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        offset=RayCasterCfg.OffsetCfg(pos=cfg.map_camera_offset),
        ray_alignment="base",  # tilts and turns with the body
        pattern_cfg=patterns.LidarPatternCfg(
            channels=channels, vertical_fov_range=(-MAP_SENSOR_PITCH - half_v, -MAP_SENSOR_PITCH + half_v),
            horizontal_fov_range=(-half_h, half_h), horizontal_res=h_res,
        ),
        max_distance=cfg.map_max_range,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
        update_period=cfg.sim.dt * cfg.decimation,  # read only on frame steps (sensors update lazily)
    )


# tests and videos use the standard course and count falls the standard way (recover_test / flip_video / getup_test set
# recover or getup themselves)
TRAINING_ONLY = ("stair_tread_range", "stairs_square_yaw", "stairs_turn_scale", "recover", "topple_push")


def apply_overrides(cfg: EvoLocomotionEnvCfg, overrides: dict | None) -> EvoLocomotionEnvCfg:
    """Re-apply the ``--env_override`` settings a run was trained with (stored in its result.json), so that playing,
    probing or filming a policy uses the same action range, gait rhythm etc. as its training."""
    for key, value in (overrides or {}).items():
        if key in TRAINING_ONLY:
            continue
        if not hasattr(cfg, key):
            raise KeyError(f"unknown environment setting {key!r} in the run's env_overrides")
        setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
    return cfg
