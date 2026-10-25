"""Articulation config for a generated robot: servo actuators with the real SG90 torque, speed and gearbox.

The servo is a stiff PD loop feeding a DC motor: torque is limited by the linear torque-speed curve (stall
torque at rest, zero torque at no-load speed). That curve, not the stall torque, is what limits a hobby servo.

Servo models (``servo_model``):
    dc_motor   (default) Isaac Lab's explicit DCMotor actuator: PD loop computed every physics step, torque
               clipped by the torque-speed curve.
    implicit   solver-side PD drive with a flat torque limit: ignores the torque-speed curve and lets servos
               deliver full torque at full speed (policies learn unrealistically strong gaits). For comparison.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from core.morphology import components as C
from core.morphology.builder import RobotSpec

# PD gains per servo type (components.ServoSpec.kp/kd): a stiff loop that saturates after ~0.1 rad of error,
# near-critically damped against the gearbox inertia, stable at the 200 Hz physics rate (omega_n * dt < 0.5).


SERVO_MODELS = ("dc_motor", "implicit")


def make_servo_actuator(servo_model: str = "dc_motor", servo_type: str = "SG90"):
    servo = C.SERVOS[servo_type]
    if servo_model == "dc_motor":
        return DCMotorCfg(
            joint_names_expr=[".*"],
            stiffness=servo.kp,
            damping=servo.kd,
            saturation_effort=servo.stall_torque,
            actuator_effort_limit=servo.stall_torque,
            actuator_velocity_limit=servo.max_speed,
            joint_effort_limit=servo.stall_torque,
            armature=servo.armature,
            friction=servo.friction,
        )
    if servo_model == "implicit":
        return ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=servo.kp,
            damping=servo.kd,
            armature=servo.armature,
            friction=servo.friction,
            joint_effort_limit=servo.usable_torque,
            joint_velocity_limit=servo.max_speed,
            actuator_velocity_limit=servo.max_speed,
        )
    raise ValueError(f"unknown servo model {servo_model!r}, choose from {SERVO_MODELS}")


def make_robot_cfg(
    spec: RobotSpec,
    usd_path: str,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    servo_model: str = "dc_motor",
    contact_sensors: bool = False,
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, spec.spawn_height), joint_pos={".*": 0.0}),
        actuators=_actuators(spec, servo_model),
    )


def _actuators(spec: RobotSpec, servo_model: str) -> dict:
    """Servos on every joint, except passive spring hinges (a dog's flexible back): a spring + damper pulling the joint
    back to straight, no motor."""
    from core.morphology.builder import is_passive_joint

    springs = [j for j in spec.joints if is_passive_joint(j.name)]
    if not springs:
        return {"servos": make_servo_actuator(servo_model, spec.servo)}
    servos = make_servo_actuator(servo_model, spec.servo)
    servos.joint_names_expr = ["^(?!spine_)(?!.*_ja$).*"]
    g = spec.genome
    stiffness, damping = {}, {}
    for j in springs:
        if j.name.startswith("spine_"):
            stiffness[j.name] = g.spine.roll_stiffness if j.name == "spine_roll" else g.spine.pitch_stiffness
            damping[j.name] = g.spine.damping
        else:
            stiffness[j.name], damping[j.name] = g.ankle.stiffness, g.ankle.damping
    spring = ImplicitActuatorCfg(
        joint_names_expr=[j.name for j in springs], stiffness=stiffness, damping=damping, armature=1e-5, friction=0.0,
        joint_effort_limit=50.0, joint_velocity_limit=20.0,
    )
    return {"servos": servos, "spine": spring}
