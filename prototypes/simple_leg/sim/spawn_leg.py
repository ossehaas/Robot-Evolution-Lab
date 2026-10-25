"""Test rig for the 2-servo leg: the hip is bolted in the air and the servos follow a slow sine sweep.

    python spawn_leg.py --viz kit              # watch it in the Isaac Sim viewport
    python spawn_leg.py --steps 600            # headless, prints how well the servos track
    python spawn_leg.py --viz kit --free       # unbolt the base and let the leg drop onto the ground

Run with the Isaac Lab environment's Python (see README.md at the repository root).
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Servo sweep on the simple leg.")
parser.add_argument("--free", action="store_true", help="free-floating base instead of the bolted test rig")
parser.add_argument("--steps", type=int, default=0, help="number of physics steps (0 = until the window closes)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple_leg", "simple_leg.usda").replace("\\", "/")

# SG90 micro servo: 0.14 N*m usable torque, ~10 rad/s, gearbox inertia at the horn (see core/morphology/components.py)
SERVO = ImplicitActuatorCfg(
    joint_names_expr=["hip_joint", "knee_joint"],
    stiffness=2.0,
    damping=0.05,
    armature=3.0e-4,
    friction=0.01,
    joint_effort_limit=0.14,
    joint_velocity_limit=10.0,
)

LEG_CFG = ArticulationCfg(
    prim_path="/World/SimpleLeg",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=not args_cli.free,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30), joint_pos={".*": 0.0}),
    actuators={"servos": SERVO},
)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, device=args_cli.device))
    sim.set_camera_view(eye=(0.45, 0.45, 0.35), target=(0.0, 0.0, 0.2))
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    leg = Articulation(LEG_CFG)

    sim.reset()
    # sim.reset() does not apply the configured initial state: write it explicitly
    leg.write_root_pose_to_sim_index(root_pose=leg.data.default_root_pose.torch.clone())
    leg.write_root_velocity_to_sim_index(root_velocity=leg.data.default_root_vel.torch.clone())
    leg.write_joint_state_to_sim_index(
        position=leg.data.default_joint_pos.torch.clone(), velocity=leg.data.default_joint_vel.torch.clone()
    )
    leg.reset()
    print(f"[simple_leg] joints {leg.joint_names}, bodies {leg.body_names}, fixed base: {not args_cli.free}")

    dt = sim.get_physics_dt()
    step, max_err = 0, 0.0
    while simulation_app.is_running():
        t = step * dt
        # hip swings +-0.6 rad at 0.5 Hz, knee +-0.8 rad at 1 Hz
        target = torch.tensor([[0.6 * math.sin(math.pi * t), 0.8 * math.sin(2 * math.pi * t)]], device=sim.device)
        leg.actuators.target_command.set_position_index(value=target)
        leg.write_data_to_sim()
        sim.step()
        leg.update(dt)
        err = (leg.data.joint_pos.torch - target).abs().max().item()
        max_err = max(max_err, err) if step > 50 else max_err
        step += 1
        if step % 100 == 0:
            q = leg.data.joint_pos.torch[0].tolist()
            z = leg.data.root_pos_w.torch[0, 2].item()
            print(f"t={t:5.2f}s  hip={q[0]:+.2f}  knee={q[1]:+.2f} rad  base z={z:.3f} m  tracking err={err:.3f} rad")
        if args_cli.steps and step >= args_cli.steps:
            break
    print(f"[simple_leg] worst tracking error after start-up: {max_err:.3f} rad")


if __name__ == "__main__":
    main()
    simulation_app.close()
