"""Environment health check: drop a rigid cube on the ground and check that it lands and comes to rest.

    IsaacLab\\.venv\\Scripts\\python.exe test_sim.py              # headless
    IsaacLab\\.venv\\Scripts\\python.exe test_sim.py --viz kit    # with the Isaac Sim viewport

Exits with code 0 when GPU physics works, 1 otherwise.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab health check.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402

CUBE = 0.05  # [m]
DROP_HEIGHT = 0.5  # [m]


def main() -> bool:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, device=args_cli.device or "cuda:0"))
    sim.set_camera_view(eye=(0.8, 0.8, 0.6), target=(0.0, 0.0, 0.1))
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE, CUBE, CUBE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.08)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, DROP_HEIGHT)),
        )
    )
    sim.reset()
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | device {sim.device}")

    dt = sim.get_physics_dt()
    for step in range(1, 401):  # 2 simulated seconds
        sim.step()
        cube.update(dt)
        if step % 50 == 0:
            z = cube.data.root_pos_w.torch[0, 2].item()
            print(f"  t={step * dt:4.2f}s  cube height {z:.3f} m")
        if not simulation_app.is_running():
            break

    z = cube.data.root_pos_w.torch[0, 2].item()
    speed = cube.data.root_lin_vel_w.torch[0].norm().item()
    ok = abs(z - CUBE / 2) < 0.005 and speed < 0.01
    print(f"{'PASS' if ok else 'FAIL'}: cube rests at {z:.4f} m (expected {CUBE / 2:.4f}), speed {speed:.4f} m/s")
    return ok


if __name__ == "__main__":
    ok = main()
    simulation_app.close()
    sys.exit(0 if ok else 1)
