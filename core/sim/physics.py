"""Physics backend selection: PhysX inside Isaac Sim (default) or Newton/MuJoCo-Warp (kitless, fast startup)."""

from __future__ import annotations

BACKENDS = ("physx", "newton")


def physics_preset(name: str = "physx"):
    if name == "physx":
        from isaaclab_physx.physics import PhysxCfg

        return PhysxCfg(bounce_threshold_velocity=0.05)
    if name == "newton":
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        return NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=200, nconmax=60, cone="pyramidal", integrator="implicitfast", impratio=1),
            num_substeps=1,
            debug_mode=False,
        )
    raise ValueError(f"unknown physics backend {name!r}, choose from {BACKENDS}")
