"""Paw part test, level 1 (journal step 48): one paw on a test stand in the simulator, moved through fixed motions, with
the paw's real colliders, materials, sole compliance and ankle spring and the dog's physics settings. No policy, no
training, no dog: one Kit process per paw design, about a minute.

The stand is a chain world -> x slider -> z slider -> shin (tilt, driven) -> ankle spring -> paw + sole (two bodies, so a
contact sensor tells what the sole carries and what the housing carries). Gravity is off: the load is a force on the z
slider. Sliders are force-driven (PD computed here), the tilt is a stiff drive; shin tilt + = toe up.

    roll   the stance of a stride: loaded, the shin sweeps +11 -> -18 deg in 0.33 s (paw 40 mm ahead -> 40 mm behind the
           hip, tools/parts/paw_geometry.py), x free. Work done at the shin, how high the ankle has to vault, share of
           the load on the housing
    drag   loaded, at a fixed shin tilt, pulled backwards at 0.1 m/s: grip (peak force / load before it slides) and
           friction (mean force / load while sliding), share on the housing
    edge   pushed forward against a 4 cm riser, the sole below its edge, then lifted at 0.1 m/s by a leg of at most 30 N:
           the force the lift takes per N of push, and whether the sole reaches the top
    drop   dropped on the floor at a quarter of the dog's mass plus a leg: peak force over 15 ms, on the housing in one
           physics step, how far the sole goes, rebound

    lab tools/parts/paw_stand.py --genome g.json --name "P 3" --out runs/exp2_dog/28_part_tests/stand.jsonl --headless
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ap = argparse.ArgumentParser()
ap.add_argument("--genome", type=Path, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out", type=Path, required=True)
add_launcher_args(ap)
args = ap.parse_args()
args.headless = True

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from core.morphology import Genome, build_robot  # noqa: E402
from core.sim.physics import physics_preset  # noqa: E402

DT = 1 / 200
LOADS = (6.0, 12.0, 18.0)  # [N] standing, a trot, a step's peak (2.45 kg dog)
TILTS = (-20.0, -10.0, 0.0, 10.0, 20.0)  # [deg] shin, drag test
EDGE_BELOW = (0.005, 0.015, 0.030)  # [m] sole below the riser's edge
EDGE_TILTS = (-10.0, 0.0, 10.0)
EDGE_PUSH = (6.0, 12.0)  # [N]
LIFT_MAX = 30.0  # [N] about what a leg lifts (2.4 N*m of usable knee torque on 80 mm)
DROPS = (0.03, 0.06)  # [m]
CARRIAGE = 0.6  # [kg] the mass above the paw in a drop
RISER_X, RISER_H = 0.15, 0.04  # [m]
T_SETTLE, T_SWEEP, T_END = 0.3, 0.33, 1.5
T_RAMP = 0.2  # [s] the sliders' speed targets ramp up over this: a step in the target would show up as a force
ROLL_FROM, ROLL_TO = 11.0, -18.0


def cases() -> list[dict]:
    out = [dict(test="roll", load=f) for f in LOADS]
    out += [dict(test="drag", load=f, tilt=t) for f in LOADS for t in TILTS]
    out += [dict(test="edge", below=b, tilt=t, push=p) for b in EDGE_BELOW for t in EDGE_TILTS for p in EDGE_PUSH]
    out += [dict(test="drop", height=h, load=CARRIAGE * 9.81) for h in DROPS]
    return out


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("+", "plus").replace(".", "_")


CASES = cases()
genome = Genome.load(args.genome)
spec = build_robot(genome)
paw = spec.link_map["L0_s3"]
ankle = next(j for j in spec.joints if j.name == "L0_ja")
sole_col = next(c for c in paw.collisions if c.kind == "sphere")  # the env's sole reference: the first sphere of the link
house_cols = [c for c in paw.collisions if c is not sole_col]
SOLE_Z = float(sole_col.transform[2, 3] - sole_col.size[0])  # lowest point of the sole, from the pivot [m]
# how far forward the paw reaches, for the start of the edge test
TOE_X = max(float(c.transform[0, 3] + (c.size[1] / 2 if c.kind == "capsule" else 0.0) + c.size[0]) for c in paw.collisions)
usd_path = args.out.parent / "stand_usd" / f"{slug(args.name)}.usda"


def write_stand(path: Path) -> None:
    """The stand as a USD articulation (Kit's pxr: call after the launch)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from core.morphology import components as C
    from core.morphology.export_usd import _physics_material, _quatf, _set_transform
    from core.morphology.geometry import MassProps, quat_wxyz_from_matrix

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    looks = Sdf.Path("/Robot/Looks")
    UsdGeom.Scope.Define(stage, looks)
    used = {c.material for c in paw.collisions}
    mats = {k: _physics_material(stage, looks, k) for k in C.MATERIALS if k in used}
    for key, (stiffness, damping) in getattr(spec, "contact_compliance", {}).items():
        if key in mats:
            prim = mats[key].GetPrim()
            prim.AddAppliedSchema("PhysxMaterialAPI")
            prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(float(stiffness))
            prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(float(damping))

    def body(name: str, mass: float, com=(0.0, 0.0, 0.0), inertia=None, collisions=()):
        xf = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        _set_transform(xf, np.eye(4))
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        api = UsdPhysics.MassAPI.Apply(xf.GetPrim())
        api.CreateMassAttr(float(mass))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        if inertia is None:
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))
        else:
            diag, axes = MassProps.principal(inertia)
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, diag)))
            api.CreatePrincipalAxesAttr(_quatf(quat_wxyz_from_matrix(axes)))
        UsdGeom.Scope.Define(stage, f"/Robot/{name}/collisions")
        for i, col in enumerate(collisions):
            cpath = f"/Robot/{name}/collisions/{col.kind}_{i}"
            if col.kind == "capsule":
                geom = UsdGeom.Capsule.Define(stage, cpath)
                geom.CreateRadiusAttr(float(col.size[0]))
                geom.CreateHeightAttr(float(col.size[1]))
                geom.CreateAxisAttr(UsdGeom.Tokens.z)
            else:
                geom = UsdGeom.Sphere.Define(stage, cpath)
                geom.CreateRadiusAttr(float(col.size[0]))
            _set_transform(geom, col.transform)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mats[col.material], UsdShade.Tokens.weakerThanDescendants, "physics")

    def joint(kind, name: str, parent: str, child: str, axis: str | None = None, lower=None, upper=None):
        j = kind.Define(stage, f"/Robot/joints/{name}")
        j.CreateBody0Rel().SetTargets([Sdf.Path(f"/Robot/{parent}")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/Robot/{child}")])
        if axis:
            j.CreateAxisAttr(axis)
            j.CreateLowerLimitAttr(float(lower))
            j.CreateUpperLimitAttr(float(upper))
        return j

    body("anchor", 1.0)
    body("carriage_x", 0.01)
    body("carriage_z", CARRIAGE)
    body("shin", 0.02)
    body("paw_house", paw.mass, paw.com, paw.inertia, house_cols)
    body("paw_sole", 0.002, collisions=[sole_col])
    UsdGeom.Scope.Define(stage, "/Robot/joints")
    UsdPhysics.FixedJoint.Define(stage, "/Robot/joints/world_fix").CreateBody1Rel().SetTargets([Sdf.Path("/Robot/anchor")])
    joint(UsdPhysics.PrismaticJoint, "slide_x", "anchor", "carriage_x", "X", -0.5, 0.5)
    joint(UsdPhysics.PrismaticJoint, "slide_z", "carriage_x", "carriage_z", "Z", -0.05, 0.3)
    for name, parent, child, lo, hi in (("tilt", "carriage_z", "shin", -60.0, 60.0),
                                        ("ankle", "shin", "paw_house", math.degrees(ankle.lower), math.degrees(ankle.upper))):
        rj = joint(UsdPhysics.RevoluteJoint, name, parent, child, "Y", lo, hi)
        UsdPhysics.DriveAPI.Apply(rj.GetPrim(), "angular").CreateTypeAttr("force")
    joint(UsdPhysics.FixedJoint, "sole_fix", "paw_house", "paw_sole")
    stage.GetRootLayer().Save()


@configclass
class StandCfg(DirectRLEnvCfg):
    episode_length_s: float = 100.0
    decimation: int = 1
    sim: SimulationCfg = SimulationCfg(dt=DT, render_interval=4)
    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=len(CASES), env_spacing=0.8, replicate_physics=True)
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1, debug_vis=False,
                                 physics_material=sim_utils.RigidBodyMaterialCfg(  # as EvoLocomotionEnvCfg
                                     friction_combine_mode="multiply", restitution_combine_mode="multiply",
                                     static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
    robot: ArticulationCfg = None
    contact: ContactSensorCfg = None


cfg = StandCfg()
cfg.sim.physics = physics_preset("physx")
cfg.robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path.resolve()).replace("\\", "/"), activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0, enable_gyroscopic_forces=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=8,
                                                                   solver_velocity_iteration_count=1)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, -SOLE_Z), joint_pos={".*": 0.0}),  # z slider 0 = the sole just on the floor
    actuators={
        "sliders": ImplicitActuatorCfg(joint_names_expr=["slide_.*"], stiffness=0.0, damping=0.0, armature=0.0, friction=0.0,
                                       joint_effort_limit=200.0, joint_velocity_limit=20.0),
        "tilt": ImplicitActuatorCfg(joint_names_expr=["tilt"], stiffness=200.0, damping=4.0, armature=1e-4, friction=0.0,
                                    joint_effort_limit=50.0, joint_velocity_limit=20.0),
        "ankle": ImplicitActuatorCfg(joint_names_expr=["ankle"], stiffness=float(genome.ankle.stiffness), damping=float(genome.ankle.damping),
                                     armature=1e-5, friction=0.0, joint_effort_limit=50.0, joint_velocity_limit=20.0),
    })
# the old ball has no housing colliders: the sensor then only sees the sole
cfg.contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/paw_(house|sole)" if house_cols else "{ENV_REGEX_NS}/Robot/paw_sole",
                               history_length=1, update_period=0.0)

with launch_simulation(cfg, args):
    from isaaclab import cloner
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sensors import ContactSensor

    class StandEnv(DirectRLEnv):
        def _setup_scene(self):
            self.robot = Articulation(self.cfg.robot)
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            riser = sim_utils.CuboidCfg(size=(0.2, 0.3, RISER_H), collision_props=sim_utils.CollisionPropertiesCfg(),
                                        physics_material=sim_utils.RigidBodyMaterialCfg(
                                            friction_combine_mode="multiply", restitution_combine_mode="multiply",
                                            static_friction=1.0, dynamic_friction=1.0, restitution=0.0))
            riser.func("/World/envs/env_0/riser", riser, translation=(RISER_X + 0.1, 0.0, RISER_H / 2))
            src, dest = "/World/envs/env_0", "/World/envs/env_{}"
            pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
            plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos, global_paths=(self.cfg.terrain.prim_path,))
            cloner.replicate(plan, stage=self.scene.stage)
            if "physx" in self.scene.physics_backend:
                self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            self.scene.articulations["robot"] = self.robot
            self.contact = ContactSensor(self.cfg.contact)
            self.scene.sensors["contact"] = self.contact

        def _pre_physics_step(self, actions):
            pass

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            none = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return none, none

    write_stand(usd_path)
    env = StandEnv(cfg)
    env.reset()
    robot, dev, n = env.robot, env.device, env.num_envs
    ix, iz, it, ia = (robot.joint_names.index(k) for k in ("slide_x", "slide_z", "tilt", "ankle"))
    body_names = list(env.contact.body_names)
    i_house = body_names.index("paw_house") if house_cols else None
    i_sole = body_names.index("paw_sole")
    tests = [c["test"] for c in CASES]
    mask = {t: torch.tensor([x == t for x in tests], device=dev) for t in set(tests)}
    col = lambda key, default=0.0: torch.tensor([float(c.get(key, default)) for c in CASES], device=dev)  # noqa: E731
    load, tilt0, below, push, height = col("load"), torch.deg2rad(col("tilt")), col("below"), col("push"), col("height")

    # start: just above the floor (the edge test beside the riser), tilted, at rest
    q = torch.zeros(n, robot.num_joints, device=dev)
    q[:, iz] = 0.004
    q[:, it] = -tilt0
    q[mask["roll"], it] = -math.radians(ROLL_FROM)
    q[mask["drop"], iz] = height[mask["drop"]]
    x_edge = RISER_X - TOE_X - 0.005
    q[mask["edge"], ix] = x_edge
    q[mask["edge"], iz] = RISER_H - below[mask["edge"]]
    robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q))
    z_ref, x_ref = q[:, iz].clone(), q[:, ix].clone()

    log = {k: [] for k in ("x", "z", "tilt", "ankle", "fx", "fz", "torque", "tilt_vel", "ankle_vel", "f_house", "f_sole")}
    for step in range(int(T_END / DT)):
        t = step * DT
        d = robot.data
        qn, vn = d.joint_pos.torch, d.joint_vel.torch
        fx, fz = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        target = -tilt0.clone()
        # roll
        m = mask["roll"]
        s = min(max((t - T_SETTLE) / T_SWEEP, 0.0), 1.0)
        target[m] = -math.radians(ROLL_FROM + s * (ROLL_TO - ROLL_FROM))
        fz[m] = -load[m]
        fx[m] = -5.0 * vn[m, ix]  # free, with the drag of a carried body
        # drag: pulled backwards
        m = mask["drag"]
        fz[m] = -load[m]
        tm = max(t - T_SETTLE, 0.0)
        ramp = min(tm / T_RAMP, 1.0)  # share of the speed reached
        travel = 0.1 * (tm * tm / (2 * T_RAMP) if tm < T_RAMP else tm - T_RAMP / 2)  # [m] at 0.1 m/s, with the ramp
        fx[m] = 2000.0 * (-travel - qn[m, ix]) + 80.0 * (-0.1 * ramp - vn[m, ix])
        # edge: pushed forward, lifted by a position loop with a limit
        m = mask["edge"]
        fx[m] = push[m] - 10.0 * vn[m, ix]
        fz[m] = torch.clamp(4000.0 * (z_ref[m] + travel - qn[m, iz]) + 100.0 * (0.1 * ramp - vn[m, iz]), -LIFT_MAX, LIFT_MAX)
        # drop: load only, held in x
        m = mask["drop"]
        fz[m] = -load[m]
        fx[m] = 2000.0 * (x_ref[m] - qn[m, ix]) - 80.0 * vn[m, ix]

        effort = torch.zeros(n, robot.num_joints, device=dev)
        effort[:, ix], effort[:, iz] = fx, fz
        pos = torch.zeros(n, robot.num_joints, device=dev)
        pos[:, it] = target
        robot.actuators.target_command.set_position_index(value=pos)
        robot.actuators.target_command.set_effort_index(value=effort)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(DT)

        forces = env.contact.data.net_forces_w.torch
        f_house = torch.linalg.norm(forces[:, i_house], dim=-1) if house_cols else torch.zeros(n, device=dev)
        for key, val in (("x", qn[:, ix]), ("z", qn[:, iz]), ("tilt", qn[:, it]), ("ankle", qn[:, ia]), ("fx", fx), ("fz", fz),
                         ("torque", d.applied_torque.torch[:, it]), ("tilt_vel", vn[:, it]), ("ankle_vel", vn[:, ia]),
                         ("f_house", f_house), ("f_sole", torch.linalg.norm(forces[:, i_sole], dim=-1))):
            log[key].append(val.detach().cpu().numpy().copy())
    trace = {k: np.stack(v) for k, v in log.items()}  # [steps, cases]

    def window(a, t0, t1):
        return a[int(t0 / DT):int(t1 / DT)]

    def share(i, t0, t1):
        h, s = window(trace["f_house"][:, i], t0, t1), window(trace["f_sole"][:, i], t0, t1)
        return float(h.sum() / max(h.sum() + s.sum(), 1e-6))

    results = []
    for i, c in enumerate(CASES):
        out = dict(c)
        if c["test"] == "roll":
            w = (T_SETTLE, T_SETTLE + T_SWEEP)
            # what the shin really passes on is the ankle spring's moment (the drive's reported torque is only an estimate)
            tq = float(genome.ankle.stiffness) * window(trace["ankle"][:, i], *w) + float(genome.ankle.damping) * window(trace["ankle_vel"][:, i], *w)
            om = window(trace["tilt_vel"][:, i], *w)
            out.update(work_mJ=float(1000 * np.sum(np.abs(tq * om)) * DT), vault_mm=float(1000 * np.ptp(window(trace["z"][:, i], *w))),
                       x_mm=float(1000 * np.ptp(window(trace["x"][:, i], *w))), on_house=share(i, *w),
                       ankle_deg=float(np.degrees(np.abs(window(trace["ankle"][:, i], *w)).max())))
        elif c["test"] == "drag":
            f = np.abs(trace["fx"][:, i])
            out.update(grip=float(window(f, T_SETTLE, T_SETTLE + 0.5).max() / c["load"]), friction=float(window(f, 0.9, T_END).mean() / c["load"]),
                       on_house=share(i, 0.9, T_END))
        elif c["test"] == "edge":
            z, lift = trace["z"][:, i], trace["fz"][:, i]  # z slider = height of the sole's lowest point
            up = np.nonzero(z >= RISER_H)[0]
            cleared = len(up) > 0
            start = int(T_SETTLE / DT)
            end = max(int(up[0]) if cleared else len(z), start + 1)
            out.update(cleared=cleared, late_mm=float(100 * (end * DT - T_SETTLE - T_RAMP / 2 - (RISER_H - z[0]) / 0.1)) if cleared else None,
                       peak_lift=float(lift[start:end].max()), lift_per_push=float(lift[start:end].max() / c["push"]),
                       mean_per_push=float(np.clip(lift[start:end], 0.0, None).mean() / c["push"]))
        else:
            tot = trace["f_house"][:, i] + trace["f_sole"][:, i]
            tot15 = np.convolve(tot, np.ones(3) / 3, "same")  # 15 ms: one physics step of a stiff contact is no force reading
            contact = np.nonzero(tot > 0.5)[0]
            first = int(contact[0]) if len(contact) else 0
            z = trace["z"][first:, i]
            k = int(np.argmin(z))
            out.update(peak_N=float(tot15.max()), peak_house_N=float(trace["f_house"][:, i].max()), lowest_mm=float(1000 * z[k]),
                       rebound_mm=float(1000 * (z[k:].max() - z[k])))
        results.append(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": args.name, "genome": str(args.genome).replace("\\", "/"), "cases": results}) + "\n")
    np.savez_compressed(args.out.parent / f"trace_{slug(args.name)}.npz", cases=json.dumps(CASES), **trace)
    print(f"[paw_stand] {args.name}: {len(results)} cases", flush=True)
    env.close()
