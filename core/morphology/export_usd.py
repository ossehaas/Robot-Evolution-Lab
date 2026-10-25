"""RobotSpec -> USD articulation using only the standard UsdPhysics schemas (readable by PhysX and Newton).

Layout of the written stage::

    /Robot                      Xform, default prim, PhysicsArticulationRootAPI
        /chassis                Xform + RigidBodyAPI + MassAPI (world aligned, at its rest position)
            /visuals/...        Mesh prims with preview-surface materials (no physics)
            /collisions/...     Cube / Capsule / Sphere prims with CollisionAPI (purpose = guide)
        /L0_s0 ...              one Xform per link, flat hierarchy
        /joints/L0_j0 ...       PhysicsRevoluteJoint, rotating about the joint frame's X axis
        /Looks/...              render and physics materials
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from . import components as C
from .builder import RobotSpec
from .geometry import MassProps, quat_wxyz_from_matrix, rotation_x_to


def _quatf(q) -> Gf.Quatf:
    return Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))


def _set_transform(xformable: UsdGeom.Xformable, tf: np.ndarray, scale=None) -> None:
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*map(float, tf[:3, 3])))
    xformable.AddOrientOp().Set(_quatf(quat_wxyz_from_matrix(tf[:3, :3])))
    if scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def _safe(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if out and not out[0].isdigit() else "_" + out


def _render_material(stage, looks: Sdf.Path, name: str, color) -> UsdShade.Material:
    path = looks.AppendChild(_safe(name))
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    rough = 0.35 if name.startswith(("servo", "pcb")) else 0.6
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, looks: Sdf.Path, key: str) -> UsdShade.Material:
    m = C.MATERIALS[key]
    mat = UsdShade.Material.Define(stage, looks.AppendChild(f"physics_{key}"))
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(m.static_friction)
    api.CreateDynamicFrictionAttr(m.dynamic_friction)
    api.CreateRestitutionAttr(m.restitution)
    return mat


def write_usd(spec: RobotSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("evolution:genome", spec.genome.name)

    looks = Sdf.Path("/Robot/Looks")
    UsdGeom.Scope.Define(stage, looks)
    render_mats: dict[tuple, UsdShade.Material] = {}
    physics_mats = {k: _physics_material(stage, looks, k) for k in C.MATERIALS}
    for key, (stiffness, damping) in getattr(spec, "contact_compliance", {}).items():
        # PhysX compliant contact (a soft material such as a TPU paw): spring + damper instead of a rigid contact
        prim = physics_mats[key].GetPrim()
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(float(stiffness))
        prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(float(damping))

    for link in spec.links:
        lpath = Sdf.Path(f"/Robot/{_safe(link.name)}")
        xf = UsdGeom.Xform.Define(stage, lpath)
        tf = np.eye(4)
        tf[:3, 3] = link.origin
        _set_transform(xf, tf)
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(link.mass))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, link.com)))
        diag, axes = MassProps.principal(link.inertia)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, diag)))
        mass_api.CreatePrincipalAxesAttr(_quatf(quat_wxyz_from_matrix(axes)))

        vis_scope = UsdGeom.Scope.Define(stage, lpath.AppendChild("visuals"))
        for i, vis in enumerate(link.visuals):
            mesh = UsdGeom.Mesh.Define(stage, vis_scope.GetPath().AppendChild(f"{_safe(vis.name)}_{i}"))
            verts = np.asarray(vis.mesh.vertices, dtype=np.float32)
            faces = np.asarray(vis.mesh.faces, dtype=np.int32)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*vis.color)])
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
            key = tuple(round(c, 3) for c in vis.color)
            if key not in render_mats:
                base = vis.name.split("_")[-1] if "_" in vis.name else vis.name
                render_mats[key] = _render_material(stage, looks, f"{base}_{len(render_mats)}", vis.color)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(render_mats[key])

        col_scope = UsdGeom.Scope.Define(stage, lpath.AppendChild("collisions"))
        for i, col in enumerate(link.collisions):
            cpath = col_scope.GetPath().AppendChild(f"{col.kind}_{i}")
            if col.kind == "box":
                geom = UsdGeom.Cube.Define(stage, cpath)
                geom.CreateSizeAttr(1.0)
                _set_transform(geom, col.transform, scale=col.size)
            elif col.kind == "capsule":
                geom = UsdGeom.Capsule.Define(stage, cpath)
                geom.CreateRadiusAttr(float(col.size[0]))
                geom.CreateHeightAttr(float(col.size[1]))
                geom.CreateAxisAttr(UsdGeom.Tokens.z)
                _set_transform(geom, col.transform)
            else:
                geom = UsdGeom.Sphere.Define(stage, cpath)
                geom.CreateRadiusAttr(float(col.size[0]))
                _set_transform(geom, col.transform)
            geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
                physics_mats[col.material], UsdShade.Tokens.weakerThanDescendants, "physics"
            )

    UsdGeom.Scope.Define(stage, "/Robot/joints")
    links = spec.link_map
    for joint in spec.joints:
        jpath = Sdf.Path(f"/Robot/joints/{_safe(joint.name)}")
        rj = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        rj.CreateBody0Rel().SetTargets([Sdf.Path(f"/Robot/{_safe(joint.parent)}")])
        rj.CreateBody1Rel().SetTargets([Sdf.Path(f"/Robot/{_safe(joint.child)}")])
        rj.CreateAxisAttr(UsdPhysics.Tokens.x)
        rot = _quatf(quat_wxyz_from_matrix(rotation_x_to(joint.axis)))
        rj.CreateLocalPos0Attr(Gf.Vec3f(*map(float, joint.origin - links[joint.parent].origin)))
        rj.CreateLocalRot0Attr(rot)
        rj.CreateLocalPos1Attr(Gf.Vec3f(*map(float, joint.origin - links[joint.child].origin)))
        rj.CreateLocalRot1Attr(rot)
        rj.CreateLowerLimitAttr(math.degrees(joint.lower))
        rj.CreateUpperLimitAttr(math.degrees(joint.upper))
        drive = UsdPhysics.DriveAPI.Apply(rj.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateMaxForceAttr(float(joint.effort))
        drive.CreateTargetPositionAttr(0.0)
        prim = rj.GetPrim()
        prim.SetCustomDataByKey("evolution:servo_direction", int(joint.servo_direction))

    stage.GetRootLayer().Save()
    return path
