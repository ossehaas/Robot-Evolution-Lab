"""Small geometry toolkit: frames, 2D profiles extruded into printable plates, primitives and mass properties."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
import trimesh
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

MIRROR_Y = np.diag([1.0, -1.0, 1.0])


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def project_perp(v, axis) -> np.ndarray:
    """Component of ``v`` perpendicular to ``axis`` (normalised)."""
    axis = unit(axis)
    return unit(np.asarray(v, dtype=float) - np.dot(v, axis) * axis)


def frame(origin, x_axis, z_axis) -> np.ndarray:
    """Right-handed 4x4 transform with the given origin, x and z axes (x is re-orthogonalised)."""
    z = unit(z_axis)
    x = project_perp(x_axis, z)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    return T


def quat_wxyz_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(c) for c in q)


def rotation_x_to(axis) -> np.ndarray:
    """A rotation matrix whose first column is ``axis`` (used for joint frames: USD joints rotate about X)."""
    a = unit(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = unit(np.cross(helper, a))
    z = np.cross(a, y)
    return np.column_stack([a, y, z])


def rotation_z_to(axis) -> np.ndarray:
    """A rotation matrix whose third column is ``axis`` (used to orient capsules, which are built along Z)."""
    a = unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = unit(np.cross(helper, a))
    y = np.cross(a, x)
    return np.column_stack([x, y, a])


# ---------------------------------------------------------------------- 2D profiles -> meshes


def stadium(p0, p1, half_width: float, end_radius: float | None = None) -> Polygon:
    """A 2D link outline between two points: a bar with rounded ends."""
    line = shapely.LineString([tuple(p0), tuple(p1)])
    bar = line.buffer(half_width, cap_style="round", quad_segs=12)
    if end_radius is not None and end_radius > half_width:
        bar = unary_union([bar, Point(tuple(p0)).buffer(end_radius, quad_segs=16)])
    return bar


def circle(center, radius: float) -> Polygon:
    return Point(tuple(center)).buffer(radius, quad_segs=12)


def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def extrude(profile, z0: float, z1: float, transform: np.ndarray | None = None) -> trimesh.Trimesh:
    """Extrude a 2D profile (holes allowed) between local heights z0 and z1, then place it with ``transform``."""
    profile = profile.buffer(0)
    polys = list(profile.geoms) if hasattr(profile, "geoms") else [profile]
    meshes = []
    for poly in polys:
        if poly.is_empty or poly.area < 1e-10:
            continue
        tris = shapely.constrained_delaunay_triangles(poly)
        verts, faces, index = [], [], {}
        for tri in tris.geoms:
            face = []
            for x, y in list(tri.exterior.coords)[:3]:
                key = (round(x, 9), round(y, 9))
                if key not in index:
                    index[key] = len(verts)
                    verts.append(key)
                face.append(index[key])
            faces.append(face)
        verts2d = np.array(verts)
        faces = np.array(faces)
        # make every triangle counter-clockwise so the extruded solid has outward normals
        a, b, c = verts2d[faces[:, 0]], verts2d[faces[:, 1]], verts2d[faces[:, 2]]
        cw = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
        faces[cw] = faces[cw][:, ::-1]
        mesh = trimesh.creation.extrude_triangulation(verts2d, faces, height=z1 - z0)
        mesh.apply_translation([0.0, 0.0, z0])
        meshes.append(mesh)
    mesh = trimesh.util.concatenate(meshes)
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def box_mesh(size, transform: np.ndarray | None = None) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=size)
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def cylinder_mesh(radius: float, z0: float, z1: float, transform: np.ndarray | None = None, sections: int = 24):
    mesh = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=sections)
    mesh.apply_translation([0.0, 0.0, 0.5 * (z0 + z1)])
    if transform is not None:
        mesh.apply_transform(transform)
    return mesh


def sphere_mesh(center, radius: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.apply_translation(center)
    return mesh


def hull(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    return trimesh.util.concatenate(meshes).convex_hull


def mirror_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Mirror across the robot's sagittal (XZ) plane, keeping outward normals."""
    out = mesh.copy()
    out.vertices = out.vertices @ MIRROR_Y.T
    out.faces = out.faces[:, ::-1]
    return out


# ---------------------------------------------------------------------- mass properties


@dataclass
class MassProps:
    """Accumulates point/solid masses into total mass, centre of mass and inertia tensor (about the COM)."""

    items: list[tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)  # (mass, com, inertia@com)

    def add(self, mass: float, com, inertia_at_com) -> None:
        if mass > 0:
            self.items.append((float(mass), np.asarray(com, float), np.asarray(inertia_at_com, float)))

    def add_mesh(self, mesh: trimesh.Trimesh, density: float) -> float:
        """Add a (possibly multi-body) printed part; bodies that are not closed volumes use their hull."""
        total = 0.0
        for body in mesh.split(only_watertight=False):
            if not body.is_volume:
                body = body.convex_hull
            body.density = density
            self.add(float(body.mass), body.center_mass, body.moment_inertia)
            total += float(body.mass)
        return total

    def add_box(self, mass: float, size, transform: np.ndarray) -> None:
        sx, sy, sz = size
        local = mass / 12.0 * np.diag([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])
        R = transform[:3, :3]
        self.add(mass, transform[:3, 3], R @ local @ R.T)

    def add_sphere(self, mass: float, center, radius: float) -> None:
        self.add(mass, center, np.eye(3) * 0.4 * mass * radius**2)

    def total(self):
        """Return (mass, com, inertia about com as 3x3)."""
        mass = sum(m for m, _, _ in self.items)
        com = sum(m * c for m, c, _ in self.items) / mass
        inertia = np.zeros((3, 3))
        for m, c, I in self.items:
            d = c - com
            inertia += I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        return mass, com, inertia

    @staticmethod
    def principal(inertia: np.ndarray):
        """Diagonal inertia and the rotation (as matrix) of its principal axes."""
        vals, vecs = np.linalg.eigh(0.5 * (inertia + inertia.T))
        if np.linalg.det(vecs) < 0:
            vecs[:, 0] = -vecs[:, 0]
        return np.maximum(vals, 1e-9), vecs
