"""Small solid-modelling toolkit on top of manifold3d (exact, robust booleans), in millimetres.

The dog's printable parts are built here as closed solids: primitives, hulls, extrusions of 2D profiles, booleans,
swept volumes (a solid rotated through a joint range) and clearance checks. Everything is in millimetres because
that is what slicers and part drawings use; ``to_trimesh(..., scale=0.001)`` converts to metres for the simulator.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import manifold3d as mf
import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, OpType

# ---------------------------------------------------------------------- conversion


def from_trimesh(mesh: trimesh.Trimesh) -> Manifold:
    m = Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))
    if m.status() != mf.Error.NoError:
        raise ValueError(f"mesh is not a closed manifold: {m.status()}")
    return m


def to_trimesh(m: Manifold, scale: float = 1.0) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3] * scale
    return trimesh.Trimesh(verts, np.asarray(mesh.tri_verts, dtype=np.int64), process=False)


# ---------------------------------------------------------------------- transforms


def translation(t) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = t
    return T


def rotation(axis, angle: float, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotation by ``angle`` [rad] about the line through ``origin`` along ``axis``."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    o = np.asarray(origin, float)
    T[:3, 3] = o - R @ o
    return T


def frame(origin, x_axis, z_axis) -> np.ndarray:
    """Right-handed 4x4 frame with the given origin, x and z axes (x is re-orthogonalised)."""
    z = np.asarray(z_axis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_axis, float)
    x = x - np.dot(x, z) * z
    x = x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, origin
    return T


def apply(m: Manifold, T: np.ndarray) -> Manifold:
    return m.transform(np.asarray(T, float)[:3, :])


def mirror(m: Manifold, normal) -> Manifold:
    """Mirror across the plane through the origin with the given normal (keeps the solid closed and outward)."""
    return m.mirror(tuple(float(v) for v in normal))


def point(T: np.ndarray, p) -> np.ndarray:
    return (np.asarray(T, float) @ np.append(np.asarray(p, float), 1.0))[:3]


# ---------------------------------------------------------------------- primitives

SEG_LEN = 1.6  # target edge length of round shapes [mm]


def _segs(r: float, lo: int = 12, hi: int = 96) -> int:
    return int(max(lo, min(hi, round(2 * math.pi * r / SEG_LEN / 4) * 4)))


def box(size, center=(0.0, 0.0, 0.0)) -> Manifold:
    return Manifold.cube(tuple(float(s) for s in size), True).translate(tuple(float(c) for c in center))


def box_between(lo, hi) -> Manifold:
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return box(hi - lo, (lo + hi) / 2)


def cyl(r: float, z0: float, z1: float, T: np.ndarray | None = None, segs: int | None = None, r1: float | None = None) -> Manifold:
    """Cylinder (or cone to ``r1``) along local z from z0 to z1, optionally placed with ``T``."""
    m = Manifold.cylinder(float(z1 - z0), float(r), float(r if r1 is None else r1), segs or _segs(max(r, r1 or 0)))
    m = m.translate((0.0, 0.0, float(z0)))
    return apply(m, T) if T is not None else m


def cyl_between(p0, p1, r: float, r1: float | None = None, segs: int | None = None) -> Manifold:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    ref = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 * length else np.array([0, 1.0, 0])
    return cyl(r, 0.0, length, frame(p0, ref, d), segs, r1)


def sphere(center, r: float, segs: int | None = None) -> Manifold:
    return Manifold.sphere(float(r), segs or _segs(r, 12, 64)).translate(tuple(float(c) for c in center))


def ellipsoid(center, radii, R: np.ndarray | None = None) -> Manifold:
    m = Manifold.sphere(1.0, _segs(max(radii), 16, 64)).scale(tuple(float(r) for r in radii))
    if R is not None:
        T = np.eye(4)
        T[:3, :3] = R
        m = apply(m, T)
    return m.translate(tuple(float(c) for c in center))


def hull(*parts: Manifold | Iterable[Manifold]) -> Manifold:
    flat: list[Manifold] = []
    for p in parts:
        if isinstance(p, Manifold):
            flat.append(p)
        else:
            flat.extend(p)
    return Manifold.batch_hull(flat)


def capsule(p0, p1, r0: float, r1: float | None = None) -> Manifold:
    return hull(sphere(p0, r0), sphere(p1, r0 if r1 is None else r1))


def rounded_box(center, size, r: float) -> Manifold:
    """Box with all edges rounded to radius ``r`` (hull of eight spheres)."""
    size = np.asarray(size, float)
    r = min(r, *(size / 2 - 1e-3))
    half = size / 2 - r
    c = np.asarray(center, float)
    return hull([sphere(c + half * np.array([sx, sy, sz]), r) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def soft_box(center, size, r_plan: float, r_top: float, r_bottom: float) -> Manifold:
    """Rounded box with separate radii: vertical edges ``r_plan``, top edges ``r_top``, bottom edges ``r_bottom``.

    Built as the hull of tori-like rings (a rounded rectangle swept by a circle) at the top and bottom.
    """
    c = np.asarray(center, float)
    sx, sy, sz = (float(v) / 2 for v in size)
    rings = []
    for z_sign, r_edge in ((1, r_top), (-1, r_bottom)):
        z_c = c[2] + z_sign * (sz - r_edge)
        n = max(6, int(r_edge / 3))
        for k in range(n + 1):
            phi = (math.pi / 2) * k / n  # 0 = equator of the edge, pi/2 = top/bottom face
            inset = r_edge * (1 - math.cos(phi))
            z = z_c + z_sign * r_edge * math.sin(phi)
            a, b = sx - inset, sy - inset
            rp = max(min(r_plan - inset, a - 1e-3, b - 1e-3), 0.5)
            prof = rounded_rect(a, b, rp)
            rings.append(Manifold.extrude(prof, 0.01).translate((c[0], c[1], z - 0.005)))
    return hull(rings)


# ---------------------------------------------------------------------- 2D profiles


def rounded_rect(half_x: float, half_y: float, r: float) -> CrossSection:
    r = max(min(r, half_x - 1e-3, half_y - 1e-3), 1e-3)
    sq = CrossSection.square((2 * (half_x - r), 2 * (half_y - r)), True)
    return sq.offset(r, mf.JoinType.Round, 2.0, _segs(r, 8, 64))


def circle2d(r: float, center=(0.0, 0.0)) -> CrossSection:
    return CrossSection.circle(float(r), _segs(r, 12, 96)).translate((float(center[0]), float(center[1])))


def poly2d(points: Sequence[Sequence[float]]) -> CrossSection:
    pts = [(float(x), float(y)) for x, y in points]
    area = 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    if area < 0:
        pts = pts[::-1]
    return CrossSection([pts])


def stadium2d(p0, p1, r: float) -> CrossSection:
    return CrossSection.batch_hull([circle2d(r, p0), circle2d(r, p1)])


def extrude(profile: CrossSection, z0: float, z1: float, T: np.ndarray | None = None) -> Manifold:
    m = Manifold.extrude(profile, float(z1 - z0)).translate((0.0, 0.0, float(z0)))
    return apply(m, T) if T is not None else m


# ---------------------------------------------------------------------- booleans and sweeps


def union(parts: Iterable[Manifold]) -> Manifold:
    parts = [p for p in parts if p is not None and not p.is_empty()]
    if not parts:
        return Manifold()
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


def difference(a: Manifold, cutters: Iterable[Manifold]) -> Manifold:
    cutters = [c for c in cutters if c is not None and not c.is_empty()]
    if not cutters:
        return a
    return a - union(cutters)


def swept(m: Manifold, origin, axis, a0: float, a1: float, step: float = math.radians(4.0)) -> Manifold:
    """Volume covered by ``m`` while it rotates from angle a0 to a1 [rad] about the given axis.

    Each pair of neighbouring poses is hulled, so convex parts are swept exactly; concave parts slightly over-estimate
    (which is the safe side for clearance cuts).
    """
    n = max(1, int(math.ceil(abs(a1 - a0) / step)))
    poses = [apply(m, rotation(axis, a0 + (a1 - a0) * k / n, origin)) for k in range(n + 1)]
    if len(poses) == 1:
        return poses[0]
    pieces = m.decompose()
    if len(pieces) > 1:  # hull each connected piece separately so disjoint parts are not bridged
        return union(swept(p, origin, axis, a0, a1, step) for p in pieces)
    return union(hull(poses[k], poses[k + 1]) for k in range(n))


def grow(m: Manifold, d: float) -> Manifold:
    """Approximate outward offset by ``d`` (Minkowski sum with a small octahedron-ish sphere)."""
    if d <= 0:
        return m
    return m.minkowski_sum(Manifold.sphere(float(d), 8))


def gap(a: Manifold, b: Manifold, search: float = 10.0) -> float:
    """Smallest distance between two solids (negative-ish 0 when they overlap), up to ``search``."""
    if not (a ^ b).is_empty() and (a ^ b).volume() > 1e-3:
        return -((a ^ b).volume()) ** (1 / 3)
    return float(a.min_gap(b, search))


def volume(m: Manifold) -> float:
    return float(m.volume())


def bbox(m: Manifold) -> tuple[np.ndarray, np.ndarray]:
    b = m.bounding_box()
    return np.array(b[:3]), np.array(b[3:])


def swept_union(m: Manifold, origin, axis, a0: float, a1: float, step: float = math.radians(1.5)) -> Manifold:
    """Union of ``m`` rotated through a0..a1 (no hulls): for non-convex solids such as a scooped body."""
    n = max(1, int(math.ceil(abs(a1 - a0) / step)))
    return union(apply(m, rotation(axis, a0 + (a1 - a0) * k / n, origin)) for k in range(n + 1))


def probe(m: Manifold, start, direction, length: float = 1000.0) -> float:
    """Coordinate along an axis-aligned ray from ``start`` where it crosses the surface of ``m``.

    Starting outside: where the ray first enters. Starting inside: where it leaves (the solid is assumed convex
    along the ray). Returns nan when the ray misses.
    """
    d = np.asarray(direction, float)
    k = int(np.argmax(np.abs(d)))
    s = np.asarray(start, float)
    inside = not (m ^ box_between(s - 0.01, s + 0.01)).is_empty()
    lo, hi = s.copy() - 0.05, s.copy() + 0.05
    if d[k] > 0:
        hi[k] = s[k] + length
    else:
        lo[k] = s[k] - length
    inter = m ^ box_between(lo, hi)
    if inter.is_empty():
        return float("nan")
    if inside:  # the piece containing the start point
        pieces = [p for p in inter.decompose() if not (p ^ box_between(s - 0.02, s + 0.02)).is_empty()]
        inter = pieces[0] if pieces else inter
    blo, bhi = bbox(inter)
    if inside:
        return float(bhi[k] if d[k] > 0 else blo[k])
    return float(blo[k] if d[k] > 0 else bhi[k])


def ray_extent(m: Manifold, origin, direction, length: float = 300.0, r: float = 0.3) -> float:
    """Farthest distance along ``direction`` from ``origin`` at which a thin rod still meets ``m`` (nan if never)."""
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    inter = m ^ cyl_between(o, o + length * d, r, segs=8)
    if inter.is_empty():
        return float("nan")
    v = np.asarray(inter.to_mesh().vert_properties)[:, :3]
    return float(((v - o) @ d).max())
