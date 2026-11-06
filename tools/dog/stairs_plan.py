"""Can the dog climb this, physically? A quasi-static proof (journal step 41), no simulator.

For a terrain profile (here: straight stairs of a rise and tread, up or down) it searches a slow crawl: one leg at a time
in the lateral-sequence order (rear left, front left, rear right, front right; each leg chooses
its own foothold), the body shifting between steps with all four paws down. Every pose of the plan is checked:

    reach        every standing paw is reached within 95 % of the hip and knee range (the walking penalty starts there)
    collisions   no part of the dog (body, head, legs, paws in the air) comes within 3 mm of the terrain; a swinging
                 paw clears every edge by 1 cm
    footholds    a standing paw rests on a tread, >= 1.5 cm from the edge behind it, not touching the riser ahead
    balance      the centre of gravity (of all links, in the pose) stays >= 1.5 cm inside the support polygon
    torque       hip and knee need at most 80 % of the servo's limit in the simulation (2.35 N*m) to hold the pose

The dog's leg joints all pitch about the sideways axis (abduction held at 0), so kinematics and collisions are exact in
the side view (2D); balance uses the real left/right paw positions. The neck is a free variable (a counterweight). If a
plan exists the cell is physically possible (sufficient, slow motion; a dynamic gait may do more). The plans (body pose,
joint angles, contacts per keyframe) are written to --out_dir as the reference for a simulator replay and as a trainer.

    lab tools/dog/stairs_plan.py --out_dir runs/exp2_dog/19_stairs_proof/plans
    lab tools/dog/stairs_plan.py --rise 0.06 --tread 0.09 --dir up --out_dir ...      one cell
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GENOME = "runs/exp2_dog/17_recover_bc/bc_s2/genome.json"
LEGS = ("L0", "R0", "L1", "R1")  # front left, front right, rear left, rear right
ORDER = (2, 0, 3, 1)  # lateral sequence: RL, FL, RR, FR
PARTNER = {0: 1, 1: 0, 2: 3, 3: 2}  # the other leg of the pair (front / rear)
JOINT_SHARE = 0.95
CLEAR = 0.003  # [m] body and legs to the terrain
SWING_CLEAR = 0.01  # [m] a swinging paw above the edges it passes
EDGE = 0.015  # [m] paw contact point from the edge behind it
RISER_GAP = 0.004  # [m] paw sphere to the riser ahead
STABILITY = 0.015  # [m] centre of gravity inside the support polygon
MAX_PITCH = 45.0  # [deg] nose-up limit of the body poses
TORQUE = 0.8 * 2.3544  # [N*m]
G = 9.81


def rot(a, x, z):
    """Rotation about +y by a, applied to (x, z) (broadcasting)."""
    c, s = np.cos(a), np.sin(a)
    return x * c + z * s, -x * s + z * c


class Dog2D:
    """Side-view model of the dog from its robot spec: links as (angle, translation) maps of rest-space points."""

    def __init__(self, genome: str):
        from core.morphology import Genome, build_robot

        spec = build_robot(Genome.load(genome))
        self.joints = {j.name: j for j in spec.joints}
        self.links = {lk.name: lk for lk in spec.links}
        self.mass = spec.total_mass
        self.parent = {j.child: j for j in spec.joints}
        self.paw_r = next(c for c in self.links["L0_s3"].collisions if c.kind == "sphere").size[0]
        xz = lambda p: np.array([p[0], p[2]])  # noqa: E731
        self.hip = {leg: xz(self.joints[f"{leg}_j1"].origin) for leg in LEGS}
        self.knee = {leg: xz(self.joints[f"{leg}_j2"].origin) for leg in LEGS}
        self.paw = {}
        for leg in LEGS:
            lk = self.links[f"{leg}_s3"]
            sph = next(c for c in lk.collisions if c.kind == "sphere")
            self.paw[leg] = xz(lk.origin + sph.transform[:3, 3])
        self.paw_y = {leg: float(self.joints[f"{leg}_ja"].origin[1]) for leg in LEGS}
        self.l1 = {leg: np.linalg.norm(self.knee[leg] - self.hip[leg]) for leg in LEGS}
        self.l2 = {leg: np.linalg.norm(self.paw[leg] - self.knee[leg]) for leg in LEGS}
        self.lim = {leg: (JOINT_SHARE * self.joints[f"{leg}_j1"].upper, JOINT_SHARE * self.joints[f"{leg}_j2"].upper)
                    for leg in LEGS}
        self.neck_o = xz(self.joints["neck_j0"].origin)
        self.neck_lim = self.joints["neck_j0"].upper
        # collision primitives per link in rest space: ("sphere"/"capsule", p0, p1, radius) or ("box", corners)
        self.prims, self.masses = {}, {}
        for name, lk in self.links.items():
            out = []
            for c in lk.collisions:
                centre = lk.origin + c.transform[:3, 3]
                if c.kind == "sphere":
                    out.append(("seg", xz(centre), xz(centre), c.size[0]))
                elif c.kind == "capsule":
                    half = c.transform[:3, 2] * c.size[1] / 2
                    out.append(("seg", xz(centre - half), xz(centre + half), c.size[0]))
                else:
                    hx, hz = c.size[0] / 2, c.size[2] / 2
                    corners = np.array([[hx, hz], [-hx, hz], [-hx, -hz], [hx, -hz]]) + xz(centre)
                    out.append(("box", corners))
            self.prims[name] = out
            self.masses[name] = (lk.mass, xz(lk.origin + lk.com))

    def link_maps(self, pitch, neck, q):
        """(angle, tx, tz) world maps per link for batched body pitch, neck angle and leg angles q[leg] = (hip, knee);
        the body origin is added by the caller."""
        a0 = -pitch
        maps = {"chassis": (a0, 0.0, 0.0), "tail": (a0, 0.0, 0.0)}

        def about(parent, o, ang):
            pa, ptx, ptz = parent
            ox, oz = o
            rx, rz = rot(ang, ox, oz)
            tx, tz = ox - rx, oz - rz  # rotation about o in rest space
            wx, wz = rot(pa, tx, tz)
            return pa + ang, ptx + wx, ptz + wz

        n = about(maps["chassis"], self.neck_o, neck)
        maps["neck"] = maps["head"] = n
        for leg in LEGS:
            maps[f"{leg}_s0"] = maps["chassis"]
            hip, knee = q[leg]
            m1 = about(maps["chassis"], self.hip[leg], hip)
            m2 = about(m1, self.knee[leg], knee)
            maps[f"{leg}_s1"], maps[f"{leg}_s2"], maps[f"{leg}_s3"] = m1, m2, m2
        return maps

    def ik(self, leg, pitch, bx, bz, px, pz):
        """Hip and knee angles that put the paw centre at world (px, pz) for body (bx, bz, pitch); valid mask."""
        rx, rz = rot(pitch, px - bx, pz - bz)  # into rest space (inverse of rot(-pitch))
        h, k, p = self.hip[leg], self.knee[leg], self.paw[leg]
        l1, l2 = self.l1[leg], self.l2[leg]
        dx, dz = rx - h[0], rz - h[1]
        d = np.hypot(dx, dz)
        cos_a = np.clip((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * np.maximum(d, 1e-9)), -1.0, 1.0)
        reach = (d <= l1 + l2 - 1e-4) & (d >= abs(l1 - l2) + 1e-4)
        u0, v0 = k - h, p - k
        bend = np.sign(u0[0] * v0[1] - u0[1] * v0[0])
        best_h = best_k = None
        for sgn in (1.0, -1.0):
            psi = np.arctan2(dz, dx) + sgn * np.arccos(cos_a)
            # thigh direction psi; knee position and shin direction
            kx, kz = h[0] + l1 * np.cos(psi), h[1] + l1 * np.sin(psi)
            vx, vz = rx - kx, rz - kz
            ok_bend = np.sign(np.cos(psi) * vz - np.sin(psi) * vx) == bend
            # rotation about +y by a turns the direction angle by -a
            a1 = np.arctan2(u0[1], u0[0]) - psi
            a2 = np.arctan2(v0[1], v0[0]) - np.arctan2(vz, vx) - a1
            a1 = (a1 + np.pi) % (2 * np.pi) - np.pi
            a2 = (a2 + np.pi) % (2 * np.pi) - np.pi
            if best_h is None:
                best_h, best_k, best_ok = a1, a2, ok_bend
            else:
                best_h = np.where(ok_bend, a1, best_h)
                best_k = np.where(ok_bend, a2, best_k)
                best_ok = best_ok | ok_bend
        lh, lk = self.lim[leg]
        valid = reach & best_ok & (np.abs(best_h) <= lh) & (np.abs(best_k) <= lk)
        return best_h, best_k, valid

    def world_points(self, maps, bx, bz):
        """Per link: list of world primitives with batched coordinates."""
        out = {}
        for name, prims in self.prims.items():
            a, tx, tz = maps[name]
            ws = []
            for p in prims:
                if p[0] == "seg":
                    x0, z0 = rot(a, *p[1])
                    x1, z1 = rot(a, *p[2])
                    ws.append(("seg", (x0 + tx + bx, z0 + tz + bz), (x1 + tx + bx, z1 + tz + bz), p[3]))
                else:
                    cs = [rot(a, cx, cz) for cx, cz in p[1]]
                    ws.append(("box", [(x + tx + bx, z + tz + bz) for x, z in cs]))
            out[name] = ws
        return out

    def com_x(self, maps, bx):
        total = 0.0
        for name, (m, c) in self.masses.items():
            a, tx, _ = maps[name]
            x, _ = rot(a, c[0], c[1])
            total = total + m * (x + tx)
        return total / self.mass + bx


class Terrain:
    """Piecewise flat side profile: segments [(x_start, x_end, height)], solid below each height."""

    def __init__(self, segments):
        self.seg = segments
        xs = [s[0] for s in segments[1:]]
        self.corners = [(x, max(a[2], b[2])) for x, a, b in zip(xs, segments[:-1], segments[1:])]

    @staticmethod
    def stairs(rise: float, tread: float, n: int = 5, up: bool = True) -> "Terrain":
        """First edge at x = 0; up: floor, n-1 steps, landing; down: landing, n-1 steps, floor."""
        h = [k * rise for k in range(n + 1)] if up else [(n - k) * rise for k in range(n + 1)]
        segs = [(-10.0, 0.0, h[0])] + [((k - 1) * tread, k * tread, h[k]) for k in range(1, n)] + [((n - 1) * tread, 10.0, h[n])]
        return Terrain(segs)

    def height(self, x):
        out = np.zeros_like(np.asarray(x, dtype=float))
        for x0, x1, h in self.seg:
            out = np.where((x >= x0) & (x < x1), h, out)
        return out

    def sdf(self, x, z):
        """Signed distance to the solid (union of the segments' columns), positive outside."""
        best = None
        for x0, x1, h in self.seg:
            cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
            cz, hz = (h - 5.0) / 2, (h + 5.0) / 2  # column down to z = -5
            qx, qz = np.abs(x - cx) - hx, np.abs(z - cz) - hz
            d = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0)
            best = d if best is None else np.minimum(best, d)
        return best

    def clearance(self, prim) -> np.ndarray:
        """Distance from a world primitive to the terrain (negative = inside)."""
        if prim[0] == "seg":
            (x0, z0), (x1, z1), r = prim[1], prim[2], prim[3]
            ts = np.linspace(0, 1, 5)
            d = self.sdf(x0, z0)
            for t in ts[1:]:
                d = np.minimum(d, self.sdf(x0 + t * (x1 - x0), z0 + t * (z1 - z0)))
            # step edges (convex corners) against the segment
            for cx, cz in self.corners:
                ex, ez = x1 - x0, z1 - z0
                t = np.clip(((cx - x0) * ex + (cz - z0) * ez) / np.maximum(ex * ex + ez * ez, 1e-12), 0, 1)
                d = np.minimum(d, np.hypot(x0 + t * ex - cx, z0 + t * ez - cz))
            return d - r
        pts = prim[1]
        d = None
        for i in range(4):
            (xa, za), (xb, zb) = pts[i], pts[(i + 1) % 4]
            for t in (0.0, 0.5):
                v = self.sdf(xa + t * (xb - xa), za + t * (zb - za))
                d = v if d is None else np.minimum(d, v)
        # a step edge inside the box (convex quad, counter-clockwise or clockwise)
        for cx, cz in self.corners:
            signs = []
            for i in range(4):
                (xa, za), (xb, zb) = pts[i], pts[(i + 1) % 4]
                signs.append((xb - xa) * (cz - za) - (zb - za) * (cx - xa))
            inside = np.all(np.stack([s > 0 for s in signs]), 0) | np.all(np.stack([s < 0 for s in signs]), 0)
            d = np.where(inside, -1.0, d)
        return d


def support_margin(points, cx):
    """Signed distance of (cx, 0) to the convex hull of the paw contact points [(x, y)] (batched x), + inside."""
    pts = list(points)
    if len(pts) == 3:
        hull = pts
    else:  # four paws: FL, FR, RR, RL around the rectangle-ish quad
        hull = [pts[0], pts[1], pts[3], pts[2]]
    # orientation of the hull
    (x0, y0), (x1, y1), (x2, y2) = hull[0], hull[1], hull[2]
    orient = np.sign((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
    m = None
    for i in range(len(hull)):
        (xa, ya), (xb, yb) = hull[i], hull[(i + 1) % len(hull)]
        ex, ey = xb - xa, yb - ya
        length = np.maximum(np.hypot(ex, ey), 1e-9)
        d = orient * (ex * (0.0 - ya) - ey * (cx - xa)) / length
        m = d if m is None else np.minimum(m, d)
    return m


def contact_forces(points, cx, mass):
    """Vertical paw forces holding the weight at (cx, 0): exact for 3 paws, least-norm for 4 (batched)."""
    w = mass * G
    n = len(points)
    xs = np.stack([np.broadcast_to(p[0], np.shape(cx)) for p in points], -1)
    ys = np.stack([np.broadcast_to(np.float64(p[1]), np.shape(cx)) for p in points], -1)
    a = np.stack([np.ones_like(xs), xs, ys], -2)  # (..., 3, n): sum f, sum f x, sum f y
    b = np.stack([np.full(np.shape(cx), w), w * np.asarray(cx), np.zeros(np.shape(cx))], -1)[..., None]
    f = np.linalg.pinv(a) @ b if n == 4 else np.linalg.solve(a, b)
    return f[..., 0]


class Planner:
    def __init__(self, dog: Dog2D, terrain: Terrain, goal_x: float):
        self.dog, self.t, self.goal_x = dog, terrain, goal_x
        r = dog.paw_r
        # footholds: paw centre x on a flat segment, contact >= EDGE from the edge behind, sphere clear of a riser ahead
        xs = np.round(np.arange(-0.5, goal_x + 0.4, 0.005), 4)
        ok = []
        for x in xs:
            seg = next(s for s in terrain.seg if s[0] <= x < s[1])
            idx = terrain.seg.index(seg)
            nxt = terrain.seg[idx + 1] if idx + 1 < len(terrain.seg) else None
            prv = terrain.seg[idx - 1] if idx > 0 else None
            good = True
            if prv is not None and prv[2] < seg[2] and x - seg[0] < EDGE:
                good = False  # too close to the edge behind (step up)
            if prv is not None and prv[2] > seg[2] and x - seg[0] < r + RISER_GAP:
                good = False  # against the riser behind (step down)
            if nxt is not None and nxt[2] > seg[2] and seg[1] - x < r + RISER_GAP:
                good = False  # against the riser ahead (step up)
            if nxt is not None and nxt[2] < seg[2] and seg[1] - x < EDGE:
                good = False  # too close to the edge ahead (step down)
            ok.append(good)
        self.footholds = xs[np.array(ok)]
        self.span = float(dog.paw["L0"][0] - dog.paw["L1"][0])  # front-rear paw distance standing
        # body pose grid (relative): body x offset from the paws' mean, chassis height above the paws' mean height,
        # pitch, neck
        g = np.meshgrid(np.linspace(-0.06, 0.06, 9), np.linspace(0.09, 0.20, 8), np.radians(np.arange(-15, MAX_PITCH + 1, 5)),
                        np.linspace(-dog.neck_lim * 0.95, dog.neck_lim * 0.95, 5), indexing="ij")
        self.grid = [v.ravel() for v in g]

    def paw_z(self, x):
        return self.t.height(x) + self.dog.paw_r

    stats: dict | None = None  # check name -> poses ruled out (diagnosis)

    def _count(self, name, before, after):
        if self.stats is not None:
            self.stats[name] = self.stats.get(name, 0) + int(before.sum() - after.sum())

    def pose_check(self, paws, bx, bz, pitch, neck, swing=None, swing_xz=None, need_stab=None):
        """Validity (batched over body poses) of one pose: stance paws at paws[x], the swinging leg (index) at
        swing_xz; returns (valid, stability margin, worst torque share, joint angles)."""
        if need_stab is None:
            need_stab = STABILITY  # read here: --stability changes the module setting after this function is defined
        dog = self.dog
        q, valid = {}, np.ones(np.shape(bx), bool)
        for i, leg in enumerate(LEGS):
            if i == swing:
                px, pz = swing_xz
            else:
                px, pz = paws[i], self.paw_z(paws[i])
            h, k, ok = dog.ik(leg, pitch, bx, bz, px, pz)
            q[leg] = (h, k)
            before = valid.copy()
            valid &= ok
            self._count("reach" + ("_swing" if i == swing else ""), before, valid)
        if not valid.any():
            return valid, np.zeros(np.shape(bx)), np.zeros(np.shape(bx)), q
        maps = dog.link_maps(pitch, neck, q)
        world = dog.world_points(maps, bx, bz)
        before = valid.copy()
        for name, prims in world.items():
            for p in prims:
                b2 = valid.copy()
                d = self.t.clearance(p)
                leg = name[:2]
                if name.endswith("_s3") and leg in LEGS:
                    i = LEGS.index(leg)
                    if i == swing:
                        # in the air: clear by SWING_CLEAR except right at lift-off / touchdown (caller)
                        valid &= d >= -1e-4
                    else:
                        valid &= d >= -1e-4  # resting on the tread (not in it)
                else:
                    valid &= d >= CLEAR
                self._count("collision_" + name, b2, valid)
        cx = dog.com_x(maps, bx)
        stance = [(paws[i], dog.paw_y[leg]) for i, leg in enumerate(LEGS) if i != swing]
        margin = support_margin(stance, cx)
        before = valid.copy()
        valid &= margin >= need_stab
        self._count("balance", before, valid)
        f = contact_forces(stance, cx, dog.mass)
        valid &= np.all(f >= -1e-6, -1)
        # joint torques of the stance legs from their vertical paw forces
        worst = np.zeros(np.shape(bx))
        j = 0
        for i, leg in enumerate(LEGS):
            if i == swing:
                continue
            a1, tx1, tz1 = maps[f"{leg}_s1"]
            hx, _ = rot(maps["chassis"][0], *dog.hip[leg])
            kx, _ = rot(a1, *dog.knee[leg])
            hip_x, knee_x = hx + bx, kx + tx1 + bx
            for jx in (hip_x, knee_x):
                worst = np.maximum(worst, np.abs(f[..., j] * (paws[i] - jx)) / TORQUE)
            j += 1
        before = valid.copy()
        valid &= worst <= 1.0
        self._count("torque", before, valid)
        return valid, margin, worst, q

    def body_candidates(self, paws):
        x0, dz, pitch, neck = self.grid
        mean_x = np.mean(paws)
        front_z = np.mean([self.paw_z(paws[0]), self.paw_z(paws[1])])
        rear_z = np.mean([self.paw_z(paws[2]), self.paw_z(paws[3])])
        return mean_x + x0, (front_z + rear_z) / 2 + dz, pitch, neck

    def swing_path(self, leg_i, x_from, x_to, n=10):
        r = self.dog.paw_r
        lo, hi = min(x_from, x_to) - r - 0.01, max(x_from, x_to) + r + 0.01
        xs = np.linspace(lo, hi, 40)
        top = max(float(np.max(self.t.height(xs))), self.paw_z(x_from) - r, self.paw_z(x_to) - r)
        z_up = top + r + SWING_CLEAR
        z0, z1 = self.paw_z(x_from), self.paw_z(x_to)
        pts = [(x_from, z0 + (z_up - z0) * s) for s in np.linspace(0.25, 1, 3)]
        pts += [(x_from + (x_to - x_from) * s, z_up) for s in np.linspace(0.2, 0.8, 4)]
        pts += [(x_to, z1 + (z_up - z1) * s) for s in np.linspace(1, 0.25, 3)]
        return pts

    def swing(self, paws, leg_i, x_to, prev_body):
        """Body poses (from the grid) for swinging leg leg_i to x_to; returns the best two as dicts. Poses are
        filtered check by check, so the expensive checks only run on the poses still valid."""
        pose = list(self.body_candidates(paws))
        new = list(paws)
        new[leg_i] = x_to

        def keep(mask, *extra):
            nonlocal pose
            pose = [v[mask] for v in pose]
            return [e[mask] for e in extra]

        # the body holds all four paws before the lift and after the touchdown (shifts happen on four paws)
        for stance in (paws, new):
            v, _, _, _ = self.pose_check(stance, *pose, need_stab=0.0)
            keep(v)
            if len(pose[0]) == 0:
                return []
        margin_min = np.full(len(pose[0]), 1.0)
        path = self.swing_path(leg_i, paws[leg_i], x_to)
        for p in path[4:6] + path[:4] + path[6:]:  # the highest points first: they rule out the most
            v, m, _, _ = self.pose_check(paws, *pose, swing=leg_i, swing_xz=p)
            margin_min = np.minimum(margin_min, m)
            (margin_min,) = keep(v, margin_min)
            if len(pose[0]) == 0:
                return []
        bx, bz, pitch, neck = pose
        dist = np.zeros(len(bx))
        if prev_body is not None:
            pbx, pbz, pp, pn = prev_body
            dist = np.abs(bx - pbx) + np.abs(bz - pbz) + 0.15 * np.abs(pitch - pp) + 0.1 * np.abs(neck - pn)  # smooth motion
        score = margin_min - 0.3 * dist
        out = []
        for k in np.argsort(-score)[:6]:
            body = (float(bx[k]), float(bz[k]), float(pitch[k]), float(neck[k]))
            if prev_body is None or self.shift_ok(paws, prev_body, body):
                out.append({"body": body, "margin": float(margin_min[k])})
            if len(out) >= 2:
                break
        return out

    def shift_ok(self, paws, a, b, n=5):
        """Body moves from pose a to pose b on all four paws."""
        for s in np.linspace(0, 1, n)[1:-1]:
            pose = [np.array([a[i] + s * (b[i] - a[i])]) for i in range(4)]
            v, _, _, _ = self.pose_check(paws, *pose, need_stab=0.005)
            if not bool(v[0]):
                return False
        return True

    def joint_angles(self, paws, body, swing=None, swing_xz=None):
        pose = [np.array([v]) for v in body]
        _, _, _, q = self.pose_check(paws, *pose, swing=swing, swing_xz=swing_xz, need_stab=-1.0)
        return {f"{leg}_j1": float(q[leg][0][0]) for leg in LEGS} | {f"{leg}_j2": float(q[leg][1][0]) for leg in LEGS} | {
            "neck_j0": float(body[3])}

    def rank(self, st) -> float:
        """Beam order: progress of the slowest paw, then of all; legs kept near their natural spacing; safety."""
        p = st["paws"]
        spread = abs(p[0] - p[2] - self.span) + abs(p[1] - p[3] - self.span) + abs(p[0] - p[1]) + abs(p[2] - p[3])
        return min(p) + 0.3 * float(np.mean(p)) - 0.5 * spread + 0.2 * st["margin"]

    def plan(self, start_paws, start_body, beam=8, max_swings=None, targets=6, partner=True):
        states = [{"paws": tuple(start_paws), "body": start_body, "phase": 0, "frames": [], "margin": 1.0}]
        use_partner = partner
        if max_swings is None:  # four swings per 3 cm the slowest paw has to go, plus slack
            max_swings = 4 * math.ceil((self.goal_x - min(start_paws)) / 0.03) + 24
        for _ in range(max_swings):
            nxt = []
            for st in states:
                leg_i = ORDER[st["phase"] % 4]
                paws = list(st["paws"])
                reach = self.footholds[(self.footholds >= paws[leg_i] + 0.02) & (self.footholds <= paws[leg_i] + 0.20)]
                cands = set(reach[np.linspace(0, len(reach) - 1, min(targets, len(reach))).round().astype(int)]) if len(reach) else set()
                partner = paws[PARTNER[leg_i]]
                if use_partner and paws[leg_i] + 0.02 <= partner <= paws[leg_i] + 0.20 and np.isclose(self.footholds, partner).any():
                    cands.add(partner)  # where the other leg of the pair stands
                cands = sorted(cands, reverse=True)
                if not cands:
                    # nothing to do for this leg (already ahead): skip it
                    nxt.append({**st, "phase": st["phase"] + 1})
                    continue
                for x_to in cands:
                    for opt in self.swing(paws, leg_i, float(x_to), st["body"]):
                        new = list(paws)
                        new[leg_i] = float(x_to)
                        frames = st["frames"] + [{"swing": LEGS[leg_i], "from": paws[leg_i], "to": float(x_to),
                                                  "paws": paws, "body": opt["body"], "margin": opt["margin"]}]
                        nxt.append({"paws": tuple(new), "body": opt["body"], "phase": st["phase"] + 1, "frames": frames,
                                    "margin": min(st["margin"], opt["margin"])})
            if not nxt:
                return None
            for st in nxt:
                if min(st["paws"]) >= self.goal_x - 1e-6:
                    return st
            # keep the beam: most progress, then the safest
            uniq = {}
            for st in nxt:
                key = (tuple(round(p, 3) for p in st["paws"]), st["phase"] % 4)
                if key not in uniq or st["margin"] > uniq[key]["margin"]:
                    uniq[key] = st
            ranked = sorted(uniq.values(), key=self.rank, reverse=True)
            states = ranked[:beam]
        return None


SEARCH = {"beam": 8, "targets": 6, "partner": True}  # search settings (--beam, --targets, --no_partner)


def apply_settings(settings: dict) -> None:
    """Module settings for this process (worker processes start fresh): limits and search."""
    g = globals()
    for key in ("JOINT_SHARE", "SWING_CLEAR", "STABILITY", "TORQUE", "MAX_PITCH", "RISER_GAP"):
        if settings.get(key) is not None:
            g[key] = settings[key]
    SEARCH.update({k: settings[k] for k in SEARCH if settings.get(k) is not None})


def solve_cell(dog: Dog2D, rise: float, tread: float, up: bool, n: int = 5) -> dict:
    t = Terrain.stairs(rise, tread, n, up)
    goal_x = (n - 1) * tread + EDGE + 0.005  # every paw on the last segment
    p = Planner(dog, t, goal_x)
    # standing before the first edge: front paws 3 cm before it, the stance of the rest pose
    span = float(dog.paw["L0"][0] - dog.paw["L1"][0])
    front = float(p.footholds[p.footholds <= -0.02].max())
    start = [front, front, front - span, front - span]
    z0 = p.paw_z(front)
    body = (float(np.mean(start)), float(z0 - dog.paw["L0"][1]), 0.0, 0.0)
    started = time.time()
    result = p.plan(start, body, beam=SEARCH["beam"], targets=SEARCH["targets"], partner=SEARCH["partner"])
    out = {"rise": rise, "tread": tread, "dir": "up" if up else "down", "feasible": result is not None,
           "settings": {"joint_share": JOINT_SHARE, "swing_clear_m": SWING_CLEAR, "stability_m": STABILITY, "max_pitch_deg": MAX_PITCH, "riser_gap_m": RISER_GAP,
                        "torque_nm": round(TORQUE, 3), **SEARCH},
           "seconds": round(time.time() - started, 1), "slope_deg": round(math.degrees(math.atan2(rise, tread)), 1)}
    if result is not None:
        frames = []
        for f in result["frames"]:
            leg_i = LEGS.index(f["swing"])
            frames.append({"swing": f["swing"], "paws_before": list(map(float, f["paws"])), "to": f["to"],
                           "body": list(f["body"]), "margin_m": round(f["margin"], 4),
                           "q": p.joint_angles(f["paws"], f["body"]),
                           "path": [list(map(float, xz)) for xz in p.swing_path(leg_i, f["from"], f["to"])],
                           "q_path": [p.joint_angles(f["paws"], f["body"], swing=leg_i, swing_xz=xz)
                                      for xz in p.swing_path(leg_i, f["from"], f["to"])]})
        out.update({"swings": len(frames), "min_margin_m": round(result["margin"], 4),
                    "max_pitch_deg": round(math.degrees(max(abs(f["body"][2]) for f in frames)), 1),
                    "start_body": list(body), "start_paws": start, "frames": frames})
    return out


def solve_column(job) -> dict:
    """One direction and tread, rising rises (stops after the first rise without a plan unless try_all)."""
    genome, d, tr, rises, try_all, out_dir, keep, settings = job
    apply_settings(settings)
    dog = Dog2D(genome)
    table, failed = {}, False
    for r in rises:
        if failed and not try_all:
            table[(d, tr, r)] = None  # not tried: a lower rise already had no plan
            continue
        name = f"{d}_r{round(r * 100)}_t{round(tr * 100)}.json"
        if keep and (Path(out_dir) / name).exists():
            res = json.loads((Path(out_dir) / name).read_text())  # solved before
        else:
            res = solve_cell(dog, r, tr, d == "up")
        failed = not res["feasible"]
        table[(d, tr, r)] = res["feasible"]
        name = f"{d}_r{round(r * 100)}_t{round(tr * 100)}.json"
        (Path(out_dir) / name).write_text(json.dumps(res, indent=1))
        print(f"[stairs_plan] {d} rise {100 * r:.0f} cm tread {100 * tr:.0f} cm ({res['slope_deg']} deg): "
              f"{'POSSIBLE' if res['feasible'] else 'no plan'} ({res['seconds']} s"
              + (f", {res['swings']} steps, margin {100 * res['min_margin_m']:.1f} cm, pitch up to "
                 f"{res['max_pitch_deg']} deg)" if res["feasible"] else ")"), flush=True)
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=GENOME)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--rise", type=float, default=None)
    ap.add_argument("--tread", type=float, default=None)
    ap.add_argument("--dir", choices=("up", "down", "both"), default="both")
    ap.add_argument("--workers", type=int, default=10, help="processes (one per direction and tread)")
    ap.add_argument("--max_rise", type=float, default=0.10, help="[m] highest rise tried")
    ap.add_argument("--per_cell", action="store_true",
                    help="one process per cell (all rises tried, no skipping): faster with many cores")
    ap.add_argument("--keep", action="store_true", help="keep cells already solved in --out_dir")
    ap.add_argument("--all", action="store_true", help="also try higher rises after a rise failed (default: skip them)")
    ap.add_argument("--joint_share", type=float, default=None, help=f"share of the joint range (default {JOINT_SHARE})")
    ap.add_argument("--stability", type=float, default=None, help=f"[m] balance margin (default {STABILITY})")
    ap.add_argument("--swing_clear", type=float, default=None, help=f"[m] swing clearance (default {SWING_CLEAR})")
    ap.add_argument("--max_pitch", type=float, default=None, help=f"[deg] nose-up limit of the body (default {MAX_PITCH})")
    ap.add_argument("--riser_gap", type=float, default=None, help=f"[m] paw sphere to the riser ahead (default {RISER_GAP})")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--targets", type=int, default=None)
    ap.add_argument("--no_partner", action="store_true", help="no 'step where the partner leg stands' candidate")
    ap.add_argument("--cells", default=None, help="only these cells, e.g. up_r5_t15,down_r4_t9")
    args = ap.parse_args()
    settings = {"JOINT_SHARE": args.joint_share, "SWING_CLEAR": args.swing_clear, "STABILITY": args.stability, "MAX_PITCH": args.max_pitch, "RISER_GAP": args.riser_gap, "beam": args.beam,
                "targets": args.targets, "partner": False if args.no_partner else None}
    rises = [args.rise] if args.rise else [round(0.01 * k, 3) for k in range(1, 11) if 0.01 * k <= args.max_rise + 1e-9]
    treads = [args.tread] if args.tread else [0.25, 0.20, 0.15, 0.12, 0.09]
    dirs = ["up", "down"] if args.dir == "both" else [args.dir]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    if args.cells:
        cells = []
        for c in args.cells.split(","):
            d, r, tr = c.split("_")
            cells.append((d, round(int(r[1:]) / 100, 3), round(int(tr[1:]) / 100, 3)))
        columns = [(args.genome, d, tr, [r], True, str(args.out_dir), False, settings) for d, r, tr in cells]
        dirs = sorted({c[0] for c in cells}, reverse=True)
        treads = sorted({c[2] for c in cells}, reverse=True)
        rises = sorted({c[1] for c in cells})
    elif args.per_cell:
        done = {f.name for f in args.out_dir.glob("*_r*_t*.json")} if args.keep else set()
        columns = [(args.genome, d, tr, [r], True, str(args.out_dir), False, settings) for d in dirs for tr in treads for r in rises
                   if f"{d}_r{round(r * 100)}_t{round(tr * 100)}.json" not in done]
        for name in done:  # already solved: into the table as they are
            res = json.loads((args.out_dir / name).read_text())
            table_pre = (res["dir"], res["tread"], res["rise"])
            globals().setdefault("_PRE", {})[table_pre] = res["feasible"]
    else:
        columns = [(args.genome, d, tr, rises, args.all, str(args.out_dir), args.keep, settings) for d in dirs for tr in treads]
    table = dict(globals().get("_PRE", {}))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(columns))) as pool:
        for part in pool.map(solve_column, columns):
            table.update(part)
    for d in dirs:
        print(f"\n{d}: quasi-static plan found (x) per rise (rows) and tread (columns)")
        print("rise  " + " ".join(f"{100 * t:4.0f}" for t in treads))
        for r in rises:
            print(f"{100 * r:4.0f}  " + " ".join({True: "   x", False: "   .", None: "   -"}[table.get((d, t, r))] for t in treads))
        print("x = plan found, . = no plan, - = not tried (a lower rise had no plan)")
    summary = [{"dir": d, "tread": t, "rise": r, "feasible": v} for (d, t, r), v in table.items()]
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
