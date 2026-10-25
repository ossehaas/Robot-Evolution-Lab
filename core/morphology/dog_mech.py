"""Mechanical design of the dog robot: every printed part as a real solid, ready to print and assemble.

All geometry here is in millimetres in the robot frame (x forward, y left, z up, chassis centre at the origin, rest
pose). ``build_mechanics(genome)`` returns a :class:`Mech` with

    parts       printable solids (PLA / TPU), each tied to the simulation link that carries it and to a print
                orientation (the world direction that points up on the bed)
    servos      the 15 STS3215 placements: frame, link carrying the case, bus id
    hardware    purchased parts (screws, inserts, bearings, ...) with the link they sit on
    electronics boards and battery (mass + placement)
    details     visual-only meshes (eye graphics on the displays, the back spot)

How it goes together (see prototypes/puppy/README.md for the step-by-step):

    body    a structural *deck* (the chassis core) carries the four abduction servos (hanging in saddles), the
            neck and tail servos, the battery bay and the electronics. A *cap* (the back) and a *bowl* (belly and
            flanks) close it; four long M3 screws from the belly clamp bowl, deck and cap together. The battery
            slides in through a belly hatch (one thumb screw).
    leg     abduction servo -> *yoke* (U-bracket on both horns) -> *shoulder* (pad + bracket + tongue, two M3
            screws into the yoke) -> hip servo -> *thigh* (hub on the hip horn, holds the knee servo across the top
            of the thigh) -> crank + pushrod + lever (1:1 parallelogram) -> *shin* on a knee axle -> TPU *ankle
            spring* (a flexure: the passive spring ankle, joint <L|R><i>_ja) -> TPU *paw* with a TPU *sole* spring
            disc (the soft paw contact); see dog_paw.py
    head    neck servo in the chassis -> *neck yoke* (both horns) -> head-tilt servo inside the head on a
            *head mount* plate, which also carries the *eye frame* with the two round displays
    tail    tail servo under the back, *tail* on its horn
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from manifold3d import Manifold

from . import cad
from . import components as C
from . import dog_paw as DP
from . import sts3215 as SV

WALL = 2.0  # body and head shells
LEG_WALL = 2.5  # leg shells
CLEAR = 0.4  # clearance between a part and a servo case
MOVE_GAP = 2.0  # gap between parts that move against each other
probe = cad.probe

# ---------------------------------------------------------------------- data model


@dataclass
class Part:
    name: str
    link: str
    material: str  # "pla" | "tpu"
    color: str  # key into the style palette
    solid: Manifold  # world, rest pose, mm
    print_up: tuple[float, float, float]  # world direction that points up on the print bed
    note: str = ""
    visual: bool = True
    infill: float | None = None  # sparse infill share for the mass model (None: dog.PRINT_INFILL[material])
    split_z: float | None = None  # a flexure: the piece below split_z belongs to link_below (mass, visuals, checks)
    link_below: str | None = None


@dataclass
class Servo:
    joint: str
    link: str  # link the case is screwed to
    S: np.ndarray  # servo frame (4x4, mm), see sts3215
    bus_id: int
    role: str


@dataclass
class Hardware:
    item: str
    link: str
    qty: int
    mass_g: float  # per piece
    use: str


@dataclass
class Electronic:
    component: C.BoxComponent
    link: str
    center: np.ndarray  # mm
    size: tuple[float, float, float]  # mm, world aligned


@dataclass
class Detail:
    name: str
    link: str
    solid: Manifold
    color: str


@dataclass
class JointGeom:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # mm
    axis: np.ndarray
    limit: float
    passive: bool = False  # spring hinge without a servo (the ankle flexure)


@dataclass
class Mech:
    parts: list[Part] = field(default_factory=list)
    servos: list[Servo] = field(default_factory=list)
    hardware: list[Hardware] = field(default_factory=list)
    electronics: list[Electronic] = field(default_factory=list)
    details: list[Detail] = field(default_factory=list)
    joints: list[JointGeom] = field(default_factory=list)
    link_origin: dict[str, np.ndarray] = field(default_factory=dict)  # mm
    collisions: list[tuple[str, str, np.ndarray, tuple, str]] = field(default_factory=list)  # link, kind, T(mm), size(mm), mat
    points: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    keepouts: dict[str, list[Manifold]] = field(default_factory=lambda: defaultdict(list))  # link -> servo solids (checks)
    ankle: object = None  # dog_paw.AnkleDesign (spring ankle + soft paw dimensions and estimates)

    def part(self, name: str) -> Part:
        return next(p for p in self.parts if p.name == name)

    def hw(self, item: str, link: str, qty: int, use: str, mass_g: float | None = None) -> None:
        self.hardware.append(Hardware(item, link, qty, HW_MASS.get(item, 0.5) if mass_g is None else mass_g, use))


# purchased hardware: unit mass [g]
HW_MASS = {
    "M2x8 self-tapping screw (pan head)": 0.25,
    "M3x6 socket head screw": 0.55,
    "M3x8 socket head screw": 0.65,
    "M3x8 button head screw": 0.6,
    "M3x10 countersunk screw": 0.6,
    "M3x20 countersunk screw": 1.1,
    "M3x12 countersunk screw": 0.7,
    "M3x20 socket head screw": 1.3,
    "M3x30 socket head screw": 1.8,
    "M3x45 socket head screw": 2.6,
    "M3x25 socket head screw": 1.6,
    "M3 nyloc nut": 0.3,
    "M3 washer": 0.15,
    "M3 heat-set insert (M3 x 5 x 4.0 mm)": 0.35,
    "M3 heat-set insert, short (M3 x 3 x 4.2 mm)": 0.25,
    "M3 knurled thumb screw (M3x10)": 2.0,
    "683ZZ ball bearing (3 x 7 x 3 mm)": 0.5,
    "M2.5 nylon standoff + screw set": 0.4,
    "M2x8 screw + nut (display / board)": 0.4,
}

# ---------------------------------------------------------------------- small helpers


def _MX() -> np.ndarray:
    return np.diag([-1.0, 1.0, 1.0])


def _MY() -> np.ndarray:
    return np.diag([1.0, -1.0, 1.0])


def mirror_solid(m: Manifold, M: np.ndarray) -> Manifold:
    """Apply a diagonal mirror (diag of +-1) to a solid."""
    out = m
    for k in range(3):
        if M[k, k] < 0:
            n = [0.0, 0.0, 0.0]
            n[k] = 1.0
            out = cad.mirror(out, n)
    return out


def mirror_frame(S: np.ndarray, M: np.ndarray) -> np.ndarray:
    """A mirrored servo is still a real servo (it is symmetric across its width), so rebuild a right-handed frame."""
    o = M @ S[:3, 3]
    x = M @ S[:3, 0]
    z = M @ S[:3, 2]
    return cad.frame(o, x, z)


def servo_world(S: np.ndarray, solid: Manifold) -> Manifold:
    return cad.apply(solid, S)


def slab_y(y0: float, y1: float, big: float = 2000.0) -> Manifold:
    return cad.box_between((-big, y0, -big), (big, y1, big))


def slab_z(z0: float, z1: float, big: float = 2000.0) -> Manifold:
    return cad.box_between((-big, -big, z0), (big, big, z1))


def slab_x(x0: float, x1: float, big: float = 2000.0) -> Manifold:
    return cad.box_between((x0, -big, -big), (x1, big, big))


def prism_y(profile_pts_xz, y0: float, y1: float) -> Manifold:
    """Extrude a polygon given in the x-z plane along y from y0 to y1."""
    # local extrusion is along +z of the (u, v) plane; map u->x, v->z, extrusion->y
    pts = [(float(p[0]), float(p[1])) for p in profile_pts_xz]
    cs = cad.poly2d(pts)
    m = cad.extrude(cs, 0.0, 1.0)
    T = np.array([[1, 0, 0, 0], [0, 0, y1 - y0, y0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    # (u, v, w) -> (u, y0 + w*(y1-y0), v): det < 0 would flip the solid, so fix orientation via mirror below
    return _affine(m, T)


def _affine(m: Manifold, T: np.ndarray) -> Manifold:
    if np.linalg.det(T[:3, :3]) < 0:
        m = cad.mirror(m, (0.0, 0.0, 1.0))
        T = T @ np.diag([1.0, 1.0, -1.0, 1.0])
    return cad.apply(m, T)


def disc_y(center_xz, r: float, y0: float, y1: float) -> Manifold:
    """Cylinder with its axis along y through (x, z) = center_xz."""
    T = cad.frame((center_xz[0], y0, center_xz[1]), (1, 0, 0), (0, 1, 0))
    return cad.cyl(r, 0.0, y1 - y0, T)


def hull_y(points_xz_r, y0: float, y1: float) -> Manifold:
    """Hull of y-axis discs (x, z, r) between y0 and y1: a rounded 2D outline extruded along y."""
    return cad.hull([disc_y((x, z), r, y0, y1) for x, z, r in points_xz_r])


def pebble_y(points_xz_r, y_base: float, y_top: float, round_r: float) -> Manifold:
    """Rounded 'pebble': outline from discs at y_base, shrinking by ``round_r`` with a quarter-round towards y_top."""
    layers = []
    n = 5
    sgn = 1.0 if y_top > y_base else -1.0
    h = abs(y_top - y_base)
    for k in range(n + 1):
        phi = (math.pi / 2) * k / n
        inset = round_r * (1 - math.cos(phi))
        y = y_base + sgn * ((h - round_r) + round_r * math.sin(phi)) if k > 0 else y_base
        pts = [(x, z, max(r - inset, 0.5)) for x, z, r in points_xz_r]
        y_lo, y_hi = (y - 0.01, y) if sgn > 0 else (y, y + 0.01)
        layers.append(hull_y(pts, y_lo, y_hi))
    layers.append(hull_y(points_xz_r, min(y_base, y_base + sgn * 0.01), max(y_base, y_base + sgn * 0.01)))
    return cad.hull(layers)



def keep_main(m: Manifold, name: str, warnings: list[str] | None = None, min_vol: float = 30.0) -> Manifold:
    """Drop slivers left by boolean cuts; warn if a sizeable separate piece would be lost."""
    pieces = m.decompose()
    if len(pieces) <= 1:
        return m
    pieces.sort(key=lambda p: -p.volume())
    keep = [pieces[0]]
    for p in pieces[1:]:
        if p.volume() >= min_vol:
            keep.append(p)
            if warnings is not None:
                warnings.append(f"{name}: separate piece of {p.volume():.0f} mm3 (kept)")
    return cad.union(keep)


# ---------------------------------------------------------------------- the builder


class DogMechBuilder:
    def __init__(self, genome):
        self.g = genome
        self.m = Mech()
        g = genome
        self.L, self.W, self.H = 1000 * g.body.length, 1000 * g.body.width, 1000 * g.body.height
        self.r_plan = min(36.0, 0.19 * self.L)
        self.r_top = 0.40 * self.H
        self.r_bot = 0.37 * self.H
        self.ya = 33.5  # abduction axis y (leaves room for the neck servo between the front abduction servos)
        self.z_hip = -0.25 * self.H
        self.za = self.z_hip + 13.75  # abduction axis z (above the hip)
        self.x_hip = 1000 * g.legs.hip_x
        self.x_yoke = min(self.x_hip, self.L / 2 - 38.0)  # abduction yoke centre: must stay clear of the rounded body corners
        self.yk = self.W / 2 + 52.0  # knee servo horn face (the leg's main plane)
        self.yh = self.yk - 4.0  # hip servo horn face
        self.deck0 = self.za + SV.FAR + 0.5  # deck underside
        self.deck1 = self.deck0 + 3.0
        a, b = 1000 * g.legs.upper, 1000 * g.legs.lower
        self.a, self.b = a, b
        self.depth = g.legs.rest_depth * (a + b)
        self.alpha = math.acos(min(1.0, (a * a + self.depth * self.depth - b * b) / (2 * a * self.depth)))
        self.rk = 32.0  # knee servo axis: distance from the hip along the thigh
        self.crank = 18.0
        self.k_off = 16.0
        self.paw_r = 1000 * g.legs.paw_radius
        self.ankle = DP.design(g, self.paw_r)  # spring ankle + soft paw dimensions (dog_paw.py)
        self.m.warnings += [f"ankle: {n}" for n in self.ankle.notes]
        self.body_outer = cad.soft_box((0, 0, 0), (self.L, self.W, self.H), self.r_plan, self.r_top, self.r_bot)
        self.body_inner = cad.soft_box((0, 0, 0), (self.L - 2 * WALL, self.W - 2 * WALL, self.H - 2 * WALL),
                                       self.r_plan - WALL, self.r_top - WALL, self.r_bot - WALL)
        self.next_id = 1

    # ------------------------------------------------------------------ shared
    def add_part(self, name, link, material, color, solid, up, note="", visual=True, **extra):
        if solid.is_empty():
            self.m.warnings.append(f"part {name} is empty")
        self.m.parts.append(Part(name, link, material, color, solid, tuple(float(v) for v in up), note, visual, **extra))

    def add_servo(self, joint, link, S, role, horn_link):
        self.m.servos.append(Servo(joint, link, S, 0, role))
        self.m.keepouts[link].append(servo_world(S, SV.case_solid()))
        self.m.keepouts[horn_link].append(servo_world(S, SV.horns()))

    def body_z_top(self, x: float, y: float) -> float:
        """Height of the body's top surface above (x, y) (ray cast from above)."""
        hits = self.body_outer.trim_by_plane((0, 0, -1), -60)  # keep z <= 60 (trivial), then probe with a thin rod
        rod = cad.box_between((x - 0.05, y - 0.05, -200), (x + 0.05, y + 0.05, 200))
        inter = hits ^ rod
        if inter.is_empty():
            return float("nan")
        return float(cad.bbox(inter)[1][2])

    def tongue_solid(self, grow: float) -> Manifold:
        """Channel from the yoke flange (inside the body) out to the shoulder bracket plate."""
        g_ = grow
        y_plate0 = self.yh + SV.IDL_PAD - 4.0
        x0 = max(self.x_yoke - 13.0, self.x_hip - 16.0)
        x1 = min(self.x_yoke + 13.0, self.x_hip + 16.0)
        return cad.hull(cad.box_between((x0 - g_, self.ya + 20.9 - g_, self.za - 15.6 - g_),
                                        (x1 + g_, self.ya + 28.0, self.za + 6.0 + g_)),
                        cad.box_between((self.x_hip - 14.0 - g_, y_plate0 - 4.0, self.za - 15.6 - g_),
                                        (self.x_hip + 14.0 + g_, y_plate0 + 0.5 + g_, self.za + 6.0 + g_)))

    # ------------------------------------------------------------------ leg (front left master)
    def leg_geometry(self):
        xh, zh, yk, yh = self.x_hip, self.z_hip, self.yk, self.yh
        al = self.alpha
        d = np.array([-math.sin(al), 0.0, -math.cos(al)])  # thigh direction hip -> knee (front leg: knee behind)
        nf = np.array([math.cos(al), 0.0, -math.sin(al)])  # perpendicular, towards the front / down
        Ph = np.array([xh, 0.0, zh])
        K = Ph + self.a * d
        F = Ph + np.array([0.0, 0.0, -self.depth])
        Pk = Ph + self.rk * d - self.k_off * nf  # knee servo axis, shifted towards the back of the thigh
        u_pk = (K := Ph + self.a * d) - Pk
        u_pk = u_pk / np.linalg.norm(u_pk)
        cb = np.array([-u_pk[2], 0.0, u_pk[0]])  # crank / lever at rest: square to the servo-knee line, towards the front
        if np.dot(cb, nf) < 0:
            cb = -cb
        e = (F - K) / np.linalg.norm(F - K)  # shin direction
        return dict(d=d, nf=nf, Ph=Ph, K=K, F=F, Pk=Pk, cb=cb, e=e,
                    Pc=Pk + self.crank * cb, Pl=K + self.crank * cb)

    def build_leg_master(self):
        """Front-left leg in world coordinates: solids, servo frames and hardware, grouped by simulation link."""
        G = self.leg_geometry()
        xh, zh, ya, za, yk, yh = self.x_hip, self.z_hip, self.ya, self.za, self.yk, self.yh
        d, nf, Ph, K, F, Pk, cb, e, Pc, Pl = (G[k] for k in ("d", "nf", "Ph", "K", "F", "Pk", "cb", "e", "Pc", "Pl"))
        out = {"hw": [], "geom": G}
        Y = np.array([0.0, 1.0, 0.0])
        X = np.array([1.0, 0.0, 0.0])
        lim_abd, lim_hip, lim_knee = 0.5, 1.2, 1.0

        def at(p, y):
            return np.array([p[0], y, p[2]])

        # ---------------- servo frames
        xy = self.x_yoke
        x_o = xy + 18.65  # abduction output horn face; the yoke spans x_o-41.3 .. x_o+4
        S_abd = cad.frame((x_o, ya, za), (0, 0, 1), (1, 0, 0))
        S_hip = cad.frame((xh, yh, zh), (0, 0, 1), (0, 1, 0))
        S_knee = cad.frame((Pk[0], yk, Pk[2]), nf, (0, 1, 0))
        out["S"] = dict(abd=S_abd, hip=S_hip, knee=S_knee)
        A = np.array([0.0, ya, za])  # a point on the abduction axis
        out["A"] = A
        hip_core = cad.union([servo_world(S_hip, SV.case_solid(CLEAR)), servo_world(S_hip, SV.horns(CLEAR)),
                              servo_world(S_hip, SV.plug_zone(0.5))])
        knee_core = cad.union([servo_world(S_knee, SV.case_solid(CLEAR)), servo_world(S_knee, SV.horns(0.8)),
                               servo_world(S_knee, SV.plug_zone(0.6))])

        # ---------------- yoke: U-bracket around the abduction servo, bolted to both horns
        def arm(z0, z1):
            pieces = [cad.cyl(11.0, z0, z1, S_abd)]
            for ws in (-9.4, 9.4):
                pieces.append(cad.cyl(3.0, z0, z1, S_abd @ cad.translation((-12.6, ws, 0))))
            return cad.hull(pieces)

        out_arm = arm(0.0, 4.0)
        idl_arm = arm(-41.3, -37.3)
        bridge = cad.box_between((x_o - 41.3, ya - 12.4, za - 15.6), (x_o + 4.0, ya + 12.4, za - 11.6))
        flange = cad.rounded_box((xy, ya + 18.0, za - 9.3), (26.0, 5.0, 12.6), 1.5)
        gusset = cad.hull(cad.box_between((xy - 13, ya + 8.0, za - 15.6), (xy + 13, ya + 13.0, za - 11.6)),
                          cad.box_between((xy - 13, ya + 15.5, za - 15.6), (xy + 13, ya + 20.5, za - 3.0)))
        yoke = cad.union([out_arm, idl_arm, bridge, flange, gusset])
        abd_case = cad.union([servo_world(S_abd, SV.case_solid(CLEAR)), servo_world(S_abd, SV.plug_zone(0.5))])
        yoke = yoke - cad.swept(abd_case, A, X, -lim_abd, lim_abd)
        yoke = yoke - servo_world(S_abd, SV.horn_bolt_cutter(0, 4.0, "output", head_from=3.6))
        yoke = yoke - servo_world(S_abd, SV.horn_bolt_cutter(-41.3, -37.3, "idler", head_from=-40.9))
        xi0, xi1 = max(xy - 13.0, xh - 16.0) + 5.0, min(xy + 13.0, xh + 16.0) - 5.0
        ins_pts = [np.array([xi0, ya + 20.5, za - 9.0]), np.array([xi1, ya + 20.5, za - 9.0])]
        for p in ins_pts:  # heat-set inserts from the outboard face
            yoke = yoke - cad.cyl_between(p + 0.5 * Y, p - 5.5 * Y, 2.0)
        out["yoke"] = yoke
        out["abd_case"] = abd_case
        out["hw"] += [("M3x6 socket head screw", "s0", 8, "yoke to both abduction horns"),
                      ("M3 heat-set insert (M3 x 5 x 4.0 mm)", "s0", 2, "yoke flange (shoulder screws)")]

        # ---------------- shoulder: rounded pad around the hip servo + bracket plate + tongue into the body
        y_pad0, y_pad1 = 57.5, yh - 0.5
        hx0, hx1 = xh - SV.HALF_W - CLEAR, xh + SV.HALF_W + CLEAR
        hz0, hz1 = zh - SV.NEAR - CLEAR, zh + SV.FAR + CLEAR
        rr = 13.0
        pad_pts = [(x, y, z) for x in (hx0 + rr - LEG_WALL - 0.5, hx1 - rr + LEG_WALL + 0.5)
                   for y in (y_pad0 + rr, y_pad1 + rr)  # beyond y_pad1 is trimmed flat below
                   for z in (hz0 + rr - LEG_WALL - 2.0, hz1 - rr + LEG_WALL + 0.5)]
        pad_E = cad.hull([cad.sphere(p, rr) for p in pad_pts] + [disc_y((xh, zh), 19.0, y_pad0 + 4, y_pad1)])
        pad_E = pad_E ^ slab_y(y_pad0, y_pad1)
        pad_I = cad.hull([cad.sphere(p, rr - LEG_WALL) for p in pad_pts] + [disc_y((xh, zh), 19.0 - LEG_WALL, y_pad0 + 4, y_pad1)])
        pad_I = pad_I ^ slab_y(y_pad0 + LEG_WALL, yh + SV.OUT_PAD - 0.1)
        pad = pad_E - pad_I
        y_plate0, y_plate1 = yh + SV.IDL_PAD - 4.0, yh + SV.IDL_PAD
        plate = (pad_I ^ slab_y(y_plate0, y_plate1))
        tongue_out = self.tongue_solid(0.0)
        tongue_in = cad.hull(cad.box_between((xi0 - 2.0, ya + 24.0, za - 12.6), (xi1 + 2.0, ya + 30.0, za + 9.0)),
                             cad.box_between((xh - 10.0, y_plate0 - 3.0, za - 12.6), (xh + 10.0, y_plate1 + 1, za + 9.0)))
        tongue = tongue_out - tongue_in
        cutters = [servo_world(S_hip, SV.case_solid(0.2)), servo_world(S_hip, SV.horns(0.8)),
                   servo_world(S_hip, cad.box_between((SV.PLUG_X0 - 0.5, -SV.PLUG_HALF_W - 0.5, -60),
                                                      (SV.PLUG_X1 + 1.0, SV.PLUG_HALF_W + 0.5, SV.IDL_MAIN))),
                   servo_world(S_hip, SV.case_screw_cutter("idler", -39.2))]
        cutters += [cad.cyl_between(p - 1.0 * Y, p + 6.0 * Y, 1.7) for p in ins_pts]  # tongue screws, from inside the channel
        out["shoulder_cutters"] = cad.union(cutters)
        out["shoulder_pieces"] = (pad_E, pad_I, plate, tongue)
        out["shoulder_raw"] = cad.union([pad, plate, tongue]) - out["shoulder_cutters"]
        out["pad_E"] = pad_E
        out["hip_core"] = hip_core
        out["hw"] += [("M2x8 self-tapping screw (pan head)", "s0", 4, "hip servo to shoulder bracket"),
                      ("M3x8 socket head screw", "s0", 2, "shoulder tongue to yoke")]

        # ---------------- moving parts of the knee drive (rest pose)
        y_crank0, y_crank1 = yk, yk + 4.0
        y_rod0, y_rod1 = yk + 4.5, yk + 9.5
        crank = hull_y([(Pk[0], Pk[2], 10.6), (Pc[0], Pc[2], 6.0)], y_crank0, y_crank1)
        crank = crank - servo_world(S_knee, SV.horn_bolt_cutter(0.0, 4.0, "output", head_from=3.6))
        crank = crank - disc_y((Pc[0], Pc[2]), 2.1, yk - 1, yk + 5)
        rod = hull_y([(Pc[0], Pc[2], 5.5), (Pl[0], Pl[2], 5.5)], y_rod0, y_rod1)
        rod = rod - disc_y((Pc[0], Pc[2]), 3.5, yk + 4, yk + 10) - disc_y((Pl[0], Pl[2]), 3.5, yk + 4, yk + 10)
        out["crank"], out["rod"] = crank, rod

        # ---------------- shin with knee boss and lever, socket for the ankle spring at the bottom
        yc = yk - 11.0
        ad = self.ankle
        Fp = np.array([F[0], yc, F[2]])  # paw centre = origin of the paw frame (dog_paw.py)
        boss = disc_y((K[0], K[2]), 12.5, yk - 26.0, yk)
        lever = hull_y([(K[0], K[2], 9.0), (Pl[0], Pl[2], 6.0)], yk, yk + 4.0)
        k6 = K + 6.0 * e  # keep the back of the knee compact (front and rear knees face each other)
        z_sock = F[2] + ad.z_block_top + DP.SPRING_BH + DP.FIT + DP.SOCKET_ROOF  # top of the socket
        sock_top = cad.rounded_box((F[0], yc, z_sock - 4.0), (2 * (DP.SPRING_BX / 2 + DP.FIT + DP.SOCKET_WALL), 30.0, 8.0), 2.5)
        sbody = cad.hull(cad.sphere((k6[0], yc, k6[2]), 15.0), sock_top) ^ slab_y(yk - 26.0, yk + 4.0)
        socket = DP.socket_outer(ad).translate(tuple(Fp))
        shin = cad.union([boss, lever, sbody, socket])
        shin_env = shin  # for clearance sweeps (before holes)
        shin = shin - disc_y((K[0], K[2]), 1.7, yk - 30, yk + 6)
        shin = shin - disc_y((K[0], K[2]), 3.55, yk - 26.5, yk - 23.0) - disc_y((K[0], K[2]), 3.55, yk + 1.0, yk + 4.5)
        shin = shin - disc_y((Pl[0], Pl[2]), 2.1, yk - 1, yk + 5)
        shin = shin - DP.socket_cutters(ad).translate(tuple(Fp))
        out["shin_raw"] = shin
        out["hw"] += [("M3 heat-set insert (M3 x 5 x 4.0 mm)", "s2", 2, "ankle spring socket (inboard cheek)"),
                      ("M3x30 socket head screw", "s2", 2, "ankle spring to shin (from outboard)")]

        # ---------------- thigh envelope: hull of spheres around everything it has to hold
        env = []  # (centre, radius)
        r_c = 7.0
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    loc = np.array([-SV.NEAR + r_c - 3.5 if sx == 0 else SV.FAR - r_c + 3.5,
                                    -SV.HALF_W + r_c - 3.5 if sy == 0 else SV.HALF_W - r_c + 3.5,
                                    SV.IDL_PAD + r_c - 3.5 if sz == 0 else 0.0])
                    env.append((cad.point(S_knee, loc), r_c))
        n_case = len(env)
        zp = SV.IDL_MAIN - SV.PLUG_DEPTH + 2.0
        for xs, ws in ((SV.PLUG_X0 + 3.0, -SV.PLUG_HALF_W + 3), (SV.PLUG_X1 - 3.0, SV.PLUG_HALF_W - 3),
                       (SV.PLUG_X0 + 3.0, SV.PLUG_HALF_W - 3), (SV.PLUG_X1 - 3.0, -SV.PLUG_HALF_W + 3)):
            env.append((cad.point(S_knee, (xs, ws, zp)), 6.5))
        for q in np.linspace(-lim_knee, lim_knee, 5):  # crank pin and lever pin paths (outer layer)
            R = cad.rotation(Y, q)[:3, :3]
            for p in (Pk + R @ (Pc - Pk), K + R @ (Pl - K)):
                env.append((at(p, yk + 5.0), 10.0))
        env.append((at(Pk, yk + 4.0), 14.5))
        env.append((at(Ph, yk + 3.0), 13.0))  # hip hub
        for t in np.linspace(0, 2 * math.pi, 6, endpoint=False):
            env.append((at(Ph + 9.0 * np.array([math.cos(t), 0, math.sin(t)]), yk + 3.0), 10.0))
        env.append((at(K, yk + 4.0), 12.0))  # knee cap (outer fork)
        env.append((at(K, yk - 28.0), 9.0))  # inner fork boss
        # big soft spheres give the rounded 'drumstick' look; the small ones guarantee everything fits inside
        def thigh_pt(along, across, y):
            return np.array([Ph[0], 0, Ph[2]]) + along * d + across * nf + np.array([0, y, 0])

        big = [(thigh_pt(0, 0, yk - 15.0), 30.0),  # dome over the hip
               (thigh_pt(self.rk, 13.0, yk - 16.0), 32.0),  # haunch around the knee servo
               (thigh_pt(self.a, 8.0, yk - 10.0), 19.0),  # knee (slim at the back: only the axle boss there)
               (thigh_pt(0.65 * self.a, 10.0, yk - 5.0), 20.0)]  # front of the thigh (pushrod)
        out["thigh_required"] = cad.hull([cad.sphere(c, r) for c, r in env])
        env = env + big
        thigh_E = cad.hull([cad.sphere(c, r) for c, r in env])
        thigh_I = cad.hull([cad.sphere(c, r - LEG_WALL) for c, r in env])
        out["thigh_E_raw"] = thigh_E
        # clearances against what it moves against: shoulder pad + hip servo (hip joint), shin (knee joint)
        hip_clear = cad.swept(cad.union([cad.grow(pad_E, 2.0), cad.grow(hip_core, 0.8), cad.grow(self.tongue_solid(0.0), 2.0)]),
                              Ph, Y, -lim_hip, lim_hip, math.radians(5))
        knee_clear = cad.swept(cad.grow(shin_env, 1.2), K, Y, -lim_knee, lim_knee, math.radians(6))
        thigh_E = thigh_E - hip_clear - knee_clear
        thigh_I = thigh_I - hip_clear - knee_clear

        # split into the three printed thigh parts
        y_out = yk - 0.4  # outer cover rim = outboard face of the structural part
        y_in = yk + SV.IDL_PAD - 0.4  # inner cover rim: the cover plate rests on the knee servo's idler pad
        shell = thigh_E - thigh_I
        rod_poses = [cad.apply(rod, cad.translation(cad.rotation(Y, q)[:3, :3] @ (Pc - Pk) - (Pc - Pk)))
                     for q in np.linspace(-lim_knee, lim_knee, 9)]
        knee_drive = cad.union([
            cad.swept(cad.grow(crank, 1.5), at(Pk, 0), Y, -lim_knee, lim_knee, math.radians(6)),
            cad.grow(cad.union(rod_poses), 2.0),
            cad.swept(cad.grow(lever, 1.5), K, Y, -lim_knee, lim_knee, math.radians(6)),
        ])
        cover_out = (shell ^ slab_y(y_out, 500)) - knee_drive
        floor = thigh_I ^ slab_y(yk - 4.0, y_out)
        pocket = hull_y([(c[0], c[2], 0.01) for c, r in env[:n_case]], y_in, y_out)
        pocket_walls = (cad.grow(pocket, 2.0) ^ thigh_I ^ slab_y(y_in, y_out)) - pocket
        fork_boss = disc_y((K[0], K[2]), 7.0, y_in, yk - 26.5) ^ thigh_I
        thigh_in = cad.union([shell ^ slab_y(y_in, y_out), floor, pocket_walls, fork_boss])
        thigh_in = thigh_in - knee_core - knee_drive - knee_clear
        thigh_in = thigh_in - servo_world(S_knee, SV.case_screw_cutter("output", 0.0))
        for hx, hy in SV.OUT_HOLES:  # counterbores for the pan heads on the outboard face
            thigh_in = thigh_in - cad.cyl(2.2, -1.3, 3.0, S_knee @ cad.translation((hx, hy, 0)))
        thigh_in = thigh_in - servo_world(S_hip, SV.horn_bolt_cutter(0.0, 3.6, "output", head_from=3.2))
        thigh_in = thigh_in - disc_y((K[0], K[2]), 1.65, yk - 60, yk + 1)
        thigh_in = thigh_in - cad.apply(cad.extrude(cad.CrossSection.circle(3.25, 6), -10.0, 3.0),
                                        cad.frame((K[0], yk - 33.5, K[2]), (1, 0, 0), (0, 1, 0)))
        cover_in = cad.union([shell ^ slab_y(-500, y_in), thigh_I ^ slab_y(y_in - LEG_WALL, y_in)])
        cover_in = cover_in - knee_core
        # loose bits of the structural part next to the inner cover belong to the cover
        pieces = sorted(thigh_in.decompose(), key=lambda q: -q.volume())
        thigh_in = pieces[0]
        for q in pieces[1:]:
            if q.volume() > 1.0 and q.min_gap(cover_in, 0.5) < 0.05:
                cover_in = cover_in + q
        cover_in = keep_main(cover_in, "thigh inner cover", None, 300.0)
        # cover screws: columns in the covers, heat-set inserts in the structural part, at spots with room
        y_top = cad.bbox(thigh_E)[1][1]
        keep_clear = cad.union([knee_drive, knee_core, knee_clear, hip_clear,
                                servo_world(S_hip, SV.horns(1.5)), servo_world(S_knee, SV.horns(1.5))])

        def screw_spots(y_rim, y_lo, y_hi, cands, n=2):
            spots = []
            for c in cands:
                rim = disc_y((c[0], c[2]), 6.0, y_rim - 6.0, y_rim + 6.0)
                col = disc_y((c[0], c[2]), 5.5, y_lo, y_hi) ^ thigh_I
                if (rim - thigh_I).volume() < 1.0 and (col ^ keep_clear).volume() < 1.0 and                         all(np.hypot(c[0] - q[0], c[2] - q[2]) > 30.0 for q in spots):
                    spots.append(c)
                if len(spots) == n:
                    break
            return spots

        cands = [Ph + al_ * d + ac_ * nf for al_ in np.arange(0.0, 76.0, 4.0) for ac_ in np.arange(-26.0, 40.0, 4.0)]
        cands.sort(key=lambda c: -np.linalg.norm(c - (Ph + 0.5 * (K - Ph))))  # prefer spots far apart, near the rim
        out_spots = screw_spots(y_out, y_out - 7.0, y_top, cands)
        in_spots = []  # the inner cover is screwed straight into the knee servo (idler face holes)
        for c in out_spots:
            col = disc_y((c[0], c[2]), 4.5, y_out, y_top) ^ thigh_I
            boss = disc_y((c[0], c[2]), 4.5, y_out - 7.0, y_out) ^ thigh_I
            cover_out = (cover_out + col) - disc_y((c[0], c[2]), 1.7, y_out - 1, y_top + 1)                 - cad.cyl(1.7, 0.0, 2.0, cad.frame((c[0], y_top - 2.0, c[2]), (1, 0, 0), (0, 1, 0)), r1=3.4)                 - disc_y((c[0], c[2]), 3.4, y_top - 0.01, y_top + 5)
            thigh_in = (thigh_in + boss) - disc_y((c[0], c[2]), 2.0, y_out - 5.5, y_out + 1)
        for c in in_spots:
            col = disc_y((c[0], c[2]), 4.5, -500.0, y_in) ^ thigh_I
            boss = disc_y((c[0], c[2]), 4.5, y_in, y_out) ^ thigh_I
            y_low = cad.bbox(col)[0][1] if not col.is_empty() else y_in - 10
            cover_in = (cover_in + col) - disc_y((c[0], c[2]), 1.7, y_low - 1, y_in + 1)                 - cad.cyl(1.7, 0.0, 2.0, cad.frame((c[0], y_low + 2.0, c[2]), (1, 0, 0), (0, -1, 0)), r1=3.4)                 - disc_y((c[0], c[2]), 3.4, y_low - 5, y_low + 0.01)
            thigh_in = (thigh_in + boss) - disc_y((c[0], c[2]), 2.0, y_in - 1, y_in + 5.5)
        out["cover_spots"] = (out_spots, in_spots)
        cover_in = cover_in - servo_world(S_knee, SV.case_screw_cutter("idler", SV.IDL_PAD - 0.4 - LEG_WALL - 1.0, which=(2, 3)))
        # cable exit from the knee servo connectors, towards the hip (the lead loops past the hip axis)
        pz = servo_world(S_knee, cad.box_between((SV.PLUG_X0, -4.0, SV.IDL_MAIN - SV.PLUG_DEPTH - 6), (SV.PLUG_X0 + 6.0, 4.0, SV.IDL_MAIN - 2.0)))
        exit_c = cad.point(S_knee, (SV.PLUG_X0 + 3.0, 0.0, SV.IDL_MAIN - SV.PLUG_DEPTH + 1))
        cover_in = cover_in - cad.cyl_between(exit_c, exit_c + 30.0 * (Ph - Pk) / np.linalg.norm(Ph - Pk) - np.array([0, 4.0, 0]), 4.0)
        # knee axle through the outer cover with a recessed head; spacer tube between lever and cover
        cover_out = cover_out - disc_y((K[0], K[2]), 1.65, yk, y_top + 1) - disc_y((K[0], K[2]), 3.3, y_top - 2.0, y_top + 1)
        spacer = (disc_y((K[0], K[2]), 3.2, yk + 4.3, y_top) - disc_y((K[0], K[2]), 1.7, yk, y_top)) ^ thigh_I
        out["thigh_in_raw"], out["cover_in_raw"], out["cover_out_raw"], out["spacer"] = thigh_in, cover_in, cover_out, spacer
        out["thigh_E"] = thigh_E
        out["hw"] += [("M3x6 socket head screw", "s1", 4, "thigh hub to hip horn"),
                      ("M2x8 self-tapping screw (pan head)", "s1", 4, "knee servo to thigh"),
                      ("M3x6 socket head screw", "s1", 4, "crank to knee horn"),
                      ("M3 heat-set insert, short (M3 x 3 x 4.2 mm)", "s1", 1, "crank pin"),
                      ("M3 heat-set insert, short (M3 x 3 x 4.2 mm)", "s2", 1, "knee lever pin"),
                      ("683ZZ ball bearing (3 x 7 x 3 mm)", "s1", 2, "pushrod eyes"),
                      ("M3x8 button head screw", "s1", 2, "pushrod pins"),
                      ("M3 washer", "s1", 4, "pushrod pins"),
                      ("683ZZ ball bearing (3 x 7 x 3 mm)", "s2", 2, "knee boss"),
                      ("M3x45 socket head screw", "s1", 1, "knee axle"),
                      ("M3 nyloc nut", "s1", 1, "knee axle"),
                      ("M3x20 countersunk screw", "s1", 2, "outer thigh cover"),
                      ("M3 heat-set insert (M3 x 5 x 4.0 mm)", "s1", 2, "outer thigh cover"),
                      ("M2x8 self-tapping screw (pan head)", "s1", 2, "inner thigh cover to knee servo")]

        # ---------------- spring ankle + soft paw (TPU), in the paw frame; placed per leg with the toes forward
        out["ankle"] = dict(spring=DP.spring_solid(ad), paw=DP.paw_housing(ad), sole=DP.sole_solid(ad))
        out["paw_center"] = Fp
        out["hw_paw"] = [("M3x30 socket head screw", "s3", 2, "ankle spring to paw (through the paw)"),
                         ("M3 nyloc nut", "s3", 2, "ankle spring to paw (in the inboard hex recesses)")]
        out["y_layers"] = dict(y_out=y_out, y_in=y_in, y_top=y_top)
        return out

    # ------------------------------------------------------------------ all four legs
    def build_legs(self):
        master = self.leg
        G = master["geom"]
        m = self.m
        knees = getattr(self.g.legs, "knees", "back_front").split("_")
        xh0 = self.x_hip
        for index, (tag, MXr) in enumerate((("front", np.eye(3)), ("rear", _MX()))):
            # the master leg has its knee backwards; the rear legs are its front-back mirror (knee forwards). A leg
            # whose knee should point the other way gets its knee parts mirrored about its own hip plane (x = x_hip)
            flip = (knees[0] == "front") if index == 0 else (knees[1] == "back")
            for side, MYr in (("L", np.eye(3)), ("R", _MY())):
                M = MXr @ MYr
                prefix = f"{side}{index}"
                mir = (lambda s, M=M: mirror_solid(s, M))

                def kp(p, flip=flip):
                    """A point of the knee side of the master leg, mirrored about the hip plane if this leg flips."""
                    p = np.asarray(p, float).copy()
                    if flip:
                        p[0] = 2.0 * xh0 - p[0]
                    return p

                def kmir(s, M=M, flip=flip):
                    """A knee-side part: (hip-plane flip, then) the usual front/rear and left/right mirror."""
                    if flip:
                        s = cad.apply(cad.mirror(s, [1.0, 0.0, 0.0]), cad.translation((2.0 * xh0, 0.0, 0.0)))
                    return mirror_solid(s, M)

                def kframe(S, M=M, flip=flip):
                    if flip:
                        S = cad.frame(kp(S[:3, 3]), _MX() @ S[:3, 0], _MX() @ S[:3, 2])
                    return mirror_frame(S, M)
                lk = {"s0": f"{prefix}_s0", "s1": f"{prefix}_s1", "s2": f"{prefix}_s2", "s3": f"{prefix}_s3", "chassis": "chassis"}
                suffix = f"{tag}_{side}"
                up = lambda v, M=M: tuple(M @ np.asarray(v, float))
                cleanup = lambda s, n: keep_main(s, n, m.warnings)
                self.add_part(f"yoke_{suffix}", lk["s0"], "pla", "frame", mir(master["yoke"]), up((0, 0, 1)),
                              "U-bracket on both abduction horns; print standing on the bridge", visual=False)
                self.add_part(f"shoulder_{suffix}", lk["s0"], "pla", "body", mir(self.shoulder_final), up((0, -1, 0)),
                              "pad + hip bracket + tongue; print on the flat outboard rim, no supports")
                self.add_part(f"thigh_{suffix}", lk["s1"], "pla", "body", kmir(cleanup(master["thigh_in_raw"], "thigh")),
                              up((0, -1, 0)), "structural thigh: hip hub, knee servo pocket, knee fork; print on the flat outboard face")
                self.add_part(f"thigh_cover_out_{suffix}", lk["s1"], "pla", "body",
                              kmir(cleanup(master["cover_out_raw"], "thigh cover")), up((0, 1, 0)),
                              "outer thigh cover over crank and pushrod; print on its rim")
                self.add_part(f"thigh_cover_in_{suffix}", lk["s1"], "pla", "body",
                              kmir(cleanup(master["cover_in_raw"], "thigh inner cover")), up((0, -1, 0)),
                              "inner thigh cover over the knee servo connectors; print on its rim")
                self.add_part(f"crank_{suffix}", lk["s1"], "pla", "frame", kmir(master["crank"]), up((0, 1, 0)),
                              "knee crank on the knee servo horn", visual=False)
                self.add_part(f"pushrod_{suffix}", lk["s1"], "pla", "frame", kmir(master["rod"]), up((0, 1, 0)),
                              "knee pushrod, 683ZZ bearing pressed into each eye", visual=False)
                self.add_part(f"knee_spacer_{suffix}", lk["s1"], "pla", "frame", kmir(master["spacer"]), up((0, 1, 0)),
                              "spacer tube on the knee axle", visual=False)
                self.add_part(f"shin_{suffix}", lk["s2"], "pla", "body", kmir(master["shin_raw"]), up((0, -1, 0)),
                              "shin with knee boss and lever; print on the flat lever face")
                # ankle spring, paw and sole: built in the paw frame, only mirrored left/right, so the toes always
                # point forward whatever way the knee points (the shin's socket is symmetric front/back)
                ad = self.ankle
                center = M @ kp(master["paw_center"])
                ysign = -1.0 if side == "R" else 1.0

                def pf(sol, ysign=ysign, center=center):
                    if ysign < 0:
                        sol = cad.mirror(sol, (0.0, 1.0, 0.0))
                    return sol.translate(tuple(center))

                ank = master["ankle"]
                self.add_part(f"ankle_spring_{suffix}", lk["s2"], "tpu", "paw", pf(ank["spring"]), (0, ysign, 0),
                              f"TPU 95A, 100 % infill (solid); print lying on its side; blade {ad.t:.2f} x {ad.L:.1f} mm "
                              f"= {ad.k_est / 1000:.2f} N*m/rad (tune: README)",
                              infill=1.0, split_z=float(center[2] + ad.zp), link_below=lk["s3"])
                self.add_part(f"paw_{suffix}", lk["s3"], "tpu", "paw", pf(ank["paw"]), (0, 0, 1),
                              "TPU 95A, 3 walls, 15 % infill; print standing on its rim (sole opening down)")
                self.add_part(f"sole_{suffix}", lk["s3"], "tpu", "paw", pf(ank["sole"]), (0, 0, -1),
                              f"TPU 95A, 100 % infill (solid); print flat, arms on the bed; arms {ad.t_sole:.2f} mm "
                              f"= {ad.k_sole_est:.1f} N/mm; snaps into the paw", infill=1.0)
                for jn, link, S, role, horn_link in (("j0", "chassis", master["S"]["abd"], "abduction", "s0"),
                                                     ("j1", "s0", master["S"]["hip"], "hip", "s1"),
                                                     ("j2", "s1", master["S"]["knee"], "knee", "s1")):
                    frame_ = kframe(S) if jn == "j2" else mirror_frame(S, M)
                    self.add_servo(f"{prefix}_{jn}", lk[link], frame_, f"{tag} {side} {role}", lk[horn_link])
                for item, link, qty, use in master["hw"] + master["hw_paw"]:
                    m.hw(item, lk[link], qty, f"{use} ({prefix})")
                # joints (axes as in the simulation model: same pitch axis both sides, mirrored abduction)
                sign = -1.0 if side == "R" else 1.0
                A = M @ np.array([self.x_hip, self.ya, self.za])
                Hp = M @ np.array([self.x_hip, self.yh, self.z_hip])
                Kp = M @ kp([G["K"][0], self.yk - 11.0, G["K"][2]])
                pivot = center + np.array([0.0, 0.0, ad.zp])  # centre of the spring's blade
                m.joints += [JointGeom(f"{prefix}_j0", "chassis", lk["s0"], A, sign * np.array([1.0, 0, 0]),
                                       self.g.legs.abduction_limit),
                             JointGeom(f"{prefix}_j1", lk["s0"], lk["s1"], Hp, np.array([0, 1.0, 0]), 1.2),
                             JointGeom(f"{prefix}_j2", lk["s1"], lk["s2"], Kp, np.array([0, 1.0, 0]), 1.0),
                             JointGeom(f"{prefix}_ja", lk["s2"], lk["s3"], pivot, np.array([0, 1.0, 0]), ad.limit, passive=True)]
                m.link_origin.update({lk["s0"]: A, lk["s1"]: Hp, lk["s2"]: Kp, lk["s3"]: pivot})
                # collision shapes (simulation): pad box, thigh capsule + haunch sphere, shin capsule, paw sphere
                lo, hi = cad.bbox(self.shoulder_final)
                c = M @ ((lo + hi) / 2)
                m.collisions.append((lk["s0"], "box", cad.translation(c), tuple(hi - lo), "pla"))
                Ph, Kk, F = G["Ph"], G["K"], G["F"]
                yc = self.yk - 11.0
                cap_a = M @ np.array([Ph[0], self.yk - 12.0, Ph[2]])
                cap_b = M @ kp([Kk[0], self.yk - 12.0, Kk[2]])
                m.collisions.append((lk["s1"], "capsule", *self._capsule(cap_a, cap_b, 20.0), "pla"))
                hc = G["Pk"] + 12.0 * G["nf"]
                m.collisions.append((lk["s1"], "sphere", cad.translation(M @ kp([hc[0], self.yk - 16.0, hc[2]])), (26.0,), "pla"))
                # shin capsule down to the socket (its bottom sits ~2 mm above the pivot), paw sphere on the paw link
                sock_c = center + np.array([0.0, 0.0, ad.zp + 2.0 + 14.0])
                m.collisions.append((lk["s2"], "capsule", *self._capsule(M @ kp([Kk[0], yc, Kk[2]]), sock_c, 14.0), "pla"))
                m.collisions.append((lk["s3"], "sphere", cad.translation(center), (self.paw_r,), "tpu"))

    @staticmethod
    def _capsule(p0, p1, r):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        dvec = p1 - p0
        length = float(np.linalg.norm(dvec))
        ref = np.array([1.0, 0, 0]) if abs(dvec[0]) < 0.9 * length else np.array([0, 1.0, 0])
        T = cad.frame((p0 + p1) / 2, ref, dvec)
        return T, (r, max(length - 2 * r, 1.0))

    # ------------------------------------------------------------------ neck, head, tail geometry (frames first)
    def head_geometry(self):
        hd = self.g.head
        L, H = self.L, self.H
        k = 1000 * hd.neck / 35.0
        N = np.array([L / 2 - 22.0, 0.0, H / 2 - 3.5])
        T = N + np.array([35.0, 0.0, 35.0]) * k
        Hc = T + np.array([30.0, 0.0, 37.0])
        size = np.array([1000 * hd.length, 1000 * hd.width, 1000 * hd.height])
        beta = math.radians(60.0)  # neck servo body points back-down, so its swing stays behind the neck
        S_neck = cad.frame((N[0], 18.65, N[2]), (-math.cos(beta), 0, -math.sin(beta)), (0, 1, 0))
        S_tilt = cad.frame(T, (0, 0, 1), (-1, 0, 0))
        x_t = -L / 2 + 24.0
        z_top = probe(self.body_outer, (x_t, 0.0, 200.0), (0, 0, -1))
        z_t = z_top - 5.0
        S_tail = cad.frame((x_t, 0.0, z_t), (1, 0, 0), (0, 0, 1))
        return dict(N=N, T=T, Hc=Hc, size=size, S_neck=S_neck, S_tilt=S_tilt, S_tail=S_tail, x_t=x_t, z_t=z_t)

    def build_neck_head(self):
        m = self.m
        hg = self.head_geometry()
        self.hg = hg
        N, T, Hc, size = hg["N"], hg["T"], hg["Hc"], hg["size"]
        S_neck, S_tilt = hg["S_neck"], hg["S_tilt"]
        hd = self.g.head
        X, Y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        lim_nod, lim_tilt = 0.6, 0.5

        # ---------------- head shells
        r_plan, r_top, r_bot = 0.30 * size[1], 0.40 * size[2], 0.32 * size[2]
        snout_c = Hc + np.array([size[0] / 2 + 1000 * hd.snout / 2 - 16.0, 0.0, -22.0])
        snout_size = np.array([1000 * hd.snout + 16.0, 0.55 * size[1], 0.42 * size[2]])
        head_O = cad.union([cad.soft_box(Hc, size, r_plan, r_top, r_bot), cad.rounded_box(snout_c, snout_size, 20.0)])
        head_I = cad.union([cad.soft_box(Hc, size - 2 * WALL - 0.5, r_plan - WALL, r_top - WALL, r_bot - WALL),
                            cad.rounded_box(snout_c, snout_size - 2 * WALL - 0.5, 20.0 - WALL)])
        head_shell = head_O - head_I
        x_m0, x_m1 = T[0] - 0.8, T[0] + 3.2  # head mount plate (on the tilt servo's output face)

        # eyes: round displays behind openings in the face
        eye_y, eye_z = 0.20 * size[1], Hc[2] + 6.0
        eyes = []
        for sy in (1, -1):
            # glass front plane: at least 1.2 mm of wall in front of the glass rim and the board corners
            rim = [(sy * eye_y + 18.5 * math.cos(t), eye_z + 18.5 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
            pcb = [(sy * eye_y + dy, eye_z + dz) for dy in (-20.4, 20.4) for dz in (-18.9, 18.9)]
            x_rim = min(probe(head_O, (Hc[0], y, z), (1, 0, 0)) for y, z in rim)
            x_pcb = min(probe(head_O, (Hc[0], y, z), (1, 0, 0)) for y, z in pcb)
            x_c = probe(head_O, (Hc[0], sy * eye_y, eye_z), (1, 0, 0))
            eyes.append(np.array([min(x_rim - 1.2, x_pcb + 0.7, x_c - WALL - 0.5), sy * eye_y, eye_z]))
        self.eyes = eyes
        disp_keep = []
        for E in eyes:
            Tx = cad.frame(E, (0, 1, 0), (1, 0, 0))  # local z = +x
            disp_keep.append(cad.union([cad.cyl(18.3, -2.0, 0.0, Tx),
                                        cad.apply(cad.rounded_box((0, 0, -2.8), (37.8, 40.8, 1.8), 6.0), Tx),
                                        cad.apply(cad.box((32.0, 34.0, 8.0), (0, 0, -7.6)), Tx)]))
        displays = cad.union(disp_keep)
        eye_bosses = cad.union([cad.cyl(23.0, 0.0, 40.0, cad.frame(E, (0, 1, 0), (1, 0, 0))) for E in eyes]) ^ head_O
        openings = cad.union([cad.cyl(17.0, -1.5, 40.0, cad.frame(E, (0, 1, 0), (1, 0, 0))) for E in eyes] +
                             [cad.cyl(17.0, 1.2, 9.0, cad.frame(E, (0, 1, 0), (1, 0, 0)), r1=22.0) for E in eyes] +
                             [cad.cyl(22.0, 9.0, 40.0, cad.frame(E, (0, 1, 0), (1, 0, 0))) for E in eyes])
        # columns for the four long screws (head back -> mount -> face), with webs to the side walls
        col_pts = [(sy * 0.32 * size[1], T[2] + 8.0) for sy in (1, -1)] + [(sy * 0.27 * size[1], T[2] + 60.0) for sy in (1, -1)]
        cols_front, cols_back = [], []
        for cy, cz in col_pts:
            web_dir = np.sign(cy)
            cf = cad.box_between((x_m1, cy - 4.5, cz - 4.5), (x_m1 + 14.0, cy + 4.5, cz + 4.5))
            webf = cad.box_between((x_m1, cy, cz - 1.5), (x_m1 + 14.0, cy + web_dir * 80, cz + 1.5))
            cols_front.append(cad.union([cf, webf]) ^ head_I)
            cb_ = cad.box_between((x_m0 - 30.0, cy - 4.5, cz - 4.5), (x_m0, cy + 4.5, cz + 4.5))
            webb = cad.box_between((x_m0 - 30.0, cy, cz - 1.5), (x_m0, cy + web_dir * 80, cz + 1.5))
            cols_back.append(cad.union([cb_, webb]) ^ head_I)
        face = cad.union([head_shell ^ slab_x(x_m1, 1000)] + cols_front)
        face = (face + eye_bosses) - openings - displays
        nose_pt = snout_c + np.array([snout_size[0] / 2 - 4.0, 0.0, snout_size[2] / 2 - 8.0])
        back = cad.union([head_shell ^ slab_x(-1000, x_m0)] + cols_back)
        mount = (head_O ^ slab_x(x_m0, x_m1))
        # tilt servo + its clearances
        tilt_ko = cad.union([servo_world(S_tilt, SV.case_solid(CLEAR)), servo_world(S_tilt, SV.plug_zone(0.5))])
        face = face - tilt_ko
        mount = mount - servo_world(S_tilt, SV.horns(1.4)) - servo_world(S_tilt, SV.case_screw_cutter("output", -3.2 - 4.0))
        mount = mount - servo_world(S_tilt, SV.case_solid(0.2))
        # long screws: insert holes in the front columns, clearance through mount and back, recessed heads
        for cy, cz in col_pts:
            p = np.array([x_m1, cy, cz])
            face = face - cad.cyl_between(p - X, p + 6.0 * X, 2.0)
            mount = mount - cad.cyl_between(p - 6 * X, p + X, 1.7)
            back = back - cad.cyl_between(p - 40 * X, p + X, 1.7)
            back_x = probe(head_O, (x_m0, cy, cz), (-1, 0, 0))
            back = back - cad.cyl_between(np.array([back_x - 5, cy, cz]), np.array([x_m1 + 6.0 - 30.0, cy, cz]), 3.1)
        m.hw("M3x30 socket head screw", "head", 4, "head back + mount + face")
        m.hw("M3 heat-set insert (M3 x 5 x 4.0 mm)", "head", 4, "face columns")
        m.hw("M2x8 self-tapping screw (pan head)", "head", 4, "tilt servo to head mount")
        # neck opening in the back of the head (stem of the neck yoke passes through)
        back = back - cad.cyl(14.5, -60.0, 10.0, cad.frame(T, (0, 1, 0), (1, 0, 0)))
        mount = mount - cad.cyl(12.0, -10.0, 10.0, cad.frame(T, (0, 1, 0), (1, 0, 0)))
        mount = mount - cad.rounded_box(T + np.array([0.0, 0.0, -17.0]), (12.0, 14.0, 7.0), 3.0)  # tilt servo + display cables
        # eye frame: holds both displays, two legs back to the mount
        x_ef1 = min(E[0] for E in eyes) - 3.6
        x_ef0 = x_ef1 - 3.0
        leg_pts = [(sy * 50.0, eye_z - 8.0) for sy in (1, -1)]
        legs_ef = []
        ef_plate = cad.union([cad.hull([cad.cyl(22.0, x_ef0, x_ef1, cad.frame((0, E[1], E[2]), (0, 1, 0), (1, 0, 0))) for E in eyes])]
                             + [cad.hull(cad.cyl(22.0, x_ef0, x_ef1, cad.frame((0, E[1], E[2]), (0, 1, 0), (1, 0, 0))),
                                         cad.box_between((x_ef0, ly - 4.5, lz - 4.5), (x_ef1, ly + 4.5, lz + 4.5)))
                                for E, (ly, lz) in zip(eyes, leg_pts)])
        ef_plate = ef_plate - cad.union([cad.cyl(15.0, x_ef0 - 1, x_ef1 + 1, cad.frame((0, E[1], E[2]), (0, 1, 0), (1, 0, 0)))
                                         for E in eyes])
        for ly, lz in leg_pts:
            legs_ef.append(cad.box_between((x_m1 + 0.2, ly - 4.0, lz - 4.0), (x_ef0 + 0.5, ly + 4.0, lz + 4.0)))
        head_I2 = cad.soft_box(Hc, size - 2 * WALL - 2.5, r_plan - WALL - 1, r_top - WALL - 1, r_bot - WALL - 1)
        eye_frame = (cad.union([ef_plate] + legs_ef) ^ head_I2) - displays - tilt_ko
        for ly, lz in leg_pts:
            p = np.array([x_m1, ly, lz])
            eye_frame = eye_frame - cad.cyl_between(p - X, p + 6 * X, 2.0)
            mount = mount - cad.cyl_between(p - 6 * X, p + X, 1.7)
        m.hw("M3x10 countersunk screw", "head", 2, "eye frame to head mount")
        m.hw("M3 heat-set insert (M3 x 5 x 4.0 mm)", "head", 2, "eye frame legs")
        m.hw("M2x8 screw + nut (display / board)", "head", 4, "displays to eye frame (or hot glue)")
        # ears (TPU, floppy): flat-backed half ellipsoids screwed to the head sides near the top
        ears, ear_up = [], []
        L_e = 1000 * hd.ear_length
        for sy in (1, -1):
            top = np.array([Hc[0] - 12.0, sy * (size[1] / 2 - 6.0), Hc[2] + size[2] / 2 - 12.0])
            wall_y = probe(head_O, (top[0], 0.0, top[2] - 0.4 * L_e), (0, sy, 0))
            if hd.ear_style == "up":
                c = top + np.array([0.0, 0.0, 0.45 * L_e])
                R = cad.rotation(X, sy * -0.35)[:3, :3]
                ear = cad.ellipsoid(c, (20.0, 5.0, 0.5 * L_e), R) - head_O
                ear_up.append(tuple(R @ np.array([0.0, sy, 0.0])))
            else:
                c = np.array([top[0], wall_y + sy * 3.0, top[2] - 0.40 * L_e + 4.0])
                R = cad.rotation(X, sy * 0.30)[:3, :3]
                ear = cad.ellipsoid(c, (26.0, 6.0, 0.5 * L_e), R)
                # flat back (prints flat): keep the outer half of the ellipsoid plus 1 mm
                Fe = np.eye(4)
                Fe[:3, :3], Fe[:3, 3] = R, c
                ear = ear ^ cad.apply(cad.box_between((-100, -1.0 if sy > 0 else -100, -100), (100, 100 if sy > 0 else 1.0, 100)), Fe)
                ear = ear - head_O
                ear_up.append(tuple(R @ np.array([0.0, sy, 0.0])))
            ears.append(ear)
            for dx in (-7.0, 7.0):
                p = np.array([top[0] + dx, sy * (size[1] / 2 - 10.0), top[2] - 8.0])
                hole = cad.cyl_between(p - sy * 25 * Y, p + sy * 25 * Y, 1.7)
                face = face - hole
                ears[-1] = ears[-1] - hole
        m.hw("M3x12 countersunk screw", "head", 4, "ears")
        m.hw("M3 nyloc nut", "head", 4, "ears")
        nose = cad.ellipsoid(nose_pt, (8.0, 12.0, 9.0)) ^ slab_x(nose_pt[0] - 3.0, 1000)  # flat back, glued
        face = face - cad.grow(nose, 0.2)

        # ---------------- neck: yoke on both neck-servo horns, stem to the tilt horn plate, cosmetic shell
        u_nt = (T - N) / np.linalg.norm(T - N)
        # neck end plane (x) just behind the back of the head, wherever the round neck end meets it
        xs = [probe(head_O, (T[0], 21.0 * math.cos(t), T[2] + 21.0 * math.sin(t)), (-1, 0, 0))
              for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        xs = [x for x in xs if not math.isnan(x)] + [probe(head_O, (T[0], 0.0, T[2]), (-1, 0, 0))]
        D = np.array([min(xs) - 2.0, 0.0, T[2]])
        neck_ko = cad.union([servo_world(S_neck, SV.case_solid(CLEAR)), servo_world(S_neck, SV.plug_zone(0.5))])
        neck_ko_sw = cad.swept(neck_ko, N, Y, -lim_nod, lim_nod, math.radians(5))
        # cross bar between the two arms: a curved plate just outside the servo's near end (r >= 16.4 about N),
        # kept behind the head so the head can tilt freely
        r_bar = 20.6
        bar_dirs = [np.array([math.cos(t), 0.0, math.sin(t)]) for t in np.radians([58.0, 72.0, 86.0])]
        bar = cad.hull([disc_y((N + r_bar * u)[0:3:2], 4.0, -22.65, 22.65) for u in bar_dirs])
        arms = [cad.hull([disc_y((N[0], N[2]), 11.0, y0, y1)] + [disc_y((N + r_bar * u)[0:3:2], 4.0, y0, y1) for u in bar_dirs])
                for y0, y1 in ((18.65, 22.65), (-22.65, -18.65))]
        Bc = N + r_bar * bar_dirs[1]
        stem = cad.union([
            cad.hull([disc_y((Bc[0], Bc[2]), 6.0, -10.0, 10.0),
                      cad.cyl(12.5, D[0] - T[0] - 8.0, D[0] - T[0] - 6.0, cad.frame(T, (0, 1, 0), (1, 0, 0)))]),
            cad.hull([cad.cyl(12.5, D[0] - T[0] - 8.0, D[0] - T[0] - 6.0, cad.frame(T, (0, 1, 0), (1, 0, 0))),
                      cad.cyl(10.5, -4.0, 0.0, cad.frame(T, (0, 1, 0), (1, 0, 0)))]),
        ])
        arms.append(bar)
        neck_yoke = cad.union(arms + [stem])
        neck_yoke = neck_yoke - neck_ko_sw
        neck_yoke = neck_yoke - servo_world(S_neck, SV.horn_bolt_cutter(0.0, 4.0, "output", head_from=3.6))
        neck_yoke = neck_yoke - servo_world(S_neck, SV.horn_bolt_cutter(-41.3, -37.3, "idler", head_from=-40.9))
        neck_yoke = neck_yoke - servo_world(S_tilt, SV.horn_bolt_cutter(0.0, 4.0, "output", head_from=3.6))
        for hx, hy in SV.HORN_HOLES:  # screwdriver access through the stem
            p = cad.point(S_tilt, (hx, hy, 3.6))
            neck_yoke = neck_yoke - cad.cyl_between(p, p - 40 * X, 3.3)
        neck_yoke = neck_yoke - cad.cyl_between(T - 40 * X, T + X, 4.0)  # centre screw access
        neck_yoke = neck_yoke - cad.capsule(T + np.array([-14.0, 0, -14.0]), N + 22.0 * u_nt, 4.5)  # head cables
        neck_yoke = keep_main(neck_yoke, "neck yoke", m.warnings)
        # neck servo mount: two cheeks on the neck servo's faces (near holes), standing on the deck top
        d1 = self.deck1
        nm = cad.union([
            cad.box_between((N[0] - 16.0, -24.0, d1), (N[0] + 16.0, 24.0, d1 + 3.0)),
            cad.box_between((N[0] - 17.0, 18.65 - 3.2, d1), (N[0] + 10.0, 18.65 - 3.2 + 4.0, N[2] + 2.0)),
            cad.box_between((N[0] - 17.0, 18.65 - 35.2 - 4.0, d1), (N[0] + 10.0, 18.65 - 35.2, N[2] + 2.0)),
        ])
        nm = nm - servo_world(S_neck, SV.case_solid(0.2)) - servo_world(S_neck, SV.plug_zone(0.5))
        nm = nm - servo_world(S_neck, SV.case_screw_cutter("output", 2.7, which=(0, 1)))
        nm = nm - servo_world(S_neck, SV.case_screw_cutter("idler", -37.9, which=(0, 1)))
        nm = nm - cad.swept(cad.grow(neck_yoke, 0.6), N, Y, -lim_nod, lim_nod, math.radians(4))
        nm = nm ^ cad.soft_box((0, 0, 0), (self.L - 2 * WALL - 1.0, self.W - 2 * WALL - 1.0, self.H - 2 * WALL - 1.0),
                               self.r_plan - WALL - 0.5, self.r_top - WALL - 0.5, self.r_bot - WALL - 0.5)
        self.nm_screws = [(N[0] - 8.0, 21.5), (N[0] - 8.0, -21.5)]
        for px, py in self.nm_screws:
            nm = nm - cad.cyl(1.7, d1 - 5, d1 + 4, cad.translation((px, py, 0)))
        self.neck_mount = nm
        m.hw("M2x8 self-tapping screw (pan head)", "chassis", 4, "neck servo to neck mount")
        # cosmetic neck shell (two halves), groove for the TPU collar
        # base: a rounded barrel on the nod axis (the gap to the back stays even while nodding), tapering to the head
        r_base, r_end, half_w = 27.5, 21.0, 23.5

        def barrel(r, hw):
            return cad.hull(disc_y((N[0], N[2]), r, -hw, hw), disc_y((N[0], N[2]), r - 4.0, -hw - 3.5, hw + 3.5))

        def neck_solid(dr):  # barrel (front trimmed so the neck stays behind the head) hulled with the end disc
            b_ = barrel(r_base - dr, half_w - dr + (1.0 if dr > 0 else 0.0))
            whole = cad.hull(b_, cad.cyl(r_end - dr, -16.0, -dr, cad.frame(D, (0, 1, 0), (1, 0, 0))))
            return whole - cad.box_between((D[0] - dr, -100, N[2] + 6.0), (1000, 100, 1000))  # stay behind the tilting head

        neck_O = neck_solid(0.0)
        neck_I = neck_solid(WALL)
        neck_S = neck_solid(1.5)  # bottom of the collar groove
        neck_shell = (neck_O - neck_I) - cad.cyl(13.5, -20.0, 5.0, cad.frame(D, (0, 1, 0), (1, 0, 0)))
        neck_shell = neck_shell - neck_ko_sw - cad.grow(neck_yoke, 0.6)
        neck_shell = neck_shell - cad.swept_union(cad.grow(nm, 1.0), N, Y, -lim_nod, lim_nod, math.radians(2.0))
        # TPU collar: a band that hugs the neck, sitting in a 1.5 mm groove
        ax = (D - N) / np.linalg.norm(D - N)
        Mc = N + 0.62 * np.linalg.norm(D - N) * ax
        collar_T = cad.frame(Mc, (0, 1, 0), ax)
        band = cad.apply(cad.box((200.0, 200.0, 9.0)), collar_T)
        band_g = cad.apply(cad.box((200.0, 200.0, 9.4)), collar_T)
        neck_shell = neck_shell - ((cad.grow(neck_O, 6.0) - neck_S) ^ band_g)
        collar = (cad.grow(neck_O, 3.0) - cad.grow(neck_S, 0.1)) ^ band
        front = np.array([ax[2], 0.0, -ax[0]])
        r_mid = probe(neck_O, Mc, (0, 1, 0)) - Mc[1]
        tag_c = Mc + front * (cad.ray_extent(collar, Mc, front) + 1.5) + np.array([0.0, 0.0, -5.0])
        tag = cad.cyl_between(tag_c - 1.2 * front, tag_c + 1.2 * front, 7.0)
        tag = cad.union([tag, cad.cyl_between(tag_c + np.array([0, 0, 5.0]), tag_c + np.array([0, 0, 9.0]), 1.5)]) - cad.grow(collar, 0.3)
        for sy in (1, -1):  # screws under the collar band, into the yoke
            neck_shell = neck_shell - cad.cyl_between(Mc + np.array([0, sy * (r_mid + 5), 0]), Mc, 1.7)
            neck_yoke = neck_yoke - cad.cyl_between(Mc + np.array([0, sy * 30, 0]), Mc + np.array([0, sy * 4, 0]), 2.0)
        self.neck_shell_raw = neck_shell
        m.hw("M3x10 countersunk screw", "neck", 2, "neck shell halves to the yoke (under the collar)")
        m.hw("M3 heat-set insert (M3 x 5 x 4.0 mm)", "neck", 2, "neck yoke (shell screws)")
        m.hw("M3x6 socket head screw", "neck", 8, "neck yoke to both neck-servo horns")
        m.hw("M3x6 socket head screw", "neck", 4, "neck yoke to the tilt horn")

        # ---------------- register parts
        self.add_part("head", "head", "pla", "body", keep_main(face, "head", m.warnings), (1, 0, 0),
                      "face + snout, eye openings; print standing on the back rim (face up)")
        self.add_part("head_back", "head", "pla", "body", keep_main(back, "head back", m.warnings), (-1, 0, 0),
                      "back of the head; print on its rim")
        self.add_part("head_mount", "head", "pla", "body", keep_main(mount, "head mount", m.warnings), (-1, 0, 0),
                      "plate between head and head back: carries the tilt servo and the eye frame; print flat")
        self.add_part("eye_frame", "head", "pla", "frame", keep_main(eye_frame, "eye frame", m.warnings), (-1, 0, 0),
                      "holds the two round displays; print on the front face", visual=False)
        self.add_part("nose", "head", "pla", "nose", nose, (1, 0, 0), "black PLA, print on the flat back, glue into its pocket")
        for sy, ear in zip((1, -1), ears):
            self.add_part(f"ear_{'L' if sy > 0 else 'R'}", "head", "tpu", "accent", ear, ear_up[0 if sy > 0 else 1],
                          "TPU 95A, print on the flat back")
        self.add_part("neck_yoke", "neck", "pla", "frame", neck_yoke, (-1, 0, 0),
                      "U-bracket on both neck horns + stem to the head-tilt horn; print on the stem plate", visual=False)
        self.add_part("collar", "neck", "tpu", "collar", collar, tuple(ax), "TPU ring, stretch over the neck into its groove")
        self.add_part("collar_tag", "neck", "pla", "tag", tag, (1, 0, 0), "yellow PLA tag, glue to the collar")
        self.add_servo("neck_j0", "chassis", S_neck, "neck nod", "neck")
        self.add_servo("head_j1", "head", S_tilt, "head tilt", "neck")
        m.electronics += [Electronic(C.ROUND_DISPLAY, "head", E - np.array([2.0, 0, 0]), (4.0, 38.0, 38.0)) for E in eyes]
        m.joints += [JointGeom("neck_j0", "chassis", "neck", N, Y, lim_nod), JointGeom("head_j1", "neck", "head", T, X, lim_tilt)]
        m.link_origin.update({"neck": N, "head": T})
        # eye graphics on the displays (visual only)
        for E in eyes:
            sy = np.sign(E[1])
            Tx = cad.frame(E + np.array([0.3, 0, 0]), (0, 1, 0), (1, 0, 0))
            m.details.append(Detail("eye", "head", cad.cyl(16.6, -0.3, 0.0, Tx), "eye"))
            m.details.append(Detail("shine", "head", cad.sphere(E + np.array([0.2, sy * 5.5, 7.5]), 4.5) ^ slab_x(E[0], 1000), "shine"))
            m.details.append(Detail("shine2", "head", cad.sphere(E + np.array([0.2, -sy * 4.5, -6.0]), 2.2) ^ slab_x(E[0], 1000), "shine"))
        # collisions
        ah, bh = 0.29 * r_plan, 0.29 * min(r_top, r_bot)  # three boxes approximate the rounded head
        for sz in ((size[0], size[1] - 2 * ah, size[2] - 2 * bh), (size[0] - 2 * ah, size[1], size[2] - 2 * bh),
                   (size[0] - 2 * ah, size[1] - 2 * ah, size[2])):
            m.collisions.append(("head", "box", cad.translation(Hc), tuple(sz), "pla"))
        m.collisions.append(("head", "box", cad.translation(snout_c), tuple(snout_size), "pla"))
        m.collisions.append(("neck", "capsule", *self._capsule(N, D, 22.0), "pla"))
        self.neck_yoke = neck_yoke
        self.neck_O = neck_O
        self.head_O = head_O
        self.neck_ko = neck_ko

    def build_tail(self):
        m = self.m
        hg = self.hg
        S_tail, x_t, z_t = hg["S_tail"], hg["x_t"], hg["z_t"]
        Z = np.array([0.0, 0.0, 1.0])
        L_t = 1000 * self.g.tail.length
        base = cad.cyl(11.0, z_t, z_t + 4.0, cad.translation((x_t, 0.0, 0.0)))
        dvec = np.array([-0.55, 0.0, 0.84])
        p0 = np.array([x_t - 1.0, 0.0, z_t + 5.0])
        tip = p0 + L_t * dvec
        body = cad.hull(cad.sphere(p0, 10.5), cad.sphere(tip, 7.0)) ^ slab_z(z_t, 1000)
        peg = cad.cyl_between(tip, tip + 7.0 * dvec, 2.9)
        tail = cad.union([base, body, peg])
        tail = tail - servo_world(S_tail, SV.horn_bolt_cutter(0.0, 4.0, "output", head_from=3.6))
        tail = tail - (servo_world(S_tail, SV.horns(0.6)) ^ slab_z(-1000, z_t))
        tail = keep_main(tail, "tail", m.warnings)
        tip_ball = cad.sphere(tip + 3.0 * dvec, 9.5) - cad.cyl_between(tip, tip + 7.5 * dvec, 3.1) - cad.grow(body, 0.3)
        tip_ball = tip_ball - cad.apply(cad.box((60, 60, 20)), cad.frame(tip + (3.0 + 8.0 + 10.0) * dvec, (0, 1, 0), dvec))  # flat to print on
        self.add_part("tail", "tail", "pla", "body", tail, (0, 0, 1), "print standing on the base disc")
        self.add_part("tail_tip", "tail", "pla", "accent", keep_main(tip_ball, "tail tip", m.warnings), tuple(-dvec), "print on the flat, glue onto the tail peg")
        self.add_servo("tail_j0", "chassis", S_tail, "tail wag", "tail")
        m.hw("M3x6 socket head screw", "tail", 4, "tail to tail horn")
        m.joints.append(JointGeom("tail_j0", "chassis", "tail", np.array([x_t, 0.0, z_t]), Z, 0.8))
        m.link_origin["tail"] = np.array([x_t, 0.0, z_t])
        m.collisions.append(("tail", "capsule", *self._capsule(p0, tip, 9.0), "pla"))
        self.tail_solid = tail
        self.tail_tip = tip

    # ------------------------------------------------------------------ body: deck, cap, bowl, hatch
    def corner_mirrors(self):
        return [np.eye(3), _MY(), _MX(), _MX() @ _MY()]

    def build_body(self):
        m = self.m
        L, W, H = self.L, self.W, self.H
        X, Y, Z = np.eye(3)
        lg = self.leg
        A = lg["A"]
        x_o = self.x_yoke + 18.65
        d0, d1 = self.deck0, self.deck1

        # scoops: room for the hip servo (+ pad wall) when the leg abducts, so the shoulder never hits the body
        hip_case = servo_world(lg["S"]["hip"], SV.case_solid(0.0))
        lo_, hi_ = cad.bbox(hip_case)
        c_, sz_ = (lo_ + hi_) / 2, hi_ - lo_
        sc_o = cad.swept(cad.rounded_box(c_, sz_ + 11.5, 11.0), A, X, -0.5, 0.5, math.radians(3))
        sc_i = cad.swept(cad.rounded_box(c_, sz_ + 15.5, 13.0), A, X, -0.5, 0.5, math.radians(3))
        scoop_o = cad.union(mirror_solid(sc_o, M) for M in self.corner_mirrors())
        scoop_i = cad.union(mirror_solid(sc_i, M) for M in self.corner_mirrors())
        outer = self.body_outer - scoop_o
        inner = self.body_inner - scoop_i
        self.body_scooped = outer
        # windows for the shoulder tongues
        tongue = self.tongue_solid(0.0)
        win = cad.swept(cad.grow(tongue, 2.0), A, X, -0.5, 0.5, math.radians(3))
        windows = cad.union(mirror_solid(win, M) for M in self.corner_mirrors())
        # yokes swing inside: keep the inner wall clear of them
        yoke_sw = cad.swept(cad.grow(lg["yoke"], 1.5), A, X, -0.5, 0.5, math.radians(3))
        yokes_sw = cad.union(mirror_solid(yoke_sw, M) for M in self.corner_mirrors())
        shell = outer - inner - windows - yokes_sw
        # neck and tail openings (the neck base is a ball around the nod axis: the gap stays even)
        hg = self.hg
        N = hg["N"]
        tail_axis = np.array([hg["x_t"], 0.0, hg["z_t"]])
        tail_cut = cad.swept(cad.grow(self.tail_solid, MOVE_GAP), tail_axis, Z, -0.8, 0.8, math.radians(5))
        shell = shell - tail_cut
        # power switch and USB-C port in the rear wall, below the tail
        x_rear = -L / 2
        switch_cut = cad.box_between((x_rear - 5, 6.0, -12.5), (x_rear + 12, 16.0, 2.5))
        usb_cut = cad.rounded_box((x_rear + 2, -11.0, -5.0), (16.0, 9.5, 4.0), 1.8)
        usb_screws = [cad.cyl_between((x_rear - 5, y, -5.0), (x_rear + 12, y, -5.0), 1.6) for y in (-25.0, 3.0)]
        shell = shell - switch_cut - usb_cut - cad.union(usb_screws)

        # long screws: belly -> bowl column -> deck -> cap column insert
        cols = [(sx * 28.0, sy * 33.0) for sx in (1, -1) for sy in (1, -1)]
        bowl_cols, cap_cols, col_holes, col_bores, col_inserts = [], [], [], [], []
        for cx, cy in cols:
            bowl_cols.append(cad.cyl(5.0, -60.0, d0, cad.translation((cx, cy, 0))) ^ self.body_inner)
            cap_cols.append(cad.cyl(5.0, d1, 80.0, cad.translation((cx, cy, 0))) ^ self.body_inner)
            col_holes.append(cad.cyl(1.7, -70.0, 90.0, cad.translation((cx, cy, 0))))
            col_bores.append(cad.cyl(3.3, -70.0, d0 - 10.0, cad.translation((cx, cy, 0))))
            col_inserts.append(cad.cyl(2.0, d1 - 1.0, d1 + 6.0, cad.translation((cx, cy, 0))))
        m.hw("M3x20 socket head screw", "chassis", 4, "belly: bowl + deck + cap (long screwdriver, >= 80 mm)")
        m.hw("M3 heat-set insert (M3 x 5 x 4.0 mm)", "chassis", 4, "cap columns")

        # belly hatch opening
        hx0, hx1, hy = -49.0, 41.0, 19.0
        hatch_cut = cad.extrude(cad.rounded_rect((hx1 - hx0) / 2, hy, 6.0), -60.0, -40.0).translate(((hx0 + hx1) / 2, 0, 0))
        # ---------------- deck: the structural core
        deck_out = cad.soft_box((0, 0, 0), (L - 2 * WALL - 0.8, W - 2 * WALL - 0.8, H - 2 * WALL - 0.8),
                                self.r_plan - WALL - 0.4, self.r_top - WALL - 0.4, self.r_bot - WALL - 0.4) - scoop_i
        deck = deck_out ^ slab_z(d0, d1)
        hang = []
        # abduction saddles (front-left master, mirrored): plates on both faces at the servo's far end + side webs
        S_abd = lg["S"]["abd"]
        ya = self.ya
        z_s0 = self.za + 19.0
        sad = cad.union([
            cad.box_between((x_o - 3.2, ya - 15.4, z_s0), (x_o + 0.8, ya + 15.4, d0 + 0.5)),
            cad.box_between((x_o - 39.2, ya - 15.4, z_s0), (x_o - 35.2, ya + 15.4, d0 + 0.5)),
            cad.box_between((x_o - 39.2, ya - 15.4, z_s0 + 6), (x_o + 0.8, ya - 12.8, d0 + 0.5)),
            cad.box_between((x_o - 39.2, ya + 12.8, z_s0 + 6), (x_o + 0.8, ya + 15.4, d0 + 0.5)),
        ])
        sad = sad - servo_world(S_abd, SV.case_solid(0.2)) - servo_world(S_abd, SV.plug_zone(0.5))
        sad = sad - cad.swept(cad.grow(lg["yoke"], 1.5), A, X, -0.5, 0.5, math.radians(3))
        sad = sad - servo_world(S_abd, SV.case_screw_cutter("output", 0.8, which=(2, 3)))
        sad = sad - servo_world(S_abd, SV.case_screw_cutter("idler", -39.2, which=(2, 3)))
        hang += [mirror_solid(sad, M) for M in self.corner_mirrors()]
        m.hw("M2x8 self-tapping screw (pan head)", "chassis", 16, "abduction servos to deck saddles")
        # battery bay: side walls, end stops, boss for the hatch thumb screw
        bay = [cad.box_between((-39.0, sy * 18.0, -44.5), (39.0, sy * 20.0, d0 + 0.5)) for sy in (1, -1)]
        bay += [cad.box_between((sx * 37.8, sy * 10.0, -43.0), (sx * 39.0, sy * 18.0, -32.0)) for sx in (1, -1) for sy in (1, -1)]
        boss = cad.union([cad.cyl(4.5, -44.5, -32.0, cad.translation((-44.0, 0.0, 0.0))),
                          cad.box_between((-44.0, -2.0, -44.5), (-38.0, 2.0, -32.0)),
                          cad.box_between((-39.0, -18.0, -42.5), (-37.8, 18.0, -32.0))])
        bay.append(boss - cad.cyl(2.0, -45.0, -38.0, cad.translation((-44.0, 0, 0))))
        hang += bay
        m.hw("M3 heat-set insert (M3 x 5 x 4.0 mm)", "chassis", 1, "hatch thumb screw")
        m.hw("M3 knurled thumb screw (M3x10)", "chassis", 1, "belly hatch")
        # tail servo shelf under its idler face, hanging from the deck
        S_tail = hg["S_tail"]
        zt = hg["z_t"]
        xt = hg["x_t"]
        shelf = cad.union([cad.box_between((xt - 12.0, -15.3, zt - 39.2), (xt + 37.0, 15.3, zt - 35.2)),
                           cad.box_between((xt - 12.0, 12.8, zt - 39.2), (xt + 37.0, 15.3, d0 + 0.5)),
                           cad.box_between((xt - 12.0, -15.3, zt - 39.2), (xt + 37.0, -12.8, d0 + 0.5))])
        shelf = shelf - servo_world(S_tail, SV.case_solid(0.2)) - servo_world(S_tail, SV.horns(1.0)) \
            - servo_world(S_tail, cad.box_between((SV.PLUG_X0 - 0.5, -SV.PLUG_HALF_W - 0.5, -60), (SV.PLUG_X1 + 1, SV.PLUG_HALF_W + 0.5, SV.IDL_MAIN))) \
            - servo_world(S_tail, SV.case_screw_cutter("idler", -39.2))
        hang.append(shelf)
        m.hw("M2x8 self-tapping screw (pan head)", "chassis", 4, "tail servo to deck shelf")
        deck = cad.union([deck] + [h ^ deck_out for h in hang])
        # passages through the deck: neck and tail servos, cables, column screws
        S_neck = hg["S_neck"]
        deck = deck - servo_world(S_neck, SV.case_solid(0.6)) - servo_world(S_neck, SV.plug_zone(0.6))
        deck = deck - servo_world(S_tail, SV.case_solid(0.6))
        for M in self.corner_mirrors():
            deck = deck - mirror_solid(cad.rounded_box((self.x_yoke - 24.5, 26.5, d0), (6.0, 10.0, 12.0), 2.5), M)
        deck = deck - cad.rounded_box((-2.0, 0.0, d0), (20.0, 12.0, 12.0), 3.0)  # battery / power cables
        deck = deck - cad.union(col_holes)
        # electronics: M2.5 heat-set inserts for nylon standoffs (board hole positions: check your boards)
        boards = [(C.BUS_ADAPTER, (20.0, 0.0)), (C.ESP32_S3, (-10.0, 24.5)), (C.IMU, (-18.0, 0.0)), (C.BUCK, (-10.0, -24.0))]
        for comp, (bx, by) in boards:
            size = tuple(1000.0 * v for v in comp.size)
            sx_, sy_ = size[0] / 2 - 2.5, size[1] / 2 - 2.5
            for px, py in ((bx - sx_, by - sy_), (bx + sx_, by + sy_)):
                deck = deck - cad.cyl(1.8, d1 - 4.0, d1 + 1.0, cad.translation((px, py, 0)))
            m.electronics.append(Electronic(comp, "chassis", np.array([bx, by, d1 + 3.0 + size[2] / 2]), size))
        m.hw("M2.5 nylon standoff + screw set", "chassis", 8, "boards on the deck")
        bat = C.BATTERY_3S if "12V" in self.g.servo else C.BATTERY_2S_DOG
        bsize = tuple(1000.0 * v for v in bat.size)
        m.electronics.append(Electronic(bat, "chassis", np.array([0.0, 0.0, -H / 2 + WALL + bsize[2] / 2]), bsize))
        # the neck base is a round barrel turning about the nod axis: keep the deck out of its wall
        r_nb = 27.5  # outer radius of the neck barrel (see build_neck_head)
        ring = disc_y((N[0], N[2]), r_nb + MOVE_GAP + 0.5, -29.5, 29.5) - disc_y((N[0], N[2]), r_nb - WALL - 2.5, -30.0, 30.0)
        deck = deck - ring
        deck = keep_main(deck, "deck", m.warnings)

        # neck shell: cut only by the fixed parts it would really meet while nodding (all below the deck, out of
        # sight), so the round base stays whole and fills its socket with an even gap
        lg = self.leg
        front_abd = cad.union(mirror_solid(cad.union([servo_world(lg["S"]["abd"], SV.case_solid(1.0)),
                                                      cad.swept(cad.grow(lg["yoke"], 1.0), lg["A"], X, -0.5, 0.5, math.radians(5))]), M)
                              for M in (np.eye(3), _MY()))
        fixed = cad.union([cad.grow(deck ^ cad.box_between((N[0] - 45, -45, d0 - 50), (N[0] + 45, 45, d1 + 1)), 1.5),
                           cad.grow(self.neck_mount, 1.0), front_abd])
        neck_shell = self.neck_shell_raw - cad.swept_union(fixed, N, Y, -0.6, 0.6, math.radians(2.0))
        neck_shell = keep_main(neck_shell, "neck shell", m.warnings)
        self.add_part("neck_L", "neck", "pla", "body", keep_main(neck_shell ^ slab_y(0.0, 100), "neck L", m.warnings),
                      (0, -1, 0), "neck shell half; print on the split face")
        self.add_part("neck_R", "neck", "pla", "body", keep_main(neck_shell ^ slab_y(-100, 0.0), "neck R", m.warnings),
                      (0, 1, 0), "neck shell half; print on the split face")
        self.neck_parts = cad.union([neck_shell, self.neck_yoke])
        # body skin: a round socket for the barrel, plus room for whatever else of the neck swings near it
        socket = disc_y((N[0], N[2]), r_nb + MOVE_GAP, -29.5, 29.5)
        neck_cut = cad.union([socket, cad.swept(cad.grow(self.neck_parts, MOVE_GAP), N, Y, -0.6, 0.6, math.radians(3))])
        shell = shell - socket
        bowl = (shell ^ slab_z(-100, d1))
        bowl = cad.union([bowl] + bowl_cols) - cad.union(col_holes + col_bores) - hatch_cut
        cap = (shell ^ slab_z(d1, 100))
        cap = cad.union([cap] + cap_cols) - cad.union(col_holes + col_inserts) - tail_cut - neck_cut


        for px, py in self.nm_screws:  # neck mount screws
            deck = deck - cad.cyl(2.0, d1 - 5.0, d1 + 0.5, cad.translation((px, py, 0)))
        m.hw("M3x8 socket head screw", "chassis", 2, "neck mount to deck")
        m.hw("M3 heat-set insert (M3 x 5 x 4.0 mm)", "chassis", 2, "deck (neck mount)")

        # hatch: flat panel in the belly opening, tongue under the front edge, thumb screw at the rear
        hatch = cad.extrude(cad.rounded_rect((hx1 - hx0) / 2 - 0.3, hy - 0.3, 5.7), -H / 2, -H / 2 + WALL)
        hatch = hatch.translate(((hx0 + hx1) / 2, 0, 0))
        tongue_h = cad.box_between((hx1 - 3.0, -10.0, -H / 2 + WALL), (hx1 + 6.0, 10.0, -H / 2 + 2 * WALL))
        ribs = [cad.box_between((hx0 + 4, sy * 8 - 1, -H / 2 + WALL), (hx1 - 4, sy * 8 + 1, -H / 2 + WALL + 2.5)) for sy in (1, -1)]
        hatch = cad.union([hatch, tongue_h] + ribs)
        hatch = hatch - cad.cyl(1.7, -60, -30, cad.translation((-44.0, 0, 0))) - cad.cyl(3.6, -60, -H / 2 + 1.2, cad.translation((-44.0, 0, 0)))

        self.add_part("deck", "chassis", "pla", "frame", deck, (0, 0, -1),
                      "structural core: servo saddles, battery bay, tail shelf; print upside down (top face on the bed)",
                      visual=False)
        self.add_part("neck_mount", "chassis", "pla", "frame", keep_main(self.neck_mount, "neck mount", m.warnings, 300.0), (0, 0, 1),
                      "holds the neck servo on the deck; print on its base", visual=False)
        self.add_part("cap", "chassis", "pla", "body", keep_main(cap, "cap", m.warnings), (0, 0, 1),
                      "the back; print on its rim (no supports, the roof bridges)")
        self.add_part("bowl", "chassis", "pla", "body", keep_main(bowl, "bowl", m.warnings), (0, 0, -1),
                      "belly and flanks; print upside down on its rim")
        self.add_part("belly_hatch", "chassis", "pla", "body", hatch, (0, 0, 1), "battery door; print flat, outside down")
        # spot on the back (decal / paint, visual only)
        spot = cad.ellipsoid((-0.15 * L, 0.18 * W, H / 2 - 4.0), (0.28 * L, 0.24 * W, 12.0)) ^ cad.grow(outer, 0.4) - outer
        m.details.append(Detail("spot", "chassis", spot, "accent"))
        # collisions: three boxes approximating the rounded body
        a = 0.29 * self.r_plan
        b_ = 0.29 * min(self.r_top, self.r_bot)
        for size in ((L, W - 2 * a, H - 2 * b_), (L - 2 * a, W, H - 2 * b_), (L - 2 * a, W - 2 * a, H)):
            m.collisions.append(("chassis", "box", np.eye(4), size, "pla"))
        m.link_origin["chassis"] = np.zeros(3)

    # ------------------------------------------------------------------ finishing touches on the shoulders
    def finish_shoulders(self):
        lg = self.leg
        A = lg["A"]
        X = np.array([1.0, 0, 0])
        tongue_zone = self.tongue_solid(0.6) - slab_y(self.yh + SV.IDL_PAD - 4.0, 1000)
        # the pad yields to the body as the leg abducts: carve its outer and inner skins separately so the wall
        # follows the clearance surface and stays closed
        sweep_o = cad.swept_union(cad.grow(self.body_scooped, MOVE_GAP), A, X, -0.5, 0.5, math.radians(2.0)) - tongue_zone
        sweep_i = cad.swept_union(cad.grow(self.body_scooped, MOVE_GAP + LEG_WALL), A, X, -0.5, 0.5, math.radians(2.0)) - tongue_zone
        pad_E, pad_I, plate, tongue = lg["shoulder_pieces"]
        pad = (pad_E - sweep_o) - (pad_I - sweep_i)
        shoulder = cad.union([pad, plate - sweep_o, tongue]) - lg["shoulder_cutters"]
        self.shoulder_final = keep_main(shoulder, "shoulder", self.m.warnings)

    # ------------------------------------------------------------------ everything
    def build(self) -> Mech:
        self.leg = self.build_leg_master()
        self.build_neck_head()
        self.build_tail()
        self.build_body()
        self.finish_shoulders()
        self.build_legs()
        # bus ids in joint order (legs FL, FR, RL, RR: abduction, hip, knee; then neck, head tilt, tail)
        sim_order = [f"{s}{i}_j{k}" for i in (0, 1) for s in ("L", "R") for k in (0, 1, 2)] + ["neck_j0", "head_j1", "tail_j0"]
        by_joint = {sv.joint: sv for sv in self.m.servos}
        self.m.servos = [by_joint[j] for j in sim_order]
        for i, sv in enumerate(self.m.servos):
            sv.bus_id = i + 1
        # the 15 servo joints in bus order, then the passive ankles (L0, R0, L1, R1) as the simulation always had them
        passive = [j for j in self.m.joints if j.passive]
        self.m.joints = sorted((j for j in self.m.joints if not j.passive), key=lambda j: sim_order.index(j.name)) + passive
        self.m.points = dict(N=self.hg["N"], T=self.hg["T"], Hc=self.hg["Hc"], tail_tip=self.tail_tip)
        self.m.ankle = self.ankle
        return self.m


def build_mechanics(genome) -> Mech:
    return DogMechBuilder(genome).build()
