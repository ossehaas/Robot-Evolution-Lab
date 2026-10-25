"""Spring ankle and soft paw of the dog robot (TPU 95A), matching the simulation's ``DogAnkle`` + compliant paw contact.

All geometry in millimetres in the *paw frame*: origin at the paw centre F (the centre of the simulation's paw sphere),
x forward, y outboard (master leg = front left, so +y points away from the body), z up. ``dog_mech`` places the result
at every leg (toes always forward, mirrored in y for the right legs).

    ankle spring  TPU I-profile, extruded along y and printed lying on its side, so the bending stress runs along the
                  printed lines: top block (in a socket at the bottom of the shin, 2 x M3 into heat-set inserts),
                  a straight blade (the flexure), bottom block (in a pocket of the paw, 2 x M3 through the paw).
                  Pivot = blade centre, ``height`` paw radii above F (the simulation's hinge point).
                  Pitch stiffness k = C * E * w * t^3 / (12 L), peak strain at the limit eps = s * limit * t / (2 L).
                  Roll and yaw are much stiffer (the blade is wide in y and short; see tools/dog/ankle_fe.py).
    stops         the shin socket ends in two faces through the pivot axis; the paw's lips are two more such faces,
                  ``limit`` lower, so the paw lands on the shin with full face contact exactly at +-limit.
    paw           TPU housing, round with toes and a slightly flared base; pocket for the spring block; a cavity;
                  a bottom opening with a snap groove for the sole disc.
    sole disc     TPU spring disc, printed flat: rim (snaps into the groove, presses on a conical seat), three
                  90 deg arc arms, a central pad with a domed tread that sticks out ``SOLE_TRAVEL`` below the paw's
                  rim. Vertical stiffness from the arms (bending + torsion of curved beams, k ~ t^3). After
                  SOLE_TRAVEL of travel the paw's rim lands on the floor (bump stop).

A round hollow paw printed sole-down cannot be this soft: it needs a printable flat of r >= 6 mm, and a shell on such a
flat is 35-55 N/mm at 1.2-1.6 mm wall (axisymmetric FE, tools/dog/ankle_fe.py). Hence the separate sole spring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

from . import cad

# ---------------------------------------------------------------------- material: Polymaker PolyFlex TPU95 (95A)
# TDS V5.4 (polymaker.com, 2024): Young's modulus (X-Y, ISO 37) 33.6 +- 0.9 MPa; stress at 100 % strain 11.1 MPa;
# elongation at break 551 +- 24 %; density 1.20-1.24 g/cm3. (TDS V4.0, 2018: 100 % modulus 9.4 MPa, tensile strength
# 29.0 MPa, elongation 330 %.) No flexural modulus, no yield point, no Z values are published. Specimens: 100 % infill.
TPU_E = 33.6  # MPa: change this after a print test (see prototypes/puppy/README.md, "Tuning the ankle and the sole")
TPU_E_FE = 33.6  # MPa: the modulus the FE calibrations below were computed with
TPU_NU = 0.45
STRAIN_DESIGN = 0.10  # peak strain at the ankle limit and at the sole stop: near-linear range, 55x below break

# ---------------------------------------------------------------------- ankle spring
SPRING_W = 20.0  # width along y
SPRING_BX = 20.0  # block length along x
SPRING_BH = 10.0  # block height
SPRING_FILLET = 2.0  # blade root fillets
SPRING_BOLT_X = 5.5  # bolt holes at x = +-5.5 mm, mid-height of each block
# FE calibration (tools/dog/ankle_fe.py, 3D hex model of this profile, voxels aligned with the blade faces): pitch
# stiffness / beam formula with the blocks held on their far + front/back faces (tight socket) / far faces only (soft
# pocket): 0.84 / 0.80 at L/t = 1.73-1.86 (t = 7.0-7.5 mm), 0.90 / 0.87 at L/t = 2.36: the blocks and roots add about
# 0.3-0.4 t of effective length. The strain rule below fixes L/t = SPRING_EPS_FE * limit / (2 eps) = 1.84, so the mean
# there (0.815, mesh-converged within 1 %) is used. Peak strain / beam strain at the blade surface: 0.82-0.93 (mesh-
# sensitive at the fillets); 0.92 is used.
SPRING_K_FE = 0.815
SPRING_EPS_FE = 0.92
FIT = 0.05  # per side: the TPU blocks go in with light pressure (snug, no play in pitch)
CHEEK_IN, SOCKET_WALL, SOCKET_ROOF = 6.0, 3.0, 3.0  # shin socket walls (the inboard cheek holds the inserts)
SHIN_HALF = 15.0  # the shin is 30 mm thick; its outboard face is flat (print bed), the socket ends flush with it
CHEEK_OUT = SHIN_HALF - SPRING_W / 2 - FIT
STOP_PAW = -0.15  # [rad] slope of the paw's stop faces (through the pivot axis); the shin's are STOP_PAW + limit
CHEEK_DROP = 1.5  # the shin cheeks reach down to 1.5 mm above the pivot beside the blade

# ---------------------------------------------------------------------- sole disc
SOLE_TRAVEL = 1.5  # pad below the paw's rim = travel to the bump stop
SOLE_PAD_R = 5.5
SOLE_ARM_R = (6.6, 10.3)  # arm band (1.1 mm slots to pad and rim)
SOLE_RIM_R = (11.4, 13.25)
SOLE_RIM_T = 5.0
SOLE_ARMS = 4
SOLE_FILLET = 0.55  # stub-root fillets (slots are 1.1 mm)
SOLE_ARM_SPAN = math.radians(55.0)
SOLE_STUB_W = 2.4
SOLE_LIP_T = 1.5  # paw lip under the sole rim
SOLE_LIP_R = 11.8  # opening in the paw's bottom (the rim snaps past it: 12 % stretch)
SOLE_SEAT = 1.4  # 45 deg conical seat between rim and paw
SOLE_GROOVE_R = SOLE_RIM_R[1] + 0.15
SOLE_CAVITY_R = SOLE_RIM_R[0] + 0.65
SOLE_TREAD_R = 20.0  # domed tread of the pad
# FE calibration of the sole (tools/dog/ankle_fe.py, 3D hex, rim held, pad pushed up, tread free to slide; sticking
# tread +3 %): k = SOLE_K_REF * (t_arm / SOLE_T_REF)^SOLE_K_EXP [N/mm] (bending + torsion of short curved arms: between
# t^2 and t^3); peak strain per mm of pad travel = SOLE_EPS[0] + SOLE_EPS[1] * (t_arm - 3.5)
SOLE_T_REF = 4.0
SOLE_K_REF = 8.83
SOLE_K_EXP = 2.19
SOLE_EPS = (0.0709, 0.0098)

PAW_ELLIPSOID = ((4.0, 0.0, 1.0), (1.3, 1.05, 1.0 + 3.0 / 18.0))  # centre [mm], radii in paw radii (r = 18: 23.4, 18.9, 21)
PAW_BASE = (1.5, 18.0, 16.0, 3.0)  # flared base: x centre, half-axes, height
PAW_BOLT_HEAD, PAW_BOLT_NUT = 13.4, 12.0  # M3x30: head seat at y = +13.4 (counterbore), nyloc nut seat at -12.0 (hex recess)


@dataclass
class AnkleDesign:
    k: float  # target pitch stiffness [N*mm/rad]
    limit: float  # [rad]
    zp: float  # pivot height above the paw centre [mm]
    t: float  # blade thickness [mm]
    L: float  # blade length [mm]
    k_sole: float  # target sole stiffness [N/mm]
    t_sole: float  # sole arm thickness [mm]
    paw_r: float
    notes: tuple = ()

    # derived
    @property
    def k_est(self) -> float:
        return spring_k(self.t, self.L)

    @property
    def eps_limit(self) -> float:
        return SPRING_EPS_FE * self.limit * self.t / (2 * self.L)

    @property
    def z_block_top(self) -> float:  # bottom face of the top block = top of the blade
        return self.zp + self.L / 2

    @property
    def z_block_bot(self) -> float:  # top face of the bottom block
        return self.zp - self.L / 2

    @property
    def phi_shin(self) -> float:
        return STOP_PAW + self.limit

    @property
    def k_sole_est(self) -> float:
        return sole_k(self.t_sole)

    @property
    def eps_sole(self) -> float:
        return sole_strain(self.t_sole)


def spring_k(t: float, L: float, E: float = TPU_E, w: float = SPRING_W) -> float:
    """Pitch stiffness [N*mm/rad] of the blade: Euler-Bernoulli under pure moment, FE-calibrated."""
    return SPRING_K_FE * E * w * t ** 3 / (12.0 * L)


def spring_dims(k: float, limit: float, E: float = TPU_E, w: float = SPRING_W, eps: float = STRAIN_DESIGN):
    """Blade (t, L) for a pitch stiffness k [N*mm/rad] with peak strain ``eps`` at the limit.

    eps = s * limit * t / (2 L)  ->  L = s * limit * t / (2 eps);  k = C E w t^3 / (12 L)  ->  t = sqrt(6 s limit k / (C E w eps))
    """
    t = math.sqrt(6.0 * SPRING_EPS_FE * limit * k / (SPRING_K_FE * E * w * eps))
    L = SPRING_EPS_FE * limit * t / (2.0 * eps)
    return round(t, 2), round(L, 1)


def sole_k(t: float, E: float | None = None) -> float:
    """Vertical stiffness [N/mm] of the sole disc with arms t [mm] thick (FE-calibrated, scales with the modulus)."""
    return SOLE_K_REF * ((TPU_E if E is None else E) / TPU_E_FE) * (t / SOLE_T_REF) ** SOLE_K_EXP


def sole_t(k: float) -> float:
    return round(SOLE_T_REF * (k / sole_k(SOLE_T_REF)) ** (1.0 / SOLE_K_EXP), 2)


def sole_strain(t: float, travel: float = None) -> float:
    """Peak strain in the sole arms after ``travel`` mm (default: all the way to the bump stop)."""
    return (SOLE_EPS[0] + SOLE_EPS[1] * (t - 3.5)) * (SOLE_TRAVEL if travel is None else travel)


def design(genome, paw_r: float) -> AnkleDesign:
    """Ankle and paw dimensions from the genome (stiffness, limit, hinge height, paw contact stiffness)."""
    an = genome.ankle
    notes = []
    limit = min(max(float(an.limit), 0.25), 0.6)
    if abs(limit - an.limit) > 1e-9:
        notes.append(f"ankle limit {an.limit:.2f} rad clamped to {limit:.2f} (printable stops)")
    k = 1000.0 * float(an.stiffness)
    t, L = spring_dims(k, limit)
    if not 4.0 <= t <= 9.0:
        t2 = min(max(t, 4.0), 9.0)
        notes.append(f"ankle stiffness {an.stiffness} N*m/rad needs a {t:.1f} mm blade; printed with {t2:.1f} mm "
                     f"({spring_k(t2, L) / 1000:.2f} N*m/rad): treat the ankle as {'rigid' if t > 9 else 'soft'}")
        t = t2
    h = min(max(float(an.height), 0.9), 1.3)
    if abs(h - an.height) > 1e-9:
        notes.append(f"ankle height {an.height:.2f} paw radii clamped to {h:.2f}")
    kc = float(getattr(genome, "paw_contact_stiffness", 0.0)) / 1000.0  # N/m -> N/mm
    ts = sole_t(kc) if kc > 0 else SOLE_T_REF * 1.25  # rigid contact in the simulation: a stiff sole
    ts = min(max(ts, 2.5), SOLE_RIM_T)
    return AnkleDesign(k, limit, h * paw_r, t, L, kc, ts, paw_r, tuple(notes))


# ---------------------------------------------------------------------- 2D helpers


def _rect(x0, x1, y0, y1) -> CrossSection:
    return CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0))


def _close(cs: CrossSection, r: float) -> CrossSection:
    """Round concave corners (fillets) with radius r."""
    return cs.offset(r, JoinType.Round, 2.0, 48).offset(-r, JoinType.Round, 2.0, 48)


def _open(cs: CrossSection, r: float) -> CrossSection:
    """Round convex corners with radius r."""
    return cs.offset(-r, JoinType.Round, 2.0, 48).offset(r, JoinType.Round, 2.0, 48)


def _extrude_y(cs: CrossSection, y0: float, y1: float) -> Manifold:
    """Profile in the (x, z) plane extruded along y from y0 to y1."""
    # (u, v, w) -> (u, y0 + w, v) is a reflection; mirror w first so the composite is a proper rotation
    m = Manifold.extrude(cs, y1 - y0).mirror((0.0, 0.0, 1.0))  # w in [-(y1 - y0), 0]
    return m.transform(np.array([[1.0, 0, 0, 0], [0, 0, -1.0, y0], [0, 1.0, 0, 0]]))


# ---------------------------------------------------------------------- ankle spring


def spring_profile(t: float, L: float, holes: bool = True) -> CrossSection:
    """I-profile in (x, z), z measured from the pivot (blade centre)."""
    bx, bh = SPRING_BX / 2, SPRING_BH
    cs = _rect(-bx, bx, L / 2, L / 2 + bh) + _rect(-bx, bx, -L / 2 - bh, -L / 2) + _rect(-t / 2, t / 2, -L / 2 - 1, L / 2 + 1)
    cs = _open(_close(cs, SPRING_FILLET), 0.6)
    if holes:
        for zb in (L / 2 + bh / 2, -L / 2 - bh / 2):
            for xb in (-SPRING_BOLT_X, SPRING_BOLT_X):
                cs = cs - cad.circle2d(1.65, (xb, zb))
    return cs


def spring_solid(d: AnkleDesign) -> Manifold:
    return _extrude_y(spring_profile(d.t, d.L), -SPRING_W / 2, SPRING_W / 2).translate((0.0, 0.0, d.zp))


def spring_bolts(d: AnkleDesign):
    """(x, z) of the four bolt axes (along y), paw frame: top pair first."""
    zt, zb = d.z_block_top + SPRING_BH / 2, d.z_block_bot - SPRING_BH / 2
    return [(x, zt) for x in (-SPRING_BOLT_X, SPRING_BOLT_X)], [(x, zb) for x in (-SPRING_BOLT_X, SPRING_BOLT_X)]


# ---------------------------------------------------------------------- shin socket


def socket_outer(d: AnkleDesign) -> Manifold:
    """Solid to add at the bottom of the shin: the socket block, already cut to its stop faces."""
    sx = SPRING_BX / 2 + FIT + SOCKET_WALL
    y0, y1 = -(SPRING_W / 2 + FIT + CHEEK_IN), SPRING_W / 2 + FIT + CHEEK_OUT
    ztop = d.z_block_top + SPRING_BH + FIT + SOCKET_ROOF
    blk = cad.rounded_box(((0.0), (y0 + y1) / 2, (d.zp - 5 + ztop) / 2), (2 * sx, y1 - y0, ztop - d.zp + 5), 2.5)
    return blk - socket_stop_cut(d)


def socket_stop_cut(d: AnkleDesign) -> Manifold:
    """Everything below the shin's stop faces: z < zp + max(tan(phi) |x|, CHEEK_DROP)."""
    tp = math.tan(d.phi_shin)
    X = 60.0
    prof = cad.poly2d([(-X, -80.0), (X, -80.0), (X, tp * X), (CHEEK_DROP / tp, CHEEK_DROP),
                       (-CHEEK_DROP / tp, CHEEK_DROP), (-X, tp * X)])
    return _extrude_y(prof, -60.0, 60.0).translate((0.0, 0.0, d.zp))


def socket_cutters(d: AnkleDesign) -> Manifold:
    """Cavity for the top block, room for the blade, bolt holes, insert holes (inboard cheek), in the paw frame."""
    bx, w2 = SPRING_BX / 2 + FIT, SPRING_W / 2 + FIT
    z0 = d.z_block_top
    cav = cad.box_between((-bx, -w2, z0 - 20.0), (bx, w2, z0 + SPRING_BH + FIT))
    blade = cad.box_between((-bx, -(SPRING_W / 2 + 0.9), z0 - 40.0), (bx, SPRING_W / 2 + 0.9, z0 + 0.6))
    cuts = [cav, blade]
    (top, _) = spring_bolts(d)
    y_in = -(w2 + CHEEK_IN)
    for x, z in top:
        cuts.append(cad.cyl_between((x, y_in - 1.0, z), (x, w2 + CHEEK_OUT + 1.0, z), 1.7))
        cuts.append(cad.cyl_between((x, y_in - 1.0, z), (x, y_in + 5.5, z), 2.0))  # M3 x 5 x 4.0 insert from inboard
    return cad.union(cuts)


# ---------------------------------------------------------------------- sole disc


def _arc_band(r0, r1, a0, a1, n=48) -> CrossSection:
    ang = np.linspace(a0, a1, n)
    pts = [(r1 * math.cos(a), r1 * math.sin(a)) for a in ang] + [(r0 * math.cos(a), r0 * math.sin(a)) for a in ang[::-1]]
    return cad.poly2d(pts)


def _radial_bar(r0, r1, a, w) -> CrossSection:
    c, s = math.cos(a), math.sin(a)
    n = np.array([-s, c]) * w / 2
    p0, p1 = np.array([c, s]) * r0, np.array([c, s]) * r1
    return cad.poly2d([tuple(p0 - n), tuple(p1 - n), tuple(p1 + n), tuple(p0 + n)])


def sole_arms_profile(n: int | None = None, span: float | None = None, band=None) -> CrossSection:
    """The arms layer of the sole disc in (x, y): pad, n arc arms with radial stubs, rim."""
    n = SOLE_ARMS if n is None else n
    span = SOLE_ARM_SPAN if span is None else span
    cs = CrossSection.circle(SOLE_PAD_R, 96)
    ra0, ra1 = SOLE_ARM_R if band is None else band
    for i in range(n):
        a0 = math.radians(90.0) + i * 2 * math.pi / n  # first pad stub straight outboard (+y)
        a1 = a0 + span
        cs = cs + _arc_band(ra0, ra1, a0 - math.radians(7), a1 + math.radians(7))
        cs = cs + _radial_bar(SOLE_PAD_R - 0.5, ra0 + 0.5, a0, SOLE_STUB_W)
        cs = cs + _radial_bar(ra1 - 0.5, SOLE_RIM_R[0] + 0.5, a1, SOLE_STUB_W)
    cs = cs + (CrossSection.circle(SOLE_RIM_R[1], 128) - CrossSection.circle(SOLE_RIM_R[0], 128))
    # slots are 0.8 mm: round the corners with radii below half of that so no slot closes
    return _open(_close(cs, SOLE_FILLET), 0.3)


def sole_solid(d: AnkleDesign) -> Manifold:
    """Sole disc in the paw frame (pad tread at z = -paw_r)."""
    zs = sole_seat_z(d)
    arms = cad.extrude(sole_arms_profile(), zs - d.t_sole, zs)
    rim = cad.extrude(cad.circle2d(SOLE_RIM_R[1]) - cad.circle2d(SOLE_RIM_R[0]), zs - SOLE_RIM_T, zs)
    # 45 deg chamfer on the rim's top outer edge (sits on the paw's conical seat)
    cham = cad.cyl(SOLE_RIM_R[1] + 1.0, zs - SOLE_SEAT, zs + 0.01) - cad.cyl(SOLE_RIM_R[1], zs - SOLE_SEAT, zs + 0.01,
                                                                         r1=SOLE_RIM_R[1] - SOLE_SEAT)
    zb = -d.paw_r
    pad = cad.cyl(SOLE_PAD_R, zb + 1.0, zs)
    tread = cad.sphere((0.0, 0.0, zb + SOLE_TREAD_R), SOLE_TREAD_R, 96) ^ cad.cyl(SOLE_PAD_R, zb - 1.0, zb + 1.5)
    return cad.union([arms, rim, pad, tread]) - cham


def sole_seat_z(d: AnkleDesign) -> float:
    """Top of the sole disc (= the paw's seat)."""
    return -d.paw_r + SOLE_TRAVEL + SOLE_LIP_T + SOLE_RIM_T


# ---------------------------------------------------------------------- paw housing


def paw_outer(d: AnkleDesign) -> Manifold:
    r = d.paw_r
    (cx, cy, cz), (ax, ay, az) = PAW_ELLIPSOID
    z_rim = -r + SOLE_TRAVEL
    ell = cad.ellipsoid((cx, cy, cz), (ax * r, ay * r, az * r))
    bx, hx, hy, hb = PAW_BASE
    base = cad.extrude(CrossSection.circle(1.0, 96).scale((hx, hy)), z_rim + 0.8, z_rim + hb).translate((bx, 0.0, 0.0))
    foot = cad.extrude(CrossSection.circle(1.0, 96).scale((hx - 0.8, hy - 0.8)), z_rim, z_rim + 0.8).translate((bx, 0.0, 0.0))
    return cad.hull(ell ^ slab_z(z_rim, 1000.0), base, foot)


def slab_z(z0, z1, big=2000.0):
    return cad.box_between((-big, -big, z0), (big, big, z1))


def paw_housing(d: AnkleDesign) -> Manifold:
    r = d.paw_r
    paw = paw_outer(d)
    (cx, _, cz), (ax, ay, az) = PAW_ELLIPSOID
    # toes: two grooves on the front
    for s in (-1, 1):
        x1 = cx + ax * r
        paw = paw - cad.cyl_between((x1 - 7.0, s * 0.36 * r, 4.0), (x1 + 3.0, s * 0.36 * r, 4.0), 1.4)
    # pocket for the spring's bottom block (open to the top), wider beside the blade
    bx, w2 = SPRING_BX / 2 + FIT, SPRING_W / 2 + FIT
    zf = d.z_block_bot - SPRING_BH - FIT
    paw = paw - cad.box_between((-bx, -w2, zf), (bx, w2, 200.0))
    paw = paw - cad.box_between((-bx, -(SPRING_W / 2 + 0.9), d.z_block_bot + 0.5), (bx, SPRING_W / 2 + 0.9, 200.0))
    # around the blade (|x| < pocket) the paw ends 2 mm above the block; beyond, the stop faces (through the pivot)
    paw = paw - cad.box_between((-bx, -50.0, d.z_block_bot + 2.0), (bx, 50.0, 200.0))
    tp = math.tan(STOP_PAW)
    prof = cad.poly2d([(-60.0, 60.0 * tp), (0.0, 0.0), (60.0, 60.0 * tp), (60.0, 80.0), (-60.0, 80.0)])
    paw = paw - _extrude_y(prof, -60.0, 60.0).translate((0.0, 0.0, d.zp))
    # sole: opening (lip), groove for the rim, conical seat, cavity with a 45 deg roof (opens into the pocket)
    z_rim = -r + SOLE_TRAVEL
    zs = sole_seat_z(d)
    zg = z_rim + SOLE_LIP_T
    paw = paw - cad.cyl(SOLE_LIP_R, z_rim - 1.0, zg + 0.01)
    paw = paw - cad.cyl(SOLE_GROOVE_R, zg, zs - SOLE_SEAT)
    paw = paw - cad.cyl(SOLE_GROOVE_R, zs - SOLE_SEAT - 0.01, zs, r1=SOLE_GROOVE_R - SOLE_SEAT)
    apex = zs + 2.5 + SOLE_CAVITY_R
    paw = paw - (cad.cyl(SOLE_CAVITY_R, zs - 0.01, apex) ^ cad.cyl(apex - zs + 0.01, zs - 0.01, apex, r1=0.01))
    # bolts for the spring's bottom block: counterbores outboard, nyloc nut recesses inboard
    (_, bot) = spring_bolts(d)
    for x, z in bot:
        paw = paw - cad.cyl_between((x, -40.0, z), (x, 40.0, z), 1.7)
        paw = paw - cad.cyl_between((x, PAW_BOLT_HEAD, z), (x, 40.0, z), 3.25)
        hexn = cad.apply(cad.extrude(CrossSection.circle(3.25, 6), 0.0, 40.0),
                         cad.frame((x, -PAW_BOLT_NUT, z), (1, 0, 0), (0, -1, 0)))  # flats top/bottom: short bridge
        paw = paw - hexn
    return paw
