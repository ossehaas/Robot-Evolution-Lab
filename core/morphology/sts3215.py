"""Feetech STS3215 serial bus servo: geometry and mounting interfaces, in millimetres.

Source: the official 2D drawing and STEP model (Waveshare wiki "ST3215 Servo", files ST3215-2D.zip / ST3215-3D.zip,
drawing "SCS215" 2022-06-08). Values taken from that model:

    case 45.22 x 24.72 mm; output shaft 10.11 mm from the near end, centred in the width
    main body 29 mm thick, with 1.5 mm raised pads on both faces (32 mm over the pads); a further raised strip on
    each face (35 mm overall); output horn and passive (idler) horn both 19.2 mm diameter, 37.25 mm over both horns
    horns: four 2.5 mm holes on a 14 mm circle (on the length and width axes), centre screw M3x6 on the output side
    case mounting holes, 1.6 mm (for M2 self-tapping screws), 20.5 mm apart across the width:
        output face at 8.3 and 32.75 mm from the shaft, idler face at 8.3 and 29.0 mm from the shaft
    the two bus connectors (5264, 3-pin) plug in from the idler face, 10.5-17.5 mm from the shaft

Servo frame S (all helpers below work in it):
    origin on the shaft axis in the outer face of the output horn (the plane a driven part bolts against)
    +z along the shaft, out of the output horn; +x along the case towards its far end; y across the width.

Assumptions to check on a real servo before printing (see prototypes/puppy/README.md):
    * the horn holes take M3 screws (2.5 mm = M3 tap drill; the metal horns are tapped or need an M3 tap)
    * M2 self-tapping screws bite at least 4 mm into the 1.6 mm case holes
"""

from __future__ import annotations

import numpy as np
from manifold3d import Manifold

from . import cad

LENGTH = 45.22
WIDTH = 24.72
NEAR = 10.11  # shaft to near end
FAR = LENGTH - NEAR  # shaft to far end (35.11)
HALF_W = WIDTH / 2

# heights along the shaft (z), relative to the outer face of the output horn
HORN_FACE = 0.0
HORN_R = 9.6
HORN_T = 2.4  # aluminium disc
HUB_R = 3.0
OUT_MAIN = -4.7  # output-side main face
OUT_PAD = -3.2  # raised pad on the output side (x >= 6), carries the mounting holes
OUT_STRIP = -2.1  # label strip on the pad
IDL_MAIN = -33.7
IDL_PAD = -35.2
IDL_STRIP = -37.1
IDL_HORN_FACE = -37.3  # outer face of the passive horn
IDL_PIN = -37.8  # bearing pin in the centre of the passive horn
PAD_X0, PAD_X1, PAD_HALF_W = 6.0, 34.4, 11.65

HORN_HOLE_R = 7.0
HORN_HOLES = [(HORN_HOLE_R, 0.0), (0.0, HORN_HOLE_R), (-HORN_HOLE_R, 0.0), (0.0, -HORN_HOLE_R)]
OUT_HOLES = [(8.3, 10.25), (8.3, -10.25), (32.75, 10.25), (32.75, -10.25)]
IDL_HOLES = [(8.3, 10.25), (8.3, -10.25), (29.0, 10.25), (29.0, -10.25)]
PLUG_X0, PLUG_X1, PLUG_HALF_W = 10.5, 17.5, 11.0  # connector zone on the idler face
PLUG_DEPTH = 11.0  # plug body + tight wire bend below the idler main face

MASS = 0.055  # kg


def case_solid(clear: float = 0.0) -> Manifold:
    c = clear
    body = cad.box_between((-NEAR - c, -HALF_W - c, IDL_MAIN - c), (FAR + c, HALF_W + c, OUT_MAIN + c))
    pads = [
        cad.box_between((PAD_X0 - c, -PAD_HALF_W - c, OUT_MAIN), (PAD_X1 + c, PAD_HALF_W + c, OUT_PAD + c)),
        cad.box_between((PAD_X0 - c, -PAD_HALF_W - c, IDL_PAD - c), (PAD_X1 + c, PAD_HALF_W + c, IDL_MAIN)),
        cad.box_between((9.5 - c, -6.9 - c, OUT_PAD), (33.9 + c, 6.9 + c, OUT_STRIP + c)),
        cad.box_between((19.6 - c, -8.9 - c, IDL_STRIP - c), (29.4 + c, 8.9 + c, IDL_PAD)),
    ]
    return cad.union([body] + pads)


def horns(clear: float = 0.0) -> Manifold:
    """Both horns as discs (they turn, so the disc is also their swept volume)."""
    c = clear
    return cad.union([
        cad.cyl(HORN_R + c, -HORN_T - 0.6 - c, HORN_FACE + c),
        cad.cyl(HUB_R + c, OUT_MAIN, -HORN_T),
        cad.cyl(HORN_R + c, IDL_HORN_FACE - c, IDL_PAD + 0.2),
        cad.cyl(HUB_R + c, IDL_PIN - c, IDL_HORN_FACE),
    ])


def plug_zone(clear: float = 0.0) -> Manifold:
    """Connectors on the idler face plus room for a tight wire bend."""
    c = clear
    return cad.box_between((PLUG_X0 - c, -PLUG_HALF_W - c, IDL_MAIN - PLUG_DEPTH - c), (PLUG_X1 + c, PLUG_HALF_W + c, IDL_MAIN))


def solid() -> Manifold:
    """Case + horns, for visuals and mass."""
    return cad.union([case_solid(), horns()])


def keepout(clear: float = 0.5, plugs: bool = True, horn_screw: bool = True) -> Manifold:
    """Everything a printed part must stay out of (except the faces it is meant to touch)."""
    parts = [case_solid(clear), horns(clear)]
    if plugs:
        parts.append(plug_zone(clear))
    if horn_screw:
        parts.append(cad.cyl(3.2, 0.0, 3.0))  # head of the M3 centre screw stands proud of the horn
    return cad.union(parts)


# ---------------------------------------------------------------------- interfaces (cutters in the S frame)


def horn_bolt_cutter(z0: float, z1: float, side: str = "output", head_from: float | None = None) -> Manifold:
    """Holes for a part bolted to a horn: 4 x M3 clearance (3.3 mm) + a 7.5 mm hole over the centre screw.

    ``side`` = "output" or "idler" (which horn). ``head_from``: z above which the screw heads need a 6 mm counterbore.
    Plate spans z0..z1 in the S frame (for the idler horn z is negative).
    """
    lo, hi = min(z0, z1) - 1, max(z0, z1) + 1
    parts = [cad.cyl(3.75, lo, hi)]
    for x, y in HORN_HOLES:
        parts.append(cad.cyl(1.65, lo, hi).translate((x, y, 0)))
        if head_from is not None:
            if side == "output":
                parts.append(cad.cyl(3.1, head_from, hi + 5).translate((x, y, 0)))
            else:
                parts.append(cad.cyl(3.1, lo - 5, head_from).translate((x, y, 0)))
    return cad.union(parts)


def case_screw_cutter(face: str, z_plate_out: float, which=(0, 1, 2, 3), head_d: float = 4.2) -> Manifold:
    """Clearance holes (2.3 mm) for M2 screws into the case holes of ``face`` ("output" or "idler").

    ``z_plate_out``: outer surface of the bracket plate in the S frame (where the screw head sits); the hole runs from
    there to the case face. A head recess is not needed for pan-head screws on a flat outer surface.
    """
    holes = OUT_HOLES if face == "output" else IDL_HOLES
    zf = OUT_PAD if face == "output" else IDL_PAD
    lo, hi = min(zf, z_plate_out) - 0.5, max(zf, z_plate_out) + 0.5
    parts = []
    for i in which:
        x, y = holes[i]
        parts.append(cad.cyl(1.15, lo - 2, hi + 2).translate((x, y, 0)))
    return cad.union(parts)


def case_hole_points(face: str, which=(0, 1, 2, 3)) -> list[np.ndarray]:
    holes = OUT_HOLES if face == "output" else IDL_HOLES
    zf = OUT_PAD if face == "output" else IDL_PAD
    return [np.array([holes[i][0], holes[i][1], zf]) for i in which]


def mass_box_S() -> tuple[tuple[float, float, float], np.ndarray]:
    """Mass box (size, centre) of the servo in the S frame [mm]."""
    return (LENGTH, WIDTH, OUT_MAIN - IDL_MAIN), np.array([(FAR - NEAR) / 2, 0.0, (OUT_MAIN + IDL_MAIN) / 2])
