"""Paw part test, level 0 (journal step 48): geometry only, CPU, seconds. From the printed CAD shape of the paw: at what
tilt of the paw, with the sole spring compressed how far, does the housing touch the floor instead of the sole? And how
far does the paw tilt in a stride?

    touch tilt   the real paw_outer and sole meshes (core.morphology.dog_paw) rotated about the paw centre, per design
                 variant: sole sticking out P mm below the base, optionally the base bevelled outside the sole's lip
    stride tilt  side-view model (tools/dog/stairs_plan.py Dog2D), body level at standing height: the shin's rotation
                 with the paw dx ahead of its rest place. The paw follows the shin unless the ankle spring gives
    base load    simple static estimate: the ankle spring needs k * (shin rotation - touch tilt) of moment to flatten
                 the paw further, the edge of the base gives that at its lever; share of the paw's load on that edge

    lab tools/parts/paw_geometry.py runs/exp2_dog/26_ring_base/genome_ring.json --out runs/exp2_dog/28_part_tests
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.morphology import Genome, cad, dog_paw as DP  # noqa: E402
from core.morphology.dog_mech import PAW_SOLE_R  # noqa: E402
from tools.dog.stairs_plan import Dog2D  # noqa: E402

COMPRESSIONS = (0.0, 0.6, 1.2, 1.8)  # [mm] sole spring: unloaded, standing, typical and high step peak (k = 10 N/mm)
DIRECTIONS = {"toe down": ((0, 1, 0), 1), "heel down": ((0, 1, 0), -1), "side": ((1, 0, 0), 1)}
BEVEL_R0, BEVEL_DEG = DP.SOLE_GROOVE_R + 0.2, 25.0  # the bevel starts just outside the sole's groove
VARIANTS = {"now (P 1.5)": (1.5, False), "P 3": (3.0, False), "P 4.5": (4.5, False), "bevel": (1.5, True),
            "P 3 + bevel": (3.0, True), "P 4.5 + bevel": (4.5, True)}
DX = (-60, -40, -20, 20, 40, 60)  # [mm] paw ahead of its rest place


def verts(m) -> np.ndarray:
    return np.asarray(m.to_mesh().vert_properties, float)[:, :3]


def bevelled(outer, z_rim: float):
    h = 40.0
    return outer ^ cad.cyl(BEVEL_R0, z_rim, z_rim + h, r1=BEVEL_R0 + h / math.tan(math.radians(BEVEL_DEG)), segs=192)


def rot(axis, a: float) -> np.ndarray:
    return cad.rotation(axis, a)[:3, :3]


def touch_tilt(housing: np.ndarray, sole: np.ndarray, axis, sign: int, comp: float) -> float:
    """First tilt [deg] where the housing is as low as the sole (compressed by comp along the paw's axis)."""
    s = sole + np.array([0.0, 0.0, comp])
    for a in np.arange(0.0, 45.01, 0.1):
        R = rot(axis, sign * math.radians(a))
        if (housing @ R.T)[:, 2].min() <= (s @ R.T)[:, 2].min():
            return round(float(a), 1)
    return 45.0


def stride_tilt(dog: Dog2D, leg: str, dx: float) -> float:
    """Shin rotation [deg, + = toe up] with the paw dx [m] ahead of its rest place, body unmoved."""
    h, k0, p0 = dog.hip[leg], dog.knee[leg], dog.paw[leg]
    p = p0 + np.array([dx, 0.0])
    l1, l2 = dog.l1[leg], dog.l2[leg]
    d = p - h
    D = np.linalg.norm(d)
    a = (D * D + l1 * l1 - l2 * l2) / (2 * D)
    n = np.array([-d[1], d[0]]) / D
    c = h + a * d / D
    b = math.sqrt(max(l1 * l1 - a * a, 0.0))
    k = min((c + b * n, c - b * n), key=lambda q: np.linalg.norm(q - k0))
    ang = lambda v: math.atan2(v[1], v[0])  # noqa: E731
    return math.degrees((ang(p - k) - ang(p0 - k0) + math.pi) % (2 * math.pi) - math.pi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("genome", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    g = Genome.load(args.genome)
    r = 1000 * g.legs.paw_radius
    d = DP.design(g, r)
    z_rim = -r + DP.SOLE_TRAVEL
    outer = DP.paw_outer(d)
    zb = -r  # only the pad reaches below the base: the sole's rim stays inside the paw whatever the pad's length
    sole = verts(DP.sole_solid(d) ^ cad.box_between((-50.0, -50.0, zb - 1.0), (50.0, 50.0, z_rim)))
    rows = []
    for name, (P, bevel) in VARIANTS.items():
        h = verts(bevelled(outer, z_rim) if bevel else outer)
        s = sole - np.array([0.0, 0.0, P - DP.SOLE_TRAVEL])
        tilt = {w: {c: touch_tilt(h, s, ax, sg, c) for c in COMPRESSIONS if c < P} for w, (ax, sg) in DIRECTIONS.items()}
        rows.append({"variant": name, "protrusion": P, "bevel": bevel, "touch_tilt": tilt, "arm_strain_at_18N": round(DP.sole_strain(d.t_sole, 18.0 / d.k_sole_est), 3),
                     "arm_strain_at_stop": round(DP.sole_strain(d.t_sole, P), 3), "force_at_stop": round(d.k_sole_est * P, 1)})

    # the simulator's colliders of this paw (dog_mech.py): sole sphere, two base rails, shell capsule
    rs, (bx, hx, hy, hb) = PAW_SOLE_R, DP.PAW_BASE
    rb = (hb + 1.5) / 2
    ball = verts(cad.sphere((0.0, 0.0, 0.0), 1.0, 96))
    (ex, _, _), (ax, _, _) = DP.PAW_ELLIPSOID
    rc = r - DP.SOLE_TRAVEL
    zc = z_rim + 2 * rb + rc
    ends = [((bx + sx * (hx - rb), sy * (hy - rb), z_rim + rb), rb) for sx in (-1, 1) for sy in (-1, 1)]
    ends += [((ex + sx * (ax * r - rc), 0.0, zc), rc) for sx in (-1, 1)]
    sim_h = np.vstack([ball * rad + np.array(c) for c, rad in ends])
    sim_s = ball * rs + np.array([0.0, 0.0, rs - r])
    rows.append({"variant": "simulator's colliders, now", "protrusion": 1.5, "bevel": False,
                 "touch_tilt": {w: {c: touch_tilt(sim_h, sim_s, ax_, sg, c) for c in COMPRESSIONS if c < 1.5} for w, (ax_, sg) in DIRECTIONS.items()},
                 "arm_strain_at_18N": rows[0]["arm_strain_at_18N"], "arm_strain_at_stop": rows[0]["arm_strain_at_stop"], "force_at_stop": rows[0]["force_at_stop"]})

    dog = Dog2D(str(args.genome))
    stride = {leg: {dx: round(stride_tilt(dog, leg, dx / 1000), 1) for dx in DX} for leg in ("L0", "L1")}
    k = float(g.ankle.stiffness)
    load = dog.mass * 9.81 / 2  # [N] a trot: two paws carry the dog
    lever = {"toe": (DP.PAW_BASE[0] + DP.PAW_BASE[1] - 0.8) / 1000, "heel": (DP.PAW_BASE[1] - 0.8 - DP.PAW_BASE[0]) / 1000}
    report = {"genome": str(args.genome).replace("\\", "/"), "mass_kg": round(dog.mass, 3), "sole_k": round(d.k_sole_est, 1), "ankle_k": k,
              "ankle_limit_deg": round(math.degrees(d.limit), 1), "paw_load_trot_N": round(load, 1), "variants": rows, "stride_tilt": stride, "lever_m": lever}

    lines = [f"Paw geometry, {args.genome}: {dog.mass:.2f} kg, sole {d.k_sole_est:.1f} N/mm, ankle {k} N*m/rad, limit "
             f"{math.degrees(d.limit):.0f} deg; a trotting paw carries {load:.1f} N", "",
             "Tilt [deg] where the housing touches the floor, at sole compression 0 / 0.6 / 1.2 / 1.8 mm (- = the base is already down):", "",
             "| variant | toe down | heel down | side | arm strain at 18 N | at the stop |", "|---|---|---|---|---|---|"]
    for row in rows:
        cells = [" / ".join(str(t[c]) if c in t else "-" for c in COMPRESSIONS) for t in row["touch_tilt"].values()]
        lines.append(f"| {row['variant']} | {cells[0]} | {cells[1]} | {cells[2]} | {row['arm_strain_at_18N']:.0%} | {row['arm_strain_at_stop']:.0%} ({row['force_at_stop']:.0f} N) |")
    lines += ["", "Shin rotation [deg, + = toe up] with the paw dx mm ahead of its rest place, body level:", "",
              "| leg | " + " | ".join(str(x) for x in DX) + " |", "|---|" + "---|" * len(DX)]
    lines += [f"| {'front' if leg == 'L0' else 'rear'} | " + " | ".join(str(stride[leg][x]) for x in DX) + " |" for leg in stride]
    lines += ["", f"Share of a trotting paw's load on the base edge (static estimate, sole at 1.2 mm), front leg:", "",
              "| variant | " + " | ".join(str(x) for x in DX) + " |", "|---|" + "---|" * len(DX)]
    for row in rows:
        cells = []
        for dx in DX:
            sh = stride["L0"][dx]
            w, lv = ("heel down", lever["heel"]) if sh > 0 else ("toe down", lever["toe"])
            t = row["touch_tilt"][w].get(1.2, 0.0)
            cells.append(f"{min(max(k * math.radians(abs(sh) - t) / (lv * load), 0.0), 1.0):.0%}")
        lines.append(f"| {row['variant']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "paw_geometry.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        (args.out / "paw_geometry.md").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
