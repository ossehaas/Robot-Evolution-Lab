"""Base skills A1-A4 (.documentation/base-skills.md): the design checks of a robot body, CPU only, no simulator.

    A1  parts do not collide, joints stay in range   core.morphology.dog_check (rest pose + every joint swept): no
                                                      part hits the body, its own leg or a servo; the front and rear
                                                      leg of one side may meet at large hip swings towards each other
                                                      (reported: the simulation has self-collision for it)
    A2  stands on a tilt without tipping              standing pose tips beyond 20 deg nose down, nose up and sideways
    A3  servos strong enough                          standing on 4 paws < 30 % of the usable torque at every hip and
                                                      knee; on 3 paws (each leg lifted in turn, the body shifted to
                                                      the best place with the centre of gravity >= 1.5 cm inside the
                                                      other three) < 60 %
    A4  printable                                     every printed part, laid flat, fits 256 x 256 x 256 mm

Side-view model (tools/dog/stairs_plan.py Dog2D): hip and knee torques from the vertical paw forces; the abduction
joints carry little while standing and are not checked. Usable torque = the joint's effort limit (80 % of stall).
Appends one JSON line {genome, checks: {A1..A4: {pass, ...}}} to --out (if given) and prints the table.

    lab tools/base/design_check.py runs/exp2_dog/17_recover_bc/bc_s2/genome.json --out runs/base/design_check.jsonl
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.dog.stairs_plan import LEGS, Dog2D, contact_forces, rot, support_margin  # noqa: E402
from tools.dog.tilt_limits import com_xz  # noqa: E402

TILT_MIN = 20.0  # [deg] A2
STAND4, STAND3 = 0.30, 0.60  # A3: share of the usable torque
MARGIN3 = 0.015  # [m] centre of gravity inside the three paws
BED = 0.256  # [m] A4


def part_leg(name: str) -> str | None:
    """Leg (L0 front left ... R1 rear right) a printed part or leg servo belongs to; None = body."""
    m = re.search(r"(front|rear)_([LR])", name)
    if m:
        return f"{m.group(2)}{0 if m.group(1) == 'front' else 1}"
    m = re.search(r"servo ([LR][01])_j[12]", name)
    return m.group(1) if m else None


def torque_shares(dog: Dog2D, usable: dict, paws: dict, stance: list[str], bx: np.ndarray, bz: float) -> tuple:
    """Worst hip/knee torque share over the stance legs, CoG margin, all forces >= 0 (batched over body x)."""
    r = dog.paw_r
    q, ok = {}, np.ones(bx.shape, bool)
    zero = np.zeros_like(bx)
    for leg in LEGS:
        h, k, valid = dog.ik(leg, zero, bx, np.full_like(bx, bz), np.full_like(bx, paws[leg]), np.full_like(bx, r))
        q[leg] = (h, k)
        ok &= valid
    maps = dog.link_maps(zero, zero, q)
    cx, _ = com_xz(dog, maps, bx, bz)
    pts = [(paws[leg], dog.paw_y[leg]) for leg in stance]
    margin = support_margin(pts, cx) if len(pts) >= 3 else np.full(bx.shape, -1.0)
    f = contact_forces(pts, cx, dog.mass)
    ok &= np.all(f >= -1e-6, -1)
    worst = np.zeros(bx.shape)
    for j, leg in enumerate(stance):
        a1, tx1, _ = maps[f"{leg}_s1"]
        hx, _ = rot(maps["chassis"][0], *dog.hip[leg])
        kx, _ = rot(a1, *dog.knee[leg])
        for name, jx in ((f"{leg}_j1", hx + bx), (f"{leg}_j2", kx + tx1 + bx)):
            worst = np.maximum(worst, np.abs(f[..., j] * (paws[leg] - jx)) / usable[name])
    return worst, margin, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("genome", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip_a1", action="store_true", help="skip the (slow) interference sweep")
    args = ap.parse_args()

    from core.morphology import Genome, build_robot

    genome = Genome.load(args.genome)
    spec = build_robot(genome)
    dog = Dog2D(str(args.genome))
    usable = {j.name: float(j.effort) for j in spec.joints}
    checks = {}

    # A1
    if args.skip_a1:
        checks["A1"] = {"pass": None, "note": "skipped"}
    else:
        from core.morphology.dog_check import check
        from core.morphology.dog_mech import build_mechanics

        issues = check(build_mechanics(genome), 4, 2.0, ("L0", "L1"), log=lambda *a, **k: None)
        legs_meet = [i for i in issues if part_leg(i[1]) and part_leg(i[2]) and part_leg(i[1]) != part_leg(i[2])]
        flaws = [i for i in issues if i not in legs_meet]
        checks["A1"] = {"pass": not flaws, "flaws": [f"{w}: {a} x {b} {v:.0f} mm3 at {math.degrees(q):.0f} deg"
                                                     for w, a, b, v, q in sorted(flaws, key=lambda x: -x[3])[:5]],
                        "legs_meet": len(legs_meet),
                        "legs_meet_from_deg": round(min((abs(math.degrees(i[4])) for i in legs_meet), default=0.0), 1)}

    # A2 (as tools/dog/tilt_limits.py, standing pose)
    r = dog.paw_r
    paws = {leg: float(dog.paw[leg][0]) for leg in LEGS}
    rest_z = r - dog.paw["L0"][1]
    q0 = {leg: (np.zeros(1), np.zeros(1)) for leg in LEGS}
    cx0, cz0 = com_xz(dog, dog.link_maps(np.zeros(1), np.zeros(1), q0), 0.0, rest_z)
    cx0, h0 = float(cx0[0]), float(cz0[0])
    tilt = {"nose_down": math.degrees(math.atan((paws["L0"] - cx0) / h0)),
            "nose_up": math.degrees(math.atan((cx0 - paws["L1"]) / h0)),
            "sideways": math.degrees(math.atan(abs(dog.paw_y["L0"]) / h0))}
    checks["A2"] = {"pass": min(tilt.values()) > TILT_MIN, **{k: round(v, 1) for k, v in tilt.items()}}

    # A3
    w4, _, ok4 = torque_shares(dog, usable, paws, list(LEGS), np.zeros(1), rest_z)
    three = {}
    bx = np.linspace(-0.08, 0.08, 161)
    for lifted in LEGS:
        w, m, ok = torque_shares(dog, usable, paws, [leg for leg in LEGS if leg != lifted], bx, rest_z)
        good = ok & (m >= MARGIN3)
        three[lifted] = round(float(w[good].min()), 3) if good.any() else None
    worst3 = max((v for v in three.values() if v is not None), default=None)
    checks["A3"] = {"pass": bool(ok4[0]) and float(w4[0]) < STAND4 and None not in three.values() and worst3 < STAND3,
                    "stand4_share": round(float(w4[0]), 3), "stand3_share": three,
                    "usable_nm": round(min(usable[f"{leg}_j1"] for leg in LEGS), 3)}

    # A4
    from core.morphology.export_print import _lay_flat

    sizes = {p.name: [round(1000 * float(s), 1) for s in _lay_flat(p).extents] for p in spec.parts}
    too_big = {n: s for n, s in sizes.items() if max(s) > 1000 * BED}
    largest = max(sizes.items(), key=lambda kv: max(kv[1]))
    checks["A4"] = {"pass": not too_big, "parts": len(sizes), "too_big": too_big,
                    "largest": f"{largest[0]} {' x '.join(f'{s:.0f}' for s in largest[1])} mm"}

    line = {"genome": str(args.genome).replace("\\", "/"), "mass_kg": round(float(spec.total_mass), 3), "checks": checks}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    mark = {True: "pass", False: "FAIL", None: "-"}
    print(f"[design_check] {args.genome} ({spec.total_mass:.2f} kg)")
    for k, c in checks.items():
        print(f"  {k}  {mark[c['pass']]:4}  " + ", ".join(f"{n}={v}" for n, v in c.items() if n != "pass"))
    return 0 if all(c["pass"] is not False for c in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
