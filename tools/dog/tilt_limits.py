"""At what tilt can the dog still stand? Static limits on a tilted plane (journal step 42), no simulator.

The dog stands on a flat plane tilted by an angle; gravity then pulls its centre of gravity (CoG) towards the low side.
It tips when the CoG's projection along gravity leaves the support rectangle of the four paws:

    tan(tilt) = (distance from the CoG to the support edge on the low side) / (CoG height above the plane)

It slides when tan(tilt) exceeds the paw grip (static friction): 45 deg on a normal floor (TPU paw, mu 1.0), 27 deg on
a smooth tiled floor (mu 0.5). Reported for the standing (rest) pose and for the best posture the legs can take with
the paws where they stand (body shifted uphill, crouched, pitched, neck as a counterweight; the leg joints within
95 % of their range): nose down (standing on a slope facing downhill, as going down stairs), nose up (facing uphill),
sideways. Abduction is held at 0 (the side-view model), so sideways only crouching helps.

    lab tools/dog/tilt_limits.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.dog.stairs_plan import GENOME, LEGS, Dog2D, rot  # noqa: E402


def com_xz(dog: Dog2D, maps, bx, bz):
    x = z = 0.0
    for name, (m, c) in dog.masses.items():
        a, tx, tz = maps[name]
        cx, cz = rot(a, c[0], c[1])
        x, z = x + m * (cx + tx), z + m * (cz + tz)
    return x / dog.mass + bx, z / dog.mass + bz


def main() -> int:
    dog = Dog2D(GENOME)
    r = dog.paw_r
    paws = {leg: float(dog.paw[leg][0]) for leg in LEGS}  # paw x at rest (front +0.08, rear -0.08)
    front, rear = paws["L0"], paws["L1"]
    half_track = abs(dog.paw_y["L0"])
    rest_z = r - dog.paw["L0"][1]  # chassis height standing on flat ground

    # rest pose
    q0 = {leg: (np.zeros(1), np.zeros(1)) for leg in LEGS}
    m0 = dog.link_maps(np.zeros(1), np.zeros(1), q0)
    cx0, cz0 = com_xz(dog, m0, 0.0, rest_z)
    cx0, h0 = float(cx0[0]), float(cz0[0])

    # posture grid: body x shift, chassis height, pitch, neck; paws stay where they stand (paw centres at z = r)
    g = np.meshgrid(np.linspace(-0.10, 0.10, 41), np.linspace(0.08, 0.20, 25), np.radians(np.linspace(-25, 25, 21)),
                    np.linspace(-dog.neck_lim * 0.95, dog.neck_lim * 0.95, 7), indexing="ij")
    bx, bz, pitch, neck = (v.ravel() for v in g)
    ok = np.ones(bx.shape, bool)
    q = {}
    for leg in LEGS:
        h, k, valid = dog.ik(leg, pitch, bx, bz, np.full_like(bx, paws[leg]), np.full_like(bx, r))
        q[leg] = (h, k)
        ok &= valid
    maps = dog.link_maps(pitch, neck, q)
    cx, cz = com_xz(dog, maps, bx, bz)
    cx, cz = cx[ok], cz[ok]
    down = np.degrees(np.arctan((front - cx) / cz))  # facing downhill: the CoG moves towards the front paws
    up = np.degrees(np.arctan((cx - rear) / cz))
    side = np.degrees(np.arctan(half_track / cz))

    def describe(i):
        sel = np.nonzero(ok)[0][i]
        return (f"body {100 * bx[sel]:+.1f} cm, height {100 * bz[sel]:.1f} cm (standing {100 * rest_z:.1f}), "
                f"pitch {math.degrees(pitch[sel]):+.0f} deg, neck {neck[sel]:+.2f} rad")

    print(f"CoG standing: {100 * cx0:+.1f} cm ahead of the body centre, {100 * h0:.1f} cm above the ground; paws "
          f"{100 * (front - rear):.0f} cm apart front-rear, {200 * half_track:.0f} cm across\n")
    print(f"{'direction':<12} {'standing':>9} {'best posture':>13}   best posture")
    for name, rest, arr in (("nose down", math.degrees(math.atan((front - cx0) / h0)), down),
                            ("nose up", math.degrees(math.atan((cx0 - rear) / h0)), up),
                            ("sideways", math.degrees(math.atan(half_track / h0)), side)):
        i = int(np.argmax(arr))
        print(f"{name:<12} {rest:6.1f} deg {arr[i]:9.1f} deg   {describe(i)}")
    print("\nsliding (paw grip): " + ", ".join(f"mu {mu}: {math.degrees(math.atan(mu)):.0f} deg" for mu in (1.0, 0.7, 0.5)))
    print("stairs for comparison: rise/tread 4/15 = 15 deg, 5/15 = 18 deg, 4/9 = 24 deg, 5/9 = 29 deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
