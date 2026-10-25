"""Hip angle at which the front and rear leg on one side first touch (the only interference of the dog's mechanism).

    lab -m tools.dog.leg_clearance prototypes/puppy/genome.json [--step 1.0]

Three cases on the left side (the right side is its mirror image): the front hip swings back towards the rear leg with
the rear leg at rest, the rear hip swings forward with the front leg at rest, and both swing towards each other by the
same angle. Each moving leg keeps its knee and ankle at rest. Reports the first angle with an overlap > --tol mm^3 and
which parts touch. Build one dog per process (the CAD library can crash when several are built in one process).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from core.morphology import cad
from core.morphology.dog_check import _bbox_overlap, _solids, _subtree


def main():
    from core.morphology.dog_mech import build_mechanics
    from core.morphology.genome import Genome

    ap = argparse.ArgumentParser(prog="python -m tools.dog.leg_clearance")
    ap.add_argument("genome", type=Path)
    ap.add_argument("--step", type=float, default=1.0, help="[deg]")
    ap.add_argument("--tol", type=float, default=2.0)
    args = ap.parse_args()
    g = Genome.load(args.genome)
    mech = build_mechanics(g)
    solids = _solids(mech)
    joints = {j.name: j for j in mech.joints}
    legs = {}
    for prefix in ("L0", "L1"):
        links = _subtree(mech, f"{prefix}_s1")
        legs[prefix] = [(n, s) for n, lk, s in solids if lk in links]
    jf, jr = joints["L0_j1"], joints["L1_j1"]

    def contact(qf, qr):
        Tf, Tr = cad.rotation(jf.axis, qf, jf.origin), cad.rotation(jr.axis, qr, jr.origin)
        front = [(n, cad.apply(s, Tf)) for n, s in legs["L0"]]
        rear = [(n, cad.apply(s, Tr)) for n, s in legs["L1"]]
        hits = []
        for na, a in front:
            ba = cad.bbox(a)
            for nb, b in rear:
                if not _bbox_overlap(ba, cad.bbox(b)):
                    continue
                v = (a ^ b).volume()
                if v > args.tol:
                    hits.append((v, na, nb))
        return sorted(hits, reverse=True)

    print(f"{g.name}: knees {getattr(g.legs, 'knees', 'back_front')}, hip limit {math.degrees(jf.limit):.0f} deg")
    lim = math.degrees(jf.limit)
    for label, f in (("front hip swings back (rear at rest)", lambda a: (a, 0.0)),
                     ("rear hip swings forward (front at rest)", lambda a: (0.0, -a)),
                     ("both swing towards each other", lambda a: (a, -a))):
        first = None
        for a in np.arange(args.step, lim + 1e-6, args.step):
            qf, qr = f(math.radians(a))
            hits = contact(qf, qr)
            if hits:
                first = (a, hits)
                break
        if first is None:
            print(f"  {label}: no contact up to {lim:.0f} deg")
        else:
            a, hits = first
            print(f"  {label}: first contact at {a:.0f} deg: " + ", ".join(f"{na} x {nb} ({v:.0f} mm3)" for v, na, nb in hits[:3]))


if __name__ == "__main__":
    main()
