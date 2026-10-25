"""Interference check of the dog's printed parts and servos: rest pose and every joint through its range.

    python -m core.morphology.dog_check runs/exp2_dog/1_design/puppy/genome.json [--steps 5] [--legs L0,L1]

For each joint the parts and servos downstream of it are turned through the joint range (the others held at the
rest pose) and intersected with everything else. The knee crank and pushrod follow the real parallelogram (the
simulator lumps them into the thigh). Overlaps above ``--tol`` mm^3 are reported, with the angle where they happen.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import cad
from . import sts3215 as SV


def _bbox_overlap(a, b, pad=0.5):
    (alo, ahi), (blo, bhi) = a, b
    return bool(np.all(alo - pad <= bhi) and np.all(blo - pad <= ahi))


def _solids(mech):
    """(name, link, solid) for every printed part and servo case. A flexure that spans two links (the ankle spring) is
    checked as two rigid halves, split at its pivot; the two halves of one part are never checked against each other."""
    out = []
    for p in mech.parts:
        if getattr(p, "split_z", None) is not None and getattr(p, "link_below", None):
            above, below = p.solid.split_by_plane((0.0, 0.0, 1.0), p.split_z)
            out += [(p.name + " (upper)", p.link, above), (p.name + " (lower)", p.link_below, below)]
        else:
            out.append((p.name, p.link, p.solid))
    for sv in mech.servos:
        out.append((f"servo {sv.joint}", sv.link, cad.apply(SV.case_solid(), sv.S)))
    return out


def _children(mech):
    kids = defaultdict(list)
    for j in mech.joints:
        kids[j.parent].append(j.child)
    return kids


def _subtree(mech, link):
    kids = _children(mech)
    out, stack = set(), [link]
    while stack:
        n = stack.pop()
        out.add(n)
        stack.extend(kids[n])
    return out


def check(mech, steps: int = 5, tol: float = 2.0, legs: tuple[str, ...] | None = None, log=print):
    solids = _solids(mech)
    boxes = {n: cad.bbox(s) for n, _, s in solids}
    issues = []

    # ---------------- rest pose
    for i in range(len(solids)):
        for k in range(i + 1, len(solids)):
            na, la, a = solids[i]
            nb, lb, b = solids[k]
            if not _bbox_overlap(boxes[na], boxes[nb]) or _same_part(na, nb):
                continue
            v = (a ^ b).volume()
            if v > tol:
                issues.append(("rest", na, nb, v, 0.0))
    log(f"rest pose: {len([x for x in issues if x[0] == 'rest'])} overlaps")

    # ---------------- joints through their range
    link_of = {n: l for n, l, _ in solids}
    for j in mech.joints:
        if legs is not None and j.name[:2] in ("L0", "R0", "L1", "R1") and j.name[:2] not in legs:
            continue
        moving_links = _subtree(mech, j.child)
        moving = [(n, l, s) for n, l, s in solids if l in moving_links]
        fixed = [(n, l, s) for n, l, s in solids if l not in moving_links]
        is_knee = j.name.endswith("_j2")
        prefix = j.name.split("_")[0]
        if is_knee:  # crank and pushrod of this leg move too
            moving += [(n, l, s) for n, l, s in fixed if n.startswith((f"crank_", f"pushrod_")) and _leg_of(n, prefix)]
            fixed = [(n, l, s) for n, l, s in fixed if not (n.startswith(("crank_", "pushrod_")) and _leg_of(n, prefix))]
        worst = {}
        for q in np.linspace(-j.limit, j.limit, 2 * steps + 1):
            if abs(q) < 1e-9:
                continue
            T = cad.rotation(j.axis, q, j.origin)
            for n, l, s in moving:
                if is_knee and n.startswith(("crank_", "pushrod_")):
                    Tm = _parallelogram(mech, n, j, q)
                else:
                    Tm = T
                sm = cad.apply(s, Tm)
                bm = cad.bbox(sm)
                for nf, lf, sf in fixed:
                    if not _bbox_overlap(bm, boxes[nf]) or _same_part(n, nf):
                        continue
                    v = (sm ^ sf).volume()
                    if v > tol:
                        key = (n, nf)
                        if key not in worst or v > worst[key][0]:
                            worst[key] = (v, q)
        for (n, nf), (v, q) in worst.items():
            issues.append((j.name, n, nf, v, q))
        log(f"{j.name:8s} +-{math.degrees(j.limit):4.0f} deg: {len(worst)} overlaps")
    return issues


def _same_part(a: str, b: str) -> bool:
    return a.split(" (")[0] == b.split(" (")[0] and a != b and "(" in a


def _leg_of(part_name: str, prefix: str) -> bool:
    side, idx = prefix[0], prefix[1]
    return part_name.endswith(f"{'front' if idx == '0' else 'rear'}_{side}")


def _parallelogram(mech, name, joint, q):
    prefix = joint.name.split("_")[0]
    sv = next(s for s in mech.servos if s.joint == f"{prefix}_j2")
    pk = sv.S[:3, 3]
    if name.startswith("crank_"):
        return cad.rotation(joint.axis, q, pk)
    # pushrod: its crank end is the rod eye nearest to the knee servo axis
    rod = next(p for p in mech.parts if p.name == name).solid
    mesh = cad.to_trimesh(rod)
    v = mesh.vertices
    dist = np.linalg.norm((v - pk)[:, [0, 2]], axis=1)
    near = v[dist < np.percentile(dist, 5)].mean(axis=0)
    # the eye centre is ~ at the crank length from pk in the x-z plane
    d = near - pk
    d[1] = 0.0
    pin = pk + d / max(np.linalg.norm(d), 1e-9) * 18.0
    R = cad.rotation(joint.axis, q)[:3, :3]
    return cad.translation(R @ (pin - pk) - (pin - pk))


def main():
    from .genome import Genome
    from .dog_mech import build_mechanics

    ap = argparse.ArgumentParser(prog="python -m core.morphology.dog_check")
    ap.add_argument("genome", type=Path)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tol", type=float, default=2.0, help="ignore overlaps below this volume [mm^3]")
    ap.add_argument("--legs", default="L0,L1", help="legs to sweep (the others are mirror images)")
    args = ap.parse_args()
    mech = build_mechanics(Genome.load(args.genome))
    issues = check(mech, args.steps, args.tol, tuple(args.legs.split(",")))
    print(f"\n{len(issues)} interferences (> {args.tol} mm^3):")
    for where, a, b, v, q in sorted(issues, key=lambda x: -x[3]):
        print(f"  {where:8s} {a:28s} x {b:28s} {v:9.1f} mm3  at {math.degrees(q):6.1f} deg")


if __name__ == "__main__":
    main()


def printability(mech, bed=(210.0, 210.0, 210.0), max_angle=50.0):
    """Per part, in its print orientation: size on the bed and the area that overhangs more than ``max_angle``
    from vertical (excluding the first layer). Large flat overhangs are bridges; check those in the slicer."""
    rows = []
    cos_lim = math.cos(math.radians(90.0 - max_angle))
    for p in mech.parts:
        up = np.asarray(p.print_up, float)
        up /= np.linalg.norm(up)
        ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        F = cad.frame((0, 0, 0), ref, up)
        mesh = cad.to_trimesh(cad.apply(p.solid, np.linalg.inv(F)))
        z0 = mesh.bounds[0][2]
        ext = mesh.extents
        n = mesh.face_normals
        c = mesh.triangles_center
        a = mesh.area_faces
        down = (n[:, 2] < -cos_lim) & (c[:, 2] > z0 + 0.3)
        flat = down & (n[:, 2] < -0.985)
        rows.append((p.name, ext, float(a[down].sum()), float(a[flat].sum()),
                     bool(np.all(np.sort(ext[:2])[::-1] <= np.sort(bed[:2])[::-1]) and ext[2] <= bed[2])))
    return rows
