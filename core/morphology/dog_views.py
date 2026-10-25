"""Pictures of the dog's mechanical design: an exploded assembly view and a layout of all printed parts.

    python -m core.morphology.dog_views runs/exp2_dog/1_design/puppy/genome.json --out runs/exp2_dog/1_design

Writes <name>_exploded.png and <name>_parts.png (matplotlib, painter's algorithm; no simulator needed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from . import cad  # noqa: E402
from . import sts3215 as SV  # noqa: E402

PALETTE = {"body": "#F2E3C9", "accent": "#B5703A", "paw": "#5A3B2A", "collar": "#D83A3A", "tag": "#F2C230",
           "frame": "#8FB0D6", "nose": "#1E1A1A", "eye": "#141414", "shine": "#FFFFFF", "servo": "#2B2B30"}
LIGHT = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _tris(solid, offset, color, tol=0.4):
    mesh = solid.simplify(tol).to_mesh()
    v = np.asarray(mesh.vert_properties, float)[:, :3] + offset
    return v[np.asarray(mesh.tri_verts)], _rgb(color)


def _draw(ax, items, azim, elev):
    a, e = np.radians(azim), np.radians(elev)
    f = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    V = np.stack([r, u, -f])
    polys, cols, depth = [], [], []
    for t, rgb in items:
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        p = t @ V.T
        keep = n @ V[2] > 0
        shade = 0.42 + 0.58 * np.clip(n @ LIGHT, 0, 1)
        polys.append(p[keep][:, :, :2])
        cols.append(np.clip(rgb[None, :] * shade[keep, None], 0, 1))
        depth.append(p[keep][:, :, 2].mean(axis=1))
    P, C, D = np.concatenate(polys), np.concatenate(cols), np.concatenate(depth)
    o = np.argsort(D)
    ax.add_collection(PolyCollection(P[o], facecolors=C[o], edgecolors="none"))
    pts = P.reshape(-1, 2)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c, s = (lo + hi) / 2, (hi - lo).max() / 2 * 1.04
    ax.set_xlim(c[0] - s, c[0] + s)
    ax.set_ylim(c[1] - s, c[1] + s)
    ax.set_aspect("equal")
    ax.axis("off")


def explode_offset(name: str, link: str) -> np.ndarray:
    """How far a part moves out in the exploded view [mm]."""
    side = -1.0 if link.startswith("R") or name.endswith("_R") else 1.0
    if link.startswith(("L", "R")) and "_s" in link:
        level = int(link[-1])
        front = 1.0 if link[1] == "0" else -1.0
        off = np.array([front * 40.0, side * (55.0 + 70.0 * level), -25.0 * level])
        if name.startswith("thigh_cover_out"):
            off[1] += side * 40.0
        elif name.startswith("thigh_cover_in"):
            off[1] -= side * 38.0
        elif name.startswith(("crank", "pushrod", "knee_spacer")):
            off[1] += side * 20.0
        elif name.startswith("paw"):
            off[2] -= 35.0
        elif name.startswith("sole"):
            off[2] -= 75.0
        elif name.startswith("ankle_spring"):
            off[2] -= 45.0
            off[1] -= side * 70.0
        elif name.startswith("yoke"):
            off[1] -= side * 40.0
        return off
    table = {"cap": (0, 0, 90), "bowl": (0, 0, -55), "belly_hatch": (0, 0, -120), "deck": (0, 0, 15),
             "neck_mount": (10, 0, 50), "neck_yoke": (60, 0, 75), "neck_L": (60, 55, 75), "neck_R": (60, -55, 75),
             "collar": (60, 0, 110), "collar_tag": (75, 0, 110), "head_back": (75, 0, 130), "head_mount": (110, 0, 130),
             "eye_frame": (150, 0, 130), "head": (200, 0, 130), "nose": (250, 0, 130), "ear_L": (200, 60, 150),
             "ear_R": (200, -60, 150), "tail": (-60, 0, 60), "tail_tip": (-80, 0, 85)}
    return np.array(table.get(name, (0, 0, 0)), float)


def render_exploded(mech, path, title=""):
    items = []
    link_off = {}
    for p in mech.parts:
        off = explode_offset(p.name, p.link)
        link_off.setdefault(p.link, []).append(off)
        items.append(_tris(p.solid, off, PALETTE[p.color]))
    for sv in mech.servos:
        offs = link_off.get(sv.link, [np.zeros(3)])
        # servos sit with the structural part of their link
        off = explode_offset({"chassis": "deck", "head": "head_mount"}.get(sv.link, "x"), sv.link) if sv.link in ("chassis", "head") \
            else np.median(np.array(offs), axis=0)
        items.append(_tris(cad.apply(SV.solid(), sv.S), off, PALETTE["servo"]))
    fig = plt.figure(figsize=(16, 10), dpi=110)
    ax = fig.add_axes([0, 0, 1, 0.95])
    _draw(ax, items, -55, 22)
    fig.suptitle(title or "exploded view", fontsize=13)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def render_parts(mech, path, title="", bed=210.0):
    """All printed parts laid flat in their print orientation on a grid, one cell per unique part."""
    seen, cells = set(), []
    for p in mech.parts:
        key = p.name.replace("_front", "").replace("_rear", "").replace("_L", "").replace("_R", "")
        count = sum(1 for q in mech.parts if q.name.replace("_front", "").replace("_rear", "").replace("_L", "").replace("_R", "") == key)
        if key in seen:
            continue
        seen.add(key)
        up = np.asarray(p.print_up, float)
        up /= np.linalg.norm(up)
        ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        F = cad.frame((0, 0, 0), ref, up)
        flat = cad.apply(p.solid, np.linalg.inv(F))
        lo, hi = cad.bbox(flat)
        flat = flat.translate(tuple(-(lo + hi) / 2 * np.array([1, 1, 0]) - np.array([0, 0, lo[2]])))
        cells.append((key, count, p.material, p.color, flat, hi - lo))
    n = len(cells)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), dpi=100)
    for ax, (key, count, mat, color, flat, size) in zip(axes.ravel(), cells):
        _draw(ax, [_tris(flat, np.zeros(3), PALETTE[color])], -60, 35)
        ax.set_title(f"{key} x{count}\n{mat.upper()}  {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title or "printed parts (print orientation, bed = 210 x 210 mm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def render_ankle(mech, path, leg: str = "front_L", title: str = ""):
    """Spring ankle and soft paw of one leg: section at rest and at the stop, exploded unit, sole from below."""
    from . import dog_paw as DP

    part = {p.name: p for p in mech.parts}
    names = [f"shin_{leg}", f"ankle_spring_{leg}", f"paw_{leg}", f"sole_{leg}"]
    if any(n not in part for n in names):
        return
    prefix = ("L" if leg.endswith("L") else "R") + ("0" if leg.startswith("front") else "1")
    ja = next(j for j in mech.joints if j.name == f"{prefix}_ja")
    d = mech.ankle
    P = np.asarray(ja.origin, float)
    C = P - np.array([0.0, 0.0, d.zp])  # paw centre
    side = 1.0 if leg.endswith("L") else -1.0
    shin, spring, paw, sole = (part[n].solid for n in names)
    zcut = C[2] + d.z_block_top + DP.SPRING_BH + 12.0
    shin = shin ^ cad.box_between(C + np.array([-40.0, -40.0, -40.0]), np.array([C[0] + 40.0, C[1] + 40.0, zcut]))
    upper, lower = spring.split_by_plane((0.0, 0.0, 1.0), P[2])
    half = cad.box_between(C + np.array([-60.0, -60.0 if side > 0 else 0.0, -60.0]),
                           C + np.array([60.0, 0.0 if side > 0 else 60.0, 80.0]))
    az = 90.0 if side > 0 else -90.0

    def bolts(T=np.eye(4), which=(0, 1)):
        out = []
        top, bot = DP.spring_bolts(d)
        for k, pts in enumerate((top, bot)):
            if k not in which:
                continue
            y0, y1 = (DP.SHIN_HALF, -DP.SHIN_HALF) if k == 0 else (DP.PAW_BOLT_HEAD, -DP.PAW_BOLT_NUT - 4.0)
            for x, z in pts:
                a = C + np.array([x, side * y0, z])
                b = C + np.array([x, side * y1, z])
                out.append(cad.apply(cad.cyl_between(a, b, 1.5), T))
        return out

    fig = plt.figure(figsize=(16, 9), dpi=110)
    specs = [(0.0, 0.5, 0.25, 0.45), (0.25, 0.5, 0.25, 0.45), (0.5, 0.0, 0.5, 0.95), (0.0, 0.0, 0.25, 0.45),
             (0.25, 0.0, 0.25, 0.45)]
    axs = [fig.add_axes(r) for r in specs]
    for ax, q, label in ((axs[0], 0.0, "section, rest"), (axs[1], d.limit, f"section at the stop (+{d.limit:.2f} rad)")):
        T = cad.rotation((0.0, 1.0, 0.0), q, P)
        items = [_tris(shin ^ half, np.zeros(3), PALETTE["body"], 0.05), _tris(upper ^ half, np.zeros(3), PALETTE["accent"], 0.05)]
        items += [_tris(cad.apply(s_ ^ half, T), np.zeros(3), PALETTE[c], 0.05)
                  for s_, c in ((lower, "accent"), (paw, "paw"), (sole, "collar"))]
        items += [_tris(b ^ half, np.zeros(3), PALETTE["servo"], 0.05) for b in bolts(np.eye(4), (0,))]
        items += [_tris(cad.apply(b ^ half, T), np.zeros(3), PALETTE["servo"], 0.05) for b in bolts(np.eye(4), (1,))]
        _draw(ax, items, az, 0.0)
        ax.set_title(label, fontsize=10)
    ex = {"shin": np.array([0, 0, 28.0]), "spring": np.array([0, side * 0.0, 0.0]), "paw": np.array([0, 0, -30.0]),
          "sole": np.array([0, 0, -62.0]), "bolts_top": np.array([0, side * 26.0, 28.0]), "bolts_bot": np.array([0, side * 32.0, -30.0])}
    items = [_tris(shin, ex["shin"], PALETTE["body"], 0.1), _tris(spring, ex["spring"], PALETTE["accent"], 0.05),
             _tris(paw, ex["paw"], PALETTE["paw"], 0.1), _tris(sole, ex["sole"], PALETTE["collar"], 0.05)]
    items += [_tris(b, ex["bolts_top"], PALETTE["servo"], 0.1) for b in bolts(which=(0,))]
    items += [_tris(b, ex["bolts_bot"], PALETTE["servo"], 0.1) for b in bolts(which=(1,))]
    _draw(axs[2], items, -60.0 if side > 0 else 60.0, 18.0)
    axs[2].set_title("ankle unit, exploded: shin socket, TPU ankle spring, TPU paw, TPU sole disc, 2 + 2 M3x30", fontsize=10)
    _draw(axs[3], [_tris(paw, np.zeros(3), PALETTE["paw"], 0.05)], -90.0, -70.0)
    axs[3].set_title("paw from below (opening, snap groove)", fontsize=10)
    _draw(axs[4], [_tris(sole, np.zeros(3), PALETTE["collar"], 0.02)], -90.0, -89.0)
    axs[4].set_title(f"sole disc from below: {DP.SOLE_ARMS} arms, pad (tread)", fontsize=10)
    fig.suptitle(title or (f"spring ankle {d.k_est / 1000:.2f} N*m/rad (blade {d.t:.2f} x {d.L:.1f} x {DP.SPRING_W:.0f} mm), "
                           f"limit +-{d.limit:.2f} rad; sole {d.k_sole_est:.1f} N/mm, {DP.SOLE_TRAVEL} mm to the stop"), fontsize=12)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    from .dog_mech import build_mechanics
    from .genome import Genome

    ap = argparse.ArgumentParser(prog="python -m core.morphology.dog_views")
    ap.add_argument("genome", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    g = Genome.load(args.genome)
    mech = build_mechanics(g)
    out = args.out or args.genome.parent
    out.mkdir(parents=True, exist_ok=True)
    render_exploded(mech, out / f"{g.name}_exploded.png", f"{g.name}: exploded assembly ({len(mech.parts)} printed parts, {len(mech.servos)} servos)")
    render_parts(mech, out / f"{g.name}_parts.png", f"{g.name}: printed parts in print orientation (all fit a 210 x 210 x 210 mm bed)")
    render_ankle(mech, out / f"{g.name}_ankle.png")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
