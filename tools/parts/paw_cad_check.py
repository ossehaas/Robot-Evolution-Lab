"""Checks of the printed paw with a longer sole and a stop pin (CPU): the parts fit, the pin stops the pad at the right
travel; a side section as a picture.

    lab tools/parts/paw_cad_check.py runs/exp2_dog/29_long_sole/genome.json runs/exp2_dog/29_long_sole/paw_section.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.morphology import Genome, cad, dog_paw as DP  # noqa: E402


def overlap(a, b) -> float:
    """Shared volume [mm3]."""
    return (a ^ b).volume()


def section(ax, part, color: str, y: float = 0.0, dz: float = 0.0) -> None:
    """Cut at y (the part's triangles crossing that plane, drawn as x, z segments)."""
    mesh = part.to_mesh()
    v, t = np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)
    for tri in t:
        p = v[tri]
        d = p[:, 1] - y
        pts = [p[i] + (p[j] - p[i]) * d[i] / (d[i] - d[j]) for i, j in ((0, 1), (1, 2), (2, 0)) if d[i] * d[j] < 0]
        if len(pts) == 2:
            ax.plot([pts[0][0], pts[1][0]], [pts[0][2] + dz, pts[1][2] + dz], color=color, lw=0.8)


def main() -> int:
    g = Genome.load(sys.argv[1])
    r = 1000 * g.legs.paw_radius
    d = DP.design(g, r)
    paw, sole, spring, pin = DP.paw_housing(d), DP.sole_solid(d), DP.spring_solid(d), DP.stop_pin(d)
    moved = lambda part, mm: part.translate((0.0, 0.0, mm))  # noqa: E731
    z_block = d.z_block_bot - DP.SPRING_BH
    z_pin = cad.bbox(pin)[1][2]
    rows = {"protrusion_mm": d.protrusion, "stop_mm": d.sole_stop, "gap_pin_block_mm": round(float(z_block - z_pin), 3),
            "pin_length_mm": round(float(z_pin - cad.bbox(pin)[0][2]), 2),
            "arm_strain_at_stop": round(DP.sole_strain(d.t_sole, d.sole_stop), 3), "force_at_stop_N": round(d.k_sole_est * d.sole_stop, 1),
            "paw_g": round(paw.volume() * 1.24e-3 * 0.55, 1), "sole_g": round(sole.volume() * 1.21e-3, 1), "pin_g": round(pin.volume() * 1.24e-3, 2)}
    checks = {"pin in the housing": overlap(pin, paw), "pin at the stop in the housing": overlap(moved(pin, d.sole_stop), paw),
              "pin in the sole": overlap(pin, sole), "pin in the spring": overlap(pin, spring),
              "pin past the stop reaches the spring": overlap(moved(pin, d.sole_stop + 0.2), spring),
              "spring in the housing": overlap(spring, paw), "sole's pad and arms at the stop in the housing":
              overlap(moved(sole ^ cad.cyl(DP.SOLE_ARM_R[1], -100.0, 100.0), d.sole_stop), paw)}
    rows["overlaps_mm3"] = {k: round(float(v), 2) for k, v in checks.items()}
    print(json.dumps(rows, indent=1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)
    for ax, y, title in ((axes[0], 0.13, "side section through the centre, sole at rest"), (axes[1], 0.13, f"sole pushed up {d.sole_stop} mm: the pin on the spring's block")):
        dz = 0.0 if ax is axes[0] else d.sole_stop
        section(ax, paw, "#555555", y)
        section(ax, spring, "#d62828", y)
        section(ax, sole, "#e8862e", y, dz)
        section(ax, pin, "#1d70b8", y, dz)
        ax.axhline(-r + dz, color="#888888", lw=0.8, ls="--")
        ax.set_aspect("equal")
        ax.set_xlim(-30, 34)
        ax.set_ylim(-22, 30)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("z [mm], paw centre = 0")
    fig.suptitle(f"sole {d.protrusion} mm below the rim, stop after {d.sole_stop} mm (housing grey, spring red, sole orange, pin blue)", fontsize=11)
    fig.tight_layout()
    fig.savefig(sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
