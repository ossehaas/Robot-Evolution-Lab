"""Render a RobotSpec to a PNG (three views) with matplotlib, no simulator needed."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .builder import RobotSpec, is_passive_joint  # noqa: E402

LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])


def _collect(spec: RobotSpec):
    tris, cols = [], []
    for link in spec.links:
        for vis in link.visuals:
            v = np.asarray(vis.mesh.vertices) + link.origin + np.array([0.0, 0.0, spec.spawn_height])
            f = np.asarray(vis.mesh.faces)
            t = v[f]
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = 0.45 + 0.55 * np.clip(n @ LIGHT, 0.0, 1.0)
            base = np.array(vis.color) ** (1 / 2.2)  # linear -> sRGB
            tris.append(t)
            cols.append(np.clip(base[None, :] * shade[:, None], 0, 1))
    return np.concatenate(tris), np.concatenate(cols)


def render_preview(spec: RobotSpec, path: str | Path, title: str | None = None) -> Path:
    path = Path(path)
    tris, cols = _collect(spec)
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center, span = (lo + hi) / 2, (hi - lo).max() * 0.55

    fig = plt.figure(figsize=(14, 7.5), dpi=110)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.9])
    views = [("perspective", 24, -60, grid[:, :2]), ("front", 0, 0, grid[0, 2]), ("top", 90, -90, grid[1, 2])]
    for name, elev, azim, slot in views:
        ax = fig.add_subplot(slot, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="none", linewidths=0))
        # ground grid (1 cm)
        g = np.arange(-0.3, 0.3001, 0.01)
        for x in g:
            if abs(x - center[0]) < span:
                ax.plot([x, x], [center[1] - span, center[1] + span], [0, 0], color="0.87", lw=0.4, zorder=0)
            if abs(x - center[1]) < span:
                ax.plot([center[0] - span, center[0] + span], [x, x], [0, 0], color="0.87", lw=0.4, zorder=0)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        top = max(hi[2] * 1.15, 0.5 * span)
        ax.set_zlim(0, top)
        ax.set_box_aspect((2 * span, 2 * span, top), zoom=1.25 if name == "perspective" else 1.0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("persp" if name == "perspective" else "ortho")
        ax.set_axis_off()
        ax.set_title(name, fontsize=10, color="0.4")
    g = spec.genome
    fig.suptitle(
        title or f"{spec.name}  |  {sum(not is_passive_joint(j.name) for j in spec.joints)} servos  |  {1000 * spec.total_mass:.0f} g  |  topology {g.topology_key()}",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
