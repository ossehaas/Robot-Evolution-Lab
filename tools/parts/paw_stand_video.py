"""Side-view video of the paw test stand from its traces (tools/parts/paw_stand.py), designs side by side, no simulator:
the paw's colliders as drawn outlines, the sole orange, the base ring red-brown, the PLA shell grey; the housing lights up
when it carries more than 0.5 N. A stance sweep and a climb past a step edge, four times slower than real time.

    lab tools/parts/paw_stand_video.py runs/exp2_dog/28_part_tests "old ball" "now" "P 3" "P 4.5"
"""

import json
import math
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.morphology import Genome, build_robot  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
DT, FPS, SLOW = 1 / 200, 30, 4
COLORS = {"tpu": "#e8862e", "tpu_rigid": "#a5522a", "pla": "#9aa3ad"}
TOUCH = "#d62828"
RISER_X, RISER_H = 0.15, 0.04


def outline(col) -> np.ndarray:
    """Side view (x, z) [mm] of a collider, in the paw's frame."""
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    c = col.transform[:3, 3] * 1000
    r = col.size[0] * 1000
    if col.kind == "sphere":
        return np.c_[c[0] + r * np.cos(a), c[2] + r * np.sin(a)]
    half = col.size[1] * 500  # capsules along x
    return np.c_[c[0] + np.where(np.cos(a) >= 0, half, -half) + r * np.cos(a), c[2] + r * np.sin(a)]


def load(folder: Path, name: str) -> dict:
    slug = name.replace(" ", "_").replace("+", "plus").replace(".", "_")
    paw = build_robot(Genome.load(folder / "genomes" / f"{slug}.json")).link_map["L0_s3"]
    d = np.load(folder / f"trace_{slug}.npz")
    sole = next(c for c in paw.collisions if c.kind == "sphere")
    return {"name": name, "cols": paw.collisions, "trace": d, "cases": json.loads(str(d["cases"])),
            "height": -1000 * float(sole.transform[2, 3] - sole.size[0])}


def scene(ax, de: dict, i: int, k: int, riser: bool, x0: float, z_span) -> None:
    tr = de["trace"]
    a = float(tr["tilt"][k, i] + tr["ankle"][k, i])  # + = toe down
    px, pz = 1000 * float(tr["x"][k, i]), 1000 * float(tr["z"][k, i]) + de["height"]
    ca, sa = math.cos(a), math.sin(a)
    ax.clear()
    ax.axhspan(-30, 0, color="#d9d4c7")
    if riser:
        ax.add_patch(plt.Rectangle((RISER_X * 1000, 0), 200, RISER_H * 1000, color="#c9c2b0"))
    ax.plot([px, px - 70 * math.sin(float(tr["tilt"][k, i]))], [pz, pz + 70 * math.cos(float(tr["tilt"][k, i]))], color="#555", lw=5)
    touched = float(tr["f_house"][k, i]) > 0.5
    for col in sorted(de["cols"], key=lambda c: c.material == "pla", reverse=True):
        p = outline(col)
        xy = np.c_[px + p[:, 0] * ca + p[:, 1] * sa, pz - p[:, 0] * sa + p[:, 1] * ca]
        ax.fill(xy[:, 0], xy[:, 1], color=COLORS[col.material], ec=TOUCH if touched and col.material != "tpu" else "#333", lw=1.5 if touched else 0.5)
    ax.plot(px, pz, "o", color="k", ms=4)
    ax.set_xlim(x0 - 60, x0 + 60)
    ax.set_ylim(*z_span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{de['name']}\nsole {float(tr['f_sole'][k, i]):.0f} N, housing {float(tr['f_house'][k, i]):.0f} N, lift {max(float(tr['fz'][k, i]), 0.0):.1f} N"
                 if riser else f"{de['name']}\nsole {float(tr['f_sole'][k, i]):.0f} N, housing {float(tr['f_house'][k, i]):.0f} N", fontsize=9)


def render(designs: list[dict], pick, title: str, out: Path, riser: bool, z_span, t0: float, t1: float) -> None:
    fig, axes = plt.subplots(1, len(designs), figsize=(3.2 * len(designs), 4.2), dpi=110)
    axes = np.atleast_1d(axes)
    idx = [next(j for j, c in enumerate(de["cases"]) if pick(c)) for de in designs]
    writer = FFMpegWriter(fps=FPS, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi=110):
        for t in np.arange(t0, t1, 1 / (FPS * SLOW)):
            k = int(t / DT)
            for ax, de, i in zip(axes, designs, idx):
                tr = de["trace"]
                x0 = RISER_X * 1000 - 15 if riser else 1000 * float(np.mean(tr["x"][:, i]))
                scene(ax, de, i, k, riser, x0, z_span)
            fig.suptitle(f"{title}   (t = {t:.2f} s, 4x slower)", fontsize=10)
            writer.grab_frame()
    plt.close(fig)
    print(f"[paw_stand_video] {out}", flush=True)


def main() -> int:
    folder = Path(sys.argv[1])
    designs = [load(folder, n) for n in sys.argv[2:]]
    out = folder / "videos"
    out.mkdir(exist_ok=True)
    render(designs, lambda c: c["test"] == "roll" and c["load"] == 12.0, "Stance under 12 N: the shin sweeps +11 to -18 deg (red outline: the housing carries load)",
           out / "stand_stance.mp4", False, (-8, 100), 0.25, 0.8)
    render(designs, lambda c: c["test"] == "edge" and c["below"] == 0.03 and c["tilt"] == 0.0 and c["push"] == 6.0,
           "Pushed 6 N against a 4 cm riser, lifted at 0.1 m/s", out / "stand_edge.mp4", True, (-8, 150), 0.25, 1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
