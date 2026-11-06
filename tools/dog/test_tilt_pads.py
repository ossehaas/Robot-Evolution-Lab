"""Checks of the tilt tiles (base skills, journal step 43), plain numpy/torch, no simulator:

    lab tools/dog/test_tilt_pads.py

1. column layout: DOG_ROUGH_STAIRS_TILT keeps the dog course's and the stairs goal columns and adds TILT_PAD_COLS
   tilt pad columns at the end; DOG_ROUGH_STAIRS is unchanged
2. every pad row and the test plates (TILT_COURSE): the mesh's top surface around the tile origin is the plane the
   environment assumes (through the origin, normal from tilt angle and downhill direction: the same formula as
   EvoLocomotionEnv.tilt_normal); the downhill side is the low side; the pad stands on the floor (lowest point 0),
   is closed (watertight), and the floor around it is flat
3. the start pose of the environment's reset (rotation from straight up to the plate normal) turns the dog's up
   axis onto the normal, and its standing height measured like ground_height() is the spawn height
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.sim.terrain import (DOG_ROUGH, DOG_ROUGH_STAIRS, DOG_ROUGH_STAIRS_TILT, TILT_ANGLES, TILT_COURSE,  # noqa: E402
                              TILT_DIRECTIONS, TILT_PAD_COLS, TILT_TRAIN_ANGLES, MeshTiltPadCfg, tilt_pad, tilted_plate)
from tools.dog.test_stairs_course import columns  # noqa: E402

FAILS = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAILS.append(what)


def normal(angle_deg: float, direction_deg: float) -> np.ndarray:
    a, phi = math.radians(angle_deg), math.radians(direction_deg)
    return np.array([math.sin(a) * math.cos(phi), math.sin(a) * math.sin(phi), math.cos(a)])


def top_z(meshes, x, y) -> float:
    """Highest surface at (x, y) over all meshes (a ray from above)."""
    best = -math.inf
    for m in meshes:
        loc, _, _ = m.ray.intersects_location(np.array([[x, y, 10.0]]), np.array([[0.0, 0.0, -1.0]]))
        if len(loc):
            best = max(best, float(loc[:, 2].max()))
    return best


# 1 layout
mix, base = columns(DOG_ROUGH_STAIRS_TILT), columns(DOG_ROUGH_STAIRS)
check(DOG_ROUGH_STAIRS_TILT.num_cols == DOG_ROUGH_STAIRS.num_cols + TILT_PAD_COLS, f"{DOG_ROUGH_STAIRS_TILT.num_cols} columns")
check(mix[:len(base)] == base, "the dog course and stairs goal columns come first, in the same order")
check(mix[len(base):] == ["tilt_pad"] * TILT_PAD_COLS, f"then {TILT_PAD_COLS} tilt pad columns: {mix[len(base):]}")
check(len(TILT_TRAIN_ANGLES) == DOG_ROUGH.num_rows, "one pad tilt per row")

# 2 geometry
rng = np.random.default_rng(0)
cases = [("pad", MeshTiltPadCfg(size=(3.0, 3.0), direction=d), tilt_pad, TILT_TRAIN_ANGLES, 0.45)
         for d in (0.0, 45.0, 90.0, 180.0, 300.0)]
cases += [("plate", TILT_COURSE.sub_terrains[f"tilt_{int(d)}"].replace(size=(3.0, 3.0)), tilted_plate, TILT_ANGLES, 0.8)
          for d in TILT_DIRECTIONS]
for kind, cfg, fn, angles, reach in cases:
    worst, low_ok, closed, floor_ok, side_ok = 0.0, True, True, True, True
    for row, ang in enumerate(angles):
        meshes, origin = fn((row + 0.5) / len(angles), cfg)
        n = normal(ang, cfg.direction)
        for _ in range(12):
            dx, dy = rng.uniform(-reach, reach, 2)
            z = top_z(meshes, origin[0] + dx, origin[1] + dy)
            plane = origin[2] - (n[0] * dx + n[1] * dy) / n[2]
            worst = max(worst, abs(z - plane))
        phi = math.radians(cfg.direction)
        if ang > 0:
            down = top_z(meshes, origin[0] + 0.3 * math.cos(phi), origin[1] + 0.3 * math.sin(phi))
            side_ok &= down < origin[2] - 0.3 * math.tan(math.radians(ang)) + 1e-3
        if kind == "pad":
            low_ok &= abs(float(meshes[1].vertices[:, 2].min())) < 1e-6
            closed &= bool(meshes[1].is_watertight)
            floor_ok &= abs(top_z(meshes, 0.1, 0.1)) < 1e-6 and abs(top_z(meshes, 2.9, 2.9)) < 1e-6
    check(worst < 1e-4, f"{kind} {cfg.direction:.0f} deg: top surface = the environment's plane (worst {1000 * worst:.3f} mm)")
    check(side_ok, f"{kind} {cfg.direction:.0f} deg: the downhill direction is the low side")
    if kind == "pad":
        check(low_ok and closed and floor_ok, f"pad {cfg.direction:.0f} deg: on the floor, watertight, flat floor around it")
meshes, origin = tilt_pad(0.99, MeshTiltPadCfg(size=(3.0, 3.0), direction=45.0))
print(f"     steepest pad (30 deg, diagonal): top centre {100 * origin[2]:.0f} cm, top edge "
      f"{100 * meshes[1].vertices[:, 2].max():.0f} cm above the floor")

# 3 start pose, as in EvoLocomotionEnv._reset_idx (quaternions in the library order, x y z w)
from isaaclab.utils.math import quat_apply, quat_from_angle_axis  # noqa: E402

worst_up = worst_h = 0.0
for ang in TILT_TRAIN_ANGLES:
    for d in (0.0, 45.0, 90.0, 180.0, 300.0):
        n = torch.tensor(normal(ang, d), dtype=torch.float32).unsqueeze(0)
        up = torch.tensor([[0.0, 0.0, 1.0]])
        axis = torch.linalg.cross(up, n)
        sin = torch.linalg.norm(axis, dim=-1)
        axis = torch.where(sin.unsqueeze(-1) > 1e-6, axis / sin.clamp(min=1e-9).unsqueeze(-1), torch.tensor([[0.0, 1.0, 0.0]]))
        q = quat_from_angle_axis(torch.atan2(sin, n[:, 2]), axis)
        worst_up = max(worst_up, float(torch.linalg.norm(quat_apply(q, up) - n)))
        origin = torch.tensor([[1.5, 1.5, 0.2]])
        pos = origin + n * 0.21
        worst_h = max(worst_h, abs(float(torch.sum((pos - origin) * n, dim=-1)) - 0.21))
check(worst_up < 1e-5, f"the start rotation turns the dog's up axis onto the plate normal (worst {worst_up:.2e})")
check(worst_h < 1e-6, "its height above the plate (ground_height on tilt tiles) is the spawn height")

print(f"\n{len(FAILS)} failed" if FAILS else "\nall checks passed")
sys.exit(1 if FAILS else 0)
