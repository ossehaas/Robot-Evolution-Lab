"""Checks of the stairs goal course (journal step 40), plain numpy/torch, no simulator:

    lab tools/dog/test_stairs_course.py

1. column layout: the mixed course keeps the dog course's columns (3 bumpy, 1 slope up, 1 slope down, 3 stairs down,
   2 stairs up, 2 blocks) and adds one column per stairs goal type
2. every tread x sides x rise: the mesh's top surface (from its triangles) equals the environment's step formula
   (terrain.straight_stairs_steps x rise) on the staircase, beside it and on the walls
3. start and goal spots: a dog-sized box (40 x 24 cm) at every start lies on flat ground at the expected height, the
   up goal is on the landing clear of the top riser and the end wall
4. the full generators (DOG_STAIRS, DOG_ROUGH_STAIRS) build, and each stairs goal row is as high as its rise says
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.sim.locomotion_env_cfg import EvoLocomotionEnvCfg  # noqa: E402
from core.sim.terrain import (DOG_ROUGH, DOG_ROUGH_STAIRS, DOG_STAIRS, STAIRS_RISES, STAIRS_TREADS,  # noqa: E402
                              MeshStraightStairsCfg, straight_stairs, straight_stairs_steps)

FAILS = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAILS.append(what)


def columns(gen) -> list[str]:
    names = list(gen.sub_terrains)
    cum = np.cumsum([gen.sub_terrains[k].proportion for k in names])
    cum = cum / cum[-1]
    return [names[int(np.nonzero(c / gen.num_cols + 0.001 < cum)[0][0])] for c in range(gen.num_cols)]


def top_surface(mesh, xy: np.ndarray) -> np.ndarray:
    """Highest point of the mesh straight above/below each xy (from its triangles)."""
    tri = np.asarray(mesh.triangles)  # (T, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    p = xy[:, None, :]
    v0, v1, v2 = (c - a)[None, :, :2], (b - a)[None, :, :2], p - a[None, :, :2]
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d02, d12 = (v0 * v2).sum(-1), (v1 * v2).sum(-1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-12  # vertical faces have no area seen from above
    den = np.where(ok, den, 1.0)
    u, v = (d11 * d02 - d01 * d12) / den, (d00 * d12 - d01 * d02) / den
    inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    z = a[None, :, 2] + u * (c - a)[None, :, 2] + v * (b - a)[None, :, 2]
    return np.where(inside, z, -np.inf).max(axis=1)


def main() -> int:
    cols = columns(DOG_ROUGH_STAIRS)
    base = columns(DOG_ROUGH)
    check(cols[: len(base)] == base, f"mixed course keeps the dog course's columns {base}")
    goal = [c for c in cols if c.startswith("goal_stairs")]
    check(len(goal) == len(set(goal)) == 2 * len(STAIRS_TREADS), f"one column per stairs goal type ({len(goal)})")
    check(columns(DOG_STAIRS) == list(DOG_STAIRS.sub_terrains), "stairs-only course: one column per type")
    check(DOG_STAIRS.num_rows == len(STAIRS_RISES), "one row per rise")

    env = EvoLocomotionEnvCfg()
    gx = np.arange(0.01, 3.0, 0.02)
    grid = np.stack(np.meshgrid(gx, gx, indexing="ij"), -1).reshape(-1, 2)
    worst = 0.0
    for tread in STAIRS_TREADS:
        for walls in (False, True):
            for row in (0, 4, 9):
                cfg = MeshStraightStairsCfg(tread=tread, walls=walls).replace(size=(3.0, 3.0))
                rise = STAIRS_RISES[row]
                meshes, origin = straight_stairs((row + 0.5) / len(STAIRS_RISES), cfg)
                import trimesh

                mesh = trimesh.util.concatenate(meshes)
                z = top_surface(mesh, grid)
                steps = straight_stairs_steps(torch.tensor(grid), torch.tensor(tread, dtype=torch.float64), cfg).numpy()
                want = steps * rise
                # away from the edges (a 2 cm grid point exactly on an edge belongs to either side)
                x, y = grid[:, 0], grid[:, 1]
                rel = (x - cfg.first_riser) / tread
                edge = (np.abs(rel - np.round(rel)) * tread < 0.005) | (np.abs(np.abs(y - 1.5) - cfg.width / 2) < 0.005)
                wall_band = (np.abs(np.abs(y - 1.5) - cfg.width / 2 - cfg.wall_thickness / 2) <= cfg.wall_thickness / 2) & \
                            (x >= cfg.first_riser - 0.3) & (x < cfg.landing_end)
                end_wall = (x >= cfg.landing_end) & (np.abs(y - 1.5) <= cfg.width / 2 + cfg.wall_thickness)
                plain = ~edge & ~end_wall & ~(wall_band if walls else np.zeros_like(edge))
                err = float(np.abs(z[plain] - want[plain]).max())
                worst = max(worst, err)
                wall_top = cfg.num_steps * rise + cfg.wall_height
                if walls:
                    check(bool(np.allclose(z[wall_band & ~edge], wall_top)), f"t{tread} r{rise} walls at {wall_top:.2f} m")
                check(bool(np.allclose(z[end_wall], wall_top)), f"t{tread} r{rise} walls={walls}: end wall") if row == 9 else None
                check(err < 1e-6 and np.allclose(origin, (1.5, 1.5, 0.0)),
                      f"t{tread} walls={walls} rise {rise}: mesh = formula (max error {err:.1e} m), origin at the centre")

                # start and goal spots: a dog-sized box around the body centre, facing the goal
                top_riser = cfg.first_riser + (cfg.num_steps - 1) * tread
                landing = cfg.num_steps * rise
                body = np.stack(np.meshgrid(np.linspace(-0.2, 0.2, 9), np.linspace(-0.12, 0.12, 7), indexing="ij"), -1).reshape(-1, 2)
                for name, cx, h in (("up start (closest)", cfg.first_riser - env.stairs_goal_approach[0], 0.0),
                                    ("up start (farthest)", cfg.first_riser - env.stairs_goal_approach[1], 0.0),
                                    ("down start", top_riser + env.stairs_goal_past + 0.1, landing),
                                    ("up goal", top_riser + env.stairs_goal_past, landing),
                                    ("down goal", cfg.first_riser - env.stairs_goal_past, 0.0)):
                    for dy in (-0.05, 0.05):
                        pts = body + np.array([cx, 1.5 + dy])
                        zz = top_surface(mesh, pts)
                        ok = bool(np.allclose(zz, h)) and pts[:, 0].min() > 0.02
                        if not ok:
                            check(False, f"t{tread} r{rise} {name} at x={cx:.2f}: ground {zz.min():.3f}..{zz.max():.3f}, want {h:.3f}")
    check(worst < 1e-6, f"all start and goal spots flat; mesh = formula everywhere (worst {worst:.1e} m)")

    # full generators
    from isaaclab.terrains import TerrainGenerator

    for name, gen in (("DOG_STAIRS", DOG_STAIRS), ("DOG_ROUGH_STAIRS", DOG_ROUGH_STAIRS)):
        tg = TerrainGenerator(gen.replace(seed=0), device="cpu")
        o = tg.terrain_origins
        check(o.shape == (gen.num_rows, gen.num_cols, 3), f"{name}: origins {o.shape}")
        if name == "DOG_STAIRS":
            v = np.asarray(tg.terrain_mesh.vertices)
            for row in range(gen.num_rows):
                for col in (0, gen.num_cols - 2):  # open sides (walls would be higher)
                    cx, cy = o[row, col, :2]
                    near = (np.abs(v[:, 0] - cx) < 1.5) & (np.abs(v[:, 1] - cy) < 0.31)
                    top = v[near, 2].max()
                    want = MeshStraightStairsCfg().num_steps * STAIRS_RISES[row]
                    if abs(top - want) > 1e-6:
                        check(False, f"{name} row {row} col {col}: landing at {top:.3f} m, want {want:.3f}")
            check(bool(np.allclose(o[..., 2], 0.0)), f"{name}: every tile origin on the floor (z = 0)")
    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
