"""Checks of the terrain map and the odometry error (plain torch, no simulator):

    IsaacLab\\.venv\\Scripts\\python.exe tools/dog/test_terrain_map.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.sim.terrain_map import OdometryError, TerrainMap, rotate_z  # noqa: E402


def ground(xy: torch.Tensor) -> torch.Tensor:
    """A 5 cm step up at x = 0.3 m."""
    return torch.where(xy[..., 0] >= 0.3, 0.05, 0.0)


def scan_points(cx: float, n: int = 4000) -> torch.Tensor:
    """Ground points in a 0.6 x 0.4 m patch ahead of x = cx (what a camera would see), shape (1, n, 3)."""
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(1, n, 2, generator=g) * torch.tensor([0.6, 0.4]) + torch.tensor([cx + 0.15, -0.2])
    return torch.cat((xy, ground(xy).unsqueeze(-1)), dim=-1)


def main() -> int:
    m = TerrainMap(1, cells=64, resolution=0.025, device="cpu")
    pts = scan_points(0.0)
    m.write(pts, torch.ones(1, pts.shape[1], dtype=torch.bool), torch.zeros(1, 2), now=0.0)

    # 1. heights where the camera looked, unknown elsewhere
    q = torch.tensor([[[0.2, 0.0], [0.5, 0.1], [0.0, 0.0], [-0.3, 0.0]]])
    h, known = m.read(q, now=0.02, memory_s=10.0)
    assert known[0].tolist() == [True, True, False, False], known
    assert abs(h[0, 0]) < 1e-6 and abs(h[0, 1] - 0.05) < 1e-6, h

    # 2. memory: still known 1 s later, forgotten with a short memory; "since" = only the latest frame
    assert m.read(q, now=1.0, memory_s=10.0)[1][0, :2].all()
    assert not m.read(q, now=1.0, memory_s=0.1)[1][0, :2].any()
    assert not m.read(q, now=1.0, memory_s=0.0, since=0.5)[1][0, :2].any()

    # 3. ring buffer: 1.6 m further on the same slots hold other cells -> unknown, until written there
    far = q + torch.tensor([1.6, 0.0])
    assert not m.read(far, now=0.02, memory_s=10.0)[1].any()
    pts2 = scan_points(1.6)
    m.write(pts2, torch.ones(1, pts2.shape[1], dtype=torch.bool), torch.tensor([[1.6, 0.0]]), now=0.04)
    assert m.read(far, now=0.05, memory_s=10.0)[1][0, :2].all()
    assert not m.read(q, now=0.05, memory_s=10.0)[1][0, :2].any(), "old cells must not read through"

    # 4. window: points too far from the centre are not stored
    m2 = TerrainMap(1, cells=64, resolution=0.025, device="cpu")
    p = torch.tensor([[[0.9, 0.0, 0.1], [0.3, 0.0, 0.1]]])
    m2.write(p, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2), now=0.0)
    assert m2.read(p[..., :2], now=0.0, memory_s=1.0)[1][0].tolist() == [False, True]

    # 5. a step edge in one cell reads as its top
    m3 = TerrainMap(1, cells=64, resolution=0.025, device="cpu")
    p = torch.tensor([[[0.301, 0.0, 0.0], [0.302, 0.0, 0.05]]])
    m3.write(p, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2), now=0.0)
    assert abs(m3.read(p[:, :1, :2], now=0.0, memory_s=1.0)[0][0, 0] - 0.05) < 1e-6

    # 6. odometry without drift: map frame = world frame
    odo = OdometryError(2, "cpu", 0.0, 0.0, 0.0, 0.0)
    odo.reset(torch.arange(2))
    pos = torch.zeros(2, 3)
    for _ in range(50):
        pos[:, 0] += 0.01
        odo.step(pos)
    assert torch.allclose(odo.to_map(pos.unsqueeze(1))[:, 0], pos, atol=1e-6)

    # 7. scale error: the estimate walks (1 + s) x as far (first call after reset counts no motion)
    odo = OdometryError(1, "cpu", 0.0, 0.0, 0.0, 0.05)
    odo.reset(torch.arange(1))
    s = float(odo.scale[0])
    pos = torch.tensor([[10.0, 5.0, 0.2]])
    odo.step(pos)
    for _ in range(100):
        pos[:, 0] += 0.01
        odo.step(pos)
    est = odo.to_map(pos.unsqueeze(1))[0, 0]
    assert abs((est[0] - 10.0) - 1.0 * (1 + s)) < 1e-4, (est, s)

    # 8. a yaw error turns the estimate about the robot: its estimated position does not jump, but a point 1 m ahead
    #    moves sideways by ~ the yaw error
    odo = OdometryError(1, "cpu", 0.0, 0.5, 0.0, 0.0)
    odo.reset(torch.arange(1))
    pos = torch.tensor([[20.0, -7.0, 0.2]])
    odo.step(pos)
    before = odo.to_map(pos.unsqueeze(1))[0, 0].clone()
    pos2 = pos + torch.tensor([[0.0001, 0.0, 0.0]])
    odo.step(pos2)
    after = odo.to_map(pos2.unsqueeze(1))[0, 0]
    assert torch.allclose(after, before + torch.tensor([0.0001, 0.0, 0.0]), atol=2e-4), (before, after)
    for _ in range(200):  # walk 2 m: yaw error std 0.5 * sqrt(2) ~ 0.7 rad
        pos2[:, 0] += 0.01
        odo.step(pos2)
    yaw = float(odo.yaw[0])
    ahead = odo.to_map((pos2 + torch.tensor([[1.0, 0.0, 0.0]])).unsqueeze(1))[0, 0]
    here = odo.to_map(pos2.unsqueeze(1))[0, 0]
    assert abs(math.atan2(float(ahead[1] - here[1]), float(ahead[0] - here[0])) - yaw) < 1e-4

    # 9. rotate_z
    r = rotate_z(torch.tensor([[[1.0, 0.0]]]), torch.tensor([math.pi / 2]))
    assert torch.allclose(r, torch.tensor([[[0.0, 1.0]]]), atol=1e-6)

    # 10. drift statistics: std after 1 m close to the setting
    odo = OdometryError(4000, "cpu", 0.03, 0.035, 0.01, 0.0)
    odo.reset(torch.arange(4000))
    pos = torch.zeros(4000, 3)
    odo.step(pos)
    for _ in range(100):
        pos[:, 0] += 0.01
        odo.step(pos)
    est = odo.to_map(pos.unsqueeze(1))[:, 0]
    err_y, err_z, yaw_std = (est[:, 1]).std(), (est[:, 2]).std(), odo.yaw.std()
    assert 0.025 < err_y < 0.045 and 0.008 < err_z < 0.012 and 0.03 < yaw_std < 0.04, (err_y, err_z, yaw_std)
    print(f"terrain map: all checks passed (drift after 1 m: sideways {100 * err_y:.1f} cm, height "
          f"{100 * err_z:.1f} cm, heading {math.degrees(yaw_std):.1f} deg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
