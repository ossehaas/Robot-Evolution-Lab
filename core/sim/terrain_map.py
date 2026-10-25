"""Terrain map with memory: the real robot's version of the perfect height scan.

A chest sensor (depth camera or 8 x 8 time-of-flight) sees the ground ahead. Its points are stored in a height grid
around the robot, placed with the robot's own motion estimate (IMU + leg odometry), which drifts. Cells the sensor no
longer sees (under the body, behind the front legs) keep their last height: the robot remembers the step it saw a
moment ago. Paws standing on the ground also write the height under them (leg kinematics).

    TerrainMap      a ring-buffer grid per robot: heights, which absolute cell each slot holds, when it was seen
    OdometryError   the drifting estimate: map frame = Rz(yaw error) * world + offset, random walk per metre walked
                    plus a per-episode scale error (the robot over- or underestimates its steps)

Plain torch (no simulator imports), so it can be tested on its own: tools/dog/test_terrain_map.py.
"""

from __future__ import annotations

import math

import torch

_KEY = 1 << 21  # absolute cell id = ix * _KEY + iy (|iy| < 2^20 cells, ~26 km at 2.5 cm)


def rotate_z(xy: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate (E, ..., 2) points by (E,) angles about the origin."""
    c, s = torch.cos(angle), torch.sin(angle)
    shape = (-1,) + (1,) * (xy.dim() - 2)
    c, s = c.view(shape), s.view(shape)
    return torch.stack((c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]), dim=-1)


class TerrainMap:
    """Per-robot height grid of cells x cells at the given resolution, kept as a ring buffer: a point at absolute cell
    (ix, iy) lands in slot (ix mod cells, iy mod cells), so the grid never has to be shifted as the robot walks. Each
    slot remembers which absolute cell it holds; a slot holding another cell (or nothing) reads as unknown."""

    def __init__(self, num_envs: int, cells: int, resolution: float, device):
        self.n, self.res, self.device = cells, resolution, device
        self.height = torch.zeros(num_envs, cells * cells, device=device)
        self.key = torch.full((num_envs, cells * cells), -1, dtype=torch.long, device=device)
        self.seen = torch.full((num_envs, cells * cells), -math.inf, device=device)
        self._env_offset = (torch.arange(num_envs, device=device) * cells * cells).unsqueeze(-1)
        self.half_window = 0.5 * cells * resolution - resolution  # points farther than this are not stored

    def clear(self, env_ids: torch.Tensor) -> None:
        self.key[env_ids] = -1
        self.seen[env_ids] = -math.inf

    def _slots(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ij = torch.floor(xy / self.res).long()
        slot = torch.remainder(ij[..., 0], self.n) * self.n + torch.remainder(ij[..., 1], self.n)
        return slot, ij[..., 0] * _KEY + ij[..., 1]

    def write(self, points: torch.Tensor, valid: torch.Tensor, centre_xy: torch.Tensor, now: float) -> None:
        """Store (E, R, 3) map-frame points where valid (E, R). Points beyond the window around centre_xy (E, 2) are
        dropped (they would share a slot with a cell inside it). Several points in one cell: the highest counts (a
        step edge reads as the top of the step)."""
        near = (points[..., :2] - centre_xy.unsqueeze(1)).abs().amax(dim=-1) < self.half_window
        sel = (valid & near & torch.isfinite(points).all(dim=-1)).flatten()
        if not bool(sel.any()):
            return
        slot, key = self._slots(points[..., :2])
        idx = (slot + self._env_offset).flatten()[sel]
        top = torch.full((self.height.numel(),), -math.inf, device=self.device)
        top.scatter_reduce_(0, idx, points[..., 2].flatten()[sel], reduce="amax")
        hit = torch.isfinite(top)
        self.height.view(-1)[hit] = top[hit]
        self.key.view(-1).scatter_(0, idx, key.flatten()[sel])  # all points in a slot share its cell (window)
        self.seen.view(-1)[hit] = now

    def read(self, xy: torch.Tensor, now: float, memory_s: float, since: float | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Heights (E, Q) and known flags (E, Q) at map-frame points xy (E, Q, 2). A cell is known if it holds this
        point's cell and was seen within memory_s (or, with ``since``, at or after that time)."""
        slot, key = self._slots(xy)
        idx = slot + self._env_offset
        h = self.height.view(-1)[idx]
        seen = self.seen.view(-1)[idx]
        fresh = seen >= since if since is not None else (now - seen) <= memory_s
        known = (self.key.view(-1)[idx] == key) & fresh
        return torch.where(known, h, torch.zeros_like(h)), known


class OdometryError:
    """The robot's drifting pose estimate. Map-frame point = Rz(yaw_err) * world point + offset. Per metre walked
    the error random-walks (std after 1 m: drift_xy, drift_yaw, drift_z); per episode the estimated distance is off
    by a constant factor (scale_error, uniform +-). Yaw errors turn the estimate about the robot itself, not about
    the world origin."""

    def __init__(self, num_envs: int, device, drift_xy: float, drift_yaw: float, drift_z: float, scale_error: float):
        self.device = device
        self.q_xy, self.q_yaw, self.q_z = drift_xy**2, drift_yaw**2, drift_z**2  # variance per metre
        self.scale_error = scale_error
        self.yaw = torch.zeros(num_envs, device=device)
        self.offset = torch.zeros(num_envs, 3, device=device)
        self.scale = torch.zeros(num_envs, device=device)
        self.last = torch.zeros(num_envs, 3, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.yaw[env_ids] = 0.0
        self.offset[env_ids] = 0.0
        self.scale[env_ids] = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * self.scale_error
        self.fresh[env_ids] = True

    def to_map(self, points: torch.Tensor) -> torch.Tensor:
        """(E, ..., 3) world points -> map frame."""
        xy = rotate_z(points[..., :2], self.yaw)
        shape = (-1,) + (1,) * (points.dim() - 2)
        off = self.offset.view(*shape, 3)
        return torch.cat((xy + off[..., :2], (points[..., 2] + off[..., 2]).unsqueeze(-1)), dim=-1)

    def step(self, true_pos: torch.Tensor, skip: torch.Tensor | None = None) -> None:
        """Advance the error by the robot's true motion since the last call. No motion counts on the first call after
        a reset, nor where ``skip`` (E,) is set (e.g. the reset step itself, whose pose may not be updated yet)."""
        still = self.fresh if skip is None else self.fresh | skip
        delta = torch.where(still.unsqueeze(-1), torch.zeros_like(true_pos), true_pos - self.last)
        self.last.copy_(true_pos)
        self.fresh.zero_()
        walked = torch.linalg.norm(delta[:, :2], dim=-1)
        self.offset += self.scale.unsqueeze(-1) * delta  # the estimate moves (1 + scale) x the true step
        self.offset[:, :2] += torch.randn_like(self.offset[:, :2]) * torch.sqrt(self.q_xy * walked).unsqueeze(-1)
        self.offset[:, 2] += torch.randn_like(walked) * torch.sqrt(self.q_z * walked)
        turn = torch.randn_like(walked) * torch.sqrt(self.q_yaw * walked)
        centre = self.to_map(true_pos.unsqueeze(1))[:, 0, :2]  # where the robot thinks it is: keep that point fixed
        self.yaw += turn
        self.offset[:, :2] = rotate_z((self.offset[:, :2] - centre).unsqueeze(1), turn)[:, 0] + centre
