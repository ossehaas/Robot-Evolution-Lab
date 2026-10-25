"""Minimal BVH reader with forward kinematics (numpy): joint world positions and rotations per frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BvhJoint:
    name: str
    parent: int  # index into the joint list, -1 for the root
    offset: np.ndarray  # rest offset from the parent [file units]
    channels: list[str] = field(default_factory=list)
    end_site: np.ndarray | None = None  # offset of the end site, if the joint ends a chain


@dataclass
class BvhClip:
    joints: list[BvhJoint]
    frame_time: float
    motion: np.ndarray  # (frames, channels)

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    def index(self, name: str) -> int:
        return self.names.index(name)


def _rot(axis: str, deg: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) about a principal axis."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def load(path: str | Path) -> BvhClip:
    tokens = Path(path).read_text().split()
    joints: list[BvhJoint] = []
    stack: list[int] = []
    i = 0
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            joints.append(BvhJoint(tokens[i + 1], stack[-1] if stack else -1, np.zeros(3)))
            i += 2
        elif t == "End":  # End Site { OFFSET x y z }
            j = i + 3  # skip "End Site {"
            assert tokens[j] == "OFFSET"
            joints[stack[-1]].end_site = np.array([float(v) for v in tokens[j + 1:j + 4]])
            i = j + 5  # past the closing brace
        elif t == "{":
            stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            joints[stack[-1]].offset = np.array([float(v) for v in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            joints[stack[-1]].channels = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    n_frames = int(tokens[tokens.index("Frames:") + 1])
    frame_time = float(tokens[tokens.index("Time:") + 1])
    start = tokens.index("Time:") + 2
    values = np.array([float(v) for v in tokens[start:]])
    motion = values.reshape(n_frames, -1)
    return BvhClip(joints, frame_time, motion)


def forward_kinematics(clip: BvhClip) -> tuple[np.ndarray, np.ndarray]:
    """World positions (frames, joints, 3) and rotations (frames, joints, 3, 3) in file units and axes."""
    n = clip.motion.shape[0]
    J = len(clip.joints)
    pos = np.zeros((n, J, 3))
    rot = np.zeros((n, J, 3, 3))
    col = 0
    for k, joint in enumerate(clip.joints):
        local_rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        local_pos = np.broadcast_to(joint.offset, (n, 3)).copy()
        for ch in joint.channels:
            v = clip.motion[:, col]
            col += 1
            if ch.endswith("position"):
                local_pos[:, "XYZ".index(ch[0])] = v
            else:
                local_rot = local_rot @ _rot(ch[0], v)  # channel order = intrinsic rotation order
        if joint.parent < 0:
            rot[:, k] = local_rot
            pos[:, k] = local_pos
        else:
            p = joint.parent
            rot[:, k] = rot[:, p] @ local_rot
            pos[:, k] = pos[:, p] + np.einsum("nij,nj->ni", rot[:, p], local_pos)
    return pos, rot


def end_positions(clip: BvhClip, pos: np.ndarray, rot: np.ndarray, joint: str) -> np.ndarray:
    """World position of a joint's end site (e.g. the tip of a paw)."""
    k = clip.index(joint)
    end = clip.joints[k].end_site
    return pos[:, k] + np.einsum("nij,j->ni", rot[:, k], end)
