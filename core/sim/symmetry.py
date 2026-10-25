"""Left-right mirror of observations and actions, for symmetry-augmented PPO (RSL-RL ``symmetry_cfg``).

Every training sample is also used mirrored, so the policy learns both sides equally (a body that is symmetric
should not turn better one way than the other). Mirroring the robot in its sagittal (x-z) plane:

    joints          left <-> right swapped; values keep their sign (abduction axes are already mirrored, pitch axes
                    lie in the mirror plane), except roll/yaw joints on the centre line (head tilt, tail wag): negated
    gravity, accel  y component negated          gyro, roll/yaw rates   x and z components negated
    command         sideways speed and turn rate negated
    gait clock      shifted by half a cycle: the mirrored diagonal pair takes the other half of the trot cycle
"""

from __future__ import annotations

import torch

CENTRE_LINE_NEGATE = ("head_j1", "tail_j0")  # rotations about x or z on the robot's centre line


def _joint_mirror(names: list[str]) -> tuple[list[int], list[float]]:
    index = {n: i for i, n in enumerate(names)}
    perm, sign = [], []
    for n in names:
        twin = ("R" + n[1:] if n[0] == "L" else "L" + n[1:]) if n[:1] in ("L", "R") and n[1:2].isdigit() else n
        perm.append(index[twin])
        sign.append(-1.0 if n in CENTRE_LINE_NEGATE else 1.0)
    return perm, sign


def _scan_mirror(cfg) -> list[int]:
    """Permutation that mirrors the height-scan grid left-right (computed from the ray positions themselves)."""
    import numpy as np

    pat = cfg.height_scanner.pattern_cfg
    x = np.arange(-pat.size[0] / 2, pat.size[0] / 2 + 1e-9, pat.resolution)
    y = np.arange(-pat.size[1] / 2, pat.size[1] / 2 + 1e-9, pat.resolution)
    gx, gy = np.meshgrid(x, y, indexing="xy" if pat.ordering == "xy" else "ij")
    pts = np.stack((gx.flatten(), gy.flatten()), -1)
    mirrored = pts * np.array([1.0, -1.0])
    return [int(np.argmin(np.linalg.norm(pts - m, axis=-1))) for m in mirrored]


def _layout(base) -> dict:
    """Index permutation and signs for the policy frame, the critic observation and the actions, plus a mask per
    observation of the entries that belong to the height scan (for widening a network that was trained without)."""
    cfg = base.cfg
    if cfg.task not in ("command", "jump"):
        raise NotImplementedError("symmetry augmentation is implemented for the command and jump tasks")
    names = list(base.obs_joint_names)
    jperm, jsign = _joint_mirror(names)
    perm: list[int] = []
    sign: list[float] = []
    scan: list[bool] = []

    def block(p, s, is_scan=False):
        off = len(perm)
        perm.extend(off + i for i in p)
        sign.extend(s)
        scan.extend([is_scan] * len(p))

    flip_y = ([0, 1, 2], [1.0, -1.0, 1.0])  # vectors: y component changes sign
    flip_xz = ([0, 1, 2], [-1.0, 1.0, -1.0])  # angular rates: x and z change sign
    block(*flip_y)  # gravity
    block(*flip_xz)  # gyro
    if cfg.sensor_suite != "ideal":
        block(*flip_y)  # accelerometer
    if cfg.sensor_suite == "imu_flow":
        block([0, 1], [1.0, -1.0])  # optical flow (vx, vy)
    block(jperm, jsign)  # joint positions
    block(jperm, jsign)  # last actions (all joints)
    # dog gaits: the mirrored foot pattern is the same pattern half a cycle later (trot pairs and the 4-beat walk)
    clock_sign = -1.0 if cfg.gait in ("trot", "walk_trot", "free") else 1.0
    block([0, 1], [clock_sign, clock_sign])
    scan_perm = _scan_mirror(cfg) if base.height_scanner is not None else []
    if scan_perm and cfg.height_scan_in_history:
        block(scan_perm, [1.0] * len(scan_perm), is_scan=True)
    if cfg.policy_sees_joint_velocity:
        block(jperm, jsign)
    if cfg.task == "command":
        block([0, 1, 2], [1.0, -1.0, -1.0])  # command (vx, vy, yaw rate)
    frame_perm, frame_sign, frame_scan = list(perm), list(sign), list(scan)
    # critic: frame + body velocity + height + joint velocities
    block(*flip_y)
    block([0], [1.0])
    block(jperm, jsign)
    tail = scan_perm and not cfg.height_scan_in_history
    if tail:  # the current scan after the critic's extras
        block(scan_perm, [1.0] * len(scan_perm), is_scan=True)
    critic_perm, critic_sign, critic_scan = perm, sign, scan

    act_names = [base.robot.joint_names[i] for i in base._act_ids.tolist()]
    aperm, asign = _joint_mirror(act_names)
    dev = base.device
    hist = max(1, cfg.obs_history)
    frame = len(frame_perm)
    n_tail = len(scan_perm) if tail else 0
    # terrain map in the policy: the scan heights read from the map, then a known flag per scan point
    n_known = n_tail if tail and cfg.terrain_map and cfg.map_policy_input else 0
    if frame * hist + n_tail + n_known != cfg.observation_space or len(critic_perm) != cfg.state_space:
        raise RuntimeError(f"mirror layout ({frame} x {hist}, critic {len(critic_perm)}) does not match the observations "
                           f"({cfg.observation_space}, critic {cfg.state_space})")
    policy_perm = [k * frame + i for k in range(hist) for i in frame_perm]
    policy_sign, policy_scan = frame_sign * hist, frame_scan * hist
    policy_known = [False] * len(policy_perm)
    if tail:  # the current scan after the history
        policy_perm += [frame * hist + i for i in scan_perm]
        policy_sign += [1.0] * n_tail
        policy_scan += [True] * n_tail
        policy_known += [False] * n_tail
    if n_known:
        policy_perm += [frame * hist + n_tail + i for i in scan_perm]
        policy_sign += [1.0] * n_known
        policy_scan += [True] * n_known
        policy_known += [True] * n_known
    return {
        "policy": (torch.tensor(policy_perm, device=dev), torch.tensor(policy_sign, device=dev)),
        "critic": (torch.tensor(critic_perm, device=dev), torch.tensor(critic_sign, device=dev)),
        "actions": (torch.tensor(aperm, device=dev), torch.tensor(asign, device=dev)),
        "scan_mask": {"policy": torch.tensor(policy_scan), "critic": torch.tensor(critic_scan)},
        "known_mask": {"policy": torch.tensor(policy_known), "critic": torch.zeros(len(critic_scan), dtype=torch.bool)},
    }


def scan_masks(base) -> dict:
    """Which policy / critic observation entries are height-scan values, including the map's known flags (bool
    tensors, CPU)."""
    return _layout(base)["scan_mask"]


def known_masks(base) -> dict:
    """Which policy / critic observation entries are the terrain map's known flags (bool tensors, CPU)."""
    return _layout(base)["known_mask"]


@torch.no_grad()
def mirror_augment(env, obs=None, actions=None):
    """RSL-RL data augmentation function: returns [original; mirrored] observations and actions."""
    base = env.unwrapped
    if not hasattr(base, "_mirror_layout"):
        base._mirror_layout = _layout(base)
    lay = base._mirror_layout
    obs_aug = actions_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():
            if key in ("policy", "critic"):
                p, s = lay[key]
                obs_aug[key][n:] = obs[key][:, p] * s
    if actions is not None:
        p, s = lay["actions"]
        actions_aug = torch.cat((actions, actions[:, p] * s), dim=0)
    return obs_aug, actions_aug
