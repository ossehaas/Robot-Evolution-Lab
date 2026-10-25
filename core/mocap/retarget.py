"""Retarget the AI4Animation dog (BVH) to a generated dog robot: joint angles, body height and velocities per frame.

Per frame: a body frame from the skeleton (forward = hips -> shoulders, left = right hip -> left hip, up = their
cross product); each paw relative to its leg root (shoulder or hip joint) in that frame; lengths scaled by the ratio of
hip heights, time by its square root (Froude similarity: a small dog with the same gait steps faster); then inverse
kinematics on the robot's real leg geometry (abduction about x, hip and knee about y, knee direction as built).

    python -m core.mocap.retarget --genome prototypes/puppy/genome.json --out data/mocap/puppy_walk.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from core.mocap.bvh import end_positions, forward_kinematics, load

BVH_DIR = Path("data/mocap/ai4animation_dog/bvh")
# leg root and paw (end site) per robot leg; BVH: Y up, units cm
LEGS = {"L0": ("LeftArm", "LeftHand"), "R0": ("RightArm", "RightHand"),
        "L1": ("LeftUpLeg", "LeftFoot"), "R1": ("RightUpLeg", "RightFoot")}
# clips with a standing dog (hips ~0.45 m) and plenty of walking / some trotting (from the survey)
DEFAULT_CLIPS = ["D1_053_KAN01_001", "D1_047z_KAN01_005", "D1_ex05_KAN02_001", "D1_ex03_KAN02_012",
                 "D1_053_KAN01_002", "D1_ex04_KAN02_003", "D1_009_KAN01_001", "D1_007_KAN01_001"]


def _to_robot_axes(v: np.ndarray) -> np.ndarray:
    """BVH (x, y up, z) [cm] -> robot world (x, y, z up) [m]; the BVH z axis becomes -y (right-handed)."""
    return np.stack((v[..., 0], -v[..., 2], v[..., 1]), -1) / 100.0


def body_frames(pos: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Origin (hip centre) and rotation (world <- body, columns x fwd, y left, z up) per frame."""
    idx = {n: names.index(n) for n in names}
    hips = 0.5 * (pos[:, idx["LeftUpLeg"]] + pos[:, idx["RightUpLeg"]])
    shoulders = 0.5 * (pos[:, idx["LeftArm"]] + pos[:, idx["RightArm"]])
    fwd = shoulders - hips
    left = (pos[:, idx["LeftUpLeg"]] - pos[:, idx["RightUpLeg"]]) + (pos[:, idx["LeftArm"]] - pos[:, idx["RightArm"]])
    x = fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)
    z = np.cross(x, left)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return hips, np.stack((x, y, z), -1)


class LegIK:
    """Inverse kinematics of one robot leg: paw target (body frame, relative to the abduction axis) -> 3 angles."""

    def __init__(self, spec, leg: str):
        J = {j.name: j for j in spec.joints}
        lm = spec.link_map
        self.abd, self.hip, self.knee = J[f"{leg}_j0"], J[f"{leg}_j1"], J[f"{leg}_j2"]
        foot = f"{leg}_s3" if f"{leg}_s3" in lm else f"{leg}_s2"
        sph = next(c for c in lm[foot].collisions if c.kind == "sphere")
        self.paw0 = lm[foot].origin + sph.transform[:3, 3]  # paw centre at rest (robot frame)
        self.paw_r = float(sph.size[0])
        self.A = self.abd.origin
        self.side = 1.0 if leg[0] == "L" else -1.0
        # planar leg (x, z) relative to the hip axis, at rest
        self.K0 = (self.knee.origin - self.hip.origin)[[0, 2]]
        self.F0 = (self.paw0 - self.hip.origin)[[0, 2]]
        self.H_rel = self.hip.origin - self.A  # hip axis relative to the abduction axis

    @staticmethod
    def _rot(v: np.ndarray, th) -> np.ndarray:
        """Rotation about +y in the (x, z) plane: (x, z) -> (x cos + z sin, -x sin + z cos)."""
        c, s = np.cos(th), np.sin(th)
        return np.stack((v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c), -1)

    def solve(self, paw: np.ndarray) -> np.ndarray:
        """paw: (n, 3) paw centre targets in the robot body frame. Returns (n, 3) joint angles (abd, hip, knee)."""
        rel = paw - self.A
        # abduction: rotate about x so that the paw lies in the leg plane of the rest pose
        rest = self.paw0 - self.A
        ang_target = np.arctan2(rel[:, 1], -rel[:, 2])
        ang_rest = math.atan2(rest[1], -rest[2])
        psi = (ang_target - ang_rest) * self.side  # positive = outwards on both sides (mirrored axes)
        c, s = np.cos(-psi * self.side), np.sin(-psi * self.side)
        y_back = rel[:, 1] * c - rel[:, 2] * s
        z_back = rel[:, 1] * s + rel[:, 2] * c
        planar = np.stack((rel[:, 0] - self.H_rel[0], z_back - self.H_rel[2]), -1)  # relative to the hip axis
        # knee: find th2 so that |K0 + R(th2)(F0 - K0)| = |target| (search over the joint range, keep the rest branch)
        D = np.linalg.norm(planar, axis=-1)
        grid = np.linspace(-self.knee.upper, self.knee.upper, 801)
        reach = np.linalg.norm(self.K0 + self._rot(np.broadcast_to(self.F0 - self.K0, (len(grid), 2)), grid), axis=-1)
        th2 = grid[np.argmin(np.abs(reach[None, :] - D[:, None]), axis=1)]
        F = self.K0 + self._rot(np.broadcast_to(self.F0 - self.K0, (len(th2), 2)), th2)
        # hip: rotate F onto the target direction (R(th) turns the polar angle by -th)
        th1 = np.arctan2(F[:, 1], F[:, 0]) - np.arctan2(planar[:, 1], planar[:, 0])
        th1 = (th1 + np.pi) % (2 * np.pi) - np.pi
        out = np.stack((psi, th1, th2), -1)
        lim = np.array([self.abd.upper, self.hip.upper, self.knee.upper])
        return np.clip(out, -lim, lim)


def retarget(genome_path: str, clips: list[str], dt_out: float = 0.02, playback: tuple[float, ...] = (1.0,)) -> dict:
    """playback: also add each clip played slower (e.g. 0.5 = half speed: same poses, half the velocities). The
    recorded dog's slowest walk is ~0.27 m/s at robot scale; slowed copies show the robot what a slow walk looks like
    (lower stride rate, longer stance) instead of letting the style reward prefer standing still."""
    from core.morphology import Genome, build_robot

    spec = build_robot(Genome.load(genome_path))
    legs = {leg: LegIK(spec, leg) for leg in LEGS}
    robot_hip_h = spec.spawn_height + legs["L1"].hip.origin[2]  # hip axis height above the floor at rest
    out = {k: [] for k in ("joints", "height", "lin_vel_b", "ang_vel_b", "gravity_b", "clip")}
    joint_names = []
    for leg in LEGS:
        joint_names += [f"{leg}_j0", f"{leg}_j1", f"{leg}_j2"]
    for name in clips:
        clip = load(BVH_DIR / f"{name}.bvh")
        pos_raw, rot_raw = forward_kinematics(clip)
        pos = _to_robot_axes(pos_raw)
        paws = {leg: _to_robot_axes(end_positions(clip, pos_raw, rot_raw, paw)) for leg, (_, paw) in LEGS.items()}
        names = clip.names
        floor = np.percentile(np.concatenate([p[:, 2] for p in paws.values()]), 2)
        origin, R = body_frames(pos, names)
        # the skeleton's shoulder-hip line leans nose-down (~5 deg on average) while the robot's chassis stands level:
        # remove the clip's average pitch, keep the pitching around it
        a = float(np.median(np.arcsin(np.clip(R[:, 2, 0], -1.0, 1.0))))
        R = R @ np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
        dog_hip_h = np.median(pos[:, names.index("LeftUpLeg"), 2] - floor)
        scale = robot_hip_h / dog_hip_h
        t_scale = math.sqrt(scale)  # Froude: time and speed scale with sqrt(length)
        # robot body frame: the robot's chassis centre sits above the hip axis height; express paws relative to
        # the leg roots and map them onto the robot's leg roots
        T = len(origin)
        joints = np.zeros((T, 12))
        for k, (leg, (root, _)) in enumerate(LEGS.items()):
            ik = legs[leg]
            rel_dog = np.einsum("nji,nj->ni", R, paws[leg] - pos[:, names.index(root)]) * scale  # body frame
            rest_root = ik.hip.origin if leg[1] == "1" else ik.hip.origin
            target = rest_root + rel_dog
            # keep the robot's own lateral stance: use the dog's lateral motion around its mean
            target[:, 1] = ik.paw0[1] + (rel_dog[:, 1] - np.median(rel_dog[:, 1]))
            joints[:, 3 * k:3 * k + 3] = legs[leg].solve(target)
        height = (origin[:, 2] - floor) * scale  # hip-centre height (robot scale)
        # velocities in the body frame (robot scale and time)
        dt_dog = clip.frame_time
        vel_w = np.gradient(origin, dt_dog, axis=0) * scale / t_scale
        lin_b = np.einsum("nji,nj->ni", R, vel_w)
        dR = np.gradient(R, dt_dog, axis=0) / t_scale
        W = np.einsum("nji,njk->nik", R, dR)  # skew(omega) in the body frame
        ang_b = np.stack((W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]), -1)
        grav_b = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -1.0]))
        # resample to the robot's control rate in robot time (slowed copies: stretched time, scaled velocities)
        for k_play, rate in enumerate(playback):
            t_robot = np.arange(T) * dt_dog * t_scale / rate
            t_new = np.arange(0.0, t_robot[-1], dt_out)
            def rs(a):
                return np.stack([np.interp(t_new, t_robot, a[:, i]) for i in range(a.shape[1])], -1) if a.ndim == 2 else np.interp(t_new, t_robot, a)
            out["joints"].append(rs(joints))
            out["height"].append(rs(height))
            out["lin_vel_b"].append(rs(lin_b) * rate)
            out["ang_vel_b"].append(rs(ang_b) * rate)
            g = rs(grav_b)
            out["gravity_b"].append(g / np.linalg.norm(g, axis=-1, keepdims=True))
            out["clip"].append(np.full(len(t_new), clips.index(name) + k_play * len(clips)))
        print(f"{name}: {T} frames -> {len(t_new)} at {1 / dt_out:.0f} Hz, scale {scale:.3f}, pitch {math.degrees(a):+.1f} deg, speed "
              f"{np.linalg.norm(lin_b[:, :2], axis=1).mean():.2f} m/s (robot), joints clipped "
              f"{np.mean(np.abs(joints) >= np.array([j.upper for leg in legs.values() for j in (leg.abd, leg.hip, leg.knee)]) - 1e-6):.1%}")
    return {k: np.concatenate(v) for k, v in out.items()} | {"joint_names": np.array(joint_names), "clips": np.array(clips),
                                                               "dt": dt_out, "playback": np.array(playback)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default="prototypes/puppy/genome.json")
    ap.add_argument("--out", default="data/mocap/puppy_dog_motion.npz")
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--playback", nargs="*", type=float, default=[1.0], help="playback rates, e.g. 1 0.75 0.5")
    args = ap.parse_args()
    data = retarget(args.genome, args.clips, playback=tuple(args.playback))
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}: {len(data['height'])} frames ({len(data['height']) * data['dt']:.0f} s)")


if __name__ == "__main__":
    main()
