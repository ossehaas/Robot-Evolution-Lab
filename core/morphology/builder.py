"""Genome -> RobotSpec: kinematics, printable parts, collision shapes and mass properties of a robot.

Conventions
    * robot frame: +x forward, +y left, +z up; the chassis is centred on the origin in the rest pose
    * every link frame is world-aligned in the rest pose and sits at the joint that drives it
      (the chassis link sits at the origin), which keeps the USD and URDF writers trivial
    * the rest pose is joint angle 0 = servo at its 90 degree centre position
    * right limbs are the exact mirror image of the left limbs. A servo cannot be mirrored, so the right
      servo is mounted as the mirror of its left twin and turns the other way for the same joint angle:
      ``Joint.servo_direction`` records the sign the firmware must apply.

How a limb is assembled (per joint, the "servo frame" S: z = shaft direction, x = along the servo body)
    * the servo body hangs below the horn plane (z < 0) and extends back towards the previous link (x > 0)
    * the link that the servo drives is a printed *horn plate* bolted to the horn at z in [0, t]
    * the link carrying the servo has a *pocket plate* the servo drops into, resting on its mounting tabs
    * a strut (convex hull) joins the horn plate of a segment to the pocket plate at its far end
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from . import components as C
from .genome import Genome
from .geometry import (
    MIRROR_Y,
    MassProps,
    box_mesh,
    circle,
    cylinder_mesh,
    extrude,
    frame,
    hull,
    mirror_mesh,
    project_perp,
    rect,
    sphere_mesh,
    stadium,
    unit,
)

SERVO = C.SG90
T = C.PLATE_THICKNESS

# colours (linear RGB) used for the simulation visuals and the preview
COLORS = {
    "chassis": (0.82, 0.83, 0.86),
    "link": (0.95, 0.45, 0.08),
    "servo": (0.10, 0.28, 0.80),
    "horn": (0.95, 0.95, 0.95),
    "foot_tpu": (0.06, 0.06, 0.07),
    "foot_pla": (0.95, 0.45, 0.08),
    "pcb": (0.05, 0.40, 0.20),
    "pcb_blue": (0.05, 0.20, 0.55),
    "battery": (0.18, 0.18, 0.22),
}


# ---------------------------------------------------------------------- data model


def is_passive_joint(name: str) -> bool:
    """Spring hinges without a servo: a dog's flexible back (spine_*) and spring ankles (*_ja)."""
    return name.startswith("spine_") or name.endswith("_ja")



@dataclass
class Collision:
    kind: str  # "box" | "capsule" | "sphere"
    transform: np.ndarray  # 4x4 in the link frame (capsules run along local z)
    size: tuple[float, ...]  # box: (x, y, z) extents; capsule: (radius, cylinder length); sphere: (radius,)
    material: str = "pla"


@dataclass
class Visual:
    name: str
    mesh: trimesh.Trimesh  # in link coordinates
    color: tuple[float, float, float]


@dataclass
class Link:
    name: str
    origin: np.ndarray  # world position in the rest pose (frames are world aligned)
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in link coordinates
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-7)  # about the com, link axes
    collisions: list[Collision] = field(default_factory=list)
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # world position in the rest pose
    axis: np.ndarray  # world direction, positive rotation by the right-hand rule
    lower: float
    upper: float
    effort: float
    velocity: float
    servo_direction: int  # +1 if a positive joint angle turns the physical servo shaft counter-clockwise


@dataclass
class PrintPart:
    name: str
    mesh: trimesh.Trimesh  # world coordinates, rest pose
    material: str
    quantity: int = 1
    print_frame: np.ndarray | None = None  # world->print transform that lays the part flat on the bed


@dataclass
class RobotSpec:
    name: str
    genome: Genome
    links: list[Link]
    joints: list[Joint]
    parts: list[PrintPart]
    spawn_height: float
    extra_bom: dict[str, int]
    servo: str = "SG90"  # key into components.SERVOS

    @property
    def link_map(self) -> dict[str, Link]:
        return {lk.name: lk for lk in self.links}

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def total_mass(self) -> float:
        return sum(lk.mass for lk in self.links)

    @property
    def foot_links(self) -> list[str]:
        """Last link of every leg (leg links are named <L|R><limb>_s<segment>; heads and tails are not feet)."""
        import re

        children = {j.parent for j in self.joints}
        return [lk.name for lk in self.links if re.fullmatch(r"[LR]\d+_s\d+", lk.name) and lk.name not in children]

    def summary(self) -> str:
        lines = [
            f"robot '{self.name}': {sum(not is_passive_joint(j.name) for j in self.joints)} servos, {len(self.links)} links, "
            f"mass {1000 * self.total_mass:.0f} g, spawn height {1000 * self.spawn_height:.0f} mm",
        ]
        for lk in self.links:
            lines.append(f"  {lk.name:<10} {1000 * lk.mass:6.1f} g")
        return "\n".join(lines)


# ---------------------------------------------------------------------- servo geometry in its own frame S


def _servo_x_range():
    return -SERVO.shaft_offset, SERVO.body_length - SERVO.shaft_offset


def _servo_center_x():
    lo, hi = _servo_x_range()
    return 0.5 * (lo + hi)


def _servo_top_z():
    return -SERVO.shaft_height


def _tab_bottom_z():
    return _servo_top_z() - SERVO.tab_from_top - SERVO.tab_thickness


def servo_visual(S: np.ndarray) -> trimesh.Trimesh:
    """SG90 body, mounting tabs, gear tower and output shaft, placed in servo frame ``S``."""
    lo, hi = _servo_x_range()
    top = _servo_top_z()
    cx = _servo_center_x()
    body = box_mesh((SERVO.body_length, SERVO.body_width, SERVO.body_height), S @ _tf([cx, 0, top - SERVO.body_height / 2]))
    tabs = box_mesh(
        (SERVO.tab_span, SERVO.body_width, SERVO.tab_thickness),
        S @ _tf([cx, 0, top - SERVO.tab_from_top - SERVO.tab_thickness / 2]),
    )
    tower = cylinder_mesh(SERVO.body_width / 2, top, top + 0.0015, S)
    shaft = cylinder_mesh(0.0024, top, 0.0, S, sections=12)
    return trimesh.util.concatenate([body, tabs, tower, shaft])


def horn_visual(S: np.ndarray) -> trimesh.Trimesh:
    return cylinder_mesh(0.0095, -0.0015, 0.0, S)


def servo_box(S: np.ndarray):
    """Collision/mass box of the servo body: (size, transform)."""
    top = _servo_top_z()
    return (SERVO.body_length, SERVO.body_width, SERVO.body_height), S @ _tf([_servo_center_x(), 0, top - SERVO.body_height / 2])


def pocket_plate(S: np.ndarray) -> trimesh.Trimesh:
    """Plate the servo drops into; its tabs rest on the plate and are screwed down."""
    cx = _servo_center_x()
    half_span = SERVO.tab_span / 2
    margin = 0.0025
    outline = rect(cx - half_span - margin, -SERVO.body_width / 2 - 0.004, cx + half_span + margin, SERVO.body_width / 2 + 0.004)
    lo, hi = _servo_x_range()
    c = C.POCKET_CLEARANCE
    pocket = rect(lo - c, -SERVO.body_width / 2 - c, hi + c, SERVO.body_width / 2 + c)
    holes = [circle((cx + s * SERVO.tab_hole_spacing / 2, 0.0), C.SCREW_HOLE / 2) for s in (-1, 1)]
    profile = outline.difference(pocket)
    for h in holes:
        profile = profile.difference(h)
    zt = _tab_bottom_z()
    return extrude(profile, zt - T, zt, S)


def pocket_far_edge(S: np.ndarray) -> trimesh.Trimesh:
    """Thin slab at the far end (+x) of a pocket plate, used as the anchor for connecting struts."""
    cx = _servo_center_x()
    x1 = cx + SERVO.tab_span / 2 + 0.0025
    zt = _tab_bottom_z()
    return box_mesh((0.004, SERVO.body_width + 0.008, T), S @ _tf([x1 - 0.002, 0, zt - T / 2]))


def _tf(t) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = t
    return M


def servo_frame(origin, shaft_axis, back_dir) -> np.ndarray:
    return frame(origin, project_perp(back_dir, shaft_axis), shaft_axis)


# ---------------------------------------------------------------------- limb construction


@dataclass
class _LimbJoint:
    origin: np.ndarray
    shaft: np.ndarray  # physical shaft direction
    kin_axis: np.ndarray  # joint axis used by the simulator
    S: np.ndarray  # servo frame
    range_rad: float


def _limb_left(genome: Genome, pair_index: int):
    """Build the left limb of a pair in world coordinates. Returns joints, per-segment pieces and foot."""
    body = genome.body
    pair = genome.limb_pairs[pair_index]
    s = math.radians(pair.splay_deg)
    u = np.array([math.sin(s), math.cos(s), 0.0])  # outward limb direction
    zup = np.array([0.0, 0.0, 1.0])
    pitch_axis = np.cross(zup, u)

    wall_point = np.array([pair.x * body.length / 2, body.width / 2, pair.z * body.height / 2])
    shoulder = SERVO.tab_span / 2 + _servo_center_x() + 0.0025
    p = wall_point + u * shoulder

    joints: list[_LimbJoint] = []
    segments = []
    prev_dir = u  # the chassis "points" outwards along u
    for seg in pair.segments:
        shaft = zup if seg.axis == "yaw" else pitch_axis
        S = servo_frame(p, shaft, -prev_dir)
        joints.append(_LimbJoint(p.copy(), shaft, shaft, S, math.radians(seg.range_deg)))
        e = math.radians(seg.elev_deg)
        d = unit(math.cos(e) * u + math.sin(e) * zup)
        end = p + seg.length * d
        segments.append((p.copy(), d, end, seg))
        prev_dir = d
        p = end
    return joints, segments, pair, wall_point, u


def _horn_plate(S: np.ndarray, start_local, end_local) -> trimesh.Trimesh:
    """Link plate bolted to a horn at the origin of ``S`` running towards ``end_local`` (2D, in S)."""
    outline = stadium(start_local, end_local, C.LINK_WIDTH / 2, end_radius=0.0095)
    for pt in [(0.0, 0.0)]:
        outline = outline.difference(circle(pt, C.HORN_HUB_HOLE / 2))
    direction = np.asarray(end_local, float) - np.asarray(start_local, float)
    n = np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]]) / n if n > 1e-9 else np.array([0.0, 1.0])
    for sgn in (-1, 1):
        outline = outline.difference(circle(sgn * C.HORN_SCREW_RADIUS * perp, C.SCREW_HOLE / 2))
    return extrude(outline, 0.0, T, S)


def _to_local2d(S: np.ndarray, point) -> tuple[float, float]:
    local = np.linalg.inv(S) @ np.append(point, 1.0)
    return float(local[0]), float(local[1])


# ---------------------------------------------------------------------- main entry point


def build_robot(genome: Genome) -> RobotSpec:
    if getattr(genome, "kind", "") == "dog":
        from .dog import build_dog

        return build_dog(genome)
    genome = genome.repair()
    # servo geometry is shared by all supported types (SERVO); mass, torque and speed depend on the chosen type
    servo_type = C.SERVOS[genome.servo]
    body = genome.body
    pla = C.PLA

    # ---------------------------------------------------------------- chassis
    chassis = Link("chassis", np.zeros(3))
    cm = MassProps()
    L, W, H = body.length, body.width, body.height
    shell_outline = rect(-L / 2, -W / 2, L / 2, W / 2)
    shell_inner = rect(-L / 2 + C.WALL, -W / 2 + C.WALL, L / 2 - C.WALL, W / 2 - C.WALL)
    tray = trimesh.util.concatenate(
        [
            extrude(shell_outline, -H / 2, -H / 2 + C.WALL),
            extrude(shell_outline.difference(shell_inner), -H / 2 + C.WALL, H / 2 - T),
        ]
    )
    lid_profile = shell_outline.difference(circle((-L / 2 + 0.010, 0.0), 0.004))  # cable pass-through
    lid = extrude(lid_profile, H / 2 - T, H / 2)
    parts = [PrintPart("chassis_tray", tray, "pla"), PrintPart("chassis_lid", lid, "pla")]
    cm.add_mesh(tray, pla.effective_density)
    cm.add_mesh(lid, pla.effective_density)
    chassis.visuals += [Visual("tray", tray, COLORS["chassis"]), Visual("lid", lid, COLORS["chassis"])]
    chassis.collisions.append(Collision("box", np.eye(4), (L, W, H), "pla"))

    # electronics: battery inside on the floor, driver + MCU + IMU stacked on the lid
    inside_floor = -H / 2 + C.WALL
    bat = C.BATTERY_2S
    bat_tf = _tf([0.0, 0.0, inside_floor + bat.size[2] / 2])
    buck_tf = _tf([-L / 2 + C.WALL + C.BUCK.size[0] / 2 + 0.001, 0.0, inside_floor + C.BUCK.size[2] / 2]) if L > 0.075 else _tf([0, 0, inside_floor + bat.size[2] + C.BUCK.size[2] / 2])
    drv = C.SERVO_DRIVER
    drv_size = (drv.size[1], drv.size[0], drv.size[2]) if L >= drv.size[1] else drv.size  # lay it along the body when it fits
    drv_tf = _tf([0.0, 0.0, H / 2 + 0.002 + drv_size[2] / 2])
    mcu_tf = _tf([0.004, 0.0, H / 2 + 0.002 + drv_size[2] + 0.004 + C.ESP32_S3.size[2] / 2])
    imu_tf = _tf([-0.018, 0.0, H / 2 + 0.002 + drv_size[2] + 0.004 + C.IMU.size[2] / 2])
    for comp, size, tf, color in [
        (bat, bat.size, bat_tf, COLORS["battery"]),
        (C.BUCK, C.BUCK.size, buck_tf, COLORS["pcb_blue"]),
        (drv, drv_size, drv_tf, COLORS["pcb_blue"]),
        (C.ESP32_S3, C.ESP32_S3.size, mcu_tf, COLORS["pcb"]),
        (C.IMU, C.IMU.size, imu_tf, COLORS["pcb"]),
    ]:
        cm.add_box(comp.mass, size, tf)
        chassis.visuals.append(Visual(comp.name, box_mesh(size, tf), color))
    cm.add_box(C.WIRING_MASS, (L * 0.6, W * 0.6, H * 0.5), np.eye(4))

    links = [chassis]
    joints: list[Joint] = []
    masses = {"chassis": cm}

    # ---------------------------------------------------------------- limbs
    for pi in range(len(genome.limb_pairs)):
        ljoints, segs, pair, wall_point, u = _limb_left(genome, pi)
        foot_mat = pair.foot.material

        # geometry of the left limb in world coordinates, grouped by the link that carries it
        carried: dict[int, list[tuple[str, trimesh.Trimesh, tuple]]] = {k: [] for k in range(-1, len(segs))}
        carried_mass: dict[int, list] = {k: [] for k in range(-1, len(segs))}
        colls: dict[int, list[tuple[str, np.ndarray, tuple, str]]] = {k: [] for k in range(-1, len(segs))}
        seg_parts: list[tuple[str, trimesh.Trimesh, np.ndarray, str]] = []

        for k, (start, d, end, seg) in enumerate(segs):
            J = ljoints[k]
            owner = k - 1  # the servo of joint k sits on the previous link (-1 = chassis)
            carried[owner].append(("servo", servo_visual(J.S), COLORS["servo"]))
            size, tf = servo_box(J.S)
            carried_mass[owner].append(("box", servo_type.mass, size, tf))
            colls[owner].append(("box", tf, size, "pla"))
            mount = pocket_plate(J.S)
            if k == 0:
                # shoulder: pocket plate on the chassis + a strut to the chassis side wall
                wall_slab = box_mesh((0.004, 0.014, min(H - 2 * T, 0.016)), frame(wall_point - u * 0.002, u, [0, 0, 1]))
                shoulder = trimesh.util.concatenate([mount, hull(pocket_far_edge(J.S), wall_slab)])
                carried[-1].append(("shoulder", shoulder, COLORS["chassis"]))
                carried_mass[-1].append(("mesh", shoulder, pla.effective_density))
                seg_parts.append((f"shoulder_p{pi}", shoulder, J.S, "pla"))
            carried[k].append(("horn", horn_visual(J.S), COLORS["horn"]))

            # the printed link of segment k: horn plate (+ strut + pocket plate for the next servo)
            end_local = _to_local2d(J.S, end)
            if k + 1 < len(segs):
                Jn = ljoints[k + 1]
                pocket = pocket_plate(Jn.S)
                reach = max(seg.length - (SERVO.tab_span / 2 + _servo_center_x() + 0.0025), 0.010)
                plate_end = start + d * reach
                horn = _horn_plate(J.S, (0.0, 0.0), _to_local2d(J.S, plate_end))
                anchor = box_mesh((0.004, C.LINK_WIDTH, T), frame(plate_end, d, J.S[:3, 2]) @ _tf([0, 0, T / 2]))
                link_mesh = trimesh.util.concatenate([horn, pocket, hull(anchor, pocket_far_edge(Jn.S))])
            else:
                link_mesh = _horn_plate(J.S, (0.0, 0.0), end_local)
                r = pair.foot.radius
                foot = sphere_mesh(end, r)
                carried[k].append(("foot", foot, COLORS["foot_" + foot_mat]))
                mat = C.MATERIALS[foot_mat]
                carried_mass[k].append(("sphere", mat.effective_density * 4.0 / 3.0 * math.pi * r**3, end, r))
                colls[k].append(("sphere", _tf(end), (r,), foot_mat))
                seg_parts.append((f"foot_p{pi}", foot, np.eye(4), foot_mat))
            carried[k].append(("link", link_mesh, COLORS["link"]))
            carried_mass[k].append(("mesh", link_mesh, pla.effective_density))
            seg_parts.append((f"link_p{pi}_s{k}", link_mesh, J.S, "pla"))
            # capsule along the segment covers the plate for contacts
            seg_len = float(np.linalg.norm(end - start))
            cap_tf = frame(0.5 * (start + end), J.S[:3, 2] if abs(np.dot(J.S[:3, 2], d)) < 0.99 else [1, 0, 0], d)
            colls[k].append(("capsule", cap_tf, (0.006, max(seg_len - 0.012, 0.001)), "pla"))

        # emit left and mirrored right sides
        for side in ("L", "R"):
            mirror = side == "R"
            M = MIRROR_Y if mirror else np.eye(3)

            def place_point(pt):
                return M @ pt

            def place_tf(tf):
                out = tf.copy()
                out[:3, :3] = M @ tf[:3, :3] @ (M if mirror else np.eye(3))
                out[:3, 3] = M @ tf[:3, 3]
                return out

            def place_mesh(mesh):
                return mirror_mesh(mesh) if mirror else mesh

            prefix = f"{side}{pi}"
            link_names = [f"{prefix}_s{k}" for k in range(len(segs))]
            for k, (start, d, end, seg) in enumerate(segs):
                J = ljoints[k]
                parent = "chassis" if k == 0 else link_names[k - 1]
                origin = place_point(J.origin)
                links.append(Link(link_names[k], origin))
                masses[link_names[k]] = MassProps()
                kin_axis = M @ J.kin_axis * (-1.0 if mirror else 1.0)
                joints.append(
                    Joint(
                        name=f"{prefix}_j{k}",
                        parent=parent,
                        child=link_names[k],
                        origin=origin,
                        axis=unit(kin_axis),
                        lower=-J.range_rad,
                        upper=J.range_rad,
                        effort=servo_type.usable_torque,
                        velocity=servo_type.max_speed,
                        servo_direction=-1 if mirror else 1,
                    )
                )
            lmap = {lk.name: lk for lk in links}
            for owner in range(-1, len(segs)):
                lname = "chassis" if owner < 0 else link_names[owner]
                link = lmap[lname]
                shift = _tf(-link.origin)
                for name, mesh, color in carried[owner]:
                    placed = place_mesh(mesh).copy()
                    placed.apply_transform(shift)
                    link.visuals.append(Visual(f"{prefix}_{name}", placed, color))
                mp = masses[lname]
                for item in carried_mass[owner]:
                    if item[0] == "mesh":
                        mp.add_mesh(place_mesh(item[1]), item[2])
                    elif item[0] == "box":
                        mp.add_box(item[1], item[2], place_tf(item[3]))
                    elif item[0] == "sphere":
                        mp.add_sphere(item[1], place_point(item[2]), item[3])
                for kind, tf, size, mat in colls[owner]:
                    local = place_tf(tf)
                    local[:3, 3] -= link.origin
                    link.collisions.append(Collision(kind, local, size, mat))
            for name, mesh, S, mat in seg_parts:
                parts.append(PrintPart(f"{name}_{side}", place_mesh(mesh), mat, print_frame=place_tf(S)))

    # ---------------------------------------------------------------- finalize masses (world -> link frames)
    for link in links:
        mp = masses[link.name]
        if not mp.items:
            mp.add_sphere(1e-3, link.origin, 0.005)
        mass, com, inertia = mp.total()
        link.mass, link.com, link.inertia = mass, com - link.origin, inertia

    spawn_height = _lowest_point(links)
    extra_bom = {
        f"{servo_type.name} micro servo": len(joints),
        C.ESP32_S3.name: 1,
        C.SERVO_DRIVER.name: 1,
        C.IMU.name: 1,
        C.BATTERY_2S.name: 1,
        C.BUCK.name: 1,
        "M2x6 self-tapping screw (servo tabs)": 2 * len(joints),
        "horn screw (supplied with servo)": 3 * len(joints),
        "M2x8 screw (lid)": 4,
    }
    return RobotSpec(genome.name, genome, links, joints, parts, spawn_height, extra_bom, servo=genome.servo)


def _lowest_point(links: list[Link]) -> float:
    """Height at which the chassis origin must spawn so no collision shape starts below the ground."""
    lowest = 0.0
    for link in links:
        for c in link.collisions:
            tf = c.transform.copy()
            tf[:3, 3] += link.origin
            if c.kind == "sphere":
                z = tf[2, 3] - c.size[0]
            elif c.kind == "capsule":
                r, length = c.size
                axis = tf[:3, 2]
                z = tf[2, 3] - abs(axis[2]) * length / 2 - r
            else:
                corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * 0.5 * np.array(c.size)
                z = (tf[:3, :3] @ corners.T)[2].min() + tf[2, 3]
            lowest = min(lowest, z)
    return float(-lowest + 0.004)
