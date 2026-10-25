"""Dog-shaped robots (experiment 2): a cute, printable quadruped of 30-50 cm with standard-size bus servos.

Unlike the spider-like genomes (genome.py / builder.py) the layout is fixed and only proportions vary:

    legs   4 x 3 servos: sideways (abduction, about x) in the chassis, hip (about y) on a shoulder bracket outside
           the chassis, knee (about y). The knee servo sits across the top of the thigh and drives the knee through
           a pushrod (a 1:1 parallelogram), so the lower leg stays light. Front knees point backwards and rear
           knees forwards, as on a real dog.
    head   neck nod (about y) and head tilt (about x); two round displays as eyes
    tail   wag (about z)

Cuteness is in the proportions: a big head (about as wide as the body), a short snout, a round body, chunky legs,
big paws. The printable parts, servo mounts, fasteners and assembly are designed in ``dog_mech.py`` (millimetres,
real solids); this module turns that design into a RobotSpec, so the USD/URDF/print exporters and the whole
simulation pipeline work unchanged. Masses come from the printed geometry (perimeters + sparse infill), the servos,
electronics, screws and cables where they really sit.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import cad
from . import components as C
from . import sts3215 as SV
from .builder import Collision, Joint, Link, PrintPart, RobotSpec, Visual, _lowest_point
from .geometry import MassProps

# ---------------------------------------------------------------------- genome


@dataclass
class DogBody:
    length: float = 0.200
    width: float = 0.110
    height: float = 0.095


@dataclass
class DogLegs:
    hip_x: float = 0.080  # front hips at +hip_x, rear hips at -hip_x [m]
    upper: float = 0.080  # thigh [m]
    lower: float = 0.085  # shin [m]
    rest_depth: float = 0.88  # hip-to-foot distance in the rest pose, as a share of upper + lower
    paw_radius: float = 0.018
    # sideways (abduction) joint range either side [rad]: 0.5 (29 deg) cannot push the dog over from lying on its side
    # (journal step 36); larger ranges are being tested for getting up
    abduction_limit: float = 0.5
    thickness: float = 1.0  # kept for older genomes; the printable legs are sized by the servos inside them
    # knee directions "<front>_<rear>": back_front = front knee backwards, rear knee forwards (as a real dog's elbow and
    # stifle; the default), back_back = all knees backwards (as Spot), front_back, front_front
    knees: str = "back_front"


@dataclass
class DogHead:
    length: float = 0.100
    width: float = 0.125
    height: float = 0.105
    snout: float = 0.032  # how far the snout sticks out
    ear_length: float = 0.055
    ear_style: str = "floppy"  # "floppy" | "up"
    neck: float = 0.045  # neck joint to head-tilt joint (up and forward)


@dataclass
class DogTail:
    length: float = 0.070


# colours in sRGB hex; converted to linear for the renderer
STYLES = {
    "puppy": {"body": "#F2E3C9", "accent": "#B5703A", "paw": "#5A3B2A", "collar": "#D83A3A", "tag": "#F2C230"},
    "terrier": {"body": "#DADADA", "accent": "#4A4A52", "paw": "#2A2A2E", "collar": "#2F7DE1", "tag": "#F2C230"},
    "bulldog": {"body": "#D8A86A", "accent": "#F5EBDD", "paw": "#6B4A2E", "collar": "#2DAA5A", "tag": "#C0C0C8"},
}
COMMON_COLORS = {"frame": "#9FB4C8", "nose": "#1E1A1A", "eye": "#141414", "shine": "#FFFFFF", "servo": "#2B2B30"}

# print model: perimeters / top-bottom skins of this thickness on every surface, sparse infill inside
PRINT_SKIN = 1.1  # mm (3 perimeters of 0.4 mm, 5 top/bottom layers of 0.2 mm)
PRINT_INFILL = {"pla": 0.15, "tpu": 0.15}
CABLE_MASS = 0.005  # kg per servo lead (150-200 mm)
HARNESS_MASS = 0.030  # kg: battery leads, XT30, switch, fuse, power distribution in the chassis


@dataclass
class DogSpine:
    """Flexible back (simulation experiment D2b): the body is split at ``split_x`` into a front half (chassis: front
    legs, neck, IMU) and a rear half (rear legs, tail), joined by passive spring hinges. No axes = stiff body."""

    axes: list[str] = field(default_factory=list)  # any of "roll" (twist about x), "pitch" (bend about y)
    roll_stiffness: float = 8.0  # [N*m/rad]
    pitch_stiffness: float = 8.0  # [N*m/rad]
    damping: float = 0.15  # [N*m*s/rad]
    roll_limit: float = 0.35  # [rad]
    pitch_limit: float = 0.30  # [rad]
    split_x: float = 0.0  # [m] position of the hinge along the body


@dataclass
class DogAnkle:
    """Spring ankle: the TPU paw hangs on a passive spring hinge just above it (pitch). The printed dog always has it:
    a TPU flexure between shin and paw (dog_paw.py), sized from ``stiffness`` and ``limit``, its pivot ``height`` paw
    radii above the paw centre. ``enabled = False`` simulates the ankle as rigid (paw parts on the shin link)."""

    enabled: bool = False
    stiffness: float = 1.5  # [N*m/rad]: ~0.1 rad under a standing paw load
    damping: float = 0.03  # [N*m*s/rad]
    limit: float = 0.4  # [rad] (the printed stops land exactly here)
    height: float = 1.0  # hinge height above the paw centre, in paw radii (printable: 0.9-1.3)


@dataclass
class DogGenome:
    name: str = "dog"
    kind: str = "dog"
    style: str = "puppy"
    body: DogBody = field(default_factory=DogBody)
    legs: DogLegs = field(default_factory=DogLegs)
    head: DogHead = field(default_factory=DogHead)
    tail: DogTail = field(default_factory=DogTail)
    servo: str = "STS3215_12V"
    lineage: list[str] = field(default_factory=list)
    spine: DogSpine = field(default_factory=DogSpine)
    ankle: DogAnkle = field(default_factory=DogAnkle)
    # springy TPU paws (list step 2): PhysX compliant contact; 0 = rigid contact as before
    paw_contact_stiffness: float = 0.0  # [N/m]
    paw_contact_damping: float = 0.0  # [N*s/m]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> DogGenome:
        return DogGenome(
            name=data.get("name", "dog"),
            style=data.get("style", "puppy"),
            body=DogBody(**data.get("body", {})),
            legs=DogLegs(**data.get("legs", {})),
            head=DogHead(**data.get("head", {})),
            tail=DogTail(**data.get("tail", {})),
            servo=data.get("servo", "STS3215_12V"),
            lineage=list(data.get("lineage", [])),
            spine=DogSpine(**data.get("spine", {})),
            ankle=DogAnkle(**data.get("ankle", {})),
            paw_contact_stiffness=data.get("paw_contact_stiffness", 0.0),
            paw_contact_damping=data.get("paw_contact_damping", 0.0),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> DogGenome:
        return DogGenome.from_dict(json.loads(Path(path).read_text()))

    def repair(self) -> DogGenome:
        if self.servo not in C.SERVOS:
            self.servo = "STS3215_12V"
        if self.style not in STYLES:
            self.style = "puppy"
        self.legs.rest_depth = min(max(self.legs.rest_depth, 0.6), 0.95)
        return self

    @property
    def num_dof(self) -> int:
        return 15

    def topology_key(self) -> str:
        return "dog-3x4+3"


# ---------------------------------------------------------------------- helpers


def _lin(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple((int(h[i : i + 2], 16) / 255.0) ** 2.2 for i in (0, 2, 4))


def printed_mass(solid, material: str, infill: float | None = None) -> float:
    """Mass [kg] of a printed part from its volume and surface (skins + sparse infill)."""
    V = float(solid.volume())  # mm^3
    A = float(solid.surface_area())  # mm^2
    skin = min(V, PRINT_SKIN * A)
    solid_mm3 = skin + (PRINT_INFILL[material] if infill is None else infill) * max(V - skin, 0.0)
    rho = C.MATERIALS[material].density  # kg/m^3
    return rho * solid_mm3 * 1e-9


def _mesh_m(solid, origin_mm, tol: float = 0.0):
    s = solid.simplify(tol) if tol > 0 else solid
    mesh = cad.to_trimesh(s, 0.001)
    mesh.apply_translation(-np.asarray(origin_mm, float) * 0.001)
    return mesh


# ---------------------------------------------------------------------- builder


def build_dog(genome: DogGenome) -> RobotSpec:
    from .dog_mech import HW_MASS, build_mechanics

    g = genome.repair()
    servo = C.SERVOS[g.servo]
    mech = build_mechanics(g)
    palette = {k: _lin(v) for k, v in {**STYLES[g.style], **COMMON_COLORS}.items()}

    # spring ankles: the paw, its sole and the lower half of the ankle spring form link <L|R><i>_s3 on the passive
    # hinge <L|R><i>_ja at the spring's pivot (dog_paw.py). With the ankle disabled they stay on the shin link (rigid).
    if not g.ankle.enabled:
        def remap(link: str) -> str:
            return link[:-1] + "2" if link.endswith("_s3") else link

        mech.joints = [j for j in mech.joints if not j.passive]
        for p in mech.parts:
            p.link = remap(p.link)
            p.link_below = None
        mech.collisions = [(remap(lk), *rest) for lk, *rest in mech.collisions]
        mech.hardware = [type(h)(h.item, remap(h.link), h.qty, h.mass_g, h.use) for h in mech.hardware]
        mech.details = [type(dt)(dt.name, remap(dt.link), dt.solid, dt.color) for dt in mech.details]

    # links (world-aligned frames at the joint that drives them)
    names = ["chassis"] + [j.child for j in mech.joints]
    sp = g.spine
    split_mm = 1000.0 * sp.split_x
    hinge_mm = np.array([split_mm, 0.0, 0.0])
    if sp.axes:
        # flexible back: the rear half of the body becomes its own link ("mid" carries the second hinge axis)
        names += (["mid"] if len(sp.axes) > 1 else []) + ["rear"]
        for n in ("mid", "rear"):
            mech.link_origin[n] = hinge_mm
    links = {n: Link(n, np.asarray(mech.link_origin[n], float) * 0.001) for n in names}
    masses = {n: MassProps() for n in names}
    origin_mm = {n: np.asarray(mech.link_origin[n], float) for n in names}

    def body_half(link: str, x_mm: float) -> str:
        """Which link carries a piece of the body: with a flexible back, chassis items behind the hinge go to the rear."""
        return "rear" if sp.axes and link == "chassis" and x_mm < split_mm else link

    # printed parts: visuals, masses, print files
    parts: list[PrintPart] = []
    part_mass: dict[str, float] = {}
    centroids = defaultdict(list)
    for p in mech.parts:
        kg = printed_mass(p.solid, p.material, p.infill)
        part_mass[p.name] = kg
        world = cad.to_trimesh(p.solid, 0.001)
        pieces = [(p.link, p.solid)]
        if p.split_z is not None and p.link_below:
            # a flexure between two links (the ankle spring): each side carries its half
            halves = p.solid.split_by_plane((0.0, 0.0, 1.0), p.split_z)  # (above, below)
            pieces = [(lk, piece) for lk, piece in zip((p.link, p.link_below), halves) if not piece.is_empty()]
        if sp.axes and p.link == "chassis":
            # cut body parts at the hinge, so both halves get their real mass and inertia
            front, back = p.solid.split_by_plane((1.0, 0.0, 0.0), split_mm)
            pieces = [(lk, piece) for lk, piece in (("chassis", front), ("rear", back)) if not piece.is_empty()]
        for lk, solid in pieces:
            piece_kg = printed_mass(solid, p.material, p.infill)
            mesh = cad.to_trimesh(solid, 0.001)
            # inertia of the piece as a uniform solid carrying its printed mass (good enough for the simulator)
            mesh.density = piece_kg / max(float(mesh.volume), 1e-12)
            props = mesh.mass_properties
            masses[lk].add(piece_kg, props["center_mass"], props["inertia"])
            centroids[lk].append((piece_kg, mesh.center_mass))
            if p.visual:
                links[lk].visuals.append(Visual(p.name, _mesh_m(solid, origin_mm[lk], 0.25), palette[p.color]))
        up = np.asarray(p.print_up, float)
        up /= np.linalg.norm(up)
        ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        frame = cad.frame((0, 0, 0), ref, up)
        parts.append(PrintPart(p.name, world, p.material, print_frame=frame))
    for d in mech.details:
        centre = cad.to_trimesh(d.solid).bounds.mean(axis=0)
        lk = body_half(d.link, float(centre[0]))
        links[lk].visuals.append(Visual(d.name, _mesh_m(d.solid, origin_mm[lk], 0.2), palette[d.color]))

    # servos: mass box where the case sits, lead mass next to it
    S_mm = {}
    for sv in mech.servos:
        size, c = SV.mass_box_S()
        T = sv.S.copy()
        T[:3, 3] = cad.point(sv.S, c)
        T[:3, 3] *= 0.001
        lk = body_half(sv.link, 1000.0 * T[0, 3])
        masses[lk].add_box(servo.mass, tuple(0.001 * v for v in size), T)
        masses[lk].add_sphere(CABLE_MASS, T[:3, 3], 0.01)
        S_mm[sv.joint] = sv
    # electronics
    for e in mech.electronics:
        masses[body_half(e.link, float(e.center[0]))].add_box(e.component.mass, tuple(0.001 * v for v in e.size),
                                                               _tf_m(e.center))
    if sp.axes:  # wiring harness: half in each body half
        masses["chassis"].add_box(HARNESS_MASS / 2, (0.04, 0.06, 0.04), _tf_m((split_mm + 20.0, 0, 0)))
        masses["rear"].add_box(HARNESS_MASS / 2, (0.04, 0.06, 0.04), _tf_m((split_mm - 20.0, 0, 0)))
    else:
        masses["chassis"].add_box(HARNESS_MASS, (0.08, 0.06, 0.04), np.eye(4))
    # screws, inserts, bearings: a small mass at the centre of the printed parts of their link
    hw_mass = defaultdict(float)
    for h in mech.hardware:
        if sp.axes and h.link == "chassis":  # screws of the body: split evenly between the halves
            hw_mass["chassis"] += 0.5 * h.qty * h.mass_g * 0.001
            hw_mass["rear"] += 0.5 * h.qty * h.mass_g * 0.001
        else:
            hw_mass[h.link] += h.qty * h.mass_g * 0.001
    for link, kg in hw_mass.items():
        ws = centroids[link]
        c = sum(w * p for w, p in ws) / sum(w for w, _ in ws) if ws else origin_mm[link] * 0.001
        masses[link].add_sphere(kg, c, 0.01)

    # collisions (link frames)
    for link, kind, T, size, mat in mech.collisions:
        T = np.array(T, float)
        boxes = [(link, T, size)]
        if sp.axes and link == "chassis" and kind == "box":
            # cut body boxes at the hinge (they are axis-aligned), with a 1 mm gap on each side
            lo, hi = T[0, 3] - size[0] / 2, T[0, 3] + size[0] / 2
            boxes = []
            for lk, a, b in (("chassis", max(lo, split_mm + 1.0), hi), ("rear", lo, min(hi, split_mm - 1.0))):
                if b - a > 2.0:
                    Tb = T.copy()
                    Tb[0, 3] = 0.5 * (a + b)
                    boxes.append((lk, Tb, (b - a, size[1], size[2])))
        elif sp.axes and link == "chassis":
            boxes = [(body_half(link, T[0, 3]), T, size)]
        for lk, Tb, sz in boxes:
            local = Tb.copy()
            local[:3, 3] = (local[:3, 3] - origin_mm[lk]) * 0.001
            links[lk].collisions.append(Collision(kind, local, tuple(0.001 * float(v) for v in sz), mat))

    # joints, with the sign the firmware applies to each servo
    joints = []
    for jg in mech.joints:
        if jg.passive:
            continue
        sv = S_mm[jg.name]
        out = sv.S[:3, 2]
        direction = 1 if float(np.dot(jg.axis, out)) > 0 else -1
        if sv.link == jg.child:  # servo body rides on the child link: the shaft turns the other way
            direction = -direction
        parent = body_half(jg.parent, float(jg.origin[0]))
        joints.append(Joint(jg.name, parent, jg.child, np.asarray(jg.origin, float) * 0.001, np.asarray(jg.axis, float),
                            -jg.limit, jg.limit, servo.usable_torque, servo.max_speed, direction))
    if sp.axes:
        # passive spring hinges of the back (no servo): chassis -> (mid ->) rear
        chain = ["chassis"] + (["mid"] if len(sp.axes) > 1 else []) + ["rear"]
        for k, ax in enumerate(sp.axes):
            axis = np.array([1.0, 0, 0]) if ax == "roll" else np.array([0, 1.0, 0])
            limit = sp.roll_limit if ax == "roll" else sp.pitch_limit
            joints.append(Joint(f"spine_{ax}", chain[k], chain[k + 1], hinge_mm * 0.001, axis, -limit, limit, 50.0, 20.0, 1))
        if "mid" in masses:
            masses["mid"].add_sphere(0.010, hinge_mm * 0.001, 0.01)  # the hinge block between the two axes
    for jg in mech.joints:
        if jg.passive:  # spring ankle (no servo): pitch, same axis on both sides like the knee; stiffness set in core/sim
            joints.append(Joint(jg.name, jg.parent, jg.child, np.asarray(jg.origin, float) * 0.001,
                                np.asarray(jg.axis, float), -jg.limit, jg.limit, 50.0, 20.0, 1))

    for n in names:
        mass, com, inertia = masses[n].total()
        links[n].mass, links[n].com, links[n].inertia = mass, com - links[n].origin, inertia
    link_list = [links[n] for n in names]
    spawn_height = _lowest_point(link_list)

    # bill of materials (purchased parts)
    hw = Counter()
    for h in mech.hardware:
        hw[h.item] += h.qty
    battery = C.BATTERY_3S if "12V" in g.servo else C.BATTERY_2S_DOG
    extra_bom = {
        f"{servo.name} serial bus servo (with horn + idler horn; see servo_map.json for ids)": len(mech.servos),
        C.BUS_ADAPTER.name: 1,
        C.ESP32_S3.name: 1,
        C.IMU.name: 1,
        battery.name + ", XT30 plug": 1,
        C.BUCK.name: 1,
        C.ROUND_DISPLAY.name: 2,
        "Bus servo extension leads 200-300 mm (5264 3-pin), daisy chains + head and tail": 6,
        "Mini rocker switch KCD11 (10 x 15 mm cutout), 10 A": 1,
        "Inline blade fuse holder + 7.5 A fuse": 1,
        "USB-C panel-mount extension (for the ESP32), 2 x M3 mounting holes 28 mm apart": 1,
        "Hook-and-loop strap 20 mm (battery)": 1,
        "Silicone wire 18 AWG red/black, 0.5 m each (battery, switch, fuse, bus power)": 1,
        "Dupont jumper wires 20 cm (displays share SCK/MOSI/DC/RST, own CS; IMU on I2C)": 30,
    }
    extra_bom.update({k: v for k, v in sorted(hw.items())})
    spec = RobotSpec(g.name, g, link_list, joints, parts, spawn_height, extra_bom, servo=g.servo)
    # extra information for the print exporter / README (not part of the RobotSpec interface)
    spec.part_mass = part_mass
    spec.part_notes = {p.name: p.note for p in mech.parts}
    spec.servo_ids = {sv.joint: sv.bus_id for sv in mech.servos}
    spec.servo_roles = {sv.joint: sv.role for sv in mech.servos}
    spec.hardware = mech.hardware
    spec.mech = mech
    spec.ankle_design = mech.ankle
    if g.paw_contact_stiffness > 0:
        spec.contact_compliance = {"tpu": (g.paw_contact_stiffness, g.paw_contact_damping)}
    spec.print_notes = PRINT_NOTES
    return spec


def _tf_m(center_mm) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(center_mm, float) * 0.001
    return T


PRINT_NOTES = [
    "Every STL is already laid flat in its print orientation (the face that goes on the bed is at z = 0); "
    "all parts print without supports. Check the first layer of the large shells in your slicer.",
    "PLA: 0.2 mm layers, 3 perimeters, 5 top/bottom layers, 15 % gyroid. The part masses in the table assume exactly that.",
    "TPU 95A (Polymaker PolyFlex TPU95; ankle springs, sole discs, paws, ears, collar) prints in its own jobs through "
    "Elegoo's TPU adapter (it cannot share a job with the CANVAS PLA): 225 C nozzle, 35-45 C textured PEI, fan on, "
    "20-30 mm/s (Polymaker: 210-230 C, bed 25-60 C, 30-50 mm/s), 0.2 mm layers, short slow retraction (~1 mm: long "
    "feed path), filament dried (70 C, 8 h).",
    "Ankle springs and sole discs are springs: print them SOLID (100 % infill, 4 walls) exactly as laid out (springs "
    "on their side, soles arms-down); their stiffness was computed for solid TPU. Paws, ears and collar: 3 walls, "
    "15 % infill.",
    "Servo case screws: M2 self-tapping into the 1.6 mm case holes (the screws that come with the servo fit if "
    "they are >= 8 mm long). Horn screws: M3 into the aluminium horns' 2.5 mm holes (tap M3 if they are not threaded).",
    "Heat-set inserts go in after printing (soldering iron at ~220 C). Holes are 4.0 mm for M3 x 5 x 4.0 inserts.",
    "Knee pushrods: press a 683ZZ bearing (3 x 7 x 3) into each eye; crank and lever pins are M3 button head screws "
    "into short heat-set inserts. Set the servo zero so that crank and lever are parallel at the rest pose.",
    "`servo_map.json` lists the bus id, joint, direction sign and limits of every servo for the firmware.",
]


# ---------------------------------------------------------------------- presets


def preset(style: str, name: str | None = None, servo: str = "STS3215_12V") -> DogGenome:
    """The three starting designs: proportions + colours."""
    if style == "terrier":
        return DogGenome(name=name or "terrier", style="terrier", servo=servo,
                         body=DogBody(0.210, 0.095, 0.075),
                         legs=DogLegs(hip_x=0.075, upper=0.105, lower=0.110, paw_radius=0.015),
                         head=DogHead(length=0.090, width=0.098, height=0.088, snout=0.055, ear_length=0.050, ear_style="up"),
                         tail=DogTail(0.085))
    if style == "bulldog":
        return DogGenome(name=name or "bulldog", style="bulldog", servo=servo,
                         body=DogBody(0.190, 0.130, 0.085),
                         legs=DogLegs(hip_x=0.065, upper=0.080, lower=0.085, paw_radius=0.019),
                         head=DogHead(length=0.095, width=0.135, height=0.100, snout=0.028, ear_length=0.035, ear_style="floppy"),
                         tail=DogTail(0.035))
    return DogGenome(name=name or "puppy", style="puppy", servo=servo)
