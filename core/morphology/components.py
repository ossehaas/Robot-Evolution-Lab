"""Physical constants of the off-the-shelf parts and print materials the robots are built from.

Everything is in SI units (m, kg, s, rad, N*m). The values are the ones the simulator uses, so a robot
that walks in simulation is being asked to do so with the torque, speed and mass of the real hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ServoSpec:
    """A hobby position servo (body dimensions exclude the mounting tabs)."""

    name: str
    body_length: float  # along the long axis, the shaft sits near one end
    body_width: float
    body_height: float  # bottom to top face, shaft excluded
    tab_span: float  # total length across both mounting tabs
    tab_thickness: float
    tab_from_top: float  # distance from the top face down to the top of the tabs
    tab_hole_spacing: float  # centre-to-centre distance of the two tab screw holes
    shaft_offset: float  # shaft centre measured from the near end of the body
    shaft_height: float  # top face to the horn seating plane
    mass: float
    stall_torque: float
    max_speed: float  # no-load speed
    armature: float  # reflected rotor inertia of the gearbox at the output shaft
    friction: float  # static friction of the gear train at the output shaft
    travel: float  # total mechanical travel
    # PD loop of the simulated servo: saturates after ~0.1 rad of error, near-critically damped against the armature
    kp: float = 2.0  # N*m/rad
    kd: float = 0.05  # N*m*s/rad

    @property
    def usable_torque(self) -> float:
        """Torque the servo can sustain; stall torque is only reached momentarily."""
        return 0.8 * self.stall_torque


# TowerPro SG90 datasheet: 1.8 kgf*cm at 4.8 V, 0.1 s/60 deg, 9 g.
# Armature: coreless motor rotor ~3e-9 kg*m^2 through a ~1:300 gear train -> ~3e-4 kg*m^2 at the horn.
SG90 = ServoSpec(
    name="SG90",
    body_length=0.0222,
    body_width=0.0118,
    body_height=0.0227,
    tab_span=0.0322,
    tab_thickness=0.0025,
    tab_from_top=0.0068,
    tab_hole_spacing=0.0278,
    shaft_offset=0.0059,
    shaft_height=0.0040,
    mass=0.009,
    stall_torque=1.8 * 9.81 / 100.0,
    max_speed=math.radians(60.0) / 0.1,
    armature=3.0e-4,
    friction=0.01,
    travel=math.radians(180.0),
)

# Other micro servos with the same mounting footprint (~23 x 12 mm), so the printed parts stay the same. Values are
# manufacturer datasheet figures at the supply voltage noted (cheap clones are often somewhat weaker). Running at 6 V
# means setting the buck converter to 6 V instead of 5 V.
MG90S = replace(SG90, name="MG90S (6 V)", mass=0.0134, stall_torque=2.2 * 9.81 / 100.0,
                max_speed=math.radians(60.0) / 0.08)  # metal gears
MG92B = replace(SG90, name="MG92B (6 V)", mass=0.0138, stall_torque=3.5 * 9.81 / 100.0,
                max_speed=math.radians(60.0) / 0.08, body_height=0.0250)  # metal gears, digital, a bit taller
SCS0009 = replace(SG90, name="Feetech SCS0009 (6 V)", mass=0.0132, stall_torque=2.3 * 9.81 / 100.0,
                  max_speed=math.radians(60.0) / 0.10, travel=math.radians(300.0))  # serial bus, position feedback

# Standard-size serial bus servos for the dog (experiment 2): 45.2 x 24.7 x 35 mm, 55 g, 12-bit magnetic encoder
# (position, speed, load and temperature feedback), daisy-chained on one bus. Vendor figures: 19.5 kgf*cm at 7.4 V,
# 30 kgf*cm at 12 V, ~0.22 s/60 deg. Armature: ~1:345 gearbox on a cored motor (rotor ~1e-7 kg*m^2) -> ~0.01 kg*m^2
# at the output; an estimate, to be measured on the real servo.
STS3215 = replace(
    SG90, name="Feetech STS3215 (7.4 V)", body_length=0.0452, body_width=0.0247, body_height=0.0350, tab_span=0.0452,
    tab_thickness=0.0, tab_from_top=0.0, tab_hole_spacing=0.0, shaft_offset=0.0110, shaft_height=0.0050, mass=0.055,
    stall_torque=19.5 * 9.81 / 100.0, max_speed=math.radians(60.0) / 0.24, armature=0.01, friction=0.05,
    travel=math.radians(360.0), kp=20.0, kd=0.6,
)
STS3215_12V = replace(STS3215, name="Feetech STS3215 (12 V)", stall_torque=30.0 * 9.81 / 100.0,
                      max_speed=math.radians(60.0) / 0.222, kp=30.0, kd=0.8)
# Hypothetical comparison actuator (simulation only): a small brushless motor with a ~6:1 planetary gearbox, as in
# research robot dogs (quasi-direct drive): fast, about as strong, and with a tiny reflected inertia. Same mass and
# size as the STS3215 here, so only the drive differs.
QDD = replace(STS3215_12V, name="quasi-direct-drive BLDC (hypothetical)", stall_torque=3.0, max_speed=30.0,
              armature=3e-4, kp=20.0, kd=0.3)
SERVOS = {"SG90": SG90, "MG90S": MG90S, "MG92B": MG92B, "SCS0009": SCS0009, "STS3215": STS3215,
          "STS3215_12V": STS3215_12V, "QDD": QDD}


@dataclass(frozen=True)
class PrintMaterial:
    name: str
    density: float  # of the solid filament [kg/m^3]
    infill: float  # effective fill fraction of a printed part (walls + infill)
    static_friction: float  # against a typical table / floor surface
    dynamic_friction: float
    restitution: float

    @property
    def effective_density(self) -> float:
        return self.density * self.infill


PLA = PrintMaterial("PLA", density=1240.0, infill=0.55, static_friction=0.45, dynamic_friction=0.35, restitution=0.1)
TPU = PrintMaterial("TPU", density=1210.0, infill=0.8, static_friction=1.0, dynamic_friction=0.85, restitution=0.0)
MATERIALS = {"pla": PLA, "tpu": TPU}


@dataclass(frozen=True)
class BoxComponent:
    """An electronics module approximated by a solid box."""

    name: str
    size: tuple[float, float, float]
    mass: float


ESP32_S3 = BoxComponent("ESP32-S3 SuperMini", (0.0230, 0.0180, 0.0045), 0.0025)
SERVO_DRIVER = BoxComponent("PCA9685 servo driver", (0.0250, 0.0620, 0.0030), 0.0090)
IMU = BoxComponent("MPU6050 IMU", (0.0210, 0.0160, 0.0030), 0.0020)
BATTERY_2S = BoxComponent("LiPo 2S 350 mAh", (0.0450, 0.0180, 0.0140), 0.0230)
BUCK = BoxComponent("5 V 3 A buck converter", (0.0220, 0.0170, 0.0060), 0.0040)
WIRING_MASS = 0.006  # servo leads, headers, screws

# dog (experiment 2)
BUS_ADAPTER = BoxComponent("Bus servo adapter (Feetech FE-URT-1 or Waveshare)", (0.040, 0.030, 0.012), 0.012)
BATTERY_3S = BoxComponent("LiPo 3S 1500 mAh (12 V servos)", (0.075, 0.035, 0.026), 0.125)
BATTERY_2S_DOG = BoxComponent("LiPo 2S 2200 mAh (7.4 V servos)", (0.105, 0.034, 0.018), 0.115)
ROUND_DISPLAY = BoxComponent("GC9A01 1.28 in round display (eye)", (0.004, 0.036, 0.036), 0.006)

# Fabrication constants used by the part generator.
PLATE_THICKNESS = 0.0035  # printed link plates
LINK_WIDTH = 0.012  # width of a link plate away from the servo pocket
WALL = 0.0020  # chassis wall thickness
POCKET_CLEARANCE = 0.0003  # per side, so the servo slides in
SCREW_HOLE = 0.0018  # M2 self-tapping pilot hole
HORN_HUB_HOLE = 0.0030  # clearance for the horn screw head / hub
HORN_SCREW_RADIUS = 0.0070  # radial position of the screws fixing a round horn to a plate
