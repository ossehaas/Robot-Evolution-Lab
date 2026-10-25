"""The genome: a compact, evolvable description of a mirror-symmetric servo robot.

A robot is a box-shaped chassis (holding battery and electronics) with one or more *limb pairs*. Each pair
is mirrored left/right and is a chain of *segments*; every segment is a printed link driven by one servo at
its proximal end. Lengths are in metres, angles in degrees (the JSON stays human-editable).

Geometric rules that keep every genome buildable and printable:
    * a yaw segment rotates about the vertical axis, so it is always horizontal (``elev_deg`` is forced to 0)
    * a pitch segment rotates about the horizontal axis perpendicular to the limb, so it may slope up/down
    * limb pairs sit at least ``MIN_PAIR_SPACING`` apart along the chassis so their hip servos fit
    * the total servo count is capped at ``MAX_DOF`` (the PCA9685 driver has 16 channels, power sets 12)
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

AXES = ("yaw", "pitch")
FOOT_MATERIALS = ("tpu", "pla")

MAX_DOF = 12
MAX_PAIRS = 4
MAX_SEGMENTS = 3
MIN_PAIR_SPACING = 0.030

# (min, max) bounds, used for sampling, mutation and clamping
BOUNDS = {
    "body.length": (0.060, 0.180),
    "body.width": (0.040, 0.100),
    "body.height": (0.022, 0.045),
    "pair.x": (-0.95, 0.95),  # fraction of the chassis half-length
    "pair.z": (-0.4, 0.4),  # fraction of the chassis half-height
    "pair.splay_deg": (-60.0, 60.0),
    "seg.length": (0.025, 0.090),
    "seg.elev_deg": (-80.0, 30.0),
    "seg.range_deg": (20.0, 90.0),
    "foot.radius": (0.004, 0.010),
}


def _clamp(value: float, key: str) -> float:
    lo, hi = BOUNDS[key]
    return min(max(value, lo), hi)


@dataclass
class Body:
    length: float = 0.100
    width: float = 0.060
    height: float = 0.030


@dataclass
class Segment:
    axis: str = "pitch"
    length: float = 0.045
    elev_deg: float = 0.0  # slope of the segment in the vertical plane of the limb, negative = downward
    range_deg: float = 60.0  # symmetric joint travel around the rest pose


@dataclass
class Foot:
    radius: float = 0.006
    material: str = "tpu"


@dataclass
class LimbPair:
    x: float = 0.0
    z: float = 0.0
    splay_deg: float = 0.0  # rotation of the limb about the vertical, positive points it forward
    segments: list[Segment] = field(default_factory=lambda: [Segment()])
    foot: Foot = field(default_factory=Foot)


@dataclass
class Genome:
    name: str = "robot"
    body: Body = field(default_factory=Body)
    limb_pairs: list[LimbPair] = field(default_factory=lambda: [LimbPair()])
    servo: str = "SG90"  # servo type for all joints, see components.SERVOS
    lineage: list[str] = field(default_factory=list)  # names of the parents, for bookkeeping only

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> Genome:
        pairs = []
        for p in data.get("limb_pairs", []):
            pairs.append(
                LimbPair(
                    x=p.get("x", 0.0),
                    z=p.get("z", 0.0),
                    splay_deg=p.get("splay_deg", 0.0),
                    segments=[Segment(**s) for s in p.get("segments", [])],
                    foot=Foot(**p.get("foot", {})),
                )
            )
        genome = Genome(
            name=data.get("name", "robot"),
            body=Body(**data.get("body", {})),
            limb_pairs=pairs,
            servo=data.get("servo", "SG90"),
            lineage=list(data.get("lineage", [])),
        )
        genome.repair()
        return genome

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> Genome:
        """Load a genome; files with "kind": "dog" give a DogGenome (core/morphology/dog.py)."""
        data = json.loads(Path(path).read_text())
        if data.get("kind") == "dog":
            from .dog import DogGenome

            return DogGenome.from_dict(data)
        return Genome.from_dict(data)

    # ------------------------------------------------------------------ properties

    @property
    def num_dof(self) -> int:
        return 2 * sum(len(p.segments) for p in self.limb_pairs)

    def topology_key(self) -> str:
        """Identifies the joint structure; policies can only be transferred between equal keys."""
        return "-".join("".join(s.axis[0] for s in p.segments) for p in self.limb_pairs)

    # ------------------------------------------------------------------ constraints

    def repair(self) -> Genome:
        """Clamp every value into its bounds and enforce the buildability rules (in place)."""
        from .components import SERVOS

        if self.servo not in SERVOS:
            self.servo = "SG90"
        b = self.body
        b.length = _clamp(b.length, "body.length")
        b.width = _clamp(b.width, "body.width")
        b.height = _clamp(b.height, "body.height")
        if not self.limb_pairs:
            self.limb_pairs = [LimbPair()]
        for pair in self.limb_pairs:
            pair.x = _clamp(pair.x, "pair.x")
            pair.z = _clamp(pair.z, "pair.z")
            pair.splay_deg = _clamp(pair.splay_deg, "pair.splay_deg")
            if not pair.segments:
                pair.segments = [Segment()]
            pair.segments = pair.segments[:MAX_SEGMENTS]
            for seg in pair.segments:
                if seg.axis not in AXES:
                    seg.axis = "pitch"
                seg.length = _clamp(seg.length, "seg.length")
                seg.range_deg = _clamp(seg.range_deg, "seg.range_deg")
                seg.elev_deg = 0.0 if seg.axis == "yaw" else _clamp(seg.elev_deg, "seg.elev_deg")
            pair.foot.radius = _clamp(pair.foot.radius, "foot.radius")
            if pair.foot.material not in FOOT_MATERIALS:
                pair.foot.material = "tpu"
        self.limb_pairs = self.limb_pairs[:MAX_PAIRS]
        # drop trailing segments until the servo budget fits
        while self.num_dof > MAX_DOF:
            longest = max(self.limb_pairs, key=lambda p: len(p.segments))
            if len(longest.segments) > 1:
                longest.segments.pop()
            else:
                self.limb_pairs.pop()
        self._space_pairs()
        return self

    def _space_pairs(self) -> None:
        """Spread limb pairs along the chassis so neighbouring hip servos do not overlap."""
        half = self.body.length / 2.0
        min_frac = MIN_PAIR_SPACING / half
        max_pairs = int(1.9 / min_frac) + 1
        self.limb_pairs.sort(key=lambda p: -p.x)  # front to back
        self.limb_pairs = self.limb_pairs[:max(1, max_pairs)]
        # forward pass then backward pass keeps the ordering and the [-0.95, 0.95] range
        for i in range(1, len(self.limb_pairs)):
            prev = self.limb_pairs[i - 1].x
            self.limb_pairs[i].x = min(self.limb_pairs[i].x, prev - min_frac)
        for i in range(len(self.limb_pairs) - 1, -1, -1):
            lower = -0.95 if i == len(self.limb_pairs) - 1 else self.limb_pairs[i + 1].x + min_frac
            self.limb_pairs[i].x = max(self.limb_pairs[i].x, lower)
        for pair in self.limb_pairs:
            pair.x = round(min(pair.x, 0.95), 4)


# ---------------------------------------------------------------------- variation operators


def _gauss(rng: random.Random, value: float, key: str, rel_sigma: float) -> float:
    lo, hi = BOUNDS[key]
    return _clamp(value + rng.gauss(0.0, rel_sigma * (hi - lo)), key)


def random_segment(rng: random.Random) -> Segment:
    axis = rng.choice(AXES)
    return Segment(
        axis=axis,
        length=rng.uniform(*BOUNDS["seg.length"]),
        elev_deg=0.0 if axis == "yaw" else rng.uniform(-75.0, 10.0),
        range_deg=rng.uniform(35.0, 80.0),
    )


def random_genome(rng: random.Random, name: str = "random") -> Genome:
    """Sample a buildable genome; biased towards shapes that have a chance to stand."""
    num_pairs = rng.choice([1, 2, 2, 2, 3, 3, 4])
    pairs = []
    for i in range(num_pairs):
        x = 0.8 - 1.6 * i / max(1, num_pairs - 1) if num_pairs > 1 else 0.0
        segments = [random_segment(rng) for _ in range(rng.choice([1, 2, 2, 3]))]
        # a downward-reaching last segment makes standing possible from generation zero
        if segments[-1].axis == "pitch" and rng.random() < 0.7:
            segments[-1].elev_deg = rng.uniform(-80.0, -40.0)
        pairs.append(
            LimbPair(
                x=x + rng.uniform(-0.1, 0.1),
                z=rng.uniform(-0.3, 0.3),
                splay_deg=rng.uniform(-40.0, 40.0),
                segments=segments,
                foot=Foot(radius=rng.uniform(*BOUNDS["foot.radius"]), material=rng.choice(["tpu", "tpu", "pla"])),
            )
        )
    genome = Genome(
        name=name,
        body=Body(
            length=rng.uniform(*BOUNDS["body.length"]),
            width=rng.uniform(*BOUNDS["body.width"]),
            height=rng.uniform(*BOUNDS["body.height"]),
        ),
        limb_pairs=pairs,
    )
    return genome.repair()


@dataclass
class MutationRates:
    param_prob: float = 0.35  # chance that an individual continuous gene is perturbed
    rel_sigma: float = 0.12  # perturbation std as a fraction of the gene's range
    add_pair: float = 0.10
    remove_pair: float = 0.08
    add_segment: float = 0.15
    remove_segment: float = 0.12
    flip_axis: float = 0.10
    swap_foot_material: float = 0.06


def mutate(genome: Genome, rng: random.Random, rates: MutationRates | None = None, name: str | None = None) -> Genome:
    """Return a mutated copy: small continuous changes plus occasional structural changes."""
    r = rates or MutationRates()
    child = copy.deepcopy(genome)
    child.name = name or f"{genome.name}_m"
    child.lineage = [genome.name]

    def maybe(value: float, key: str) -> float:
        return _gauss(rng, value, key, r.rel_sigma) if rng.random() < r.param_prob else value

    b = child.body
    b.length, b.width, b.height = maybe(b.length, "body.length"), maybe(b.width, "body.width"), maybe(b.height, "body.height")
    for pair in child.limb_pairs:
        pair.x = maybe(pair.x, "pair.x")
        pair.z = maybe(pair.z, "pair.z")
        pair.splay_deg = maybe(pair.splay_deg, "pair.splay_deg")
        pair.foot.radius = maybe(pair.foot.radius, "foot.radius")
        if rng.random() < r.swap_foot_material:
            pair.foot.material = "pla" if pair.foot.material == "tpu" else "tpu"
        for seg in pair.segments:
            seg.length = maybe(seg.length, "seg.length")
            seg.elev_deg = maybe(seg.elev_deg, "seg.elev_deg")
            seg.range_deg = maybe(seg.range_deg, "seg.range_deg")
            if rng.random() < r.flip_axis:
                seg.axis = "yaw" if seg.axis == "pitch" else "pitch"
                if seg.axis == "pitch":
                    seg.elev_deg = rng.uniform(-60.0, 0.0)

    # structural mutations
    if rng.random() < r.add_pair and len(child.limb_pairs) < MAX_PAIRS:
        donor = copy.deepcopy(rng.choice(child.limb_pairs))
        donor.x = rng.uniform(-0.9, 0.9)
        child.limb_pairs.append(donor)
    if rng.random() < r.remove_pair and len(child.limb_pairs) > 1:
        child.limb_pairs.pop(rng.randrange(len(child.limb_pairs)))
    if rng.random() < r.add_segment:
        pair = rng.choice(child.limb_pairs)
        if len(pair.segments) < MAX_SEGMENTS:
            new = random_segment(rng)
            new.axis, new.elev_deg = "pitch", rng.uniform(-80.0, -30.0)
            pair.segments.append(new)
    if rng.random() < r.remove_segment:
        pair = rng.choice(child.limb_pairs)
        if len(pair.segments) > 1:
            pair.segments.pop(rng.randrange(len(pair.segments)))
    return child.repair()


def crossover(a: Genome, b: Genome, rng: random.Random, name: str | None = None) -> Genome:
    """Uniform crossover over whole limb pairs (pairs are aligned front-to-back) and body dimensions."""
    child = Genome(name=name or f"{a.name}x{b.name}", lineage=[a.name, b.name])
    child.body = copy.deepcopy(rng.choice([a.body, b.body]))
    num_pairs = rng.choice([len(a.limb_pairs), len(b.limb_pairs)])
    pairs = []
    for i in range(num_pairs):
        options = [g.limb_pairs[i] for g in (a, b) if i < len(g.limb_pairs)]
        pairs.append(copy.deepcopy(rng.choice(options)))
    child.limb_pairs = pairs
    return child.repair()
