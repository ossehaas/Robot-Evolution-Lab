"""Rough terrain sized for palm-sized robots (chassis ~5 cm off the ground, legs ~8 cm long).

Isaac Lab's stock rough terrain is made for 50 cm tall quadrupeds (stairs up to 23 cm). Here every feature is
scaled down: at the hardest level stairs are 3 cm, blocks 2.5 cm, bumps 1.2 cm and slopes ~19 degrees.

The terrain is a grid of 2 x 2 m tiles: rows are difficulty levels (0 = almost flat, 9 = hardest), columns mix
the terrain types. With the curriculum enabled, a robot that follows its commands well on its tile moves one
row up at the next reset, a robot that falls or lags moves one row down.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

NUM_LEVELS = 10


def scaled_bumps(difficulty: float, cfg):
    """Random bumps whose height grows with difficulty (Isaac Lab's version ignores difficulty, which would put
    the full roughness on the easiest level)."""
    from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain

    lo, hi = cfg.noise_range
    top = max(lo + (hi - lo) * difficulty, lo + cfg.noise_step)
    return random_uniform_terrain(difficulty, cfg.replace(noise_range=(lo, top)))


def tread_stairs(stairs_function, wide: float, narrow: float):
    """Stairs whose treads shorten as the difficulty grows: wide treads (a curb, all four paws fit on one step) on the
    easy rows, ``narrow`` ones on the hardest (the dog spans several steps and climbs continuously)."""

    def stairs(difficulty: float, cfg):
        return stairs_function(difficulty, cfg.replace(step_width=wide + (narrow - wide) * difficulty))

    return stairs


SMALL_ROBOT_ROUGH = TerrainGeneratorCfg(
    size=(2.0, 2.0),
    border_width=4.0,  # flat margin around the grid: a dog walks ~4 m in an episode and must not walk off the world
    num_rows=NUM_LEVELS,
    num_cols=16,
    horizontal_scale=0.02,  # 2 cm height-field cells: a foot is ~1.2 cm across
    vertical_scale=0.001,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "bumpy": terrain_gen.HfRandomUniformTerrainCfg(
            function=scaled_bumps, proportion=0.25, noise_range=(0.0, 0.012), noise_step=0.002, border_width=0.1
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.35), platform_width=0.5, border_width=0.1
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.35), platform_width=0.5, border_width=0.1
        ),
        # robots start on the top platform: walking outwards goes down the steps
        "stairs_descend": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.002, 0.03), step_width=0.12, platform_width=0.5, border_width=0.1,
            holes=False, convert_to_heightfield=True,
        ),
        # robots start at the bottom of a pit: walking outwards climbs the steps
        "stairs_climb": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.002, 0.03), step_width=0.12, platform_width=0.5, border_width=0.1,
            holes=False, convert_to_heightfield=True,
        ),
        "blocks": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.075, grid_height_range=(0.002, 0.025), platform_width=0.5,
            convert_to_heightfield=True,
        ),
    },
)


TERRAIN_TYPES = tuple(SMALL_ROBOT_ROUGH.sub_terrains)


# Rough terrain for the dog (experiment 2, ~36 cm long, hips 16 cm up): 3 x 3 m tiles. Stairs scaled like a Labrador
# on human stairs (rise ~1/3 of hip height): up to 6 cm rise with 15 cm treads at the hardest level; bumps up to
# 2.5 cm, blocks up to 4 cm, slopes up to 20 degrees.
DOG_ROUGH = TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=5.0,
    num_rows=NUM_LEVELS,
    num_cols=12,
    horizontal_scale=0.02,
    vertical_scale=0.001,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "bumpy": terrain_gen.HfRandomUniformTerrainCfg(
            function=scaled_bumps, proportion=0.2, noise_range=(0.0, 0.025), noise_step=0.003, border_width=0.1
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.36), platform_width=0.8, border_width=0.1
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.36), platform_width=0.8, border_width=0.1
        ),
        "stairs_descend": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.005, 0.06), step_width=0.15, platform_width=0.8, border_width=0.1,
            holes=False, convert_to_heightfield=True,
        ),
        "stairs_climb": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.005, 0.06), step_width=0.15, platform_width=0.8, border_width=0.1,
            holes=False, convert_to_heightfield=True,
        ),
        "blocks": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.11, grid_height_range=(0.005, 0.04), platform_width=0.8,
            convert_to_heightfield=True,
        ),
    },
)


def rough_terrain_importer(max_init_level: int = 2, types: list[str] | None = None, dog: bool = False) -> TerrainImporterCfg:
    """Importer for the rough terrain (``dog``: the dog-sized version); ``types`` restricts it to some terrain types
    (e.g. a stairs specialist)."""
    generator = DOG_ROUGH if dog else SMALL_ROBOT_ROUGH
    if types:
        unknown = set(types) - set(generator.sub_terrains)
        if unknown:
            raise ValueError(f"unknown terrain types {sorted(unknown)}, choose from {tuple(generator.sub_terrains)}")
        generator = generator.replace(sub_terrains={k: v for k, v in generator.sub_terrains.items() if k in types})
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=max_init_level,
        collision_group=-1,
        # the robot's own materials decide the friction (multiply with 1.0)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.36), roughness=0.9),
        debug_vis=False,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Obstacle course (roadmap M3): flat floor with pillars too tall to climb; difficulty = how many pillars.
# Goals are picked from fixed spots on a ring around the start, which are kept clear so every goal is reachable.

import math  # noqa: E402

import numpy as np  # noqa: E402
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg  # noqa: E402
from isaaclab.terrains.height_field.utils import height_field_to_mesh  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

OBSTACLE_TILE = 3.0  # [m]
GOAL_RING_RADIUS = 1.0  # [m] goals sit on this ring around the tile centre (where robots start)
GOAL_SPOTS = tuple(
    (GOAL_RING_RADIUS * math.cos(2 * math.pi * k / 8), GOAL_RING_RADIUS * math.sin(2 * math.pi * k / 8)) for k in range(8)
)


@height_field_to_mesh
def _obstacle_heights(difficulty: float, cfg) -> np.ndarray:
    """Pillars of fixed height at random places; the start area and the goal spots stay clear."""
    nx, ny = int(cfg.size[0] / cfg.horizontal_scale), int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((nx, ny))
    count = int(round(cfg.num_obstacles_range[0] + difficulty * (cfg.num_obstacles_range[1] - cfg.num_obstacles_range[0])))
    height = int(cfg.obstacle_height / cfg.vertical_scale)
    wmin, wmax = (int(w / cfg.horizontal_scale) for w in cfg.obstacle_width_range)
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    # cell centres in metres relative to the tile centre
    cx, cy = (xs + 0.5) * cfg.horizontal_scale - cfg.size[0] / 2, (ys + 0.5) * cfg.horizontal_scale - cfg.size[1] / 2
    keep_clear = np.hypot(cx, cy) < cfg.start_clear_radius
    for gx, gy in GOAL_SPOTS:
        keep_clear |= np.hypot(cx - gx, cy - gy) < cfg.goal_clear_radius
    rng = np.random.default_rng()
    if cfg.floor_bump_max > 0:
        # bumpy floor: random heights that grow with difficulty (2 cm cells, like the rough-terrain bumps)
        top = int(cfg.floor_bump_max * difficulty / cfg.vertical_scale)
        if top > 0:
            hf += rng.integers(0, top + 1, size=hf.shape)
    for _ in range(count):
        w, l = rng.integers(wmin, wmax + 1, size=2)
        x0, y0 = rng.integers(0, nx - w), rng.integers(0, ny - l)
        hf[x0 : x0 + w, y0 : y0 + l] = height
    hf[keep_clear] = 0  # flat start area and goal spots (the robot starts at floor height 0)
    return np.rint(hf).astype(np.int16)


def obstacle_field(difficulty: float, cfg):
    """Mesh + start position. The helper takes the highest point within 1 m of the centre as start height, which
    would be a pillar top; robots start on the cleared floor, so the start height is the floor (0)."""
    meshes, origin = _obstacle_heights(difficulty, cfg)
    origin[2] = 0.0
    return meshes, origin


@configclass
class HfObstacleFieldCfg(HfTerrainBaseCfg):
    function = obstacle_field
    num_obstacles_range: tuple[int, int] = (4, 60)
    obstacle_width_range: tuple[float, float] = (0.06, 0.14)  # [m]
    obstacle_height: float = 0.10  # [m] taller than the robot: must be walked around
    start_clear_radius: float = 0.30  # [m]
    goal_clear_radius: float = 0.15  # [m]
    floor_bump_max: float = 0.0  # [m] at the hardest level; 0 = flat floor


OBSTACLE_COURSE = TerrainGeneratorCfg(
    size=(OBSTACLE_TILE, OBSTACLE_TILE),
    border_width=1.0,
    num_rows=NUM_LEVELS,
    num_cols=8,
    horizontal_scale=0.02,
    vertical_scale=0.001,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={"pillars": HfObstacleFieldCfg(proportion=1.0)},
)


OBSTACLE_COURSE_ROUGH = OBSTACLE_COURSE.replace(
    sub_terrains={"pillars_bumpy": HfObstacleFieldCfg(proportion=1.0, floor_bump_max=0.012)}
)


def obstacle_terrain_importer(max_init_level: int = 1, rough_floor: bool = False) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBSTACLE_COURSE_ROUGH if rough_floor else OBSTACLE_COURSE,
        max_init_terrain_level=max_init_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.36), roughness=0.9),
        debug_vis=False,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Enclosure (unreachable-goal test): a closed ring wall in the middle of each tile. The task puts the goal inside the
# ring (off centre) or somewhere outside it; robots start outside. Nothing changes with difficulty.

ENCLOSURE_RADIUS = 0.40  # [m] outer surface of the wall, around the tile centre
ENCLOSURE_WALL = 0.06  # [m] wall thickness


@height_field_to_mesh
def _enclosure_heights(difficulty: float, cfg) -> np.ndarray:
    nx, ny = int(cfg.size[0] / cfg.horizontal_scale), int(cfg.size[1] / cfg.horizontal_scale)
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = (xs + 0.5) * cfg.horizontal_scale - cfg.size[0] / 2, (ys + 0.5) * cfg.horizontal_scale - cfg.size[1] / 2
    r = np.hypot(cx, cy)
    wall = (r <= cfg.radius) & (r >= cfg.radius - cfg.wall_thickness)
    if cfg.cup_arcs:
        # open ring ("cup"): only an arc of the wall, centred on -x; the row (difficulty level) picks the arc length
        arc = cfg.cup_arcs[min(int(difficulty * len(cfg.cup_arcs)), len(cfg.cup_arcs) - 1)]
        from_left = np.abs(np.angle(np.exp(1j * (np.arctan2(cy, cx) - np.pi))))  # angle away from -x, 0..pi
        wall &= from_left <= np.deg2rad(arc) / 2
    if cfg.pocket_depth > 0:
        # a slot from the outside inwards along +x: its bottom is closer to the centre than the rest of the wall
        t, half, bottom = cfg.wall_thickness, cfg.pocket_width / 2, cfg.radius - cfg.pocket_depth
        block = (cx >= bottom - t) & (np.abs(cy) <= half + t) & (r <= cfg.radius + 0.01)
        slot = (cx >= bottom) & (np.abs(cy) <= half)
        wall = (wall | block) & ~slot
    return np.where(wall, int(cfg.wall_height / cfg.vertical_scale), 0).astype(np.int16)


def enclosure(difficulty: float, cfg):
    meshes, origin = _enclosure_heights(difficulty, cfg)
    origin[2] = 0.0  # the floor, not the wall top
    return meshes, origin


@configclass
class HfEnclosureCfg(HfTerrainBaseCfg):
    function = enclosure
    radius: float = ENCLOSURE_RADIUS
    wall_thickness: float = ENCLOSURE_WALL
    wall_height: float = 0.10  # [m] like the pillars: cannot be climbed
    pocket_width: float = 0.0  # [m] free width of the slot (0 = plain ring)
    pocket_depth: float = 0.0  # [m] from the outer radius inwards
    cup_arcs: tuple[float, ...] = ()  # [deg] open ring: wall arc per difficulty row, closed side towards -x


POCKET_RADIUS, POCKET_WIDTH, POCKET_DEPTH = 0.50, 0.30, 0.30


ENCLOSURE_COURSE = OBSTACLE_COURSE.replace(num_rows=4, curriculum=False, sub_terrains={"ring": HfEnclosureCfg(proportion=1.0)})


ENCLOSURE_POCKET_COURSE = ENCLOSURE_COURSE.replace(
    sub_terrains={"ring": HfEnclosureCfg(proportion=1.0, radius=POCKET_RADIUS, pocket_width=POCKET_WIDTH, pocket_depth=POCKET_DEPTH)}
)


# Detour test: a goal inside an open ring ("cup") whose open side faces away from the robot. Walking straight at the
# goal ends against the closed side; the way in is round the back. Rows = longer arcs (180 = a half circle).
CUP_RADIUS = 0.45
CUP_ARCS = (60.0, 90.0, 120.0, 150.0, 180.0, 210.0)
CUP_COURSE = ENCLOSURE_COURSE.replace(
    num_rows=len(CUP_ARCS), curriculum=True,
    sub_terrains={"ring": HfEnclosureCfg(proportion=1.0, radius=CUP_RADIUS, cup_arcs=CUP_ARCS)},
)


def enclosure_terrain_importer(pocket: bool = False, cup: bool = False) -> TerrainImporterCfg:
    if cup:
        return obstacle_terrain_importer().replace(terrain_generator=CUP_COURSE, max_init_terrain_level=1)
    course = ENCLOSURE_POCKET_COURSE if pocket else ENCLOSURE_COURSE
    return obstacle_terrain_importer().replace(terrain_generator=course, max_init_terrain_level=None)
