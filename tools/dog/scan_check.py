"""What does the height scan see? Dogs on the hardest stair tiles; print the scan along the body axis."""
import argparse, torch
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation
from core.evolution.train import build_robot_files
from core.morphology import Genome, build_robot
from core.sim.locomotion_env_cfg import make_env_cfg

parser = argparse.ArgumentParser(); add_launcher_args(parser); args = parser.parse_args([])
g = Path("prototypes/puppy/genome.json")
usd = build_robot_files(g.resolve(), Path("runs/scratch/scan_check/robot"))
spec = build_robot(Genome.load(g))
cfg = make_env_cfg(spec, str(usd.resolve()).replace("\\", "/"), 12, False, task="command", terrain="dog_rough",
                   sensors="imu", obs_history=8, height_scan=True)
cfg.reset_yaw_range = 0.0
with launch_simulation(cfg, args):
    from core.sim.locomotion_env import EvoLocomotionEnv
    env = EvoLocomotionEnv(cfg)
    t = env.terrain
    gen = cfg.terrain.terrain_generator
    names = list(gen.sub_terrains)
    cum = torch.cumsum(torch.tensor([gen.sub_terrains[n].proportion for n in names]), 0); cum = cum / cum[-1]
    col_type = [names[int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
    climb = [c for c, n in enumerate(col_type) if n == "stairs_climb"][0]
    flat = [c for c, n in enumerate(col_type) if n == "bumpy"][0]
    env.curriculum_enabled = False
    t.terrain_levels[:] = 9
    t.terrain_types[:6] = climb
    t.terrain_types[6:] = flat
    t.terrain_levels[6:] = 0
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
    env.command_override = torch.zeros(3, device=env.device)
    obs, _ = env.reset()
    for _ in range(10):
        obs, *_ = env.step(torch.zeros(env.num_envs, cfg.action_space, device=env.device))
    sd = env.height_scanner.data
    print("RAW sensor pos z:", [round(v, 3) for v in sd.pos_w.torch[[0, 6], 2].tolist()],
          "hit z (first 5):", [[round(v, 3) for v in sd.ray_hits_w.torch[i, :5, 2].tolist()] for i in (0, 6)],
          "offset", cfg.height_scan_offset, "root z", [round(v, 3) for v in env.robot.data.root_pos_w.torch[[0, 6], 2].tolist()],
          "origin z", [round(v, 3) for v in t.env_origins[[0, 6], 2].tolist()])
    n_scan = cfg.height_scan_points
    scan = obs["policy"][:, -n_scan:]
    pat = cfg.height_scanner.pattern_cfg
    nx = round(pat.size[0] / pat.resolution) + 1
    ny = round(pat.size[1] / pat.resolution) + 1
    grid = scan.reshape(-1, ny, nx)  # "xy" ordering: x is the inner loop
    for i, label in ((0, "stairs_climb level 9"), (6, "bumpy level 0")):
        print(f"{label}: scan min {grid[i].min():.2f} max {grid[i].max():.2f} (x {cfg.height_scan_clip:.2f} m); "
              f"centre row along x (back -> front):")
        print("   ", " ".join(f"{v:+.2f}" for v in grid[i, ny // 2].tolist()))
    env.close()
