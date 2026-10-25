# Evolution lab

Design small servo-driven robots, teach them to walk in simulation, let evolution improve their bodies, then
3D-print the winners.

```
genome.json ──► core.morphology ──► USD (simulation) + URDF + printable STL parts + BOM + preview.png
                                         │
                                         ▼
                 core.sim + core.evolution.train ──► PPO on 4096 copies in Isaac Lab ──► policy + fitness
                                         │
                                         ▼
                           core.evolution.evolve ──► mutate / cross the best bodies, repeat
```

Every robot is built from real parts: SG90 micro servos (torque-speed curve, gearbox inertia, 20 ms control
latency), PLA links and TPU feet (masses and friction), an ESP32-S3, PCA9685 servo driver, MPU6050 IMU and a
2S LiPo. What walks in simulation is being asked to do so with the hardware you can actually buy.

## Quick start

All commands run from this folder in PowerShell or cmd. `lab.cmd` is Python from the Isaac Lab environment.

```bat
:: 1. health check: GPU physics works (drops a cube, prints PASS)
lab test_sim.py

:: 2. build a robot from its genome: USD, URDF, STL parts, BOM, preview.png
lab -m core.morphology build prototypes\crawler_q4\genome.json

:: 3. teach it to walk (~10-30 min on the RTX 5080), then watch it
lab -m core.evolution.train --genome prototypes\crawler_q4\genome.json --out runs\try\crawler_q4
lab -m core.evolution.play --run_dir runs\try\crawler_q4
lab -m core.evolution.play --run_dir runs\try\crawler_q4 --video --num_envs 4

:: 3b. steerable walking: learn to follow velocity commands, then drive it
lab -m core.evolution.train --genome prototypes\crawler_q4\genome.json --out runs\try\crawler_q4_steer --task command --iterations 800
:: drive it with the keyboard (arrows = move, Z/X = turn, L = stop), or record a captioned scripted tour
lab -m core.evolution.play --run_dir runs\try\crawler_q4_steer
lab -m core.evolution.play --run_dir runs\try\crawler_q4_steer --video

:: measure what a steerable robot really does per direction (forward/backward/sideways/turning)
lab -m core.evolution.probe --run_dir runs\try\crawler_q4_steer

:: 3c. rough terrain (bumps, slopes, stairs, blocks; difficulty curriculum), continuing from the steering policy
lab -m core.evolution.train --genome prototypes\crawler_q4\genome.json --out runs\try\crawler_q4_rough --task command --terrain rough --iterations 2000 --init_checkpoint runs\exp1_crawler\1_walking\crawler_q4_steer\model_final.pt
:: watch it on a given difficulty level (0 = easiest, 9 = hardest)
lab -m core.evolution.play --run_dir runs\try\crawler_q4_rough --level 5

:: 3d. walk to goals through pillar fields with an 8x8 distance sensor (roadmap M3)
lab -m core.evolution.train --genome runs\exp1_crawler\2_evolution\evo_rough_002\g05_025\genome.json --out runs\try\nav --task goal --terrain obstacles --iterations 2000 --sensors imu --obs_history 4 --env_override collision_weight=6.0
:: ... or only train a navigator that steers an existing walker (hierarchical)
lab -m core.evolution.train --genome runs\exp1_crawler\2_evolution\evo_rough_002\g05_025\genome.json --out runs\try\nav_hier --task goal --terrain obstacles_rough --low_level_policy runs\exp1_crawler\6_navigation\poc\student_v2\policy.pt --sensors imu --obs_history 4

:: 3d'. unreachable goal (inside a closed ring): get as close as possible; watch it with a far-side start
lab -m core.evolution.train --genome runs\exp1_crawler\6_navigation\poc\nav_camera\genome.json --out runs\try\ring --task goal --terrain enclosure --iterations 600 --sensors imu --obs_history 4 --init_checkpoint runs\exp1_crawler\6_navigation\poc\nav_camera\model_final.pt --env_override goal_signal=camera --env_override collision_weight=1.0
lab -m core.evolution.play --run_dir runs\try\ring --video --num_envs 1 --far_start
:: (--terrain enclosure_pocket: a ring with a slot = local minimum; add --memory gru for a recurrent policy)

:: 3e. combine specialist skills into one model (distillation; train the teachers first: the kept experiment-1 runs
::     only have their results), and compare robots side by side on video
lab -m core.evolution.distill --genome runs\exp1_crawler\2_evolution\evo_rough_002\g05_025\genome.json --out runs\try\student --teacher stairs=runs\exp1_crawler\4_skills\spec_stairs\model_final.pt:stairs_climb,stairs_descend --fallback runs\exp1_crawler\4_skills\generalist\model_final.pt --sensors imu --obs_history 8
lab -m core.evolution.compare_video --out runs\try\cmp.mp4 --terrain_type stairs_climb --level 5 --run "Specialist=runs\exp1_crawler\4_skills\spec_stairs" --run "Student=runs\exp1_crawler\4_skills\student"

:: 5. experiment 2, the robot dog (prototypes\puppy: printable design, 15 STS3215 servos)
lab -m core.morphology build prototypes\puppy\genome.json --out prototypes\puppy
:: walk on the dog-sized rough course (stairs up to 6 cm), seeing the ground (height scan), symmetric gait
lab -m core.evolution.train --genome prototypes\puppy\genome.json --out runs\try\dog --task command --terrain dog_rough --iterations 3000 --num_envs 2048 --sensors imu --obs_history 8 --symmetry --height_scan --init_checkpoint runs\exp2_dog\2_walking\calm_v2\model_final.pt --env_override action_scale=0.75
:: watch the best dog so far, or compare two runs side by side on stairs
lab -m core.evolution.play --run_dir runs\exp2_dog\4_stairs_seeing\seeing_v3 --video --num_envs 1 --terrain_type stairs_descend --level 7
:: jump training (reward = highest point of the lowest paw, upright)
lab -m core.evolution.train --genome prototypes\puppy\genome.json --out runs\try\jump --task jump --iterations 1500 --num_envs 2048 --sensors imu --obs_history 8 --symmetry
:: diagnostics: tools\dog\ (power and jitter, knee height, mirror test, scan check, jump check, leg reach)

:: 4. evolve bodies and gaits together (hours; resumable with --resume; add --task command to evolve for steering)
lab -m core.evolution.evolve --run runs\try\evo_001 --population 12 --generations 8 --seed_genomes prototypes\crawler_q4\genome.json
:: ... or on rough terrain with steering as fitness
lab -m core.evolution.evolve --run runs\try\evo_rough --population 8 --generations 6 --iterations 600 --num_envs 2048 --task command --terrain rough --seed_genomes prototypes\crawler_q4\genome.json
```

Training logs go to TensorBoard: `lab -m tensorboard.main --logdir runs`.

## Repository layout

| Path | What lives there |
|---|---|
| `core/morphology/` | Genome format, robot builder, exporters (USD, URDF, STL, BOM), preview renderer |
| `core/sim/` | Isaac Lab task: servo actuator model, locomotion environment, physics backend presets |
| `core/evolution/` | Training one robot (`train`), watching it (`play`), the genetic algorithm (`evolve`), reports |
| `core/hardware/` | (planned) shared ESP32 firmware |
| `prototypes/<robot>/` | One folder per robot under active development (`genome.json`, notes, trained policies) |
| `releases/` | Robots that have proven themselves on the real table |
| `runs/` | Training output (not versioned): `exp1_crawler/`, `exp2_dog/` per experiment step, `try/` for new runs; `runs/INDEX.md` lists what is kept and what was removed |
| `tools/` | Diagnostic and maintenance scripts (`tools/dog/`: measurements on trained dogs) |
| `IsaacLab/` | NVIDIA Isaac Lab checkout (tag `v3.0.0-EA`), including the Python environment in `IsaacLab/.venv` |
| `.env_init/readme.txt` | How the environment was installed and how to rebuild it |
| `.documentation/roadmap.md` | Milestones: steerable walking, rough terrain, obstacle avoidance |
| `.documentation/project-journal.md` | Project log |

## The genome

A robot is a chassis (battery and electronics) with 1-4 mirrored *limb pairs*; each limb is a chain of 1-3
*segments*, each driven by one servo. See `prototypes/crawler_q4/genome.json`:

| Gene | Meaning |
|---|---|
| `body.length / width / height` | Chassis box [m] |
| `limb_pairs[].x`, `.z` | Mount position along the chassis (fraction of half-length / half-height) |
| `limb_pairs[].splay_deg` | Limb rotated forward (+) or backward (-) about the vertical |
| `segments[].axis` | `yaw` swings horizontally (segment stays level), `pitch` lifts / lowers |
| `segments[].length`, `.elev_deg` | Segment length [m] and slope in the rest pose (negative = down) |
| `segments[].range_deg` | Servo travel either side of the rest pose (SG90 max: 90) |
| `foot.radius`, `foot.material` | Toe sphere [m], `tpu` (grippy) or `pla` (slippery) |

Limits and variation operators live in `core/morphology/genome.py`; at most 12 servos per robot.

## What the tasks reward

`core/sim/locomotion_env.py` has two tasks (`--task`):

- `forward` (default): walk along +x (capped at 0.5 m/s), stay on course and upright, spend little mechanical
  energy, move smoothly, respect joint limits; falling over ends the episode. Fitness = speed.
- `command`: follow a velocity command (forward -30..35 cm/s, sideways +-30 cm/s, turn +-1.2 rad/s) given in
  the robot's own frame; commands change every 4 s and include standing still and turning in place. The command
  is part of the policy input, so the real robot can be steered. Fitness = tracking score (0..1).

The policy only sees what the real robot can measure (IMU and its own servo commands), so a trained `policy.onnx` can in principle
run on the ESP32. The critic gets privileged simulator state during training only.

## Status and known limits

- Isaac Sim PhysX is the default physics backend. Newton (`--physics newton`) runs without Isaac Sim but its
  contact/friction handling differs and is not validated for these robots yet.
- The STL parts are generated drafts: plates print flat without support, struts between plates at an angle
  may need support. Check each part in a slicer before printing.
- SG90s have no position feedback; the policy's "joint angle" input would need either feedback servos or the
  commanded angle on the real robot. See `.agents/project-info.md` for the sim-to-real plan.
