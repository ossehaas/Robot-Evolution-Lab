# Evolutionary Robotics: Project Journal

## The Master Plan
The objective is to master the Sim-to-Real pipeline by designing, simulating, evolving, and physically manufacturing robotic crawlers. The workflow strictly adheres to the following sequence:

1. **Environment & Infrastructure**: Establish a deterministic, hardware-accelerated physics simulator (Isaac Lab on RTX 5080).
2. **Asset Definition**: Define the physical properties and kinematics of the robots using URDF (Unified Robot Description Format).
3. **Simulation & Physics Testing**: Load the URDF into the simulator to validate joint limits, gravity response, and physics constraints.
4. **Brain Evolution (RL/NEAT)**: Vectorize the environment to simulate thousands of robots in parallel. Train a neural network policy using a defined reward function and Domain Randomization (varying mass/friction).
5. **Hardware Fabrication**: 3D print the chassis and assemble the physical robot using micro-servos and an ESP32-S3.
6. **Deployment**: Export the trained model to the microcontroller. Validate that the digital gait translates to physical locomotion.

---

## Log Entries

### Step 1: Environment Initialization & Physics Engine Stabilization
**Status**: Completed
**Objective**: Establish a stable Isaac Lab 3.0 workspace backed by Isaac Sim 6.0, capable of executing vectorized physics operations on the RTX 5080.

**Execution Details**:
- Configured a clean Python 3.12 Conda environment.
- Installed `isaacsim==6.0.0.0` and the local Isaac Lab `develop` branch.
- **Dependency Resolution**: Downgraded `numpy` to `<2.0` to resolve fatal ABI conflicts preventing the physics engine from starting.
- **Warp-Lang Patching**: Surgically patched `omni.replicator.core` scripts (`annotators.py`, `writers.py`) to bypass deprecated `wp.context.Kernel` APIs removed in `warp-lang 1.13.0`.
- **Circular Dependency Fix**: Patched `base_writer_node.py` in `isaacsim.core.nodes` to resolve runtime evaluation failures of Replicator type hints by injecting `from __future__ import annotations`.
- **Hardware Bottleneck Resolution**: Identified that standard PyTorch distributions lack native `sm_120` (Blackwell) CUDA kernels. Bypassed the resulting `torch.AcceleratorError` by forcing the installation of the experimental `cu130` PyTorch Nightly build (`torch-2.13.0.dev20260517+cu130`), natively unlocking the RTX 5080 for tensorized physics operations.
- **Validation**: Executed `test_sim.py --viz kit`. The Isaac Sim viewport initialized, and the physics timeline ticked successfully via GPU tensor operations.

> **Correction (2026-09-30), superseded by Step 4.** Several conclusions above were wrong. The Isaac Lab
> `develop` branch did not match the released Isaac Sim 6.0.0.0 (it needed extensions that release lacks,
> which were then commented out of `apps/isaaclab.python.kit`). The original sm_120 failure came from installing
> a PyTorch wheel without Blackwell kernels, not from PyTorch lacking them: the stable cu128 wheels support
> sm_120. Isaac Lab requires numpy >= 2, so the numpy downgrade broke its declared requirements. And
> `test_sim.py` spawned shapes without rigid bodies or colliders, so it never exercised physics.

### Step 2: Mono-Repo Architecture Initialization
**Status**: Completed
**Objective**: Scaffold a rigid, scalable directory structure to separate shared machine learning logic from isolated robotic prototypes and open-source release candidates.

**Execution Details**:
- Generated `/core/` to house generic simulation wrappers (`/sim`), RL loops (`/evolution`), and base ESP32 firmware (`/hardware`).
- Generated `/prototypes/` to isolate active R&D. Scaffolded the first robot folder `prototypes/simple_leg/` containing strictly separated directories for `/cad` (3D models), `/sim` (URDF config), `/train` (ML checkpoints), and `/firmware` (specific code).
- Generated `/releases/` to serve as the "Hall of Fame" for completed, verified robots packaged for open-source distribution.

### Step 3: First URDF Asset Definition (Simple Leg)
**Status**: Completed
**Objective**: Define the digital twin of a 2-servo robotic leg using the Unified Robot Description Format (URDF).

**Execution Details**:
- Wrote `prototypes/simple_leg/sim/simple_leg.urdf`.
- Designed a 3-link system: `base_link` (hip mount), `thigh_link` (upper leg), and `calf_link` (lower leg).
- Defined two `revolute` joints (`hip_joint` and `knee_joint`) to mimic the rotational constraints of SG90 micro servos.
- Explicitly defined the `<inertial>` mass matrix and `<collision>` geometry for all links, which is a hard requirement for the PhysX engine to calculate gravity and contact mechanics.

**Next Action Queue**:
- Manually export the successfully imported URDF from the Isaac Sim stage as a native `simple_leg.usd` file.
- Update `spawn_leg.py` to use `UsdFileCfg` and the native Isaac Lab `ArticulationCfg` physics wrapper, completely bypassing the buggy Python URDF converters.
- Validate that the `.usd` leg spawns, falls, and collides with the ground plane autonomously via Python.

### Step 4: Environment Rebuild on Isaac Lab 3.0 EA
**Status**: Completed (2026-09-30)
**Objective**: Replace the patched environment with a clean, reproducible install.

**Execution Details**:
- Switched the `IsaacLab/` checkout to the tagged release `v3.0.0-EA` (local branch `evolution-v3.0.0-EA`),
  which pins Isaac Sim 6.1.0.0, PyTorch 2.11 (cu128), numpy 2.x and warp 1.16 in its `uv.lock`.
- Installed with `uv sync --extra isaacsim --extra rsl-rl --extra skrl --extra video` into `IsaacLab/.venv`.
  No downgrades, no nightly builds, no edits to site-packages or kit files.
- Verified: stable PyTorch runs natively on the RTX 5080 (sm_120); official Cartpole trains on both Isaac Sim
  PhysX and kitless Newton; the new `test_sim.py` drops a rigid cube and checks it comes to rest on the ground.
- Pitfall found: importing standalone `pxr` before Isaac Sim starts breaks Isaac Sim's USD libraries, so
  configs and environment classes live in separate modules and USD files are built in a subprocess.

### Step 5: Simple Leg Scripts Fixed
**Status**: Completed (2026-09-30)
**Execution Details**:
- Found that the imported leg asset contained its own fixed `root_joint` bolting the base to the world, so
  the leg could never fall. Removed it; `spawn_leg.py` now chooses explicitly (`--free` drops the leg).
- `spawn_leg.py` applies the initial state after `sim.reset()` (it was silently ignored), uses SG90-like
  actuators (0.14 N*m, 10 rad/s, gearbox armature) and reports servo tracking (worst error 0.12 rad on a sine sweep).
- `train_leg.py` is now a complete RL example: the bolted leg learns to reach random target points with PPO
  (150 iterations, ~1 minute; mean reach error 9.8 mm).

### Step 6: Generative Robots (genome -> sim + print)
**Status**: Completed (2026-09-30)
**Objective**: Stop hand-drawing robots; describe them by a genome that drives simulation, printing and evolution.

**Execution Details**:
- `core/morphology`: mirror-symmetric robots with 1-4 limb pairs, 1-3 servo segments per limb, yaw/pitch
  joints, TPU or PLA feet. The builder generates real printable parts (link plates with servo pockets, horn
  holes and tab screw holes; shoulder mounts; chassis tray and lid), places SG90 servos and electronics, and
  computes mass and inertia of every link from the actual geometry.
- Exports: USD with plain UsdPhysics schemas (no converter needed), URDF + STL visuals, print-ready STL set in
  mm, `BOM.md`, `servo_map.json` for firmware, and a preview render.
- Stress test: 80 random/mutated/crossed genomes all built without errors (105-269 g, 2-12 servos).

### Step 7: Training Walkers and the Evolution Loop
**Status**: In progress (2026-09-30)
**Execution Details**:
- `core/sim`: forward-locomotion task for any generated robot. Policy at 50 Hz (servo PWM rate) with 20 ms
  command latency; SG90 modelled as a DC motor (stall torque 0.177 N*m falling linearly to zero at 10.5 rad/s).
  The policy observes only IMU gravity + gyro, joint angles, its last command and a gait clock; the critic
  additionally sees body velocity, height and joint velocities.
- First crawler run (implicit servo model, no latency): real walking gait, but 55 cm/s, too fast for SG90s.
  That motivated the DC-motor torque-speed model and the latency.
- Crawler retrained with the DC-motor model + 20 ms latency (500 iterations, 4096 envs, 27 min): 47 cm/s,
  no falls, mechanical power dropped from ~6 W to ~2 W. The gait is vigorous (full hip sweeps); expect the
  real robot to be slower (servo deadband, backlash, PLA flex are not modelled).
- Training throughput depends heavily on other GPU users: with other programs keeping the GPU at 70-90 %,
  an iteration took ~2.8 s instead of ~0.9 s.
- Newton backend runs these robots but is slower here and showed suspiciously fast early learning (different
  friction handling), so PhysX stays the default.
- Evolution loop verified end to end on a small run (seed + random genomes, breeding, resume, reports).
- `core/evolution/evolve.py`: (mu + lambda) GA over genomes; each candidate trains in its own process;
  Lamarckian warm start when the joint layout matches a parent; leaderboard, fitness plot and gallery per run.

### Step 8: Steerable Walking (Roadmap M1)
**Status**: Completed (2026-09-30)
**Execution Details**:
- New `command` task: the policy follows (forward, sideways, turn-rate) commands in the robot frame, resampled every
  4 s, including standing still (10 %) and turning in place (15 %). The command is a policy input, so the real
  robot can be steered from outside. Reward: exponential tracking terms plus the existing posture/energy/smoothness terms.
- Evaluation reports velocity and turn-rate errors and a tracking score (0..1), which `evolve --task command` uses as fitness.
- `play` detects steerable runs: keyboard driving in the viewport, or `--video` for a scripted tour (stand, forward,
  curve, turn in place, sidestep, backward, full speed) with captions and command/actual velocity arrows.
- Result for crawler_q4 (800 iterations, 4096 envs, ~50 min with the GPU shared): tracking score 0.85, velocity error
  2.8 cm/s, turn-rate error 0.15 rad/s, no falls.

### Step 9: Symmetric Steering
**Status**: Completed (2026-10-02)
- Sideways felt slow because of my own limits: sideways commands were trained only up to 10 cm/s and the keyboard
  asked for 8 cm/s. A probe (`core.evolution.probe`, fixed commands per robot) showed the body can strafe as fast as
  it walks forward. Ranges are now forward -30..35, sideways +-30 cm/s; keyboard 25 cm/s in all directions.
- Fine-tuned from the previous policy (400 iterations, ~4 min with the GPU to itself: 0.6 s/iteration vs 3.7 s while
  shared). All directions now track within ~7 % up to 30 cm/s.
- Fixed warm starts (`--init_checkpoint`), which the evolution loop also uses: rsl-rl 5 takes `load_cfg`, not
  `load_optimizer`.

### Step 10: Rough Terrain (Roadmap M2, first part)
**Status**: Completed (2026-10-02)
- `core/sim/terrain.py`: 2 x 2 m tiles in 10 difficulty rows: bumps (to 1.2 cm), slopes (to ~19 deg), stairs up/down
  (2 mm to 3 cm), random blocks (2 mm to 2.5 cm), on a 2 cm height grid. Curriculum: robots that track their commands
  (mean error < 6 cm/s, no fall) move a row up, robots that fall or lag (> 12 cm/s) move down.
- Found and fixed: Isaac Lab's random-bump terrain ignores difficulty, so "level 0" was full of 1.5 cm spikes. Own
  `scaled_bumps` now scales bump height with difficulty.
- Found: the flat-ground gait shuffles; even 5 mm stair edges catch the 12 mm feet. Added foot contact sensors and a
  feet air-time reward (steps longer than 0.12 s rewarded while moving), which also suits the real robot better.
- Evaluation spreads robots over all levels and reports scores per level and per terrain type.
- crawler_q4, 2000 iterations from the steering policy (~57 min): tracking score 0.46 over all levels (flat-ground
  policy 0.39), level 0: 0.58, level 9: 0.40, falls < 1 %. Stairs and blocks remain the hardest.
- Evolution on terrain started: `runs/exp1_crawler/2_evolution/evo_rough_001`, 8 bodies x 6 generations, 600 iterations per new body
  (300 for warm-started children), 2048 robots per training, ~5 h.

### Step 11: First Terrain Evolution Failed: It Evolved a Stander
**Status**: Lesson learned (2026-10-02)
- Run `runs/exp1_crawler/2_evolution/evo_rough_001` (8 bodies x 6 generations, ~7 h). The "winner" g01_008 (tall body, 6 servos, `pp-p`)
  scored 0.348 while the hand-made crawler scored 0.309. Checking before celebrating: its velocity error was
  20 cm/s (about the commanded speed) and its curriculum level 0.04. It did not walk; it stood still, stably.
- Causes: (1) 600 iterations from scratch on rough terrain is too little for any body to learn walking; (2) the
  fitness (tracking score) gave credit for standing: the evaluation included stand-still and turn-in-place
  commands, and slow commands score well when not moving; (3) from generation 1 on, the population collapsed onto
  near-copies of the stander (all elites `pp-p`), so evolution could not explore.
- Fixes:
  - evaluation commands always ask for movement (10-30 cm/s in a random direction + turn rate);
  - new fitness `walk_score`: share of the commanded speed delivered in the commanded direction (0..1), discounted
    for sideways drift, times (0.5 + 0.5 x turn accuracy). Re-scored: stander 0.06, crawler (2000 it) 0.35;
    also reported: `progress_ratio` (stander 0.02, crawler 0.54);
  - two-stage training for new bodies: `--pretrain_flat N` first learns to walk on flat ground, then rough terrain;
  - elite selection keeps the best of each body type first; mutations are stronger;
  - `evolve --parallel N` (untested at scale) to train several bodies at once when RAM allows.
- Fair head-to-head (500 flat + 500 rough iterations each) showed the next problem: the crawler walked (walk score
  0.24) but the tall 6-servo and long 8-servo bodies learned nothing, even on flat ground. Their total reward rose
  while linear tracking fell: they farmed the turn-rate reward. Cause: the linear tracking reward was a narrow
  kernel (sigma 8 cm/s), which gives a body that is not walking yet almost no learning signal.
- Fix: dense `progress` reward (speed made good along the commanded direction, capped at the commanded speed,
  15 per m/s) and a wider tracking kernel (12 cm/s). Flat check, 500 iterations: crawler 0.53, long 8-servo 0.28,
  tall 6-servo 0.25 (before: 0.02-0.05 for the latter two). All bodies now learn to walk.
- Memory: one training process (2048 robots, rough terrain) uses ~4.4 GB VRAM but commits ~11.8 GB of system
  memory (6.2 GB resident). With 31 GB RAM and VS Code etc. open, two parallel trainings would exceed the Windows
  commit limit, so evolution runs sequentially (`--parallel` stays at 1). The GPU itself would have room for ~3.
- Evolution run 2 started: `runs/exp1_crawler/2_evolution/evo_rough_002`, 8 bodies x 6 generations, walk-score fitness, 500 flat + 500 rough
  iterations per new body, ~6 h.

### Step 12: Evolution Run 2 Works
**Status**: Completed (2026-10-03); a PC restart interrupted it once, `--resume` continued without loss.
- `runs/exp1_crawler/2_evolution/evo_rough_002`: 8 bodies x 6 generations with walk-score fitness. Best fitness rose every generation:
  0.251 (crawler seed) -> 0.271 -> 0.294 -> 0.306 -> 0.307 -> 0.311.
- Winner g05_025 (`yp-p`): evolution removed the two rear yaw servos in generation 1 and then tuned proportions:
  6 servos, 162 g (crawler: 8 servos, 180 g). Better on every terrain type: stairs (levels 3+) 0.27 vs 0.20,
  blocks 0.17 vs 0.13, easiest level 0.42 vs 0.35, hardest 0.24 vs 0.20.
- Caveat: warm-started descendants accumulate training (the winner's line ~2000 iterations vs 1000 for the
  crawler); a from-scratch comparison with equal budgets is running (`runs/exp1_crawler/3_comparisons/compare2`).
- Very small bodies (1-2 limb segments, 2-4 servos) scored ~0: they cannot walk at all.
- Video: `runs/exp1_crawler/2_evolution/evo_rough_002/videos/crawler_vs_evolved_stairs.mp4` (side by side, climbing out of a stair pit).
- Terrain types renamed for clarity: `stairs_descend` (start on top) and `stairs_climb` (start in a pit).

### Step 13: Better Senses (Roadmap M2b), implementation
**Status**: Implemented (2026-10-03), comparison running
- `--sensors imu`: Isaac Lab IMU sensor on the chassis (at the MPU6050 position): noisy gravity direction, gyro with
  per-robot bias, raw accelerometer readings (noise 0.08 m/s^2, bias up to 0.15 m/s^2).
- `--sensors imu_flow`: plus an optical-flow sensor (speed over ground, 5 % scale noise, 1 cm/s noise, 5 % dropouts
  with the last reading held).
- `--obs_history N`: the policy sees the last N sensor frames (N = 8: 160 ms).
- The sensor setup is stored in `result.json`; `play` and `probe` rebuild the same senses automatically.
- `evolve --video` records the best body climbing stairs after each generation (`<run>/videos/`).

### Step 14: Fair Comparison, a Memory Bug, and the Specialist Plan
**Status**: 2026-10-03
- From scratch with equal budgets (500 flat + 1000 rough iterations): evolved body g05_025 walk score 0.30 vs crawler
  0.25. The evolved body is genuinely better, not just longer trained.
- The first sensor runs (IMU + 8-frame memory) collapsed into standers. Isolation runs (300 flat iterations): IMU
  without memory 0.46 (fine), ideal senses with memory 0.01 (broken). Cause: the policy observation was a view of the
  history buffer, which is modified in place each step; the trainer kept references, so stored observations no
  longer matched the actions. Fix: return a copy. After the fix ideal + memory scores 0.45. Sensor runs re-started.
- Plan agreed (roadmap M2c): specialists per skill (`--terrain_types`, e.g. stairs only) to get target numbers, then
  one student distilled from them (`core/evolution/distill.py`: teachers routed by terrain type, the student only gets
  its own senses, DAgger-style), then RL polish, then the same with different senses to see which sensors the real
  robot needs to recognize its situation.
- Sensor comparison after the fix (evolved body, 500 flat + 1000 rough iterations, ~22 min training each), walk score:
  ideal senses 0.30, realistic IMU + 8-frame memory 0.26, IMU + optical flow + memory 0.26 (crawler, ideal: 0.25).
  Realistic sensor noise costs a little; memory and optical flow did not pay off within this budget (8x larger input,
  single runs; differences of ~0.03 may be seed noise). The ideal-sense numbers are optimistic for a real robot.
- Distillation check: one teacher (IMU + memory generalist, 0.264) copied into a fresh student in 150 iterations,
  3 minutes: student 0.272. Imitating a skill is ~7x faster than learning it with RL.
- Skills experiment started (`runs/exp1_crawler/4_skills/run_all.cmd`, ~2 h): stairs specialist, blocks specialist, generalist with
  the same total budget, distilled student, student + RL polish.

### Step 15: Specialists, Generalist and a Distilled Student (Roadmap M2c, first round)
**Status**: Completed (2026-10-03). Evolved body g05_025, realistic IMU + 8-frame memory, 2048 robots.
Mean walk score over difficulty levels 3-9 per terrain type:

| Run | Overall | Stairs climb | Stairs descend | Blocks | Bumps | Slopes | Training |
|---|---|---|---|---|---|---|---|
| Stairs specialist (stairs only) | - | 0.160 | 0.223 | - | - | - | 19 min |
| Blocks specialist (blocks only) | - | - | - | 0.159 | - | - | 33 min |
| Generalist (all terrain, same budget) | 0.256 | 0.168 | 0.232 | 0.154 | 0.276 | ~0.30 | 33 min total |
| Distilled student (2 specialists + generalist) | 0.278 | 0.186 | 0.252 | 0.178 | 0.302 | ~0.31 | 6 min |
| Student + 500 it RL polish | 0.235 | 0.156 | 0.211 | 0.146 | 0.251 | ~0.28 | 12 min |

- Specialists are not better than the generalist on their own terrain: the "all-terrain compromise / local optimum"
  explanation for weak stair climbing is rejected. The limit is shared by all: body/servos, blindness, or the common
  training setup (penalties, exploration).
- The distilled student beats every teacher on every terrain type (ensemble + DAgger effect) in 6 minutes.
- RL polish after distillation hurt: the polish starts with an untrained critic whose value estimates damage the
  policy. To do: warm up the critic with the actor frozen before fine-tuning.
- Videos per ability (specialist / generalist / student): `runs/exp1_crawler/4_skills/videos/`.

### Step 16: What Limits Stair Climbing? (servo types, foot lift, seeing)
**Status**: Completed (2026-10-03, `runs/exp1_crawler/5_stairs/run_all.cmd`, ~22 min training per run)
- Servo catalog in `core/morphology/components.py` (same mounting footprint, so the printed parts stay the same;
  datasheet values): SG90 1.8 kg*cm / 0.10 s per 60 deg / 9 g; MG90S @6 V 2.2 kg*cm / 0.08 s / 13.4 g (metal gears;
  the user owns these); MG92B @6 V 3.5 kg*cm / 0.08 s / 13.8 g; Feetech SCS0009 @6 V 2.3 kg*cm / 0.10 s / 13.2 g
  (serial bus with position feedback). The genome now has a `servo` field (default SG90); mass, torque, speed and
  the bill of materials follow it.
- Tests, all as stairs specialists with identical settings to the baseline `runs/exp1_crawler/4_skills/spec_stairs` (SG90, climb
  0.160, descend 0.223): A1 MG90S, A2 MG92B, B foot-lift reward (swing foot up to 2.5 cm above its last stance
  height), C terrain height scan (7 x 5 grid around the chassis; simulation-only perfect sensor).
- Video tool `core/evolution/compare_video.py` (side by side / grid of up to 6 robots on the same terrain).
- Results (walk score; climb/descend = mean over difficulty levels 3-9):

| Test | Overall | Climb | Descend | Climb level 9 | Mass |
|---|---|---|---|---|---|
| Baseline SG90 | 0.218 | 0.160 | 0.223 | 0.124 | 162 g |
| A1 MG90S @6 V | 0.184 | 0.140 | 0.182 | 0.110 | 188 g |
| A2 MG92B @6 V | 0.190 | 0.154 | 0.181 | 0.121 | 190 g |
| B foot-lift reward | 0.195 | 0.144 | 0.197 | 0.103 | 162 g |
| C sees the terrain | 0.268 | 0.212 | 0.265 | 0.163 | 162 g |

- Conclusion: blindness is the main limit. Seeing the terrain gives +23 % overall and +33 % climbing, better at every
  difficulty level. Stronger servos do not help (torque was not the bottleneck; the extra 26 g costs); a blind
  foot-lift reward does not help either. Single runs (noise ~0.03), but the +0.05 for seeing is consistent across levels.
- Consequence: a terrain sensor is worth it (roadmap M3). Next: model a real sensor (VL53L5CX 8x8 time-of-flight,
  looking down/forward, with its field of view, range and noise) instead of the perfect height grid.
- Video: `runs/exp1_crawler/5_stairs/videos/stairs_tests_climb.mp4`.

### Step 17: Obstacle Avoidance and Goal Navigation (Roadmap M3), setup and first result
**Status**: first result (2026-10-04)
- Direction agreed: "seeing" is kept as a sense; stair climbing is not a goal in itself; continue with obstacle
  avoidance using the light evolved body (162 g).
- Obstacle course (`core/sim/terrain.py`): 3 x 3 m tiles, pillars 10 cm tall (cannot be climbed) and 6-14 cm wide;
  4 pillars at level 0 up to ~60 at level 9 (0.4 % to 5 % of the floor). The start area and 8 goal spots on a 1 m
  ring stay clear, so every goal is reachable.
- Distance sensor model (VL53L5CX-like): 8 x 8 rays over 45 x 45 degrees at the chassis front, centre tilted 20
  degrees down, 2 m range, 3 % + 5 mm noise, refreshed every 3rd policy step (~16.7 Hz; real sensor 15 Hz in 8x8
  mode). Checked: downward rows see the floor 0.1-0.5 m ahead, upper rows see pillars at 0.7-1.9 m.
- Goal task: the policy gets the goal relative to itself (direction + distance) and the 64 distances. Reward:
  approach speed, +5 per goal reached (then a new goal), -2 per second in obstacle contact; air time; usual penalties.
  Obstacle contact = a body pushing sideways (horizontal force > 1.2 x vertical + 0.2 N); checked: 11 % of steps when
  robots are shoved through the densest pillar field, 0 % on open floor.
- Curriculum: >= 2 goals in an episode -> more pillars; 0 goals or a fall -> fewer.
- Evaluation: goals per minute, share reaching a goal, share of time in collision, per difficulty level.
- First run `runs/exp1_crawler/6_navigation/nav/goal_v1` (evolved body, SG90, realistic IMU + 4-frame memory + distance sensor, 2000
  iterations from scratch, 24 min training): 8.9 goals per minute (goals ~1 m apart), 95 % reach a goal within 20 s,
  no falls. It reached ~7.5 goals/min after only 100 iterations (~2 min).
  Goals/min by level 0..9: 11.2 11.3 10.0 9.7 8.9 9.4 7.8 6.8 7.4 6.7; time in collision: 3 % rising to 19 %.
  It often brushes past pillars instead of avoiding them cleanly: next to improve.

### Step 18: POC Finish: Combining Skills and Obstacle Navigation (batch 1)
**Status**: 2026-10-04
- RL polish after distillation, two fixes: the student now inherits the teacher's exploration noise (it kept the
  initial 0.6), and `train --critic_warmup N` trains only the critic first. Result: distilled student 0.278, old polish
  0.235, fixed polish 0.255, RL generalist 0.256. Even fixed, RL polish pulls the student back to what RL alone
  reaches with this setup; the distillation ensemble is better than that. Use the distilled student without polish.
- Navigation, +1500 iterations from goal_v1: control (collision weight 2) 8.8 goals/min, 10.6 % in collision;
  strict (`--env_override collision_weight=6.0`) 9.0 goals/min, 8.2 % in collision (-26 %; densest level 19 % -> 13 %).
  More training alone does not reduce collisions; the stricter penalty does, without slowing it down.
- New: `train --env_override key=value` for any environment setting; camera-like goal signal (`goal_signal="camera"`:
  25 Hz, 1 cm / 2 deg noise, 80 ms delay); hierarchical navigator (`train --low_level_policy policy.pt`: the trained
  network only outputs a steering command for a frozen walking policy); obstacle course on a bumpy floor
  (`--terrain obstacles_rough`, bumps up to 1.2 cm).
- Batch 2 (all with collision weight 6):

| Run | Training | Goals/min | Reaches a goal | In collision |
|---|---|---|---|---|
| End-to-end, flat pillars (nav_strict) | 3500 it total | 9.0 | 95 % | 8.2 % |
| Hierarchical, flat pillars (hier_flat) | 1000 it from scratch, 17 min | 7.6 | 93 % | 12.5 % |
| End-to-end, bumpy floor + pillars (e2e_rough) | 3500 + 1500 it | 7.4 | 94 % | 7.7 % |
| Hierarchical, bumpy floor + pillars (hier_rough) | 1500 it from scratch, 20 min | 6.0 | 84 % | 11.3 % |

  The hierarchical navigator (only steers the frozen distilled walker) learns very fast: ~8 goals/min after 50
  iterations (< 1 min), because walking comes ready-made. It ends lower: it can only steer as precisely as the walker
  follows commands (~13 cm/s error on rough ground), and the walker never learned about pillars. Hierarchy = fast way
  to add a skill on top of existing ones (scales to many skills); end-to-end = better result when affordable.
- `--env_override` now accepts plain text (cmd strips quotes); the camera-signal run was re-started.
- Camera-like goal signal (overhead camera model: 25 Hz, 1 cm / 2 deg noise, 80 ms delay), nav_strict + 1000 it:
  9.3 goals/min, 96 % reach a goal, 6.9 % in collision: the best navigation result, so a delayed, noisy camera
  position is good enough (part of the gain is the extra training).
- Final POC video: `runs/exp1_crawler/6_navigation/poc/videos/navigation_final.mp4` (hierarchical vs end-to-end on bumpy pillars, camera variant).

### Step 19: Unreachable Goal: Find the Closest Reachable Point
**Status**: 2026-10-05: ring solved (stage 1), pocket not solved yet (stage 2)
Question: if the goal cannot be reached (inside a closed wall, off centre), does the robot end at the closest point
it can reach, on the correct side of the wall?
- New course `--terrain enclosure`: a closed ring wall (outer radius 40 cm, 10 cm tall) per 3 m tile. Half of the
  goals lie inside the ring, 10-30 cm off centre in a random direction (unreachable); the other half are normal goals
  outside it. Robots start 0.75-1.1 m from the ring centre in a random direction. Evaluation: even robots get an
  unreachable goal, odd ones a reachable one; one 20 s episode each.
- Measures: end within 5 cm of the closest reachable distance (radius + robot gap - goal offset), angle between
  where it ended and the goal direction (seen from the ring centre), split by start on the near or far side.
- Note: for a ring (convex) the robot does not need to go all the way round: along the wall the goal distance has
  one minimum, so "follow the wall in the direction the distance shrinks" always ends at the closest point.
  The harder case is a shape with a dent (stage 2).

| Navigator | Within 5 cm of closest | Correct side, far-side start | Reachable goals reached |
|---|---|---|---|
| Zero-shot (nav_camera, never saw this) | 53 % | 50 % | 98 % |
| v1: +600 it, reward exp(-distance/0.25 m), collision 2 | 38 % | 18 % | 98 % |
| v2: +600 it, reward vs true closest distance, collision 1 | 70 % | 86 % | 100 % |

- v1 got worse: the video showed it walking to the wall and parking where it first touched. Touching the wall cost
  more than the small gain from sliding closer. v2 rewards exp(-excess / 6 cm), excess = distance beyond the closest
  reachable one. The simulator knows that distance; only the reward uses it, the robot never sees it.
- Video with a far-side start (`play --terrain enclosure --far_start`): it walks to the wall, follows it round and
  stops opposite the goal. `runs/exp1_crawler/7_closest_point/ring_v2/videos/ring_v2_far.mp4`. Training time ~9 min (600 it).

- Stage 2, `--terrain enclosure_pocket`: ring radius 50 cm with a 30 cm wide, 30 cm deep slot cut inwards from one
  side. The slot bottom is a second candidate for the closest point; when the goal sits on the other side, it is a
  local minimum (a robot walking straight at the goal from the slot side ends up in it). Episodes 30 s. Of the
  unreachable goals 27 % have the slot bottom as closest point, 73 % the outer wall.

| Navigator (pocket course) | Training | Within 5 cm of closest | Caught in slot (outer best) | Slot best: reached | Reachable goals |
|---|---|---|---|---|---|
| ring_v2 zero-shot | - | 32 % | 15 % | 9 % | 95 % |
| P1: ring_v2 + 800 it, 80 ms history | 12 min | 37 % | 13 % | 5 % | 95 % |
| P2: navigator over walker, GRU memory, from scratch | 1500 it, 27 min | 21 % | 9 % | 5 % | 71 % |
| P3: same, 80 ms history (control for P2) | 1500 it, 23 min | 18 % | 8 % | 5 % | 80 % |

- Not solved. Being caught in the slot is not the main failure; not exploring at all is. The video
  (`runs/exp1_crawler/7_closest_point/pocket_p1_mlp_warm/videos/pocket_p1_far.mp4`) shows the robot parking at the first wall it reaches,
  while the closest spot is at the slot bottom round the other side. When the slot is the best place, almost no
  robot goes into it.
- GRU memory vs none: no difference (21 vs 18 %). Memory does not help while the policy never explores: there is
  nothing to remember. The hierarchical navigators trained from scratch also failed to go round the larger ring
  (2 % correct side from far-side starts), so they never learned stage 1 here either.
- Likely next steps: a curriculum (ring first, then a shallow slot that deepens); reward the best distance reached
  so far in the episode (so exploring is not punished on the way); longer training for the GRU; a slot wide enough
  that walking in is easy.

### Step 20: Detour Test: Goal Inside a Half Circle, Open Side Away from the Robot
**Status**: 2026-10-05, done
- `--terrain cup`: an open ring (radius 45 cm, 10 cm wall) whose closed side faces the robot's start. Half of the
  goals lie inside, 10-25 cm left of the centre (reachable only round the back, through the open side), half are
  normal goals outside. Walking straight at the goal ends against the closed side: a local minimum of the distance.
  The wall arc grows with the difficulty level: 60, 90, 120, 150, 180 (half circle), 210 degrees; curriculum: up
  after reaching a goal in an episode, down after none. Normal goal reward only (no reward for being close).
- Evaluation: robots start on the closed side (within +-26 deg), 30 s, spread over all levels.

| Wall arc | 60 | 90 | 120 | 150 | 180 | 210 |
|---|---|---|---|---|---|---|
| Zero-shot nav_camera: reaches goal inside | 100 % | 44 % | 28 % | 1 % | 1 % | 0 % |
| Zero-shot: stuck at the closed side | 0 % | 13 % | 70 % | 98 % | 99 % | 100 % |
| cup_v1 (+1500 it, 20 min): reaches goal inside | 92 % | 78 % | 68 % | 78 % | 73 % | 68 % |
| cup_v1: stuck at the closed side | 8 % | 11 % | 30 % | 22 % | 27 % | 32 % |

- Outside goals still 99 %; time in collision 9.5 % -> 3.7 %. The pillar navigator only knew small detours; with the
  curriculum it learned to walk away from the goal, round the wall end and in through the opening.
- Video (180 degrees, start on the closed side): `runs/exp1_crawler/8_detour/cup_v1/videos/cup_180.mp4`: walks to the wall,
  turns away, follows it round the upper end and walks in to the goal.

## Experiment 2: a cute robot dog

### Step 21: Dog Generator and Design
**Status**: 2026-10-05
- Goal: a toy dog of at most 50 cm for a 12-year-old, natural movement, cute for everyone; really printable.
- `core/morphology/dog.py`: fixed dog layout with variable proportions, presets puppy / terrier / bulldog. Output is
  the same RobotSpec, so simulation, training and exports work unchanged. The owner chose the puppy.
- Servos: Feetech STS3215 (12 V, 30 kgf*cm, magnetic encoder: the robot knows its joint angles). 12 for the legs
  (sideways, hip, knee via pushrod), 3 for expression (nod, head tilt, tail wag).
- A design agent turned the puppy into a printable design (`prototypes/puppy/`, README with assembly order): 59
  parts on a 210 mm bed without supports, servo mounts from the official STS3215 drawing, knee crank + pushrod with
  683ZZ bearings, heat-set inserts, battery hatch in the belly, round displays as eyes. 2.37 kg, 36 x 25 x 37 cm.
  Interference check (`python -m core.morphology.dog_check`): all joints clear except front and rear leg on one side
  when the hips swing towards each other by > ~45 deg in total. To verify on a real servo first: horn thread, case
  hole depth, connector side, display and board hole patterns, battery length <= 75 mm.

### Step 22: Teaching the Dog to Walk
**Status**: 2026-10-05, in progress
- Expression joints (nod, tilt, tail) are not driven by the walking policy; during training they move randomly
  (poses, slow looks, fast wagging) so the walker keeps its balance whatever they do. Scriptable afterwards.
- Run 1: every dog fell over within 0.3 s. The energy penalty was tuned for the 0.12 kg spider; for a 1.9 kg dog,
  falling at once was cheaper than walking. Fix: energy per kg (weight x 0.15 kg / mass). Run 2 then walked
  (walk score 0.46, no falls) but crawled low with sprawled legs.
- Natural gait (`gait = "trot"`, automatic for the dog): trot rhythm reward (diagonal pairs in stance by turns,
  2 Hz), standing height, penalty for thighs/shoulders touching, legs under the body; body touching the ground or
  tilt > 60 deg = fall.
  - trot_v1 (from scratch): lay on its side kicking its legs in the air (half the trot reward, no fall detected).
  - trot_v2/v3 (from the crawler): stayed ~9 cm too low on its thighs; a narrow height reward gives no gradient.
  - trot_v4: actions mapped to half the joint range around the standing pose (as ANYmal-style training does),
    linear height reward, from scratch, 2000 it (~36 min, 512 robots): stands tall, trots, no body contact;
    tracking still weak (walk score 0.23, speed error 19 cm/s).
- Smaller runs need less RAM: PhysX GPU buffers now scale with the robot count (512 robots instead of 2048 while
  the PC has ~6 GB free).
- puppy_print_v1: the printable design (2.37 kg, real joint positions, leg-clash penalty, real IMU position),
  +3000 it from trot_v4 (33 min, 512 robots): walk score 0.28, speed error 17 cm/s, 1 % falls; stands tall, trots,
  no body contact, leg clash rare. Video: `runs/exp2_dog/2_walking/videos/puppy_print_v1__tour.mp4`.
- Probe of print_v1 (`core.evolution.probe`): too fast at low speeds (asked 10 cm/s, walks 13-18 cm/s: a fixed 2 Hz
  rhythm forces a minimum step) and lopsided turning (right 76-90 %, left 30-44 % of the command) although the body is
  exactly symmetric (centre of mass on the centre line): a learned bias.
- New: symmetry augmentation (`train --symmetry`, `core/sim/symmetry.py`): every sample is also trained mirrored.
  Gait rhythm now follows the commanded speed (1.4-2.6 Hz); tighter velocity tracking (sigma 8 cm/s) for the dog.
- print_v2 (+3000 it from print_v1 with symmetry, 37 min): the bias flipped (left 75-98 %, right 9-33 %), still too
  fast at 10 cm/s. Physical mirror test (dog B gets the mirrored actions of dog A; `tools/dog/mirror_test.py`):
  all sensor signs match, differences are contact chaos, so the mirror rules are right. Starting from a biased policy
  works against the augmentation; next: symmetric training from scratch (puppy_sym_v1, 4000 it).
- puppy_sym_v1 (symmetry from scratch, 4000 it, ~75 min): left and right now equal in every probe; forwards and
  backwards 0.8-1.3x the command; sideways 1.4-2.4x too fast; turning only 22-33 %.
- puppy_sym_v2 (+3000 it, turn reward 1.5 -> 4, sigma 0.3 rad/s): turning 92-96 % of the command both ways (was
  22-33 %); slow speeds still 1.5-2.2x too fast; 4 % falls in the evaluation. Video:
  `runs/exp2_dog/2_walking/videos/puppy_sym_v2__tour.mp4`. Next: sym_v3 with a calmer slow rhythm (1.0 Hz) and double speed
  tracking weight.
- puppy_sym_v3 (+3000 it, 38 min; slow rhythm 1.0 Hz, speed tracking weight 6): forwards/backwards 0.88-1.00x the
  command from 10 to 30 cm/s (slow walking fixed); turning 90-93 % both ways; sideways still 1.3-2.0x too fast
  (accepted for now: dogs rarely sidestep); 3 % falls.
- Robustness: random pushes for the dog (velocity change up to 0.3 m/s every 4-8 s) and the rough terrain of
  experiment 1 (bumps up to 1.2 cm, slopes, 3 cm steps: gently uneven for this dog). puppy_rough_v1 started.
- puppy_rough_v1/v2 (+3000 it on rough terrain with pushes from sym_v3): 46 % falls overall, 14 % already on level 0.
  v1's curriculum never left level 0 (the spider's promotion threshold of 6 cm/s is out of reach); with dog
  thresholds (15/25 cm/s) v2 reached level ~1 and drifted back. Flat-floor check: pushes cost little (sym_v3 2 % falls
  without, 4 % with pushes), the uneven floor is the problem.
- Root cause found: the policy's raw actions were beyond the +-1 limit 99 % of the time (mean |action| 43): bang-bang
  control, hidden because actions are clipped before the environment sees them. That explains the stiff gait, the
  overshoot at low speeds and the fragility. Fix: actions passed unclipped, the environment clips and penalizes the
  excess (`action_limit_weight`, 1 per second per unit beyond the limit). The spider policies of experiment 1 probably
  had the same issue (their symmetry losses suggest it); not re-checked.
- puppy_smooth_v1: from scratch with the action limit penalty (flat, symmetry, 4000 it).
- puppy_smooth_v1 (from scratch, flat, action limit penalty, 4000 it, ~70 min): actions beyond the limit 9 % (was
  99 %), mean |action| 0.66 (was 43); walk score 0.37 (best so far), speed error 15 cm/s, turning 97-99 % both ways,
  2 % falls; slow forward walking 1.5x too fast again (one stage instead of three).
- puppy_rough_v3 (+3000 it from smooth_v1 on rough terrain with pushes, 54 min): falls 29 % overall (bang-bang
  rough_v2: 46 %); per level 0-9: 6, 2, 6, 14, 28, 33, 37, 47, 59, 63 %; curriculum reached level ~2.5. Bumps are
  fine at every level; slopes down and climbing steps are the hardest. Video:
  `runs/exp2_dog/2_walking/videos/puppy_rough_v3__tour_level3.mp4`.
- puppy_rough_v4 (+2000 it from rough_v3 with 2048 dogs, 37 min): falls 19 % overall (rough_v3: 29 %); per level 0-9:
  9, 3, 2, 2, 10, 13, 29, 38, 38, 45 %; curriculum reached level 3.8. Levels 1-3 now below 5 %.
- Scaling on this laptop (RTX 5080 Laptop 16 GB, 31 GB RAM), dog on rough terrain with symmetry augmentation:
  512 dogs 0.82 s/it; 2048 dogs 1.34 s/it (~1530 dog-steps/s), process 6.9 GB RAM in use / 13.0 GB committed, VRAM
  4.7 GB; 4096 dogs 4.04 s/it (~1010 dog-steps/s: slower per sample), 8.2 / 15.7 GB RAM, VRAM 6.1 GB. 2048 is the
  sweet spot here; memory is not the limit (8192 would need roughly 9 GB VRAM and ~21 GB committed RAM).
- Level-0 falls (9 %) were dogs walking off the world: level 0 is the edge row of the terrain grid and the flat margin
  was only 1 m (a dog walks ~4 m per episode; the slow spider never got there). Border widened to 4 m: rough_v4
  re-evaluated without training gives 1, 2, 3, 3, 9, 20, 20, 33, 46, 47 % falls per level (18 % overall).
- New metric: cost of transport (mechanical power / (weight x speed), dimensionless) in the command-task evaluation.
- puppy_rough_v5 (+2000 it from rough_v4, 2048 dogs, 38 min): falls 16 % overall; per level 0-9: 2, 2, 2, 4, 8, 13,
  21, 34, 36, 40 %. First energy figure: cost of transport 3.6 (22.8 W mechanical) - wasteful (walking dogs ~0.2-0.5,
  good robot dogs ~1-2): a later item.

### Step 23: Flexible Back (D2b)
**Status**: 2026-10-07, first round done: a passive back makes the dog less stable
- Idea from the owner: would a back that can bend make rough ground easier (and later stairs, jumping)? Spot, ANYmal
  and toy robots have a stiff body. Test passive first: spring hinges, no motor.
- `DogSpine` in `core/morphology/dog.py`: the body is cut at the middle (printed parts split with manifold3d, so both
  halves get their real mass: front 478 g, rear 366 g), rear legs and tail hang on the rear half; hinge axes roll
  (twist) and/or pitch (bend) with a spring + damper (`spine` actuator group in `core/sim/robot_cfg.py`). The hinges
  are not driven and not observed; the policy sees the joints in the stiff dog's order, so a stiff-dog policy runs
  on the flexible dog unchanged.
- Zero-shot (rough_v5 policy, twist spring 8 N*m/rad): walks at once, 14 % falls (stiff: 16 %, within noise).
- Experiment: stiff (control), roll k = 4 / 8 / 16 N*m/rad, pitch k = 8, roll + pitch k = 8; each +1000 it from
  rough_v5 with identical settings (`runs/exp2_dog/3_flexible_back/batch.py`).
- Result (each variant +1000 it from rough_v5, ~25 min each; then 3 evaluations = ~6000 episodes, falls +-0.5 %;
  `runs/exp2_dog/3_flexible_back/compare.py`, `runs/exp2_dog/3_flexible_back/comparison.txt`):

| Variant | Falls | Walk score | Cost of transport |
|---|---|---|---|
| stiff (control) | 17.9 % | 0.305 | 3.66 |
| twist k16 | 20.5 % | 0.309 | 3.71 |
| bend k8 | 20.4 % | 0.300 | 3.79 |
| twist k4 | 21.5 % | 0.302 | 3.71 |
| twist k8 | 21.8 % | 0.296 | 3.66 |
| twist + bend k8 | 24.6 % | 0.276 | 3.70 |

- The more flexible, the worse; the difference is on the hard levels (6-9), slopes and steps. A back the policy cannot
  sense or drive is a disturbance: balance is kept through the legs and a swaying body underneath makes that harder
  (likely part of why Spot and ANYmal are stiff). Not ruled out: a back helps when the policy senses it (angle sensor),
  when trained from scratch with it, or for bigger obstacles, stairs and jumping.

### Step 24: Energy and Jitter
**Status**: 2026-10-07, running
- Energy weight x3 / x10 (+1000 it from rough_v5): cost of transport 3.74 / 3.26 (control 3.66), falls 19 / 21 %:
  the penalty barely moves the energy use.
- Diagnosis (`tools/dog/measure_power.py`): the dog uses 20.2 W standing still and 20.8 W walking; joints move
  3-4 rad/s while "standing" and the commands change ~0.2 per 20 ms step: constant jitter. The action-rate penalty
  (0.05) amounts to ~0.03 per second at that jitter, and nothing rewards standing calmly.
- Test: action-rate weight 0.05 -> 1.0, a stand-still penalty (joint speed while the command is zero), and both;
  each +1000 it from rough_v5 (`runs/exp2_dog/2_walking/smooth_batch.py`).
- Results (+1000 it each; power measured with head and tail still):

| Variant | Standing still | Walking 25 cm/s | Cost of transport | Falls |
|---|---|---|---|---|
| control (rough_v5) | 20.2 W | 20.8 W | 3.66 | 17.9 % |
| action rate x20 | 18.7 W | 17.8 W | 3.23 | 21 % |
| stand-still penalty | 8.4 W | 21.3 W | 3.42 | 26 % |
| both | 0.4 W | 16.7 W | 3.10 | 20 % |

  Together the two terms make the dog really stand still (50x less power when idle); walking is still restless.
  Next: calm_v2 from calm_both with action rate 3.0.
- calm_v2 (+1500 it from calm_both, action rate 3.0, stand-still 0.3, 2048 dogs): falls 10 % overall (rough_v5:
  17.9 %), per level 0-9: 2, 0, 2, 1, 2, 8, 9, 18, 24, 34 %; cost of transport 2.57 (was 3.66); standing 0.0 W,
  walking 25 cm/s 12.4 W (was 20.8 W). Probe: forward 0.95-1.13x, sideways 0.92-1.35x (was 1.3-2.0x), turning
  104-106 % both ways; backward overshoots 1.24-1.52x. A jittery gait was as much a stability problem as an energy
  problem. Video: `runs/exp2_dog/2_walking/calm_v2/videos/tour_level4.mp4`.
- Smoothness changes the backward tracking: calm_v2 overshoots backwards (1.24-1.52x). Open.

### Step 25: What Can These Legs Do? Stairs and Jumping
**Status**: 2026-10-08
- Paw lift (geometry, `tools/dog/leg_reach.py`): with 50 % of the joint range (what the walking policy may
  use) 39 mm, 75 % 56 mm, 100 % 78 mm. Dog-scaled stairs (a Labrador on human stairs: rise ~1/3 of hip height; hips
  here 16 cm) are ~5 cm with ~15 cm treads: possible with 75 % of the range, no hardware change.
- Jumping (`tools/dog/jump_test.py`, scripted: stand, crouch, all hips and knees to full extension):

| Actuator | Body rises | All paws off the ground |
|---|---|---|
| STS3215 12 V | 19 mm (= straight legs) | 0 ms |
| STS3215, reflected inertia / 3 | 19 mm | 20 ms |
| same motor, gear ratio / 2 (2x faster, 2x weaker, inertia / 4) | 19 mm | 0 ms |
| same motor, gear ratio / 4 | 32 mm | 80 ms |
| quasi-direct-drive BLDC (30 rad/s, 3 N*m, low inertia) | 81 mm | 200 ms (~5 cm jump) |

  The STS3215 cannot jump: it is slow (0.22 s / 60 deg) and its 1:345 gearbox gives a reflected rotor inertia
  (~0.01 kg*m^2) ten times the leg's own, so an explosive extension mostly spins up the motors. A "4x faster" servo
  that keeps that inertia does not jump either. Real robot dogs (Unitree, Spot, MIT Mini Cheetah) use low-ratio
  brushless actuators for this reason (~EUR 60-150 each at this size). Learning a jump (height / flight-time reward
  with a curriculum, or imitating a reference jump) only makes sense with such actuators.
- Dog-sized rough terrain (`--terrain dog_rough`, `terrain.DOG_ROUGH`): 3 m tiles; stairs up to 6 cm rise with 15 cm
  treads, bumps up to 2.5 cm, blocks up to 4 cm, slopes up to 20 deg. calm_v2 without training: 28 % falls, 55-62 % on
  the hardest levels; climbing stairs is the weakest type.
- Stairs training: joint range 75 % vs 50 %, each +3000 it from calm_v2 (`runs/exp2_dog/4_stairs_seeing/batch.py`).
- Stairs results (+3000 it from calm_v2 on dog_rough, ~75 min each): joint range 75 %: 18 % falls (calm_v2 without
  training 28 %), range 50 %: 21 %; levels 7-9 still 39-51 %; curriculum stuck at level ~4.4. CoT 3.0 vs 2.5.
- Falls per type (new in the evaluation: `type_fall_rates`), range75, levels 6-9: stairs down 55-88 %, stairs up
  43-74 %, blocks 26-74 %; bumps and slopes up below 10 %. Descending is worse than climbing: a blind dog only notices
  a drop when a paw steps into it.
- Next: a seeing dog. Height scan at dog size (80 x 40 cm, 5 cm grid, 153 points, +-20 cm), mirrored for the symmetry
  augmentation; `train.widen_checkpoint` adds zero-weight inputs to a checkpoint trained without the scan, so the
  network starts exactly as the blind one (checked: 18 % falls, CoT 3.03, identical). seeing vs blind, each +3000 it
  from range75 (`runs/exp2_dog/4_stairs_seeing/batch_scan.py`).
- seeing (first try): 33 % falls, curriculum collapsed to level 0 in the first ~60 iterations and never recovered;
  4 h training (blind: 78 min). Two design errors: the scan was part of all 8 history frames (1224 new inputs next to
  352: early updates on their weights shook the gait), and the scanner ran at the physics rate (200 Hz).
  Fix: for the dog only the current scan is observed, once, after the history (153 inputs; `height_scan_in_history`
  False), and the scanner updates once per control step (50 Hz). Widened network still neutral (19 % falls).
  seeing_v2: 1.57 s/it, curriculum rising (level 3.0 after 300 it).
- seeing_v2: 16.2 % falls, the same as blind (16.0 %). Reason: a bug. The scan read -1 (clipped) everywhere: the ray
  caster's reported position (`pos_w`) does not include its 0.5 m mounting offset, while the height formula assumed it
  did, so every height came out ~0.5 m too low. Fix: heights relative to the chassis root (`root_z - hit_z - standing
  height`); checked on the hardest stairs: flat floor reads -0.01, the first 6 cm step 40-50 cm ahead reads 6.2 cm.
  **Consequence for experiment 1:** the spider's height scan (stairs test C, step 13) used the same formula and very
  likely saw nothing either; its +33 % stairs result must have had another cause (e.g. the bigger network input or
  training noise). Treat that result as unconfirmed.
- seeing_v3 started with the working scan (+3000 it from range75).
- **seeing_v3 (working scan) vs blind**, both +3000 it from range75 (~80 min each, 2048 dogs):

| | Blind | Seeing |
|---|---|---|
| Falls overall | 16.0 % | 7.1 % |
| Stairs down, levels 6-9 (3.6-6 cm) | 35 / 61 / 80 / 94 % | 2 / 12 / 20 / 39 % |
| Stairs up, levels 6-9 | 26 / 35 / 44 / 56 % | 6 / 26 / 26 / 32 % |
| Blocks, levels 7-9 | 40 / 56 / 44 % | 34 / 26 / 47 % |
| Curriculum reached | 4.5 | 5.7 |
| Walk score | 0.316 | 0.344 |

  Seeing halves the falls; the biggest gain is going down stairs (a blind dog steps into the drop). Video:
  `runs/exp2_dog/4_stairs_seeing/videos/blind_vs_seeing_stairs_down_level8.mp4`.
- Bug found along the way: `play` and `probe` ignored a run's `env_overrides` (action range, gait rhythm). Fixed
  (`locomotion_env_cfg.apply_overrides`). Earlier dog videos and probes since sym_v3 ran with the default gait rhythm
  (1.4-2.6 Hz instead of the trained 1.0-2.6 Hz); the training evaluations themselves were correct.

### Step 26: Jump Training
**Status**: 2026-10-09
- `--task jump` (dog): 4 s episodes; reward = the episode's best height of the LOWEST paw while nothing touches the
  ground (paid when the record is beaten, so the episode's jump reward = 50 x best height), plus small terms for body
  rise and time in the air; body contact or tilt > 60 deg = fall. Full joint range. "Airborne" was first defined by
  paw/shin contacts only (a kneeling dog could count); now no sensed body part may touch anything.
- STS3215 (1500 it from scratch, 2048 dogs, ~20 min): lowest paw 9.7 cm on average (best 10.4 cm), every attempt
  leaves the ground, 2.0 s of 4 s airborne (continuous hopping), 0 % falls. Checked step by step
  (`tools/dog/jump_check.py`): truly airborne, the lowest point of the whole dog 5-8 cm up. The trick: a very deep
  crouch (body down to 8 cm instead of the 15 cm of my scripted test) gives twice the push-off stroke, and the legs
  are tucked in the air. The body itself barely gets above standing height (+0.6 cm): a hop with tucked legs, not a
  leap. The scripted "no jump" verdict (step 25) was too pessimistic: learning found a better motion.
  Video: `runs/exp2_dog/5_jumping/videos/v1__jump_sts3215.mp4`.
- QDD (hypothetical fast brushless motors, same training): lowest paw 43 cm, body up to 35 cm, but 100 % falls: it
  somersaults (at the top the head is the lowest part; upside down the paws are on top) - the fall penalty (3) was far
  below the reward (50 x 0.43 m). Fix: jump height only counts while upright (tilt < ~35 deg), fall penalty 20 for
  jumping. Both variants retrained (first round moved to `runs/exp2_dog/5_jumping/v1/`).
- Retrained with the upright rule (1500 it each from scratch, ~20 min each):

| | STS3215 (real servo) | QDD (fast brushless, hypothetical) |
|---|---|---|
| Lowest paw, average best | 8.1 cm (best 10.1) | 19.2 cm (best 21.4) |
| Body above standing height | 0.8 cm | 4.1 cm |
| Time in the air per 4 s | 1.4 s | 2.4 s |
| Falls | 0 % | 0 % |

  Most of the height comes from tucking the legs in the air. The fast motors jump 2.4x higher and really lift the
  body. Video: `runs/exp2_dog/5_jumping/videos/jump_sts3215_vs_qdd.mp4`.

### Step 27: Housekeeping - Runs per Experiment
**Status**: 2026-10-10
- `runs/` reorganized (`tools/reorganize_runs.py`): `runs/exp1_crawler/{1_walking .. 8_detour}` and
  `runs/exp2_dog/{1_design .. 5_jumping}`; new trainings go to `runs/try/`, scratch output of tools to `runs/scratch/`.
  The best run of each successful line is kept in full; comparisons and failed experiments keep only their results
  (result.json, videos, summaries); superseded versions were removed (their videos were kept, named after the run).
  4.6 GB -> 0.74 GB. `runs/INDEX.md` lists every kept, reduced and removed run with its key numbers and the old -> new
  paths. Paths in this journal, the README, the roadmap and the puppy README were updated (`tools/update_doc_paths.py`);
  references to removed runs keep their (new-style) path for the record.
- Diagnostic scripts moved to `tools/dog/` (power and jitter, knee height, mirror test, scan check, jump check, leg
  reach, scripted jump). The hierarchical navigators' stored walker path was rewritten; checked: hier_flat still runs.

### Step 28: Natural Gait (list step 1)
**Status**: 2026-10-10
- New: gait modes `walk_trot` (lateral-sequence 4-beat walk when slow, 3 paws down; trot when fast; blended over
  0.25-0.40 m/s) and `free` (no rhythm prescribed); flight penalty (no paw down while walking); smooth commands
  (`command_accel` 0.5 m/s^2, yaw 1.5 rad/s^2); top speed 0.45 m/s. Froude scaling: with 16 cm legs a real animal
  would only start trotting around 0.75 m/s, so at 0.1-0.45 m/s a walk is the natural gait.
- Gait analysis tool `tools/dog/gait_check.py` (paws down, stance per paw, paired touchdowns, step rate, CoT per speed;
  `--course` on the run's terrain, `--dr` with randomization, a geometric cross-check of the paw soles).
- seeing_v3 (before): ~0.9 paws on the ground while walking, rear paws down only 12-16 % of the time: a hopping,
  one-paw-at-a-time gait; stands still at 0.10 m/s.
- walk_trot / free (+3000 it from seeing_v3, ~2 h each): walk_trot 4.5 % falls (best so far), speed error 12.5 cm/s,
  walk score 0.383; free 10.9 % falls. Gait unchanged in both (~0.95 paws by contact sensor, ~1.2 by geometry):
  the warm start keeps the habit.
- Measurement flaw found: the paw sits on the shin link, so the contact sensor counts a shin on the ground as "paw
  down" and the rewards never penalized kneeling on the rear shins (the trained dog rests on its hocks when standing).
  Fix: the paw as its own link (ankle hinge; a very stiff spring for a "rigid" ankle), so contacts are paw-only and
  shins count as body contact.

### Step 29: Gait and Body from Scratch (list steps 1-3 combined)
**Status**: 2026-10-10, running
- New genome options: `ankle` (passive spring hinge above the paw; the paw becomes link s3), `paw_contact_stiffness /
  damping` (PhysX compliant contact for soft TPU paws), `legs.knees` (back_front as before, back_back like Spot,
  front_back, front_front: the knee-side parts are mirrored about the hip plane). Zero-shot: seeing_v3 walks on spring
  ankles + soft paws at once (walk score 0.36, 9 % falls at 512 dogs).
- Batch (`runs/exp2_dog/9_gait_body/batch.py`), all 4000 it from scratch with one recipe (dog course, height scan,
  symmetry, walk_trot, flight penalty, smooth commands): A stiff ankle / knees as now, B springy ankle + soft paws,
  C Spot knees.
- First batch attempt failed: from scratch every dog fell within ~1 s (curriculum at level 0) and iterations took
  4.4 s (constant resets). Not the ankle (standing still with zero actions: 0 falls, ~51 ms per physics step for 2048
  dogs with or without ankle). Cause: the calmness penalties (action rate 3.0, stand-still 0.3) were made for
  fine-tuning a calm dog; for a randomly flailing new network they cost ~25 per second while a fall costs 3 once, so
  falling at once is optimal (the same "rather fall" exploit as the very first dog run). Fix: two stages - stage 1
  (2000 it) learn to walk with mild penalties and 50 % joint range, stage 2 (2500 it) strict calmness and 75 % range.
  Stage 1 now learns normally (episode length 103 -> 431 steps in 186 it, 1.7 s/it).
- Round 1 (two-stage, from scratch, ~2.2 h per variant):

| | A stiff ankle | B springy ankle + soft paws | C Spot knees, stiff ankle |
|---|---|---|---|
| Falls (dog course) | 8.9 % | 0.6 % | 2.5 % |
| Falls levels 7-9 | 12-17 % | 2 % | 2-7 % |
| Speed error | 11.3 cm/s | 7.4 cm/s | 12.4 cm/s |
| Walk score | 0.328 | 0.473 | 0.327 |
| CoT (evaluation) | 14.4 (61.7 W) | 4.5 | 13.4 (57.5 W) |
| Paws down 0.1-0.45 m/s (gait_check) | ~1.0 | 1.8-2.1 | ~1.0 |
| Diagonal pairs together at 0.3 m/s | 4 % | 99 % (a clean trot) | 0 % |

  B is the best dog so far: a real trot with all four paws ~50 % on the ground, 0.6 % falls, best speed tracking. A and
  C still hop and draw ~60 W: the "stiff" ankle (50 N*m/rad on a 13 g paw) probably vibrates, so it is a poor stand-in
  for a rigid leg and muddies the knee comparison (C is still more stable than A with the same ankle).
- Round 2 started: D Spot knees + springy ankle + soft paws, E springy ankle with hard paws, F stiff ankle with soft
  paws (`batch2.py`).
- Round 2 (D Spot knees + springy ankle + soft paws, E springy ankle + hard paws, F stiff ankle + soft paws), same
  two-stage recipe. Both rounds together:

| Variant | Knees | Ankle | Paws | Paws down | Falls | Speed err | Turn err | CoT |
|---|---|---|---|---|---|---|---|---|
| A | as now | stiff | hard | ~1.0 (hops) | 8.9 % | 11.3 | 0.54 | 14.4 |
| B | as now | springy | soft | ~2.0 (trot) | 0.6 % | 7.4 | 0.45 | 4.5 |
| C | Spot | stiff | hard | ~1.0 (hops) | 2.5 % | 12.4 | 0.38 | 13.4 |
| D | Spot | springy | soft | ~2.0 (trot) | 1.7 % | 9.5 | 0.65 | 4.3 |
| E | as now | springy | hard | ~1.0 (hops) | 10.1 % | 11.4 | 0.49 | 5.9 |
| F | as now | stiff | soft | ~1.9 | 2.8 % | 9.1 | 0.72 | 12.0 |

  Conclusions: soft paws decide the gait (every soft-paw variant walks on ~2 paws, every hard-paw variant hops, even
  with a springy ankle); the springy ankle decides the energy (stiff ankles vibrate: ~55-60 W vs ~19 W); both together
  (B) are best on almost everything, 2 % falls even on the highest stairs. Spot knees are not better for this dog
  (D: more falls, worse turning, stands still at 0.1 m/s): keep the current knee direction. One run per variant, so
  small differences are uncertain. Videos: `runs/exp2_dog/9_gait_body/videos/gait_A_B_C.mp4`, `soft_vs_hard_paws.mp4`.
- Next: make the printable puppy match B: a TPU 95A ankle coupling with ~1.5 N*m/rad and paws as soft as the simulated
  ones (~1e4 N/m contact stiffness, a guess to be measured on a printed paw); then imitation of real dog motion.
- Owner's choice (2026-10-11, from `soft_vs_hard_paws.mp4`): **D (Spot knees + springy ankle + soft paws)** for the
  puppy: "B looks hectic; D seems more in control, turning might be slower but it looks more natural". The numbers
  agree on energy (CoT at 0.3 / 0.45 m/s: D 2.13 / 1.44, B 2.56 / 1.50). The design agent makes back_back the primary
  printable design. B and D are being re-run with seed 2 to see how much of the stability/turning gap is luck.

### Step 30: The Printable Springy Lower Leg (Variant D)
**Status**: 2026-10-11
- Design agent (`core/morphology/dog_paw.py`, `dog_mech.py`; report `runs/exp2_dog/1_design/puppy_ankle_fe.md`):
  - Ankle: a TPU95 I-shaped flexure (blade 7.78 x 14.3 x 20 mm, printed on its side), held in a PLA shin socket and
    the paw by 2 + 2 M3x30; pivot 18 mm above the paw centre (= the simulation's hinge). Pitch 1.50 N*m/rad (beam
    theory 1.84, own 3D FE 1.46-1.54), roll 4.6x, yaw 1.6x stiffer; 10 % strain at the +-0.4 rad stops (55x below
    break). Material: PolyFlex TPU95 datasheet V5.4, E = 33.6 MPa.
  - Soft paw: the round paw is a housing; a TPU sole disc (rim snaps into a groove, four curved arms carry a domed
    pad 1.5 mm proud) gives 10 N/mm (0.6 mm at 6 N, 1.2 mm at 12 N); the paw rim is the bump stop at ~15 N.
  - Tuning after a test print (ruler + weights on the ankle, kitchen scale + feeler gauges on the paw): set
    `TPU_E = 33.6 x k_measured / 1.50` in dog_paw.py and rebuild; stiffness ~ thickness^3 (tables in the README).
  - Genome now Spot knees (back_back) + ankle 1.5 / 0.03 / 0.4 + soft paws 1e4 / 100; robot 2.45 kg (paw link 26 g,
    shin 35 g); 67 STL parts. Checks clean except front/rear leg contact at large hip angles (24 deg each when both
    swing towards each other; front alone 64 deg with back_back vs 46 deg before).
  - Printed TPU damps far less than the trained 0.03: ankle damping is now randomized 0.005-0.03 in training.
- The B seed-2 replicate was invalid (its second stage was built from the design mid-edit) and was stopped; B is no
  longer needed after the owner's choice.
- Training the printable D (`runs/exp2_dog/10_printable_dog`, two seeds, two-stage recipe) started.
- Printable D results (two seeds, ~2.2 h each): seed 1: 3.0 % falls, turning error 0.82 rad/s, uneven stance (one
  front paw 0.59 vs 0.46), trot pairing only ~28 %; seed 2: **0.3 % falls (0-2 % even on levels 7-9), speed error
  8.0 cm/s, turning error 0.41 rad/s, CoT 4.16 (18.7 W), a clean trot at 0.3 m/s (98 % paired), ~2 paws down, walks
  slowly (0.09 m/s at 0.10)** - the best dog so far, with the real printable leg, masses and randomized TPU damping.
  Lesson: the spread between seeds of one recipe is large; train important variants at least twice and keep the best.
  Videos: `runs/exp2_dog/10_printable_dog/D_print_s2/videos/tour_flat.mp4`, `tour_stairs_up_l7.mp4`.

### Step 31: Imitating a Real Dog (list step 4, AMP)
**Status**: 2026-10-12, done; waiting for the owner's verdict on the look
- Motion data: AI4Animation dog motion capture (Zhang et al. 2018, CC BY-NC 4.0, non-commercial; kept out of git in
  `data/mocap/`, source in `data/mocap/ai4animation_dog/SOURCE.md`). 51 BVH clips of one dog; 8 chosen with plenty
  of walking and some trotting (~12 min walk, ~2 min trot).
- Retargeting (`core/mocap/bvh.py`, `core/mocap/retarget.py`): body frame from the hips and shoulders, each paw relative
  to its leg root, lengths scaled by the hip-height ratio (~0.44), time by its square root (Froude: a small dog with
  the same gait steps faster), inverse kinematics on the puppy's real legs (Spot knees). The skeleton's shoulder-hip
  line leans 3-11 deg nose-down per clip while the robot's chassis stands level: the clip's average pitch is removed.
  Result `data/mocap/puppy_dog_motion.npz`: 282 s at 50 Hz, robot walking speeds 0.27-0.44 m/s, <=1 % of angles at a
  joint limit. Kinematic replay: `runs/exp2_dog/11_imitation/videos/replay_walk.mp4`.
- Style reward (`core/sim/amp.py`, adversarial motion priors, Peng et al. 2021 / Escontrela et al. 2022): a small
  network (the "judge") learns to tell transitions (state, next state) of the real dog from the robot's; state = 12 leg
  angles and speeds, body velocity, turn rate, gravity direction (no height: stairs vs flat recordings). The robot
  earns 0..1 style per second times `amp_weight`. The judge trains inside the environment (every 4 steps), so the PPO
  trainer is unchanged; it is saved with the run (`amp_disc.pt`) and the evaluation reports a `style_score`. A saved
  judge can score any policy (`amp_disc_file`, frozen; `--iterations 0` warm start = evaluation only).
- Round 1 (`runs/exp2_dog/11_imitation/batch.py`): fine-tune D_print_s2 (1500 iterations, 50 critic warm-up) with no
  prescribed gait rhythm (`gait=free`, the dog's style decides walk or trot), style weight 3 and 6, plus a control
  without the style reward (same settings and length: separates imitation from "free gait + more training").

| run | falls | walk score | speed err | turn err | CoT | power | style (judge w3 / w6) |
|---|---|---|---|---|---|---|---|
| D_print_s2 (start) | 0.3 % | 0.463 | 8.0 cm/s | 0.41 | 4.16 | 18.7 W | 0.13 / 0.12 |
| free_ctrl | 0.3 % | 0.472 | 8.2 cm/s | 0.36 | 4.16 | 18.1 W | 0.09 / 0.11 |
| amp_w3 | 1.1 % | 0.418 | 8.9 cm/s | 0.41 | 3.96 | 17.1 W | 0.26 / 0.23 |
| amp_w6 | 1.4 % | 0.387 | 9.0 cm/s | 0.52 | 3.53 | 15.4 W | 0.71 / 0.47 |

  Style = mean 0..1 reward of a frozen judge during the evaluation (`cross_score.py`, `cross_score.json`); each judge
  was trained against its own dog, but both rank the dogs the same. Gait analysis on flat ground (`*_gait.txt`):
  - free_ctrl: a clean trot from 0.2 m/s (96-100 % paired), CoT 1.21 at 0.45 m/s (D_print_s2 1.44, 37 % paired):
    without the prescribed rhythm the trot gets cleaner at top speed; at slow speeds nothing changes (CoT 5.2 at 0.1).
  - amp_w3: like free_ctrl when fast, but stands still when asked for 0.1 m/s (the recorded dog never walks that
    slowly at robot scale: its slowest walk maps to ~0.27 m/s).
  - amp_w6: walks at 0.1 m/s on ~2.8 W (D_print_s2 ~11 W), CoT 1.9 / 1.5 / 1.6 / 1.3 at 0.1 / 0.2 / 0.3 / 0.45 m/s
    (D_print_s2 5.1 / 3.0 / 2.2 / 1.4); front paws longer on the ground than the rear ones (stance 0.5-0.7 vs
    0.4-0.5; real dogs carry ~60 % on the front legs); fewer paired touchdowns; overshoots slow speed commands
    (0.34 at 0.25: drawn to the recorded dog's walking speed); turns less accurately.
  - The gait tool divided the stride rate by 4 twice (now fixed: strides per paw per second, ~2 at 0.3 m/s); no
    earlier journal numbers used it.
- Defect found in the video (`videos/imitation_2x2.mp4`): amp_w6 swings ~60 deg to the right in 3 s when it starts
  walking straight from standing (steady walking drifts only +-0.03 rad/s). First suspicion: the recorded dog turns
  right more than left (-0.09 rad/s on average while walking, clips up to -0.27). Fix for that: the judge also learns
  left-right mirrored recordings (`amp_mirror`, default on).
- Measured with a new tool (`tools/dog/start_turn.py`: 64 dogs stand 2 s, then "straight ahead at 0.25 m/s" for 3 s;
  heading change; `--expr` head/tail moving, `--course N` per tile type, `--hold G` heading hold). (A first version
  read the heading with a w-first quaternion formula: Isaac Lab 3 stores x, y, z, w, so it always read ~180 deg;
  now via `quat_apply` on the forward axis. `play.py` now logs heading and velocity at every tour segment.)

| dog | mean change | mean abs change | worst | with heading hold (gain 2): abs / worst |
|---|---|---|---|---|
| D_print_s2 | +1.4 deg | 5.1 deg | 15 deg | 1.8 / 6.7 deg |
| free_ctrl | -1.2 deg | 4.8 deg | 17 deg | |
| amp_w3 | +0.3 deg | 3.8 deg | 12 deg | |
| amp_w6 | -1.2 deg | 9.8 deg | 43 deg | 2.4 / 8.8 deg |
| amp_w6m_s1 (mirrored) | -2.5 deg | 8.8 deg | 29 deg | 2.8 / 9.3 deg |

  So no right bias (the mean is ~0) but at style 6 the heading wanders about twice as much (the recorded dog never
  walks perfectly straight: half its walking frames turn noticeably); the mirror does not change that.
- Heading hold (`heading_hold_gain`, `play.py --heading_hold`, `compare_video.py --heading_hold`): while the command
  asks to walk without turning, the heading at that moment is kept and a drift becomes a corrective turn command
  (gain x error, max 0.5 rad/s) that the policy sees and tracks like any turn command (as Isaac Lab's heading
  commands; on the real robot the heading comes from the IMU). Works on the trained dogs without retraining (table).
  Use it for videos and in training from now on.
- Round 2 (`batch2.py`): style 6 with mirrored recordings, two seeds, plus one with slowed copies of the recordings
  (0.75x and 0.5x playback, `retarget.py --playback`, `data/mocap/puppy_dog_motion_slow.npz`) as examples of slow
  walking. Round 3 (`batch3.py`): the slow variant with seed 2.

| run | falls | walk score | speed err | turn err | CoT | power | at 0.1 m/s | style (judges w3 / w6 / w6m) |
|---|---|---|---|---|---|---|---|---|
| amp_w6m_s1 | 1.7 % | 0.372 | 9.5 cm/s | 0.49 | 3.24 | 13.9 W | stands | 0.78 / 0.52 / 0.47 |
| amp_w6m_s2 | 2.3 % | 0.405 | 8.6 cm/s | 0.47 | 3.12 | 13.3 W | stands | 0.77 / 0.55 / 0.52 |
| amp_w6m_slow_s1 | 0.4 % | 0.395 | 9.0 cm/s | 0.47 | 3.60 | 15.5 W | 0.08 m/s, CoT 1.6 | 0.61 / 0.45 / 0.44 |
| amp_w6m_slow_s2 | 1.8 % | 0.393 | 9.0 cm/s | 0.48 | 3.36 | 14.4 W | 0.07 m/s, CoT 2.4 | |

  - All style-6 dogs are 3-4x more dog-like than D_print_s2 / free_ctrl by all three judges (0.44-0.78 vs
    0.09-0.15) and need 25-30 % less power over the evaluation mix.
  - The slowed copies do what they should: both slow seeds walk at 0.1 m/s with the front paws down longer
    (stance 0.7 vs 0.35-0.4), while the normal-speed imitation dogs stand still there (as amp_w3).
  - Falls of the imitation dogs are mostly on the blocks tiles (5.8-7.3 %, hardest rows 7-10 %; D_print_s2 0.9 %;
    slow_s1 1.8 %): the recorded dog walked on a flat floor only, its style does not include careful stepping over
    rubble. Stairs, slopes and bumps stay at 0-2 %.
  - Starting from standstill without heading hold, slow_s1 swerves 19 deg on average (worst 45 deg); with heading
    hold (gain 2) 3.3 / 11 deg; slow_s2 with hold 1.6 / 5.3 deg (the best of all dogs).
- Video (flat ground, heading hold 2, tour now with a "slow 10 cm/s" segment): `videos/imitation_final_2x2.mp4`
  (D_print_s2 / amp_w6m_s2 / slow seed 1 / slow seed 2). By eye at 10 cm/s: D_print_s2 walks with legs spread wide and
  long reaching steps; the imitation dogs keep the legs under the body with short steps, more like a small dog.
- Candidate: `amp_w6m_slow_s1` (0.4 % falls, dog-like, walks slowly) with heading hold. Open: rubble (blocks), speed
  overshoot at 0.2-0.3 m/s; ideas: imitation weight lower on rough tiles, foot-clearance on blocks, recordings of a dog
  on uneven ground, heading hold also during training.

### Step 32: Benchmark of the Candidates, Robust Imitation
**Status**: 2026-10-12, training
- Owner (2026-10-12): all four look good, no clear favourite; the narrow stance of the imitation dogs needs testing on
  other terrain; rubbery paws give grip. Directions: running and back, legs / grip, terrain. Camera question: the
  owner means seeing a step or obstacle and lifting the right paw high enough - that is what the height scan already
  does in simulation (seeing halved the falls on the course); the real robot needs a depth camera to build that scan
  (next big step after choosing the walker); a room map is only needed later for navigation.
- Benchmark (`tools/dog/benchmark.py`, `runs/exp2_dog/11_imitation/benchmark_batch.py`, `benchmark.md`): the standard
  evaluation (1024 dogs, one episode, real walking commands, all difficulty rows), heading hold 2, in five scenarios:
  flat, course, slippery (grip 0.5), grippy (1.4), pushes (0.4 m/s every 2-4 s + mass/servo randomization).

| falls | flat | course | slippery | grippy | pushes | power flat / course |
|---|---|---|---|---|---|---|
| D_print_s2 | 0.0 % | 0.1 % | 0.5 % | 0.4 % | 0.4 % | 18.9 / 19.0 W |
| amp_w6m_s2 | 0.0 % | 1.1 % | 2.1 % | 1.9 % | 11.2 % | 11.9 / 13.3 W |
| amp_w6m_slow_s1 | 0.0 % | 0.5 % | 1.5 % | 0.5 % | 2.4 % | 14.7 / 15.7 W |
| amp_w6m_slow_s2 | 0.0 % | 1.2 % | 2.3 % | 1.8 % | 6.0 % | 12.2 / 14.5 W |

  The wide stance of the plain RL dog is the most robust (shoves, slippery rubble); the imitation dogs walk on 25-37 %
  less power with the same tracking on flat ground, but topple more easily. The grip was never varied in training
  (always TPU 1.0); a smooth tiled floor is ~0.5.
- Robust imitation (`batch4.py`): fine-tune amp_w6m_slow_s1 (two seeds) and, as the fair comparison, D_print_s2 with
  paw grip randomized 0.5-1.4 (`paw_friction_range`), shoves 0.3 m/s every 3-6 s (`push_velocity`, `push_interval`)
  and heading hold during training; then the benchmark again.
- Paused (2026-10-12/13): the disk was full; all three runs crashed at the start. Superseded by step 33, which
  includes robust_slow_s1 as its control (`robust_scan`).
- Servos (owner, 2026-10-20): the STS3215 stays. Torque was never the limit (step 25: only jumping needs low-ratio
  brushless actuators); hardware is bought once the simulation shows a build-worthy dog.

### Step 33: Terrain Map with Memory, and a Learned Memory
**Status**: 2026-10-20, done (`runs/exp2_dog/12_terrain_map/`, tables in `results.md`)
- Owner's idea: walking up stairs you look at them once, then climb without looking: you know where the step is and
  how high to lift your legs. The dog should know the terrain around it (also under and behind its body) from what it
  saw, and decide from the height whether it can climb something. Objects (wires, things on wheels) come later.
- What the seeing dog had: the perfect height scan (17 x 9 points, 5 cm apart, 35 cm behind to 45 cm ahead of the
  body centre), cast straight down, also under the body; no noise, no delay. Its memory: 8 frames of IMU and own
  commands (0.16 s); a feed-forward network. A real camera cannot see under the body, so a real dog needs memory.
- Agreed approach: (1) an explicit map that makes it possible, (2) a learned memory that learns from it.
- (1) Terrain map (`core/sim/terrain_map.py`, `terrain_map` / `map_*` settings, `--terrain_map depth|tof`):
  - a chest sensor at the centre of the chassis front, below the neck, 40 deg down (fixed to the body; a head camera
    would swing with the neck): depth camera (RealSense D405 / OAK-D-Lite class, 87 x 58 deg as 32 x 24 rays, 25 Hz,
    0.15-2 m, 1 % range noise) or VL53L5CX 8 x 8 ToF (45 x 45 deg, 15 Hz, 3 % noise); 40 ms until a frame is in the
    map, 1 deg orientation error per frame
  - points go into a 1.6 x 1.6 m grid (2.5 cm cells, a ring buffer per dog) placed with a drifting motion estimate:
    after 1 m walked ~3 cm sideways, 1 cm height, 2 deg heading, plus a per-episode distance misjudgement of up to
    5 %. Seen cells keep their height (memory); paws on the ground write the height under them (leg kinematics)
  - the policy reads the same 153 scan points from the map plus a known flag per point (306 inputs); the critic keeps
    the perfect scan. Warm start from a perfect-scan dog: the map heights take the scan's place, the known flags start
    with zero weights (`widen_checkpoint`); mirror augmentation covers the flags
  - sensor blackouts (`map_dropout_rate`, 0.3-1.5 s): in training 0.1 per second, in the "dark" benchmark 0.5
  - `tools/dog/test_terrain_map.py` (no simulator): grid, memory, ring buffer, window, step edges, drift statistics
    (after 1 m: 3.6 cm sideways, 1.0 cm height, 2.0 deg heading; the extra sideways comes from the heading error)
  - `tools/dog/map_check.py`: share of scan points the map knows ahead / under / behind the body and its height error,
    with memory and latest frame only, plus pictures (true ground / map / latest frame)
- (2) Learned memory (`core/evolution/distill_memory.py`): a GRU student that gets only this step's body senses,
  command, clock and the scan points the latest frame covers. It copies the map dog's actions (DAgger) and must
  rebuild the perfect height scan from its memory, also under its body (a picture shows what it "thinks" the ground
  is). 64-step sequences (1.3 s), exported as TorchScript (actions, hidden = student(inputs, hidden)).
- Batch (`runs/exp2_dog/12_terrain_map/batch.py`, ~6 h): smoke tests of every piece first (stops at a failure), then
  from amp_w6m_slow_s1 with the robust recipe of batch4, 1500 it each: robust_scan (perfect scan, control), map_depth,
  map_tof, map_depth_nomem (no memory); student_depth (from map_depth); map quality; benchmark course / dark /
  slippery / pushes (`benchmark.py` has the new "dark" scenario and runs memory students); stairs videos (level 7);
  tables in `results.md` (`summary.py`).
- Questions it answers: how much does a real sensor cost against the perfect scan; does memory matter (map_depth vs
  map_depth_nomem, mostly under the body and on stairs down); is the cheap ToF enough; can a learned memory replace
  the explicit map. Next after that: the climbing limit per step height and a "can I climb this?" rule on the map.
- Smoke run caught one bug (observation history size with the map's extra inputs); fixed, then the batch ran clean.
- Results (benchmark, 1024 dogs, real walking commands in a RANDOM direction; falls, walk score):

| run | course | dark (blackouts) | slippery | pushes | stairs down (rows 6-9) | stairs up (6-9) |
|---|---|---|---|---|---|---|
| robust_scan (perfect scan) | 5.8 % (0.33) | 4.5 % | 4.0 % | 8.8 % | 3 % | 22 % |
| map_depth (memory) | 10.4 % (0.29) | 9.9 % | 9.6 % | 14.9 % | 35 % | 47 % |
| map_tof (memory) | 12.8 % (0.31) | 13.1 % | 10.6 % | 16.1 % | 63 % | 41 % |
| map_depth_nomem | 9.6 % (0.39) | 10.8 % | 7.3 % | 14.5 % | 48 % (dark 58 %) | 38 % |
| student_depth (GRU) | 8.3 % (0.31) | 8.4 % | 6.2 % | 12.5 % | 34 % | 25 % |

  - The robust recipe itself cost course performance: amp_w6m_slow_s1 (re-checked with the new code) 0.4 % on the
    course, robust_scan 5.8 %. Shared by all five runs, so the comparison holds; the recipe needs another look.
  - A real sensor costs about half: 8-13 % vs 5.8 % falls, almost all on stairs (mostly going down) and the uphill
    slopes; bumps and rubble are as good or better.
  - Memory matters where expected: stairs down with blackouts 33 % (memory) vs 58 % (no memory). Overall the dog
    without memory was not worse (9.6 vs 10.4 %): it walks faster (walk score 0.39) and does better on rubble.
  - The learned memory (GRU student) is the best real-sensor dog: 8.3 % falls, stairs up as good as the perfect scan
    (25 % vs 22 % on rows 6-9), blackouts cost nothing (8.4 %). Its rebuilt ground under the body is off by 0.6 cm
    on average but blurred: levels and slopes right, step edges smeared (`student_depth/rebuilt_terrain.png`).
  - ToF 8x8: map knows 63 % ahead / 37 % under the body (depth camera 90 % / 51 %); worst on stairs down (63 %).
  - Why the map knows only ~half of the ground under the body: the evaluation walks in random directions (sideways,
    backwards); a forward-looking chest sensor only maps where the dog walks forwards. The perfect scan sees all round.
- Videos (stairs up / down, level 7, perfect scan / depth / ToF / no memory): `videos/stairs_climb_l7.mp4`,
  `videos/stairs_descend_l7.mp4`.
- Next ideas: evaluate with mostly-forward walking (as a real dog walks); train the map dogs longer (they started
  below the scan dog on the curriculum); RL fine-tune of the GRU student; a second, rear-facing sensor; revisit the
  robust recipe.


### Step 34: Walking Where the Sensor Looks
**Status**: 2026-10-23, done (`results_forward_test.md`, `results_forward_training.md`)
- Owner's questions: (1) is the gap between the depth-camera dog and the perfect scan mostly the walking direction
  (you would not walk down stairs backwards without looking first)? (2) Should the dog walk mainly forwards, and a
  bit sideways, where it can see, to get the most out of the sensor?
- New: `eval_heading_range` (walking direction of the evaluation commands); benchmark scenarios "forward" (every
  command within +-30 deg of straight ahead) and "forward_dark" (with blackouts); `map_check.py --forward`;
  `summary.py --runs/--scenarios/--out`.
- A: the step 33 dogs on forward / forward_dark (+ map quality when walking forwards) -> results_forward_test.md.
- B: +1500 iterations with mostly forward commands (forward up to 0.45 m/s, backwards and sideways at most 0.1 m/s,
  turning unchanged): fwd_scan (from robust_scan), fwd_depth (from map_depth), fwd_tof (from map_tof), and
  student_fwd_depth (GRU, from fwd_depth). Benchmarks, map quality, stairs videos -> results_forward_training.md.
- A (step 33 dogs, walking forwards): the map knows far more around the body (depth camera, under the body 51 -> 82 %,
  behind 35 -> 72 %). Gap to the perfect scan: depth + memory +4.7 -> +3.1 points, GRU student +2.5 -> -0.2 (7.6 %
  vs 7.8 % falls; stairs down 4.3 % vs 9.0 %), ToF unchanged (+7.8). Memory on stairs down with blackouts (rows 6-9):
  36 % with, 61 % without. The forward test is harder for every dog (perfect scan 5.8 -> 7.8 %): they walk into the
  stairs instead of along them; compare gaps, not absolute numbers.
- B (+1500 it with mostly forward commands), forward test:

| run | forward | forward_dark | gap to fwd_scan |
|---|---|---|---|
| fwd_scan (perfect scan) | 13.7 % | 15.1 % | - |
| fwd_depth | 12.3 % | 14.1 % | -1.4 |
| fwd_tof | 13.4 % | 19.0 % | -0.3 (dark +3.9) |
| student_fwd_depth (GRU) | 9.2 % | 11.3 % | -4.5 |

  - Trained the same way, the sensor dogs are now as good as or better than the perfect-scan dog: the gap is gone.
    The GRU student is the best dog of this batch (and its rebuilt ground under the body: 0.4 cm off, was 0.6).
  - But every forward-trained dog is worse in absolute terms than its step 33 version, also on the forward test
    (perfect scan 7.8 -> 13.7 %), and they fall 35-43 % with random directions (backwards/sideways up to 0.3 m/s,
    hardly practised: expected). Training curves look alike (row ~6, ~20 % falls with shoves and slippery floors);
    each warm start drops back to row ~1.7. Not resolved; suspects: (a) the robust recipe keeps pulling the gait away
    (the original candidate 0.4 % -> robust_scan 5.8 % on the course), (b) the forward test asks for up to 0.15 m/s
    sideways, trained only up to 0.1, (c) single seeds.
  - map_check of fwd_tof: the simulator crashed at start-up (not reproduced; the other checks ran).
- Videos: `videos/fwd_stairs_climb_l7.mp4`, `videos/fwd_stairs_descend_l7.mp4`.


### Step 35: Deliberate Stair Stepping
**Status**: 2026-10-23, prepared (`runs/exp2_dog/13_stairs/batch.py`, ~10 h, runs when the owner frees RAM)
- Owner (videos of step 34): the dogs walk into the stairs without lifting the paws high enough; a step up should be
  a controlled move (three paws steady until the fourth stands on the tread and pushes up), a step down a placed
  move that helps the balance. Stairs are continuous climbing for this dog: ~16-20 cm paw span on 15 cm treads.
  Reality check: house stairs (17-19 cm rise) are higher than the whole leg (hips at 16 cm); this dog can learn curbs,
  doorsteps, thresholds and small steps up to ~6-7 cm.
- Why the rewards did not ask for it: lift reward capped at 4 cm (steps at level 7 ~5 cm); the style reward comes from
  a dog on a flat floor; joint range 75 % (56 mm lift); no cost for stubbing a toe; same speed and rhythm on stairs.
- New (`locomotion_env_cfg.py`, all off by default): `stair_tread_range` (treads shortening over the rows, training
  only, `terrain.tread_stairs`; checked offline: 40 cm treads on the easiest row, 15 cm on the hardest),
  `foot_clearance_terrain` (lift target = highest true ground 3 cm behind to 10 cm ahead of the paw + 1.5 cm),
  `stumble_weight`, `foothold_weight` (touchdown within ~2 cm of a step edge; checked offline on a synthetic step),
  `stairs_support_weight` (three paws down while moving on stairs), `amp_skip_types` (no style reward and no
  discriminator samples on those tiles). Tests and videos always use the standard course.
- Batch: from amp_w6m_slow_s1 without the robust recipe (suspect for steps 32-34), depth camera + memory, mostly
  forward, 2000 it, seeds 1 and 2: base / stairs (treads, lift, stumble, foothold, no style on stairs and rubble,
  joint range 85 %) / support (stairs + three paws down). Then benchmark forward / forward_dark / course, a GRU
  student from the best stairs dog, stairs videos (with step 33's map_depth as "before"), results.md.
- Owner (watching again): it goes wrong when the dog turns on the stairs: it should go straight up; and a paw set
  down right in front of the next riser should rise straight up, not push forwards (that pushes the dog the wrong
  way). Added to the stairs variants: `stairs_square_yaw` (on stair tiles start facing the steps, +-15 deg) and
  `stairs_turn_scale` 0 (no turn commands on stairs; heading hold keeps it straight), both training only;
  `lift_first_weight` (per m/s a swinging paw moves forwards while still below a step up within 6 cm ahead).
  New benchmark scenario `stairs_straight` (straight on, facing the steps; step 33's map_depth measured too as
  "before"); videos with `--drive straight` (stand, 15 cm/s, 25 cm/s straight on).
- First stairs_s1 attempt stopped at ~it 200: the style reward was exactly 0 (base dog: 0.36). Not a code bug: the
  discriminator separated the gait perfectly (policy loss ~0.11), and the LSGAN style reward is 0 with no gradient
  for d <= -1. Cause: action_scale 0.75 -> 0.85 made the warm-started gait 13 % bigger from the first step. Fix:
  `--init_action_scale 0.75` (`train.rescale_actions`: the actor's output layer and noise are scaled by 0.75/0.85, so
  the servo targets start identical; checked offline). Log kept as `stairs_s1_attempt1_style0.log`. The penalties
  work (it 150: foothold -0.30, stumble -0.06, lift_first -0.01).
- Second attempt (with the rescale): style 0.23 at it 50 (base 0.50), then 0 from it 100. So the stairs terms
  themselves changed the gait on every tile (e.g. foothold also fires on bumps). Fix: `stairs_terms_types` = the
  amp_skip_types (stairs up/down, blocks): terrain-aware clearance, stumble, foothold and lift first only count on the
  tiles without style reward; elsewhere the dog keeps the base rewards and its own style. Log kept as
  `stairs_s1_attempt2_style0.log`.
- Results (2026-10-24, `runs/exp2_dog/13_stairs/results.md`; falls; stairs_straight = straight on, facing the steps):

| run | stairs_straight | forward | forward_dark | course (random dirs) | stairs up / down (straight, rows 6-9) | blocks (rows 6-9) | power |
|---|---|---|---|---|---|---|---|
| before (step 33 map_depth) | 9.6 % | 10.9 % | 11.8 % | 10.4 % | 52 % / 25 % | 22 % | 15.4 W |
| base s1 / s2 | 2.3 / 2.2 % | 1.2 / 2.5 % | 2.1 / 2.2 % | 8.8 / 8.3 % | 3 / 0-4 % | 22-24 % | 12-17 W |
| stairs s1 / s2 | 1.7 / 0.3 % | 1.2 / 0.3 % | 1.3 / 0.7 % | 7.5 / 7.5 % | 1 / 1-3 % | 1-6 % | 18 W |
| support s1 / s2 | 0.6 / 0.4 % | 0.5 / 0.7 % | 0.5 / 0.4 % | 11.3 / 11.9 % | 1 / 0 % | 0-1 % | 17-18 W |
| student (GRU, from stairs_s2) | 0.5 % | 0.6 % | 0.4 % | 6.5 % | 4 / 0 % | 3 % | 17 W |

  - Dropping the robust recipe (+ mostly forward) is most of the gain: base falls 1-2.5 % walking forwards, vs 10.9 %
    for step 33's dog. With random directions (backwards/sideways up to 0.3 m/s, hardly trained) all dogs stay at
    7-12 %.
  - The stairs rewards fix the rubble (blocks rows 6-9: 22-24 % -> 0-6 %) and make stairs reliable in both seeds;
    three paws down (support) adds little beyond that.
  - Costs: style on normal ground lower (style averaged over all tiles 0.08-0.12 vs 0.36-0.38 for base; per style
    tile ~0.2-0.3), and more power (17-18 W vs 12-17 W): bigger, higher steps (85 % joint range).
  - The memory student (GRU, latest depth frame only) is as good as its teacher on stairs and the best on random
    directions (6.5 %). Rebuilt ground under the body 0.4 cm off.
- Videos (straight on, level 7): `videos/stairs_climb_l7.mp4`, `videos/stairs_descend_l7.mp4` (before / base /
  stairs / stairs + 3 paws down).
- Owner (watching the videos): the "stairs" dog (bottom left) never climbed a step, it circled; base and support
  never got a rear paw on the first step. So the low fall rates on stairs mostly mean "did not try". The benchmark
  counted falls only. The top-left ("before") dog looked best and climbed, but tipped backwards (rearing).
- Proportions (measured from the design): centre of gravity 17.9 cm high, paws 16 cm apart front-rear and 19 cm
  across; standing on all four it tips backwards at 28 deg, the hardest stairs slope 18 deg: 10 deg margin. Height /
  wheelbase 1.12 (a Labrador ~0.85). The rearing in the video (~40 deg) is control, not only geometry. Ideas for later:
  relax the standing-height reward on stairs (crouch), let the policy use the neck as a counterweight, evolve a
  longer/lower body.

### Goals for experiment 2 (agreed 2026-10-24)
1. Get up from any fallen position (first: if it cannot, the design has a flaw).
2. Walk naturally and calmly (forwards, turning, slow and fast) on flat and uneven ground, with moderate shoves.
3. Steps and stairs at its own scale, found with a curriculum from easy to hard: the limit is what the STS3215 legs can
   do without too many problems (expected around 6 cm steps, 25-30 deg dog-scaled stairs). House stairs (17-19 cm
   steps) are higher than the whole leg: not for this dog. The help hound is a later, bigger design.
4. Decide whether something can be climbed (stop or go around).
5. Ready for the real robot (sim2real realism, measured servo model).
Training uses the perfect height scan from now on (accepted: it also sees under the body); a real sensor comes at the
end through a memory student (step 33-35: the student matched its teacher).

### Step 36: Getting Up, Being Pushed, and What Was Really Climbed
**Status**: 2026-10-24, prepared (`runs/exp2_dog/14_getup/batch.py`, ~4 h, runs on the owner's go)
- Climb measure (`evaluate`: `type_level_climb_m`, largest height change from the start per terrain type and row;
  kept by `benchmark.py` with `type_level_scores`): the stairs dogs are re-measured on stairs_straight.
- Get-up mode (`getup`, `getup_*` settings): starts standing / dropped upright / on the side / on the back with the
  legs anywhere in their range (harder starts later), no fall termination; rewards: upright, height (upright only),
  rising (potential-based), standing on four paws at height, near the rest pose; costs: energy, smoothness (low),
  joint limits, servo stall (> 2.5 N*m at < 0.5 rad/s). Trained from scratch, perfect scan, same inputs as the walkers,
  full joint range, 3000 it, seeds 1 and 2. `tools/dog/getup_test.py`: share standing (held 0.5 s) per start.
- Balance: `tools/dog/push_test.py`, one shove of 0.2-1.0 m/s while walking at 20 cm/s, falls per strength.
- `play.py --drive stand --env_set key=value` for a video of the four starts.
- Next (step 40, after the get-up work; owner's idea 2026-10-24): stairs as a goal task, like the crawler's goal
  task. One straight staircase per tile (flat approach ~1 m, N steps, top landing; some with open sides, some with
  walls; rise and tread harder over the rows; also downwards from the landing), a goal at the other end. The goal
  goes through the walking command (a navigation layer: turn towards it first, then straight on), so the walker stays
  one policy and cannot avoid the stairs. Falls: soft on the stairs (small penalty, get up and go on), hard when
  tumbling down (height more than one step below the best reached) or off the side. Measure: share reaching the goal,
  time, falls per kind, per rise and tread: the climbing-limit table for the STS3215 legs.
- First run stopped before the get-up training: the walker test from a standing start counted 1 % "standing". Check
  (`runs/scratch/standing_check.py`, 64 walkers standing still): upright and height 100 % right, nothing else
  touching, but all four paws down only 16 %: the walker rests one paw lightly (0.7 N vs ~6 N on the others).
  Standing now means at least three paws carrying > 0.5 N: 81 % of the standing walkers count. Climb and push results
  of that run are kept (unaffected); the walkers' get-up tests are redone.
- Results (2026-10-24, `runs/exp2_dog/14_getup/results.md`):
  - Climb measure: the step 35 dogs climb only 0.5-0.7 steps on the hardest rows (they avoid the stairs: the owner's
    observation); the original candidate climbs 3.4 steps (9 % falls), step 33's map dog 3.7 (54 % falls). Nobody
    really goes down (< 1.3 steps). Step 35's "0 % falls" meant "did not try".
  - Balance (one shove while walking at 20 cm/s; falls at 0.5 / 0.8 / 1.0 m/s): original 9 / 40 / 59 %, base 15 / 41
    / 64 %, stairs_s2 0 / 10 / 18 %, support_s2 14 / 25 / 39 %, memory student 1 / 8 / 23 %. The stairs dogs are
    much steadier under shoves.
  - Getting up (share standing within 8 s): walkers without get-up training 100 % standing, 97 % dropped upright,
    0 % side, 0 % back. Get-up training (3000 it from scratch, both seeds): 100 % / 100 % (0.1-0.3 s), side 0 %,
    back 0 %. Video (`videos/getup_getup_s1.mp4`): from the back it rolls onto its side within 1.5 s, then stays on
    the side; from the side it never gets over onto the belly.
  - Cause (design): the abduction joint is limited to +-0.5 rad (29 deg, `dog_mech.py`); hip and knee swing the legs
    parallel to the floor when the dog lies on its side, so no limb can push the body over; neck, head and tail move
    randomly (not driven by the policy). A design flaw as the owner expected: the dog cannot recover from a fall onto
    its side or back.

### Step 37: Which Design Change Lets the Dog Get Up From Its Side?
**Status**: 2026-10-24, prepared (`runs/exp2_dog/15_getup_design/batch.py`, ~5 h, runs on the owner's go)
- Owner: apply changes and see what works. New: genome `legs.abduction_limit` (default 0.5 rad, `dog_mech.py` used a
  fixed 0.5), variants 29 / 50 / 80 deg (`genomes/abd*.json`); `policy_drives_expression` (neck, head and tail become
  policy actions instead of random motions); `self_collisions` (on for all step 37 runs, so a leg cannot pass through
  the body or another leg). Action mapping made safe for when the policy drives every joint (`_remap_actions`).
- Mechanics (`dog_check --legs L0`): the current shoulder hits the deck, bowl and its own servo at +50 deg (10
  overlaps); at 80 deg 33 overlaps, inwards the paw also reaches the other front leg's shin at -60 deg. A larger range
  needs a redesigned shoulder; the simulation first shows which range is needed.
- Runs (get-up from scratch, 3000 it, seed 1): abd29 + head, abd50, abd50 + head, abd80, abd80 + head; get-up test per
  start, videos of the best and of abd80 + head, `results.md` (with step 36's dogs for reference).
- Smoke run 1 stopped: the self-collision check (walker standing, 3 s) gave 62 %; without self-collision the same
  3-s test gave 75 % (with: 81 % on 256 dogs): the test was too short, not self-collision. Check lengthened to 8 s
  (100 %).
- Results (2026-10-24, `runs/exp2_dog/15_getup_design/results.md`; share standing within 8 s, median time):

| variant | standing | dropped upright | on its side | on its back |
|---|---|---|---|---|
| step 36: 29 deg, head random, no self-collision | 100 % | 100 % | 0 % | 0 % |
| 29 deg + neck/head/tail by the policy | 100 % | 100 % | 100 % (1.1 s) | 100 % (1.1 s) |
| 50 deg, head random | 100 % | 97 % | 0 % | 0 % |
| 50 deg + head | 100 % | 100 % | 0 % | 0 % |
| 80 deg, head random | 100 % | 100 % | 100 % (1.3 s) | 100 % (1.2 s) |
| 80 deg + head | 100 % | 100 % | 100 % (0.8 s) | 100 % (0.9 s) |

  - The design flaw is solved without changing the shoulder: with neck, head and tail driven by the policy (all
    STS3215 12 V, as in the printable design) the dog gets up from its side and its back every time, in ~1.1 s.
    Video `videos/getup_abd29_head.mp4`: a head/neck swing with leg moves rolls it onto its belly by 0.8 s, it
    pushes up and stands calmly at 1.5 s. A larger abduction range (80 deg) also works on its own, but would need a
    redesigned shoulder.
  - 50 deg failed with and without head (single seeds): getting up is all-or-nothing (the trick is found or not),
    so this most likely means the training did not discover it, not that 50 deg is worse than 29 deg.
  - Consequence: neck, head and tail must be policy-driven in the walking dog as well (for getting up, and possibly
    as a counterweight on stairs). Expression (nodding, tail wagging) then becomes a command the policy follows when
    nothing else needs the head, as Microduck does with its head command.
- Flip video (owner's request; `tools/dog/flip_video.py`): getup_abd29_head stands; each time it has stood 1 s it is
  thrown up with a 14 rad/s tumble (1.3 m/s up), each flip 36 deg further round (relative to its heading), 10 flips.
  Frames confirm real flips (upside down 0.25-0.5 s after the kick). It stood up 10/10, on average 1.5 s after the
  kick (0.8-2.9 s; slowest at 108 deg). `runs/exp2_dog/15_getup_design/videos/flips_abd29_head.mp4` (+ .json).

### Step 38: Keep Balance, and Get Up After a Fall
**Status**: 2026-10-24, running (`runs/exp2_dog/16_recover/batch.py`, ~3.5 h; the stairs goal task moves to step 39)
- Goal: one dog that walks, catches itself after a shove and, if it does go over, gets up and walks on.
- New: recover mode (`recover`, `recover_*`: a fall does not end the episode unless the dog stays down 6 s; get-up
  rewards while fallen; a fall costs once; the style judge ignores fallen states), topple shoves (`topple_push`,
  `topple_interval`, `topple_spin`), both training-only for tests/videos. `core/evolution/distill_recover.py`:
  DAgger from two teachers chosen by posture (walker while upright, get-up dog from a fall until it stands), actions
  rescaled from the walker's 0.75 range to the student's full range, each teacher sees what it was trained with
  (walker: last actions in its scale; get-up dog: a zero command). `tools/dog/recover_test.py` (one shove while
  walking: stays up / falls and gets up / stays down), `flip_video.py --vx --push` (walking shove and flip videos).
  The style judge now looks at the legs only (the recording has no neck, head or tail).
- Batch: distill 400 it -> RL polish 2000 it (walker recipe + recover, topple 0.8-1.3 m/s every 6-12 s), tests,
  shove/flip videos while walking, flat tour before/after.
- Results (2026-10-24, `runs/exp2_dog/16_recover/results.md`). One shove while walking at 20 cm/s (stays up / falls
  and gets up / stays down):

| dog | 0.5 m/s | 0.9 m/s | 1.3 m/s | 1.5 m/s |
|---|---|---|---|---|
| original walker | 90 / 1 / 10 % | 55 / 3 / 42 % | 7 / 1 / 92 % | 1 / 7 / 92 % |
| distill (two teachers) | 81 / 19 / 0 % | 45 / 55 / 0 % | 23 / 76 / 1 % | 4 / 95 / 1 % |
| recover (+2000 it RL with shoves) | 94 / 6 / 0 % | 62 / 37 / 1 % | 25 / 67 / 8 % | 12 / 75 / 13 % |

  - Getting up from the side / back: distill 99 / 100 % (1.9 s), recover 70 / 88 %.
  - Walking (standard benchmark, a fall ends the episode; falls, walk score): original forward 0.7 % (0.41), course
    0.8 % (0.41), 3.6 steps up on the hardest stairs; distill 6.5 % (0.39), 3.9 % (0.37), 1.7 steps; recover 4.4 %
    (0.25), 6.9 % (0.22), 1.3 steps.
  - The combination works: the distilled dog almost never stays down after a shove and walks nearly as well as the
    original (walk score 0.37-0.39 vs 0.41), but it falls a little more easily and climbs less. The RL polish made
    it steadier under shoves of 0.5-1.3 m/s, but it partly forgot getting up from the side (70 %) and walks worse
    (0.22-0.25). It had not converged: the curriculum was still rising (row 4.4 at the end vs ~6 for the original)
    and the dogs were down ~19 % of the time (topple shoves every 6-12 s plus the course): too many shoves.
  - Style stayed alive during the polish (0.23-0.30 vs 0.36 for the original).
  - Videos (polished dog, walking at 20 cm/s): 10 shoves of 1.0 m/s: went down 6x, stayed on its feet 4x, got up 6/6
    (1.7 s on average); 10 flips: got up 8/10 (1.4 s). Flat tour before/after: `videos/tour_flat.mp4`.
- Next: polish longer while keeping the teachers as an anchor (Microduck's gated behaviour cloning: still copy the
  get-up dog while fallen and the walker while upright, RL on top), and far fewer topple shoves.

### Step 39: Balance Training With the Teachers as an Anchor
**Status**: 2026-10-24, running (`runs/exp2_dog/17_recover_bc/batch.py`, ~4.5 h; the stairs goal task moves to step 40)
- `core/evolution/gated_bc.py`: PPO with gated behaviour cloning (Microduck's VelStand recipe). After each PPO update
  a few copying epochs on the same rollout: samples tilted > 35 deg copy the get-up dog (zero command), < 25 deg the
  original walker (actions rescaled 0.75 -> 1.0, neck/head/tail at rest), in between RL only; separate Adam (3e-4) on
  the actor, skipped during the critic warm-up. `train.py --bc_walker --bc_getup` switches it on. Checked offline with
  the real teacher networks: upright samples go to the walker, lying ones to the get-up dog.
- From step 38's distilled dog, topple shoves 0.8-1.3 m/s every 15-25 s (was 6-12 s), 3000 it, seeds 1 and 2; tests as
  in step 38; videos for the better seed; flat tour original / step 38 polish / this one.
- Results (2026-10-25, `runs/exp2_dog/17_recover_bc/results.md`), both seeds alike:
  - Walking (standard benchmark): forward 1.1 / 0.4 % falls, walk score 0.45 (original walker 0.7 %, 0.41); course
    0.7 / 1.0 % (0.43); stairs straight 1.3 / 1.2 %, 3.5 / 3.9 steps up on the hardest rows (original 3.6); 12.4 W
    (original 14.3 W). As good as the original walker or better, at less power.
  - Getting up from the side / back: 100 / 100 % in 1.5-1.7 s (step 38 polish: 70 / 88 %).
  - One shove while walking (stays up / gets up / stays down): 0.5 m/s 86-90 / 10-14 / 0 %, 0.9 m/s 52-53 / 47-48 /
    0 %, 1.3 m/s 3-6 / 91-95 / 2-3 %, 1.5 m/s 0-1 / 93-97 / 2-7 %. It almost never stays down; it stays on its feet
    about as often as the original walker (which stays down 42-92 % after shoves of 0.9-1.5 m/s).
  - Videos (bc_s2, walking at 20 cm/s): 10 shoves of 1.0 m/s: on its feet 3x, down 7x, up again 7/7 (1.5 s); 10 flips:
    up again 10/10 (1.9 s). `videos/tour_flat.mp4`: original / step 38 polish / bc_s2.
  - The anchor did what it should: nothing was forgotten (walking, climbing, getting up), and getting up after shoves
    is now near-certain. Catching itself without falling did not improve beyond the original walker.
