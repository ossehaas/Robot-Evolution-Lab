# Project Evolution: Evolutionary Robotics Lab

## Project Goal
Build, simulate and physically manufacture tiny robots (crawlers/walkers) whose bodies and controllers are
found by evolution and reinforcement learning. The workflow bridges simulation (Sim) and reality (Real),
ending in 3D-printed robots driven by the trained policies.

## Toolchain (as installed, 2026-09-30)
- **NVIDIA Isaac Lab 3.0** (tag `v3.0.0-EA`) on **Isaac Sim 6.1**, Python 3.12, PyTorch 2.11 (cu128, native
  sm_120 support for the RTX 5080 Laptop GPU, 16 GB). Environment: `IsaacLab/.venv`, created with `uv sync`
  from Isaac Lab's lockfile. Setup details: `.env_init/readme.txt`.
- **Physics**: PhysX 5 through Isaac Sim (default). Newton / MuJoCo-Warp runs without Isaac Sim and starts
  faster, but is not yet validated for these robots.
- **RL**: RSL-RL PPO (skrl also installed).
- Run everything with `lab.cmd` (Python of the Isaac Lab environment, repository on the import path).

## Repository Architecture (mono-repo)

### 1. `/core/` (the engine, shared by all robots)
- `core/morphology/`: the **genome** (compact JSON description of a robot) and the builder that turns it
  into a simulation USD, a URDF, printable STL parts, a bill of materials, a servo map for firmware and a
  preview image. Physical constants of the parts (SG90, PLA, TPU, electronics) are in `components.py`.
- `core/sim/`: the Isaac Lab task. Servo actuator model (DC-motor torque-speed curve, gearbox armature,
  friction), forward-locomotion environment with asymmetric actor/critic observations, domain randomization.
- `core/evolution/`: `train` (one robot: build, PPO, evaluate, export policy), `play` (viewport or video),
  `evolve` ((mu + lambda) genetic algorithm with Lamarckian warm starts), `report` (leaderboard, plots).
- `core/hardware/`: (planned) base ESP32 firmware and communication.

### 2. `/prototypes/` (the lab)
One folder per robot under development.
- `prototypes/simple_leg/`: the first 2-servo leg (hand-written URDF + USD). `sim/spawn_leg.py` is a servo test
  rig, `sim/train_leg.py` the smallest complete RL example (reach task).
- `prototypes/crawler_q4/`: 4-legged, 8-servo crawler defined by `genome.json`; reference robot for the
  locomotion pipeline and a seed for evolution.

### 3. `/releases/` (the hall of fame)
Robots that walked on the real table: final STLs, frozen policy, firmware and a build README.

---

## The Sim-to-Real Pipeline

### Phase 1: Virtual sandbox (done)
Isaac Lab installed cleanly, physics validated (`test_sim.py`), first leg simulated and trained.

### Phase 2: Bodies and brains (current)
- **Generative bodies**: robots come from genomes instead of hand-drawn URDFs, so the same description
  drives simulation, printing and evolution. The genome is the source of truth; URDF is an export.
- **Brains**: PPO with thousands of parallel copies per robot. Reward: forward speed, heading, uprightness,
  energy, smoothness. Domain randomization: link masses +-15 %, servo gains +-25 %.
- **Evolution**: a genetic algorithm over genomes (mutation of dimensions, adding/removing limbs and
  segments, crossover of limb pairs). Each candidate is trained and scored by walking distance; children
  with the same joint layout as a parent inherit its policy as a warm start.
- NEAT remains an option for evolving controller topology, but PPO + morphology evolution comes first.

### Phase 3: Hardware for tiny robots (bill of materials)
Per robot, generated in `build/print/BOM.md`:
- **Controller**: ESP32-S3 SuperMini (runs the small MLP policy; ONNX export is produced by training).
- **Servo driver**: PCA9685 (16 PWM channels over I2C).
- **Sensing**: MPU6050 IMU (gravity direction + gyro are the policy's body observations).
- **Actuators**: SG90 micro servos (1.8 kg*cm stall, 0.1 s/60 deg); up to 12 per robot.
- **Power**: LiPo 2S 7.4 V, 350 mAh, with a 5 V / 3 A buck converter for the servos.
- **Print**: PLA links and chassis, TPU feet.

### Phase 4: Sim-to-real
1. Pick a winner from evolution; print the parts from its `build/print/stl/`.
2. Assemble; `servo_map.json` gives each joint's PWM channel, direction and limits.
3. Deploy `policy.onnx` on the ESP32 at 50 Hz: inputs are IMU gravity + gyro, joint angles, last command and a
   gait clock; outputs are servo angles.
4. Open question: SG90s have no angle feedback. The policy already gets no joint velocities, but it does get
   joint angles. Options: replace the measured angle by the commanded angle and retrain, tap the servo
   potentiometer, or use feedback servos (e.g. Feetech SCS0009). Decide before the first physical build.

## Next Steps
Milestones are tracked in `.documentation/roadmap.md`:
- **M1** steerable walking (command following) - done
- **M2** rough terrain + evolution on terrain - next
- **M3** obstacle avoidance and goal navigation with a VL53L5CX 8x8 ToF sensor
- In parallel: build the real crawler and the ESP32 firmware skeleton.
