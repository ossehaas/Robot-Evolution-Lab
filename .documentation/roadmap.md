# Roadmap

Agreed on 2026-09-30, after the crawler learned to walk straight ahead. Each milestone builds on the previous
one. Update the status line when a milestone starts or finishes; details go in `project-journal.md`.

## M1. Steerable walking
**Status**: done (2026-09-30). crawler_q4 tracks commands with score 0.85 (untrained 0.38): mean velocity
error 2.8 cm/s, turn-rate error 0.15 rad/s, no falls. Video: `runs/exp1_crawler/1_walking/crawler_q4_steer/videos/crawler_q4_steering_tour.mp4`.
2026-10-02: command ranges made symmetric (forward/backward/sideways up to 30-35 cm/s; sideways was capped at 10 cm/s
by design, not by the body). The robot now tracks all four directions within ~7 % up to 30 cm/s (`core.evolution.probe`).

The policy follows a command (forward speed, sideways speed, turn rate) instead of always walking along +x.
- Commands resampled during episodes, including standing still and turning in place.
- Command is part of the policy observation, so the real robot can be driven like an RC car
  (later: from a phone over the ESP32's Wi-Fi).
- Evaluation and fitness: how well commands are tracked, not only top speed.
- Demo: scripted drive (forward, turn, turn in place, backward) recorded to video; keyboard driving in the viewport.

**Done when**: the crawler tracks forward/turn commands in simulation, shown in a video, and evolution can use
command tracking as fitness.

## M2. Rough terrain + evolution on it
**Status**: in progress (2026-10-02). Terrain + curriculum + foot air-time reward done; crawler_q4 trained on it
(tracking score 0.46 over all levels vs 0.39 for the flat-ground policy). First evolution run (`runs/exp1_crawler/2_evolution/evo_rough_001`)
failed instructively: it evolved a body that stands still (see journal step 11). Fitness, reward and training
schedule fixed; evolution run 2 (`runs/exp1_crawler/2_evolution/evo_rough_002`) started 2026-10-02 evening.

- Isaac Lab terrain generator: slopes, stairs, random bumps, gaps, with a difficulty curriculum.
- Blind locomotion: no new sensors, the robot feels the ground through its body (IMU + servos).
- Run evolution on terrain. On flat ground almost any four-legged body works; on steps and slopes body shape
  (leg length, extra segments, ground clearance) should matter and give visibly different winners.

**Done when**: an evolution run on terrain produces bodies that clearly beat the flat-ground crawler on it.

## M2b. Better senses: realistic IMU, memory, optical flow
**Status**: in progress (agreed 2026-10-03)

The robot already has an MPU6050 IMU (accelerometer + gyro) and the policy uses its gravity direction and turn
rate. What it lacks is knowing whether it actually moves: a robot pushing against a stair edge does not notice it
is stuck. Integrating the accelerometer to get speed drifts within seconds, so:
- Realistic IMU in simulation: raw accelerometer + gyro readings with MPU6050-like noise and bias (the current
  simulated IMU is perfect, which the real one is not).
- Short memory (observation history): from recent IMU + joint readings the policy can infer its speed and contacts
  (how ANYmal / Unitree controllers work without a speed sensor).
- Optional optical-flow sensor (e.g. PMW3901, 1-2 g, ~EUR 10-15) looking down at the floor: measures speed over
  ground directly, like a mouse sensor. Simulated with noise and dropouts.
- Real world (with the first physical build): overhead camera + ArUco marker as ground truth for test runs.
- Videos: evolution records its best body after every generation; milestone videos on stairs/blocks.

**Done when**: on rough terrain, the crawler with the new senses clearly beats the current one (walk score,
especially on stairs and blocks), and the comparison is shown in a video.

## M2c. Specialists first, then one combined model
**Status**: done for the POC (2026-10-04). Distilled student beats all teachers (0.278 vs generalist 0.256); RL polish
afterwards, even fixed (critic warm-up, inherited exploration noise), falls back to RL's own level (0.255). Hierarchy
(navigator steering the frozen walker) adds a skill on top of existing ones in minutes. Open: students with other senses.

1. Train specialists per skill (flat walking, stairs, blocks, ...) and record their scores: these are the target
   numbers for what this body can do at each skill.
2. Combine them into one model without starting from scratch: distillation (each specialist teaches a single
   student; the student also practises in its own situations and is corrected there, then is fine-tuned with RL on
   mixed terrain). Alternative to test if needed: keep the specialists and add a small selector (mixture of experts).
3. Measure the combined model per skill against its specialist ("stairs: x % of the specialist").
4. Sensor question: the student never gets the terrain label, only its own senses. Distil into students with
   different senses (IMU only / + memory / + downward distance sensor) to find which sensors the real robot needs
   to recognize its situation.

**Done when**: one model reaches a known share of each specialist's score, and we know which senses that requires.

## M3. Obstacle avoidance and goal navigation
**Status**: done for the POC (2026-10-04): 9.3 goals/min through pillar fields, 96 % reach a goal, 6.9 % of the time
in contact with a pillar, with a modelled VL53L5CX and a camera-like goal signal; also on a bumpy floor (7.4 goals/min).
Earlier note: Motivated by the stairs tests (2026-10-03): a robot that sees the terrain climbs 33 % better, while
stronger servos and a blind foot-lift reward gave nothing. First step: simulate the real VL53L5CX for terrain.

- Add a distance sensor: VL53L5CX time-of-flight, 8x8 zones, ~4 m range, ~3 g, I2C, ~10 EUR, fits the ESP32.
- Simulate it with Isaac Lab's ray-cast sensor using the same 8x8 grid and field of view.
- Task: reach a goal point in a room with random boxes and walls without touching them.
- Sensor mount position and tilt become genes, so evolution decides where the "eye" goes.

**Done when**: a robot reaches goals through random obstacle fields in simulation with a low collision rate.

# Experiment 2: a cute robot dog (started 2026-10-05)

Goal: a natural-moving, cute toy dog of at most 50 cm that can really be printed and built (for a 12-year-old, and
cute enough for anyone). Generator: `core/morphology/dog.py` (presets puppy / terrier / bulldog), renders in
`runs/exp2_dog/1_design/`.

- Hardware: 15 Feetech STS3215 bus servos (12 V, 30 kgf*cm, position feedback): 12 for the legs (sideways, hip,
  knee; knee driven by a pushrod from a servo at the top of the thigh), 3 for expression (nod, head tilt, tail wag).
  ESP32-S3, bus servo adapter, IMU, 3S LiPo, two round displays as eyes. ~35 cm long, ~1.9 kg.
Long-term vision (2026-10-06, with the co-founders): a robot assistance / guide dog for disabled or blind people
(real ones are expensive, scarce and slow to train). Not now; in steps. Everything the toy dog learns about walking
well should transfer to a bigger dog. Priority now: walking well (D2).
Goals for this dog (agreed 2026-10-24, journal "Goals for experiment 2"): get up from any fall (first: a design
check), walk naturally with moderate shoves, steps and dog-scaled stairs up to what the STS3215 legs can do (found
with an easy-to-hard curriculum; house stairs are higher than its leg: that is for the later, bigger help hound),
decide what it can climb, sim2real readiness. Train with the perfect height scan; a real sensor comes at the end
through a memory student.

- D1 Design: pick a style (with the future owner), make the parts printable (servo mounts, split shells).
- D2 Walk well (priority): walk and trot with a dog-like gait; head and tail move independently while walking.
  Done when (measured with `core.evolution.probe` and the evaluation):
  - speed within 15 % of the command from 10 to 35 cm/s, forwards, backwards and sideways
  - turning equally well both ways (left and right within 20 % of each other and of the command)
  - no falls on flat ground; < 5 % falls on a gently rough floor (1 cm bumps, 10 deg slopes) and under small pushes
  - natural: trot rhythm, body at standing height, only paws touch the ground (knees stay > 4 cm up)
  - efficient: report the cost of transport (energy per kg per metre) so gaits can be compared
- D2b Flexible back (idea from the owner, 2026-10-07): does a back that can bend make walking on rough ground more
  stable (and later stairs and jumping easier)? Spot, ANYmal and toy robots all have a stiff body. First test
  passive: body split into a front and rear half joined by a spring hinge, no motor, so the walking policy can be
  fine-tuned from the stiff dog. Variants: stiff / twist (roll) / bend (pitch) / both, each with a few spring
  stiffnesses. Measure falls per terrain level and type, energy, paw contact. Expectation: twist helps on uneven
  floor, bend on steps (and jumping). Later: stiffness and spine axes as genes for evolution.
  First round (2026-10-07): a passive back the policy cannot sense makes it less stable (stiff 17.9 % falls, twist
  20.5-21.8 %, bend 20.4 %, both 24.6 %). Open follow-ups: back angle as a sensor, training from scratch with a back,
  stairs/jumping (where a back should matter most).
- D2c Stairs (dog-scaled: ~5 cm rise, 15 cm treads) on the dog-sized course `dog_rough`; needs 75 % of the joint range
  (paw lift 56 mm), no hardware change. Jumping is not possible with STS3215 servos (scripted test: no flight; slow and
  high reflected gearbox inertia); it needs low-ratio brushless (quasi-direct-drive) actuators, which matters for a
  bigger guide dog. Possible later: learn a jump in simulation with such actuators to learn the method.
  2026-10-09: seeing (ideal height scan, 80 x 40 cm) halves the falls on the dog course (7.1 % vs 16.0 % blind);
  stairs down at level 8: 20 % vs 80 %. Next: a realistic sensor instead of the ideal scan (chest-mounted 8x8 ToF or a
  depth camera), and more training to push the curriculum beyond level ~6.
  2026-10-20 (journal step 33): terrain map with memory from a chest depth camera or 8x8 ToF with drifting odometry,
  and a learned (GRU) memory taught by it. Then: climbing limit per step height and a "can I climb this?" rule.
- D2d Gait and body (2026-10-11): soft TPU paws give a real gait (trot on ~2 paws instead of hopping), a springy
  ankle saves energy; the owner chose Spot-style knees (all backwards) for the natural look. Printable version of the
  springy ankle + soft paws in progress. Next: imitation of real dog motion (AMP style reward, AI4Animation dog mocap,
  non-commercial licence) on the chosen leg design.
  2026-10-11: printable D (D_print_s2): 0.3 % falls, the best plain RL dog.
- D2e Imitation of a real dog (2026-10-12): AMP style reward from retargeted dog motion capture (mirrored, plus
  slowed copies for slow walking): 3-4x more dog-like by the discriminators, 25-30 % less power, legs under the body
  and front paws loaded longer (as a real dog), walks at 0.1 m/s on a third of the energy. Needs heading hold (IMU
  heading -> corrective turn command) against swerving at the start. Open: careful stepping over rubble (blocks
  falls 2-7 % vs 1 %: the recordings are flat floor only), speed overshoot at 0.2-0.3 m/s, the owner's verdict.
- D3 Tricks and getting up: sit, lie down, give a paw, play bow; get up after falling over.
- D4 Personality: idle behaviour (look around, head tilt, tail wag, blink), come when called / follow a ball.

## Later ideas
- Real environments from video (owner's idea, 2026-10-10): film a real room, staircase or garden and turn it into a 3D
  environment to train and test in. Three techniques to compare:
  1. 3D Gaussian Splatting: fast, photorealistic rendering from a phone video; great for camera-based perception and
     visual realism, but it gives no clean surfaces for physics (a collision mesh has to be extracted separately).
  2. NeRF (Neural Radiance Fields): similar idea (neural scene from video), slower to train and render; meshes can be
     extracted, quality varies.
  3. Photogrammetry (e.g. Meshroom, RealityCapture, phone LiDAR apps): classic mesh + texture from photos; directly
     usable as a collision mesh in Isaac Sim (closest to what the physics needs).
  Likely combination: photogrammetry (or a mesh extracted from a splat) for the physics, Gaussian splatting for what a
  camera on the robot sees. Fits the guide-dog vision: train in a copy of the real home or street.
- Kinect (owner has an Xbox 360 Kinect = v1, 2026-10-11; libfreenect or Kinect SDK 1.8 + USB/power adapter): live metric depth + human skeleton tracking (v2). Useful for
  teaching *what* to do (gestures as commands: sit, come, point at a goal for the navigator), "follow me" (guide dog),
  tracking the real robot as ground truth, and quick room scans (KinectFusion) for the simulator. Not for teaching
  *how* to move like a dog (its body tracking is human-only): use dog videos + animal pose estimation or dog motion
  capture for that (list step 4). Drivers: v1 (Xbox 360) libfreenect + USB/power adapter; v2 (Xbox One) Kinect SDK 2.0
  + Kinect Adapter for Windows. Indoors only (IR washes out in sunlight).
- Get up after falling (trained recovery behaviour).
- Energy-efficient gaits: evolve for distance per battery charge instead of speed (calmer, more realistic gaits).
- Following or chasing a moving target (camera or beacon).

## In parallel: build the real crawler
- Print `prototypes/crawler_q4` (parts and BOM are generated) to measure how optimistic the simulation is.
- Decide first how the robot knows its joint angles: SG90s report none. Options: use commanded angles as policy
  input, tap the servo potentiometer, or switch to feedback servos (e.g. Feetech SCS0009).
- ESP32 firmware skeleton in `core/hardware/`: IMU read, ONNX policy at 50 Hz, PCA9685 output via `servo_map.json`.
