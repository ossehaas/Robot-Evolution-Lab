# Starting a new robot: the base point (2026-11-06)

What we know after the crawler and 53 journal steps with the dog, as the checklist for the next body. In order. The
entry exam itself is in `base-skills.md`; this list is what comes around it. Change it when a lesson turns out wrong,
and note why.

## 1. Before drawing

- The goal in a sentence and the skills it needs (walk, stairs up / down, get up, stand still, ...). One model for all of
  them in the end, a recognizer outside it only sends a signal (step 53).
- Hard limits: the printer (Centauri Carbon 2: 256 x 256 x 256 mm, PLA + TPU 95 in one print), the servo (its real
  torque, speed, latency; STS3215 12 V for the dog), mass budget, battery.
- Which part of the robot is new, and what in the simulation stands in for it. Every stand-in is a risk (step 46-47).

## 2. The simulated robot is the printed robot

- Colliders, masses and inertia come from the CAD, not from a simpler shape. Every dog up to step 46 learned on a ball paw that
  does not exist; the real shape cost a third of its walk (steps 46-47).
- Check colliders against the CAD (within ~0.3 deg, `tools/parts/paw_cad_check.py` as the model) and materials per part
  (PLA vs TPU grip: it decided the stuck paws, step 46).
- What the simulator does not model, written down per robot: backlash, flex of printed parts, servo heat, sole
  compression beyond its sphere model. Domain randomization for what we can (latency jitter 15-30 ms, +-1 deg backlash,
  friction 0.7-1.3: the Microduck / BAM lessons).

## 3. Parts first (the funnel, step 48)

- Level 0, CAD geometry (seconds), level 1, the part alone on a stand in the simulator (minutes), level 2, trained
  robots on the new part without training (~1.5 h), level 3, retraining (hours). Many designs in at the top.
- Rank by what predicted the dogs: the load on the hard housing did, the stand's work per step did not.
- A new part type gets its own stand (`tools/parts/paw_stand.py` is the example).

## 4. The entry exam

- `base-skills.md`: design checks A1-A4 (`tools/base/design_check.py`, CPU, seconds), then the learned basics B1-B8.
  Nothing harder before it passes.

## 5. Prove a skill is possible before teaching it

- Plans without learning (`tools/dog/stairs_plan.py`, replay open loop) say what the body can do. Then a failed training
  is the training's fault, not the body's (steps 41, 50).

## 6. Training

- Start from the last recipe that worked, not from defaults; record every override in `result.json`.
- New skills come from a teacher to copy (a plan, a dog, a demonstration), not from a small extra reward: a reward ~35 x
  smaller than the walking pay taught nothing (step 50), a stalling cost moved nothing (step 52), and removing the walking
  teacher wrecked the dog (step 49).
- Every training forgets what it does not practise (step 51: going down cost the walk). Keep old skills in: copy from a
  frozen version (the owner's 80 / 20) or merge by copying specialists (`core/evolution/distill_skills.py`).
- Every situation the robot must handle gets practice starts, including lying on its side and back: step 53 lost
  getting up because no episode started fallen.
- Measure what it achieves, not only what it avoids: few falls also fits a robot that never tries (step 35).

## 7. Tests and comparing

- The standard set for a dog: `benchmark.py` (forward, course, cost of transport), `recover_test.py`, `getup_test.py`
  (all four starts), `tilt_test.py`, `stairs_goal_test.py` (clean = without any fall), `lunge_test.py`. A "nothing
  forgotten" table against the base in every summary.
- Two seeds for anything that matters; a difference within the seeds' spread is no difference.
- Videos of every result; the owner judges the motion by eye (calm, controlled beats a small metric gain).

## 8. Running it

- Smoke test first, running every test tool and a video, not only training (step 51 lost a night of tests to a crash no
  smoke saw). The batch stops when a test leaves no result.
- GPU only on the owner's go; never interrupt a run for speed; record each step in the journal, with the date it ran.
- Check the simulator itself after any environment change: something must fall and collide (step 1 "passed" without
  physics).

## 9. Before building

- The robot passes the entry exam and its goal tasks in simulation with the real parts, across two seeds.
- Test prints of the risky parts first (the 4.5 mm sole, its stop pin), measured against the simulation (stiffness, grip,
  travel), then buy the rest.
