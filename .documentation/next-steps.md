# Next steps: checklist (from 2026-10-25)

Continue the robot dog (experiment 2) from **dog v1**. Read this first in a new chat, then the end of
`.documentation/project-journal.md` (steps 33-39) for the details. The public story is at
https://robotevolutionlab.azurewebsites.net/ (source: `C:\ProjectsHome\Robots\EvolutionBlog`).

## Where things are

| What | Where |
|---|---|
| **Dog v1** (best dog: walks, climbs, gets up, recovers from shoves) | `runs/exp2_dog/17_recover_bc/bc_s2` (`policy.pt`, `model_final.pt`, `genome.json`, printable parts in `robot/print/`) |
| Its teachers | walker `runs/exp2_dog/11_imitation/amp_w6m_slow_s1`, get-up dog `runs/exp2_dog/15_getup_design/getup_abd29_head` |
| Its recipe | `runs/exp2_dog/17_recover_bc/batch.py` (RL polish with gated behaviour cloning, `core/evolution/gated_bc.py`) |
| Dog v1's settings | 15 STS3215 12 V servos, neck/head/tail driven by the policy (`policy_drives_expression`), self-collision on, action scale 1.0, perfect height scan (`--height_scan`), IMU + 8-frame history, heading hold 2 |
| Tests | `tools/dog/benchmark.py` (scenarios course, forward, forward_dark, stairs_straight, slippery, pushes, dark, ...; reports falls, walk score, `type_level_climb_m`), `recover_test.py` (shove + recovery), `getup_test.py`, `push_test.py`, `map_check.py`, `gait_check.py` |
| Videos | `core/evolution/compare_video.py` (`--drive tour / straight / stand`), `tools/dog/flip_video.py` (`--vx --push --spin`) |
| Terrain map / memory student | `core/sim/terrain_map.py`, `core/evolution/distill_memory.py` (step 33-34) |
| Stairs rewards (exist, off by default) | `stair_tread_range`, `foot_clearance_terrain`, `stumble_weight`, `foothold_weight`, `lift_first_weight`, `stairs_support_weight`, `stairs_square_yaw`, `stairs_turn_scale`, `amp_skip_types`, `stairs_terms_types` in `core/sim/locomotion_env_cfg.py` (step 35) |
| Launcher | `lab.cmd` (Windows) / `./lab.sh` (Linux); batch scripts pick it themselves |

## Working rules (agreed)

- **Prepare batches fully, but start Isaac Sim only when the owner says go** (the PC is shared; ~13 GB RAM per
  training with 2048 robots). CPU-only checks are fine any time.
- Every batch starts with a **smoke test** (64 robots, a few iterations) and stops at the first failure.
- **Two seeds** for anything important; single seeds can miss an all-or-nothing skill.
- **Measure what the robot achieves, not only what it avoids** (step 35: few falls on stairs meant "did not try").
- **Always make videos**; the owner chooses gaits by eye (calm, controlled, natural) over small metric gains.
- Keep the STS3215; no actuator changes. Body changes only when a task really fails, and minimal.
- Write every step in the journal (results, failures, lessons); update the blog afterwards.
- Lessons that cost time before: a style reward collapses when a warm-started gait changes abruptly (rescale with
  `--init_action_scale`, limit stair terms to stair tiles); shoves stronger than walking speed make a nervous gait;
  RL polish without the teachers as an anchor forgets skills.

## 0. Base skills (every robot, `.documentation/base-skills.md`; agreed 2026-10-27)

- [x] The list with pass criteria: design checks A1-A4 (CPU) and learned basics B1-B8.
- [x] A1-A4 for dog v1 (`tools/base/design_check.py`): all pass.
- [x] Step 43 (`runs/exp2_dog/21_base_skills/`): B1 + B2 still fail; nothing gained, nothing forgotten (the tilt
      dogs lost "get up from standing"). Likely cause: the gait clock keeps ticking while standing.
- [x] Step 44 (2026-10-31, `runs/exp2_dog/22_still_clock/`): gait clock held while standing: the stepping is gone
      (0.1 cm in 10 s, 0.12 W), a tremor of ~0.6 rad/s summed remains, so B1 still just fails; B2 unchanged; the
      training added nothing over holding the clock. Next for B1: judge the tremor by eye, then decide (servo deadband,
      a lower-gain stand, or a stand teacher).

## 0b. One model, many skills (agreed 2026-10-28)

Goal: one end model (one `policy.pt`) that holds all skills and picks the right one from its sensors, where a new
skill reuses what the model already knows without overwriting or levelling out the old ones. Design:
- base (walk + get up + balance) trained first, then frozen;
- one small expert per new skill: reads the base's internal features (reuse) and adds a correction to the base's
  action (its own strategy); only the new expert learns, so old skills cannot be overwritten;
- a gate that says how much each expert adds: trained with the true situation as a label (tile type, fallen), at run
  time it predicts it from the sensors (height scan, IMU, command); the gate output is logged as proof of switching;
- each expert has its own reward set during training (e.g. no trot term on the stairs);
- optional at the end: a short training of the whole model with a small learning rate (handovers), tests guard it.
Model size grows only when shown necessary: experts at 2-3 sizes, and a size test by distillation.

- [x] Step 45 (2026-10-31, `runs/exp2_dog/23_stairs_expert/`): the gate switches cleanly (0.98 on the stairs, 0.02-0.05
      before / after, 0.2 % on the course) and the base's skills are unchanged, but the expert learned no lunge (its
      correction stays ~0.01: the quiet loss outweighs the sparse plan reward).
- [x] Step 46 (2026-10-31, `runs/exp2_dog/25_hard_paw/`): hard paw front. The simulated paw was a sphere; on the
      printed shape every policy walks worse. Owner's choice: the ring paw (PLA shell printed onto a 4.5 mm TPU base
      ring, one print with a filament change).
- [x] Step 47 (2026-11-01, `runs/exp2_dog/26_ring_base/`): retrained on the ring paw: get-up and balance back, walking
      not (walk score 0.33, cost of transport 3.8 vs 2.6): the flat base drags. Did not pass; 45b waits.
- [ ] Next for the paw (owner): sole sticking out further / rounded base (test only), and/or training without the
      sphere-paw walking teacher.
- [ ] Step 48 (`runs/exp2_dog/28_part_tests/`): part tests: geometry, paw test stand and trained dogs on longer soles done
  (level 2: walk score 0.30 -> 0.33 at 3 mm -> 0.36 at 4.5 mm, old ball 0.45). Chosen: 4.5 mm with a 2 mm stop pin, CAD done. Step 49
  (`runs/exp2_dog/29_long_sole/`): retraining done: with the walking teacher 0.38 (mark 0.40), without it the dog fails. Owner to decide: a round with
  a teacher from this paw, or go on to step 45b with teach_s1. Owner's test print later.
  Step 50 first (`runs/exp2_dog/30_level_body/`): a level body with the weight ahead on the stairs: the reward was too small, dog unchanged. Step 51 (`31_copy_plans/`, the standing legs copy the plans): the dogs freeze at the stairs, but go down much further, with falls. Step 52 (`32_no_stalling/`, stalling costs on a descent): paid for, still refuses. Step 53 (`33_one_model/`, walking, up and down in one model): works, up as teach_s1, down better than copy_s1; the signal helps walking only; getting up from side or back lost (never practised in training). Owner to decide what is next.
- [ ] Step 45b (prepared, `runs/exp2_dog/27_stairs_expert_b/`, chained after 47 if it passes): quiet loss off + copy
      loss on the plan's joint pose; base = step 47's best dog; the same tests and videos.
- [ ] Print a test paw (owner): TPU first 4.5 mm, then PLA; check the bond and the sole's snap.
- [ ] Step 46: proof of concept on a mixed parcours (flat, stairs up, flat): base skills unchanged, stairs up better
      than the base alone (limit table, the lunge visible and measured), the gate on near the first riser and off on
      the landing.
- [ ] Step 47: stairs down the same way (plans first, then an expert; does the gate tell up from down? beyond the
      1 cm short of step 42?).
- [ ] Then more skills, one expert each, watching the expert size.
- [ ] B7 (robustness to delay, masses, servo strength) and B8 (sensor blackout: stop and stand) tests.

## 1. Stairs goal task + climbing-limit table (journal step 40)

Status 2026-10-25: run done (`runs/exp2_dog/18_stairs_goal/results.md`, journal step 40). Best dog: free_s2 (up to
4 cm up on treads >= 15 cm, 5 cm on 25 cm treads; down only 2-3 cm: it stops or creeps rather than falls).

Goal: find how steep and how high dog v1 can climb (up and down) without too many problems, and the sweet spot.

- [x] Terrain: one straight staircase per tile (flat approach ~1 m, N steps, top landing; some with open sides, some
      with walls), rise and tread varied over the rows (e.g. rise 2-8 cm, tread 25 -> 9 cm, i.e. ~5 to ~40 deg), and
      the reverse (start on the landing, goal at the bottom).
- [x] Goal through the walking command: a simple navigation layer (turn towards the goal, then straight on), so the
      walker stays one policy and cannot avoid the stairs.
- [x] Falls: soft on the stairs (small penalty; dog v1 can get up and go on), hard when tumbling down (height more
      than one step below the best reached) or off the side.
- [x] Measure per (rise, tread): share reaching the goal, time, steps climbed, falls per kind -> **the climbing-limit
      table** (up and down), plus the sweet spot.
- [x] First zero-shot with dog v1, then train on the goal task (from dog v1, keep the gated-BC anchor) with a
      curriculum over rise and tread; compare before/after.
- [ ] Stair ideas still open from step 35/36: crouch on stairs (done in the "free" variant: it helped), neck as a
      counterweight (now possible: the policy drives it).
- [ ] NEW (step 40): descending is the weak point (stops / creeps, few falls): train the descent on purpose.
- [x] Physically possible? (step 41, `tools/dog/stairs_plan.py`): up to 5 cm up on every tread, 4-5 cm down on 12-25 cm
      treads; 6 cm is the limit of the legs. Mid-climb starts from the plans: no gain overall; stair terms: worse.
- [x] Step 42 (`runs/exp2_dog/20_balance_and_style/`): no reward term punishes going down (refusal = caution + posture
      at its tipping limit); the trot term (3/s) is the biggest reward and holds the 2-paw gait; trained dogs tip at
      the same tilts as the bare body (~20-25 deg) and never stand still; desc curriculum worse, balance-margin reward
      = start dog (better than desc on hard descents); open-loop replay of 3 cm plans unreliable (0/29 nominal).
- [ ] NEXT candidate: a 4-beat walk instead of the trot on the stairs (`gait=walk_trot` or trot term off on stairs
      tiles) together with the margin reward, so the plans' 3-paw strategy is not paid less than the trot.
- [ ] Side tour: stand still while stable (until it senses instability), less power; measure first (tilt test).
- [x] Videos straight up and down at the sweet spot and at the limit (+ beyond the limit going up).
- Done when: a table "up to X cm rise / Y deg it climbs and descends with < Z % problems", with videos.

## 2. "Can I climb this?" decision

Goal: the dog only tries what it can do; it stops or goes around otherwise.

- [ ] From the limit table: a rule on the height map (step height, slope, gap) -> go / stop / go around.
- [ ] Test terrain with steps and blocks beyond the limit (e.g. 10-18 cm, house-stair height) next to climbable ones.
- [ ] Measure: never attempts the impossible (falls at too-high steps ~0), still climbs everything within the limit.
- [ ] Later option: a learned climbability estimate from the simulation's successes and failures.
- Done when: in a mixed parcours it climbs what it can and stops or detours at what it cannot.

## 3. The real sensor (depth camera + memory student)

Goal: choose the sensor and its placement that get closest to the perfect height scan.

- [ ] Teacher: dog v1 (or its stairs-trained successor) with the perfect scan.
- [ ] Students (`core/evolution/distill_memory.py`, GRU, rebuild loss): depth camera vs 8x8 ToF vs two ToFs, at
      different mount positions and pitch angles (chest, under the chin, head), field of view and update rate.
- [ ] Add sim2real realism while doing it (Microduck lessons, see memory/journal): varying latency 15-30 ms, IMU
      misalignment ~6 deg, encoder bias, gearbox backlash, sensor blackouts.
- [ ] Measure the gap to the teacher on forward walking, stairs (goal task) and blackouts; map coverage pictures.
- Done when: a recommendation "this sensor, here, at this angle" with the gap to the perfect scan, plus the student
  policy that uses it.

## 4. Household obstacles from Blender (parcours)

Goal: realistic obstacles and parcours to train on and to show what the dog can do.

- [ ] Blender (installed) via its Python API: a library of obstacles of known heights (doorstep, rug edge, cable,
      shoe, toy, chair legs, curb, small steps, a ramp), exported to USD; add them as terrain/obstacles in Isaac Lab.
- [ ] Parcours scenes combining walking, stairs (within the limit), obstacles to step over and ones to go around.
- [ ] Use them for steps 2 and 3, and for show-off videos for the blog.
- Done when: a few parcours the dog completes, with videos.

## 5. Body fine-tuning by evolution (ongoing, only when needed)

- [ ] Only if a task above really fails, or a small change clearly helps: small mutations of the current genome
      (e.g. leg lengths, body length/height, abduction range) evaluated on that task; keep the printable design intact.
- [ ] Every change goes through the interference check (`python -m core.morphology.dog_check`) and keeps the parts
      printable on the owner's printer (256 mm bed).
- Done when: either "no change needed" or a minimal change with a measured gain.

## Also pending (before the first real robot)

- [ ] Limits spec sheet of dog v1 (highest step, steepest slope, top speed; partly covered by step 1).
- [ ] Sim2real realism round (see step 3) and later a measured STS3215 servo model (one servo on a pendulum bench,
      Rhoban BAM method).
- [ ] Software on the robot (policy runtime, servo bus, safety limits, emergency stop). On a flat floor the height
      scan is known (flat), so standing, getting up and walking can be tested without a camera.
- [ ] Printing/ordering only on the owner's go (the design is final apart from fine-tuning).
