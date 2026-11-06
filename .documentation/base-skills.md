# Base skills: what every robot must be able to do (agreed 2026-10-27)

The entry exam for any future robot body. If a robot cannot pass these, the design has a flaw, and we find it before
spending GPU hours on harder tasks (stairs, parcours, sensors). The list may change when a test turns out to be wrong
or too strict; change it here and note why.

Order matters: the design checks (A) cost seconds on the CPU and run first; the learned basics (B) need training.

## A. Design checks (CPU, no training)

| # | Check | Passes when | Tool |
|---|---|---|---|
| A1 | Parts do not collide, joints stay in range | rest pose and every joint through its range: no part hits the body, its own leg or a servo (the front and rear leg of one side may meet at large hip swings towards each other: reported, the simulation has self-collision for it and the policy avoids it) | `tools/base/design_check.py` (uses `core.morphology.dog_check`) |
| A2 | Stands on a tilt without tipping (standing pose) | tips beyond 20 deg in every direction (nose down, nose up, sideways) | `tools/base/design_check.py` (uses `tools/dog/tilt_limits.py`) |
| A3 | Servos strong enough | standing on 4 paws < 30 % of the usable servo torque at every joint; on 3 paws (any leg lifted) < 60 % | `tools/base/design_check.py` |
| A4 | Printable | every printed part fits the printer's build plate (256 x 256 x 256 mm) | `tools/base/design_check.py` |

Usable torque = 80 % of the stall torque (STS3215 12 V: 30 kg cm stall -> 2.35 N m usable). Holding torque also costs
battery and heat on the real servo, even standing still (the simulator's mechanical power is 0 when nothing moves), so
A3 doubles as the standing-power check.

## B. Learned basics (training + test)

| # | Skill | Passes when | Tool |
|---|---|---|---|
| B1 | Stand balanced and still | on a flat floor, command "stand": still within 2 s (summed joint speed < 0.5 rad/s) and still >= 90 % of the time after that; moves < 2 cm in 10 s | `tools/dog/tilt_test.py` (0 deg row) |
| B2 | Stay balanced when tilted, move only when unstable | on plates tilted up to 80 % of its tilt limit (A2) in every direction: never falls, still >= 80 % of the time; after a small shove (0.3 m/s) it steps or leans to recover, then is still again within 3 s | `tools/dog/tilt_test.py --shove 0.3` |
| B3 | Get up after a fall | from standing, dropped, its side and its back: standing within 8 s, >= 95 % each | `tools/dog/getup_test.py` |
| B4 | Take a shove while walking | 0.5 m/s: stays up >= 75 %; up to 1.5 m/s: if it falls, it gets up (stays down < 10 %) | `tools/dog/recover_test.py` |
| B5 | Walk, turn and stop on command | follows speed and turn (walk score as dog v1 or better); after "stop" it stands still (B1) | `tools/dog/benchmark.py forward`, `tilt_test.py` |
| B6 | Small unevenness | a 2 cm threshold, rough and slippery floor: falls < 3 % | `tools/dog/benchmark.py course / slippery` |
| B7 | Robust to the real world | B1-B5 still pass with control delay 15-30 ms, masses +-15 %, servo strength +-25 %, paw grip 0.7-1.0 | the tests with randomization on (partly: training randomization) |
| B8 | Safe when blind | the height scan drops out while walking: it stops and stands instead of walking on | new (not built yet) |

The plateau (B2) is first done with fixed plates (a plate per tile, tilted 0-30 deg, the dog placed standing on it) plus
small shoves, which force "unstable -> react -> still again". A plate that tilts slowly while the dog stands on it is the
nicer test; it needs a moving mesh the height scanner can see (Isaac Lab's multi-mesh ray caster), so it is left for
later.

## Status per robot

### Dog v1 (`runs/exp2_dog/17_recover_bc/bc_s2`) and its stairs successor free_s2 (`runs/exp2_dog/18_stairs_goal/free_s2`)

| # | Result | Pass? |
|---|---|---|
| A1 | no flaws; front and rear leg of one side meet from 52 deg hip swing towards each other (16 overlaps, known since step 21) | yes |
| A2 | 20 deg nose down, 28 nose up, 29 sideways (`tilt_limits.py`, step 42) | yes, just (nose down is the weak side) |
| A3 | standing 12 % of the usable torque (2.35 N m), on 3 paws 27 % (front leg lifted) / 12 % (rear) | yes |
| A4 | parts made for the Centauri Carbon (step 30) | yes |
| B1 | with the gait clock held while standing (step 44): moves 0.1 cm in 10 s at 0.12 W, but a tremor of ~0.6 rad/s summed keeps it 'still' only 33-42 % of the time (limit 0.5 rad/s) | **no** (nearly) |
| B2 | stands to 15 deg, falls at 20-25 deg, still 0-20 % on a tilt; tilted pads and the held clock did not help (steps 42-44) | **no** |
| B3 | 100 % from every start (steps 39-42) | yes |
| B4 | 0.5 m/s: 77-83 % stay up; 1.5 m/s: 0-12 % stay down (step 42) | yes |
| B5 | walk score 0.45, falls ~1 % | yes (stop part: see B1) |
| B6 | course falls 0.7 %, slippery: see benchmark | yes |
| B7 | trained with randomization; not tested separately | ? |
| B8 | not built | ? |

Design checks: `lab tools/base/design_check.py <genome> --out runs/base/design_check.jsonl` (dog v1: all pass, 2026-10-27).

First target: dog v1's line passes completely. Step 43 (`runs/exp2_dog/21_base_skills/`) did not get B1 / B2; next
try: a gait clock that stops while standing (step 44).
