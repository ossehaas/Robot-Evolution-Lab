# The base on the ring paw (step 47)

hold_s2_tpu = hold_s2 on the old sphere paw, hold_s2_ring = the same policy on the ring paw (step 46); ring_s1 / ring_s2 = hold_s2 trained on with the ring paw (3000 it). Stumble = share of paw-steps on the way up with a paw pushed sideways more than twice as hard as it is pressed down (a toe against a riser); stuck = such a stumble lasting more than 0.3 s, per climb.

## Stairs up (lunge test, rises 1-5 cm, all treads)

| dog | climbed | stumble | stuck per climb | shift front cm | shift rear cm | nose up |
|---|---|---|---|---|---|---|
| hold_s2_tpu | 0.82 | 0.0223 | 0.38 | -1.4 | -0.2 | 8 deg |
| hold_s2_ring | 0.72 | 0.0200 | 0.06 | -1.7 | -0.5 | 8 deg |
| ring_s1 | 0.73 | 0.0187 | 0.04 | -2.1 | -0.5 | 10 deg |
| ring_s2 | 0.74 | 0.0190 | 0.06 | -2.0 | -0.4 | 9 deg |

## Stairs goal test (clean = reached without any fall, open sides)

| dog | mean clean | up | down | limits up (treads 25-9) | limits down |
|---|---|---|---|---|---|
| hold_s2_tpu | 0.361 | 0.415 | 0.308 | 4 / 4 / 3 / 3 / 2 | 3 / 2 / 2 / 1 / 1 |
| hold_s2_ring | 0.327 | 0.394 | 0.260 | 4 / 4 / 4 / 3 / 2 | 3 / 3 / 2 / 1 / 1 |
| ring_s1 | 0.323 | 0.385 | 0.261 | 4 / 3 / 3 / 3 / 2 | 2 / 2 / 1 / 1 / 1 |
| ring_s2 | 0.323 | 0.379 | 0.266 | 4 / 2 / 3 / 2 / 2 | 3 / 2 / 2 / 1 / 1 |

## Walking, getting up, shoves, standing on tilted plates

| dog | forward falls (score) | course falls (score) | power |
|---|---|---|---|
| hold_s2_tpu | 1.0 % (0.45) | 0.9 % (0.43) | 12.7 W |
| hold_s2_ring | 0.4 % (0.30) | 0.8 % (0.34) | 13.5 W |
| ring_s1 | 0.8 % (0.33) | 0.6 % (0.35) | 14.5 W |
| ring_s2 | 0.4 % (0.32) | 0.6 % (0.34) | 14.2 W |

Getting up within 8 s:

| dog | standing | dropped upright | on its side | on its back |
|---|---|---|---|---|
| hold_s2_tpu | 99 % | 92 % | 100 % | 100 % |
| hold_s2_ring | 80 % | 84 % | 100 % | 99 % |
| ring_s1 | 100 % | 100 % | 100 % | 99 % |
| ring_s2 | 87 % | 91 % | 100 % | 100 % |

One shove while walking (stays up / gets up / stays down):

| dog | 0.3 m/s | 0.5 m/s | 0.7 m/s | 0.9 m/s | 1.1 m/s | 1.3 m/s | 1.5 m/s |
|---|---|---|---|---|---|---|---|
| hold_s2_tpu | 100 / 0 / 0 % | 97 / 3 / 0 % | 67 / 33 / 0 % | 58 / 42 / 0 % | 27 / 68 / 4 % | 3 / 92 / 6 % | 0 / 93 / 7 % |
| hold_s2_ring | 100 / 0 / 0 % | 100 / 0 / 0 % | 94 / 6 / 0 % | 75 / 25 / 0 % | 52 / 48 / 0 % | 18 / 79 / 3 % | 3 / 94 / 3 % |
| ring_s1 | 100 / 0 / 0 % | 100 / 0 / 0 % | 91 / 9 / 0 % | 73 / 27 / 0 % | 53 / 47 / 0 % | 16 / 82 / 1 % | 10 / 86 / 4 % |
| ring_s2 | 100 / 0 / 0 % | 99 / 1 / 0 % | 94 / 6 / 0 % | 75 / 25 / 0 % | 37 / 63 / 0 % | 22 / 78 / 0 % | 8 / 90 / 3 % |

Standing on tilted plates (calm / shoved 0.3 m/s): still % up to 15 deg, falls up to 15 deg, falls at 20 deg:

| dog | calm still | calm falls <= 15 | calm falls 20 | shoved falls <= 15 | shoved falls 20 |
|---|---|---|---|---|---|
| hold_s2_tpu | 20 % | 0 % | 35 % | 0 % | 43 % |
| hold_s2_ring | 12 % | 0 % | 32 % | 0 % | 22 % |
| ring_s1 | 14 % | 0 % | 18 % | 0 % | 8 % |
| ring_s2 | 11 % | 0 % | 12 % | 0 % | 25 % |

Best and pass mark for step 45b: {"best": "ring_s1", "ok": false, "scores": {"walk_score": 0.3318294882774353, "fall_rate": 0.0078125, "getup": 0.988, "stairs_up": 0.38549999999999995}, "pass_mark": {"walk_score": 0.4, "fall_rate": 0.02, "getup": 0.95, "stairs_up": 0.37}}

Videos: `videos/compare_down_r4_t15.mp4`, `videos/compare_stand_flat.mp4`, `videos/compare_stop_go.mp4`, `videos/compare_up_r4_t9.mp4`
