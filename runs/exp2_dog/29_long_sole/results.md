# The base on the 4.5 mm sole (step 49)

hold_s2_tpu = hold_s2 on the old ball (step 46); ring_s1 / ring_s2 = retrained on the current paw (step 47); hold_s2_p45 = untrained, 4.5 mm sole without a stop (step 48); start = untrained, 4.5 mm sole with its stop; teach = retrained with the walking teacher, free = retrained without it (3000 it). Stumble = share of paw-steps on the way up with a paw pushed sideways more than twice as hard as it is pressed down (a toe against a riser); stuck = such a stumble lasting more than 0.3 s, per climb.

## Stairs up (lunge test, rises 1-5 cm, all treads)

| dog | climbed | stumble | stuck per climb | shift front cm | shift rear cm | nose up |
|---|---|---|---|---|---|---|
| hold_s2_tpu | 0.82 | 0.0223 | 0.38 | -1.4 | -0.2 | 8 deg |
| ring_s1 | 0.73 | 0.0187 | 0.04 | -2.1 | -0.5 | 10 deg |
| ring_s2 | 0.74 | 0.0190 | 0.06 | -2.0 | -0.4 | 9 deg |
| hold_s2_p45 | 0.81 | 0.0226 | 0.07 | -1.6 | -0.4 | 8 deg |
| start | 0.83 | 0.0222 | 0.05 | -1.6 | -0.4 | 8 deg |
| teach_s1 | 0.83 | 0.0200 | 0.04 | -1.5 | -0.2 | 8 deg |
| teach_s2 | 0.79 | 0.0189 | 0.02 | -1.6 | -0.4 | 8 deg |
| free_s1 | 0.00 | 0.0106 | 0.01 | -0.9 | 0.3 | 4 deg |
| free_s2 | 0.01 | 0.0322 | 0.00 | -0.4 | -0.4 | -0 deg |

## Stairs goal test (clean = reached without any fall, open sides)

| dog | mean clean | up | down | limits up (treads 25-9) | limits down |
|---|---|---|---|---|---|
| hold_s2_tpu | 0.361 | 0.415 | 0.308 | 4 / 4 / 3 / 3 / 2 | 3 / 2 / 2 / 1 / 1 |
| ring_s1 | 0.323 | 0.385 | 0.261 | 4 / 3 / 3 / 3 / 2 | 2 / 2 / 1 / 1 / 1 |
| ring_s2 | 0.323 | 0.379 | 0.266 | 4 / 2 / 3 / 2 / 2 | 3 / 2 / 2 / 1 / 1 |
| hold_s2_p45 | 0.360 | 0.415 | 0.305 | 4 / 4 / 3 / 3 / 2 | 3 / 2 / 2 / 1 / 1 |
| start | 0.360 | 0.412 | 0.307 | 4 / 4 / 3 / 2 / 2 | 3 / 2 / 2 / 1 / 1 |
| teach_s1 | 0.343 | 0.432 | 0.255 | 5 / 4 / 3 / 3 / 2 | 3 / 2 / 2 / 1 / 1 |
| teach_s2 | 0.336 | 0.403 | 0.270 | 4 / 3 / 3 / 3 / 2 | 3 / 2 / 2 / 1 / 1 |
| free_s1 | 0.102 | 0.029 | 0.174 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| free_s2 | 0.133 | 0.008 | 0.258 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |

## Walking, getting up, shoves, standing on tilted plates

| dog | forward falls (score) | course falls (score) | power | cost of transport |
|---|---|---|---|---|
| hold_s2_tpu | 1.0 % (0.45) | 0.9 % (0.43) | 12.7 W | 2.61 |
| ring_s1 | 0.8 % (0.33) | 0.6 % (0.35) | 14.5 W | 3.82 |
| ring_s2 | 0.4 % (0.32) | 0.6 % (0.34) | 14.2 W | 3.85 |
| hold_s2_p45 | 0.3 % (0.36) | 0.9 % (0.39) | 13.7 W | 3.45 |
| start | 0.6 % (0.36) | 0.3 % (0.39) | 13.6 W | 3.44 |
| teach_s1 | 0.4 % (0.38) | 0.4 % (0.39) | 13.8 W | 3.29 |
| teach_s2 | 0.5 % (0.37) | 0.4 % (0.39) | 13.8 W | 3.43 |
| free_s1 | 21.2 % (0.15) | 90.0 % (0.07) | 28.6 W | 5.50 |
| free_s2 | 10.4 % (0.16) | 78.0 % (0.07) | 17.0 W | 3.60 |

Getting up within 8 s:

| dog | standing | dropped upright | on its side | on its back |
|---|---|---|---|---|
| hold_s2_tpu | 99 % | 92 % | 100 % | 100 % |
| ring_s1 | 100 % | 100 % | 100 % | 99 % |
| ring_s2 | 87 % | 91 % | 100 % | 100 % |
| hold_s2_p45 | 96 % | 92 % | 100 % | 100 % |
| start | 98 % | 93 % | 100 % | 100 % |
| teach_s1 | 100 % | 99 % | 100 % | 100 % |
| teach_s2 | 98 % | 99 % | 100 % | 100 % |
| free_s1 | 0 % | 0 % | 0 % | 0 % |
| free_s2 | 0 % | 0 % | 0 % | 0 % |

One shove while walking (stays up / gets up / stays down):

| dog | 0.3 m/s | 0.5 m/s | 0.7 m/s | 0.9 m/s | 1.1 m/s | 1.3 m/s | 1.5 m/s |
|---|---|---|---|---|---|---|---|
| hold_s2_tpu | 100 / 0 / 0 % | 97 / 3 / 0 % | 67 / 33 / 0 % | 58 / 42 / 0 % | 27 / 68 / 4 % | 3 / 92 / 6 % | 0 / 93 / 7 % |
| ring_s1 | 100 / 0 / 0 % | 100 / 0 / 0 % | 91 / 9 / 0 % | 73 / 27 / 0 % | 53 / 47 / 0 % | 16 / 82 / 1 % | 10 / 86 / 4 % |
| ring_s2 | 100 / 0 / 0 % | 99 / 1 / 0 % | 94 / 6 / 0 % | 75 / 25 / 0 % | 37 / 63 / 0 % | 22 / 78 / 0 % | 8 / 90 / 3 % |
| hold_s2_p45 | 100 / 0 / 0 % | 99 / 1 / 0 % | 89 / 11 / 0 % | 67 / 33 / 0 % | 47 / 53 / 0 % | 17 / 82 / 1 % | 4 / 92 / 3 % |
| start | 100 / 0 / 0 % | 100 / 0 / 0 % | 90 / 10 / 0 % | 70 / 30 / 0 % | 39 / 60 / 1 % | 16 / 80 / 4 % | 8 / 90 / 2 % |
| teach_s1 | 100 / 0 / 0 % | 100 / 0 / 0 % | 83 / 17 / 0 % | 66 / 34 / 0 % | 38 / 60 / 2 % | 16 / 80 / 5 % | 3 / 93 / 4 % |
| teach_s2 | 100 / 0 / 0 % | 100 / 0 / 0 % | 86 / 14 / 0 % | 71 / 28 / 1 % | 36 / 63 / 1 % | 18 / 77 / 4 % | 7 / 88 / 5 % |
| free_s1 | 100 / 0 / 0 % | 99 / 1 / 0 % | 99 / 1 / 1 % | 90 / 10 / 0 % | 84 / 16 / 1 % | 80 / 18 / 2 % | 67 / 30 / 3 % |
| free_s2 | 100 / 0 / 0 % | 99 / 1 / 0 % | 99 / 1 / 0 % | 92 / 8 / 0 % | 80 / 20 / 0 % | 68 / 32 / 0 % | 62 / 38 / 0 % |

Standing on tilted plates (calm / shoved 0.3 m/s): still % up to 15 deg, falls up to 15 deg, falls at 20 deg:

| dog | calm still | calm falls <= 15 | calm falls 20 | shoved falls <= 15 | shoved falls 20 |
|---|---|---|---|---|---|
| hold_s2_tpu | 20 % | 0 % | 35 % | 0 % | 43 % |
| ring_s1 | 14 % | 0 % | 18 % | 0 % | 8 % |
| ring_s2 | 11 % | 0 % | 12 % | 0 % | 25 % |
| hold_s2_p45 | 18 % | 0 % | 30 % | 0 % | 18 % |
| start | 19 % | 0 % | 20 % | 0 % | 18 % |
| teach_s1 | 15 % | 0 % | 35 % | 0 % | 38 % |
| teach_s2 | 19 % | 0 % | 18 % | 0 % | 10 % |
| free_s1 | 0 % | 100 % | 100 % | 99 % | 100 % |
| free_s2 | 0 % | 100 % | 100 % | 100 % | 100 % |

Best and pass mark for step 45b: {"best": "teach_s1", "ok": false, "scores": {"walk_score": 0.37934425473213196, "fall_rate": 0.00390625, "getup": 0.992, "stairs_up": 0.43200000000000005}, "pass_mark": {"walk_score": 0.4, "fall_rate": 0.02, "getup": 0.95, "stairs_up": 0.37}}

Videos: `videos/compare_down_r4_t15.mp4`, `videos/compare_stand_flat.mp4`, `videos/compare_stop_go.mp4`, `videos/compare_up_r4_t9.mp4`
