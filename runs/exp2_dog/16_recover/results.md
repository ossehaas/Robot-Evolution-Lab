# Keep balance and get up (step 38)

## One shove while walking at 20 cm/s: stays up / falls and gets up / stays down

| dog | 0.3 m/s | 0.5 m/s | 0.7 m/s | 0.9 m/s | 1.1 m/s | 1.3 m/s | 1.5 m/s |
|---|---|---|---|---|---|---|---|
| amp_w6m_slow_s1 | 100 / 0 / 0 % | 90 / 1 / 10 % | 74 / 3 / 23 % | 55 / 3 / 42 % | 34 / 3 / 62 % | 7 / 1 / 92 % | 1 / 7 / 92 % |
| distill | 98 / 2 / 0 % | 81 / 19 / 0 % | 63 / 37 / 0 % | 45 / 55 / 0 % | 43 / 57 / 0 % | 23 / 76 / 1 % | 4 / 95 / 1 % |
| recover | 99 / 1 / 0 % | 94 / 6 / 0 % | 75 / 24 / 1 % | 62 / 37 / 1 % | 40 / 51 / 10 % | 25 / 67 / 8 % | 12 / 75 / 13 % |

## Getting up: share standing within 8 s (median time)

| dog | standing | dropped upright | on its side | on its back |
|---|---|---|---|---|
| distill | 100 % (1.4 s) | 100 % (1.1 s) | 99 % (1.92 s) | 100 % (1.94 s) |
| recover | 100 % (0.5 s) | 99 % (0.48 s) | 70 % (2.56 s) | 88 % (1.68 s) |

## Walking (falls, walk score; stairs: steps climbed up on the hardest rows)

| dog | forward | course | stairs straight | steps up (rows 6-9) | power forward |
|---|---|---|---|---|---|
| amp_w6m_slow_s1 | 0.7 % (0.41) | 0.8 % (0.41) | 1.3 % (0.40) | 3.6 | 14.3 W |
| recover | 4.4 % (0.25) | 6.9 % (0.22) | 3.2 % (0.24) | 1.3 | 13.4 W |
| distill | 6.5 % (0.39) | 3.9 % (0.37) | 5.5 % (0.38) | 1.7 | 11.8 W |

Video shoves_walking.mp4: went down 6x, stayed on its feet 4x, got up 6/6 (mean 1.72 s)

Video flips_walking.mp4: went down 10x, stayed on its feet 0x, got up 8/10 (mean 1.43 s)
