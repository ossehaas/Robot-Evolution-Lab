# Trained for forward walking (batch.py dogs for comparison)

## Falls per scenario (walk score)

| run | forward | forward_dark | course | pushes | power forward |
|---|---|---|---|---|---|
| robust_scan | 7.8 % (0.33) | 8.4 % (0.33) | 5.8 % (0.33) | 8.8 % (0.31) | 14.3 W |
| map_depth | 10.9 % (0.30) | 11.8 % (0.30) | 10.4 % (0.29) | 14.9 % (0.28) | 15.4 W |
| student_depth | 7.6 % (0.32) | 10.5 % (0.32) | 8.3 % (0.31) | 12.5 % (0.29) | 14.4 W |
| fwd_scan | 13.7 % (0.30) | 15.1 % (0.29) | 39.0 % (0.25) | 43.3 % (0.24) | 16.6 W |
| fwd_depth | 12.3 % (0.32) | 14.1 % (0.32) | 36.8 % (0.26) | 42.2 % (0.25) | 15.7 W |
| fwd_tof | 13.4 % (0.30) | 19.0 % (0.31) | 35.0 % (0.26) | 40.9 % (0.24) | 16.4 W |
| student_fwd_depth | 9.2 % (0.34) | 11.3 % (0.34) | 22.0 % (0.28) | 30.3 % (0.26) | 14.8 W |

## Gap to the perfect scan (extra falls, percentage points; perfect-scan dog of the same training)

| run | reference | forward | forward_dark | course | pushes |
|---|---|---|---|---|---|
| map_depth | robust_scan | +3.1 | +3.4 | +4.7 | +6.2 |
| student_depth | robust_scan | -0.2 | +2.1 | +2.5 | +3.7 |
| fwd_depth | fwd_scan | -1.4 | -1.1 | -2.1 | -1.1 |
| fwd_tof | fwd_scan | -0.3 | +3.9 | -4.0 | -2.3 |
| student_fwd_depth | fwd_scan | -4.5 | -3.8 | -17.0 | -13.0 |

## Falls per terrain type (forward): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| robust_scan | 0.4 % (1 %) | 2.4 % (6 %) | 3.8 % (9 %) | 9.0 % (14 %) | 15.3 % (32 %) | 14.7 % (24 %) |
| map_depth | 1.2 % (2 %) | 3.5 % (6 %) | 7.5 % (19 %) | 11.3 % (20 %) | 25.9 % (56 %) | 15.9 % (32 %) |
| student_depth | 1.2 % (3 %) | 0.0 % (0 %) | 7.5 % (19 %) | 4.3 % (7 %) | 20.0 % (46 %) | 14.1 % (29 %) |
| fwd_scan | 1.6 % (2 %) | 4.4 % (11 %) | 8.5 % (21 %) | 15.9 % (32 %) | 34.7 % (68 %) | 14.7 % (22 %) |
| fwd_depth | 2.0 % (3 %) | 0.0 % (0 %) | 8.5 % (21 %) | 12.5 % (19 %) | 35.2 % (63 %) | 12.9 % (29 %) |
| fwd_tof | 1.2 % (2 %) | 0.0 % (0 %) | 11.9 % (22 %) | 15.6 % (28 %) | 33.5 % (65 %) | 15.9 % (26 %) |
| student_fwd_depth | 0.8 % (2 %) | 1.2 % (0 %) | 2.4 % (3 %) | 7.8 % (17 %) | 22.9 % (44 %) | 17.6 % (35 %) |

## Falls per terrain type (forward_dark): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| robust_scan | 0.8 % (2 %) | 2.5 % (3 %) | 3.8 % (9 %) | 9.8 % (14 %) | 16.5 % (37 %) | 15.3 % (31 %) |
| map_depth | 0.4 % (0 %) | 1.1 % (3 %) | 8.5 % (21 %) | 18.7 % (36 %) | 26.4 % (56 %) | 11.2 % (24 %) |
| student_depth | 0.4 % (1 %) | 2.2 % (6 %) | 5.0 % (12 %) | 14.8 % (33 %) | 19.4 % (46 %) | 17.6 % (34 %) |
| fwd_scan | 0.0 % (0 %) | 10.3 % (17 %) | 7.2 % (15 %) | 17.2 % (27 %) | 41.7 % (81 %) | 14.7 % (29 %) |
| fwd_depth | 1.6 % (1 %) | 4.6 % (8 %) | 6.1 % (12 %) | 19.5 % (40 %) | 30.6 % (66 %) | 17.1 % (34 %) |
| fwd_tof | 1.5 % (1 %) | 6.9 % (11 %) | 13.1 % (22 %) | 30.0 % (66 %) | 37.0 % (76 %) | 20.0 % (37 %) |
| student_fwd_depth | 0.0 % (0 %) | 5.7 % (11 %) | 3.8 % (9 %) | 11.7 % (23 %) | 29.9 % (59 %) | 15.9 % (25 %) |

## Map quality: share of scan points known (height error of known points)

| dog / sensor | memory: ahead | under body | behind | latest frame: ahead | under body | behind |
|---|---|---|---|---|---|---|
| amp_w6m_slow_s1 / depth (walks on perfect scan) | 91 % (0.6 cm) | 52 % (0.7 cm) | 34 % (0.9 cm) | 65 % (0.5 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) |
| amp_w6m_slow_s1 / tof (walks on perfect scan) | 64 % (0.8 cm) | 35 % (0.8 cm) | 20 % (0.9 cm) | 16 % (0.7 cm) | 1 % (0.2 cm) | 0 % (0.3 cm) |
| map_depth / depth | 90 % (0.6 cm) | 51 % (0.7 cm) | 35 % (0.9 cm) | 63 % (0.6 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) |
| map_depth / depth, forward | 95 % (0.7 cm) | 82 % (0.7 cm) | 72 % (0.8 cm) | 63 % (0.6 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) |
| fwd_depth / depth, forward | 95 % (0.7 cm) | 82 % (0.7 cm) | 72 % (0.8 cm) | 65 % (0.6 cm) | 1 % (0.3 cm) | 0 % (0.1 cm) |

## Memory student student_depth

- training 500 iterations, 60 min; final action loss 0.0049, rebuild error under the body 0.6 cm
- own evaluation: falls 9.2 %, walk score 0.294; with blackouts falls 10.0 %
- picture: student_depth/rebuilt_terrain.png

## Memory student student_fwd_depth

- training 500 iterations, 126 min; final action loss 0.0048, rebuild error under the body 0.4 cm
- own evaluation: falls 22.9 %, walk score 0.263; with blackouts falls 22.7 %
- picture: student_fwd_depth/rebuilt_terrain.png
