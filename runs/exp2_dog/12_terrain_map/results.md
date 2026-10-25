# Terrain map experiments

## Falls per scenario (walk score)

| run | course | dark | slippery | pushes | power course |
|---|---|---|---|---|---|
| robust_scan | 5.8 % (0.33) | 4.5 % (0.33) | 4.0 % (0.34) | 8.8 % (0.31) | 14.7 W |
| map_depth | 10.4 % (0.29) | 9.9 % (0.29) | 9.6 % (0.32) | 14.9 % (0.28) | 15.4 W |
| map_tof | 12.8 % (0.31) | 13.1 % (0.31) | 10.6 % (0.33) | 16.1 % (0.30) | 15.7 W |
| map_depth_nomem | 9.6 % (0.39) | 10.8 % (0.39) | 7.3 % (0.41) | 14.5 % (0.36) | 14.1 W |
| student_depth | 8.3 % (0.31) | 8.4 % (0.30) | 6.2 % (0.33) | 12.5 % (0.29) | 14.6 W |

## Gap to the perfect scan (extra falls, percentage points; perfect-scan dog of the same training)

| run | reference | course | dark | slippery | pushes |
|---|---|---|---|---|---|
| map_depth | robust_scan | +4.7 | +5.4 | +5.6 | +6.2 |
| map_tof | robust_scan | +7.0 | +8.6 | +6.6 | +7.3 |
| map_depth_nomem | robust_scan | +3.8 | +6.3 | +3.3 | +5.7 |
| student_depth | robust_scan | +2.5 | +3.9 | +2.2 | +3.7 |

## Falls per terrain type (course): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| robust_scan | 1.2 % (2 %) | 0.0 % (0 %) | 2.5 % (6 %) | 2.7 % (3 %) | 11.8 % (22 %) | 15.9 % (29 %) |
| map_depth | 1.2 % (2 %) | 4.4 % (11 %) | 4.9 % (12 %) | 15.5 % (35 %) | 21.1 % (47 %) | 11.8 % (22 %) |
| map_tof | 0.8 % (1 %) | 3.3 % (8 %) | 2.5 % (6 %) | 30.6 % (63 %) | 19.4 % (41 %) | 7.1 % (13 %) |
| map_depth_nomem | 0.0 % (0 %) | 0.0 % (0 %) | 1.2 % (3 %) | 19.8 % (48 %) | 18.8 % (38 %) | 8.2 % (18 %) |
| student_depth | 0.0 % (0 %) | 3.5 % (6 %) | 2.4 % (3 %) | 15.1 % (34 %) | 11.7 % (25 %) | 12.3 % (26 %) |

## Falls per terrain type (dark): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| robust_scan | 0.4 % (0 %) | 2.2 % (6 %) | 2.5 % (6 %) | 1.5 % (3 %) | 11.7 % (24 %) | 10.0 % (18 %) |
| map_depth | 0.8 % (2 %) | 3.3 % (8 %) | 5.0 % (12 %) | 14.7 % (33 %) | 15.9 % (32 %) | 15.9 % (35 %) |
| map_tof | 0.4 % (0 %) | 9.0 % (23 %) | 4.7 % (6 %) | 30.7 % (67 %) | 14.1 % (28 %) | 10.6 % (19 %) |
| map_depth_nomem | 0.0 % (0 %) | 0.0 % (0 %) | 5.0 % (12 %) | 26.1 % (58 %) | 14.7 % (31 %) | 8.8 % (19 %) |
| student_depth | 0.4 % (1 %) | 2.2 % (6 %) | 2.5 % (6 %) | 15.2 % (35 %) | 11.2 % (24 %) | 13.5 % (26 %) |

## Map quality: share of scan points known (height error of known points)

| dog / sensor | memory: ahead | under body | behind | latest frame: ahead | under body | behind |
|---|---|---|---|---|---|---|
| amp_w6m_slow_s1 / depth (walks on perfect scan) | 91 % (0.6 cm) | 52 % (0.7 cm) | 34 % (0.9 cm) | 65 % (0.5 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) |
| amp_w6m_slow_s1 / tof (walks on perfect scan) | 64 % (0.8 cm) | 35 % (0.8 cm) | 20 % (0.9 cm) | 16 % (0.7 cm) | 1 % (0.2 cm) | 0 % (0.3 cm) |
| map_depth / depth | 90 % (0.6 cm) | 51 % (0.7 cm) | 35 % (0.9 cm) | 63 % (0.6 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) |
| map_tof / tof | 63 % (0.8 cm) | 37 % (0.8 cm) | 22 % (1.0 cm) | 15 % (0.8 cm) | 1 % (0.2 cm) | 0 % (0.2 cm) |
| map_depth_nomem / depth | 64 % (0.6 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) | 64 % (0.6 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) |
| map_depth / depth, forward | 95 % (0.7 cm) | 82 % (0.7 cm) | 72 % (0.8 cm) | 63 % (0.6 cm) | 1 % (0.3 cm) | 0 % (0.2 cm) |
| map_tof / tof, forward | 67 % (0.9 cm) | 62 % (0.9 cm) | 54 % (0.9 cm) | 15 % (0.8 cm) | 1 % (0.2 cm) | 0 % (0.2 cm) |

## Memory student student_depth

- training 500 iterations, 60 min; final action loss 0.0049, rebuild error under the body 0.6 cm
- own evaluation: falls 9.2 %, walk score 0.294; with blackouts falls 10.0 %
- picture: student_depth/rebuilt_terrain.png
