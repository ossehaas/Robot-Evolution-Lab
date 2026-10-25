# Walking forwards: the batch.py dogs

## Falls per scenario (walk score)

| run | forward | forward_dark | course | dark | power forward |
|---|---|---|---|---|---|
| robust_scan | 7.8 % (0.33) | 8.4 % (0.33) | 5.8 % (0.33) | 4.5 % (0.33) | 14.3 W |
| map_depth | 10.9 % (0.30) | 11.8 % (0.30) | 10.4 % (0.29) | 9.9 % (0.29) | 15.4 W |
| map_tof | 15.6 % (0.33) | 16.9 % (0.33) | 12.8 % (0.31) | 13.1 % (0.31) | 15.8 W |
| map_depth_nomem | 11.4 % (0.43) | 13.0 % (0.43) | 9.6 % (0.39) | 10.8 % (0.39) | 13.8 W |
| student_depth | 7.6 % (0.32) | 10.5 % (0.32) | 8.3 % (0.31) | 8.4 % (0.30) | 14.4 W |

## Gap to the perfect scan (extra falls, percentage points; perfect-scan dog of the same training)

| run | reference | forward | forward_dark | course | dark |
|---|---|---|---|---|---|
| map_depth | robust_scan | +3.1 | +3.4 | +4.7 | +5.4 |
| map_tof | robust_scan | +7.8 | +8.5 | +7.0 | +8.6 |
| map_depth_nomem | robust_scan | +3.6 | +4.6 | +3.8 | +6.3 |
| student_depth | robust_scan | -0.2 | +2.1 | +2.5 | +3.9 |

## Falls per terrain type (forward): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| robust_scan | 0.4 % (1 %) | 2.4 % (6 %) | 3.8 % (9 %) | 9.0 % (14 %) | 15.3 % (32 %) | 14.7 % (24 %) |
| map_depth | 1.2 % (2 %) | 3.5 % (6 %) | 7.5 % (19 %) | 11.3 % (20 %) | 25.9 % (56 %) | 15.9 % (32 %) |
| map_tof | 0.4 % (0 %) | 6.7 % (17 %) | 8.6 % (19 %) | 34.3 % (71 %) | 24.6 % (56 %) | 9.4 % (21 %) |
| map_depth_nomem | 0.0 % (0 %) | 1.1 % (3 %) | 4.9 % (9 %) | 17.3 % (42 %) | 37.5 % (66 %) | 1.8 % (3 %) |
| student_depth | 1.2 % (3 %) | 0.0 % (0 %) | 7.5 % (19 %) | 4.3 % (7 %) | 20.0 % (46 %) | 14.1 % (29 %) |

## Falls per terrain type (forward_dark): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| robust_scan | 0.8 % (2 %) | 2.5 % (3 %) | 3.8 % (9 %) | 9.8 % (14 %) | 16.5 % (37 %) | 15.3 % (31 %) |
| map_depth | 0.4 % (0 %) | 1.1 % (3 %) | 8.5 % (21 %) | 18.7 % (36 %) | 26.4 % (56 %) | 11.2 % (24 %) |
| map_tof | 1.2 % (1 %) | 5.6 % (14 %) | 8.6 % (22 %) | 39.0 % (81 %) | 25.3 % (53 %) | 8.8 % (21 %) |
| map_depth_nomem | 0.0 % (0 %) | 1.2 % (3 %) | 3.8 % (9 %) | 27.1 % (61 %) | 27.6 % (51 %) | 7.1 % (15 %) |
| student_depth | 0.4 % (1 %) | 2.2 % (6 %) | 5.0 % (12 %) | 14.8 % (33 %) | 19.4 % (46 %) | 17.6 % (34 %) |

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
