# Deliberate stair stepping

## Falls per scenario (walk score)

| run | stairs_straight | forward | forward_dark | course | power stairs_straight |
|---|---|---|---|---|---|
| map_depth | 9.6 % (0.30) | - | - | - | 15.4 W |
| base_s1 | 2.3 % (0.40) | 1.2 % (0.44) | 2.1 % (0.43) | 8.8 % (0.40) | 12.1 W |
| base_s2 | 2.2 % (0.35) | 2.5 % (0.37) | 2.2 % (0.37) | 8.3 % (0.31) | 17.0 W |
| stairs_s1 | 1.7 % (0.32) | 1.2 % (0.33) | 1.3 % (0.33) | 7.5 % (0.28) | 18.0 W |
| stairs_s2 | 0.3 % (0.33) | 0.3 % (0.34) | 0.7 % (0.34) | 7.5 % (0.30) | 17.7 W |
| support_s1 | 0.6 % (0.31) | 0.5 % (0.34) | 0.5 % (0.34) | 11.3 % (0.31) | 17.3 W |
| support_s2 | 0.4 % (0.34) | 0.7 % (0.35) | 0.4 % (0.35) | 11.9 % (0.31) | 17.7 W |
| student | 0.5 % (0.33) | 0.6 % (0.35) | 0.4 % (0.35) | 6.5 % (0.30) | 16.9 W |

## Falls per terrain type (stairs_straight): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| map_depth | 0.4 % (1 %) | 1.1 % (3 %) | 2.5 % (6 %) | 13.6 % (25 %) | 22.9 % (52 %) | 11.8 % (22 %) |
| base_s1 | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 1.8 % (3 %) | 12.4 % (24 %) |
| base_s2 | 0.0 % (0 %) | 0.0 % (0 %) | 1.2 % (3 %) | 1.5 % (4 %) | 1.2 % (3 %) | 9.4 % (22 %) |
| stairs_s1 | 2.0 % (4 %) | 0.0 % (0 %) | 0.0 % (0 %) | 1.5 % (3 %) | 1.2 % (1 %) | 3.5 % (6 %) |
| stairs_s2 | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.4 % (1 %) | 0.6 % (1 %) | 0.6 % (1 %) |
| support_s1 | 0.8 % (2 %) | 0.0 % (0 %) | 1.1 % (0 %) | 0.0 % (0 %) | 0.6 % (1 %) | 1.2 % (1 %) |
| support_s2 | 0.8 % (1 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.6 % (1 %) | 0.6 % (0 %) |
| student | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 1.8 % (4 %) | 1.2 % (3 %) |

## Falls per terrain type (forward): all rows (hardest rows 6-9)

| run | bumpy | slope_up | slope_down | stairs_descend | stairs_climb | blocks |
|---|---|---|---|---|---|---|
| base_s1 | 0.4 % (1 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 1.2 % (1 %) | 5.3 % (12 %) |
| base_s2 | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 4.7 % (12 %) | 10.6 % (22 %) |
| stairs_s1 | 1.6 % (4 %) | 0.0 % (0 %) | 1.2 % (0 %) | 0.0 % (0 %) | 1.2 % (3 %) | 3.0 % (4 %) |
| stairs_s2 | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.6 % (1 %) | 1.2 % (1 %) |
| support_s1 | 0.4 % (1 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.6 % (1 %) | 1.8 % (3 %) |
| support_s2 | 0.8 % (1 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.4 % (1 %) | 2.3 % (4 %) | 0.0 % (0 %) |
| student | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 0.0 % (0 %) | 2.4 % (6 %) | 1.2 % (0 %) |

## Memory student student

- training 500 iterations, 59 min; final action loss 0.0042, rebuild error under the body 0.4 cm
- own evaluation: falls 7.3 %, walk score 0.289; with blackouts falls 8.2 %
- picture: student/rebuilt_terrain.png
