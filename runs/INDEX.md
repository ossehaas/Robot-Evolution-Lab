# Runs index

Training output, sorted per experiment (reorganized 2026-10-10). Only the best run of each
successful line is kept in full; comparisons and failed experiments keep their results (result.json, videos,
summaries) without models; superseded versions were removed. Details and numbers: `.documentation/project-journal.md`.

## Kept in full

| Run | What | Result |
|---|---|---|
| `exp1_crawler/1_walking/crawler_q4` | first walk (forward only) | falls 0%, speed 0.47 m/s, 1617 s training |
| `exp1_crawler/1_walking/crawler_q4_steer` | steerable walking (M1) | tracking 0.75, falls 0%, speed 0.00 m/s, 252 s training |
| `exp1_crawler/1_walking/crawler_q4_rough` | crawler on rough terrain (baseline for evolution) | tracking 0.46, falls 0%, speed -0.00 m/s, 3405 s training |
| `exp1_crawler/2_evolution/evo_rough_002/g05_025` | best evolved body (used for skills and navigation) | walk 0.31, tracking 0.35, falls 0%, speed 0.00 m/s, 287 s training |
| `exp1_crawler/3_comparisons/compare2/evolved_imu_h8` | evolved body with realistic IMU + memory (the sensor setup used afterwards) | walk 0.26, tracking 0.32, falls 0%, speed -0.00 m/s, 1281 s training |
| `exp1_crawler/4_skills/student` | distilled student: beats all teachers (0.278) | walk 0.28, tracking 0.31, falls 0%, speed 0.00 m/s, 375 s training |
| `exp1_crawler/6_navigation/poc/e2e_rough` | best navigator on a bumpy floor: 7.4 goals/min | 7.4 goals/min, falls 0%, speed -0.00 m/s, 1341 s training |
| `exp1_crawler/6_navigation/poc/hier_flat` | best hierarchical navigator: 7.6 goals/min | 7.6 goals/min, falls 0%, speed 0.00 m/s, 989 s training |
| `exp1_crawler/6_navigation/poc/nav_camera` | best navigator: 9.3 goals/min, camera-like goal signal | 9.3 goals/min, falls 0%, speed -0.00 m/s, 1024 s training |
| `exp1_crawler/6_navigation/poc/student_v2` | walker used by the hierarchical navigators | walk 0.28, tracking 0.31, falls 0%, speed 0.00 m/s, 368 s training |
| `exp1_crawler/7_closest_point/ring_v2` | unreachable goal in a ring: 70 % end within 5 cm of the closest point | 4.6 goals/min, falls 0%, within 5 cm 70%, speed -0.00 m/s, 557 s training |
| `exp1_crawler/8_detour/cup_v1` | detour round a half circle: 73 % reach the goal | 144.7 goals/min, falls 0%, inside goal 76%, speed 0.02 m/s, 1197 s training |
| `exp2_dog/2_walking/calm_v2` | best dog walker (calm, 10 % falls on rough, CoT 2.57, 0 W standing) | walk 0.31, tracking 0.37, falls 10%, CoT 2.57, speed 0.00 m/s, 1803 s training |
| `exp2_dog/4_stairs_seeing/seeing_v3` | best dog on the dog course: seeing (height scan), 7.1 % falls | walk 0.34, tracking 0.40, falls 7%, CoT 2.91, speed 0.00 m/s, 5033 s training |
| `exp2_dog/5_jumping/qdd` | jump with hypothetical fast motors: lowest paw 19 cm | falls 0%, jump 0.192 m, speed 0.00 m/s, 1296 s training |
| `exp2_dog/5_jumping/sts3215` | jump with the real servos: lowest paw 8 cm | falls 0%, jump 0.081 m, speed -0.00 m/s, 1267 s training |

## Results only (models removed)

| Run | What | Result |
|---|---|---|
| `exp1_crawler/3_comparisons/compare/crawler` | body/sensor comparison runs (journal steps 11-14) | walk 0.24, tracking 0.29, falls 0%, speed -0.00 m/s, 539 s training |
| `exp1_crawler/3_comparisons/compare/flat_crawler` | body/sensor comparison runs (journal steps 11-14) | walk 0.53, tracking 0.51, falls 0%, speed -0.00 m/s, 291 s training |
| `exp1_crawler/3_comparisons/compare/flat_long8` | body/sensor comparison runs (journal steps 11-14) | walk 0.28, tracking 0.31, falls 0%, speed -0.00 m/s, 334 s training |
| `exp1_crawler/3_comparisons/compare/flat_tall6` | body/sensor comparison runs (journal steps 11-14) | walk 0.25, tracking 0.36, falls 0%, speed -0.00 m/s, 300 s training |
| `exp1_crawler/3_comparisons/compare/long8` | body/sensor comparison runs (journal steps 11-14) | walk 0.02, tracking 0.27, falls 0%, speed 0.00 m/s, 946 s training |
| `exp1_crawler/3_comparisons/compare/reeval_crawler_rough2000` | body/sensor comparison runs (journal steps 11-14) | walk 0.35, tracking 0.40, falls 0%, speed 0.00 m/s, 0 s training |
| `exp1_crawler/3_comparisons/compare/reeval_stander_g01_008` | body/sensor comparison runs (journal steps 11-14) | walk 0.06, tracking 0.31, falls 0%, speed -0.00 m/s, 0 s training |
| `exp1_crawler/3_comparisons/compare/tall6` | body/sensor comparison runs (journal steps 11-14) | walk 0.05, tracking 0.32, falls 0%, speed -0.00 m/s, 758 s training |
| `exp1_crawler/3_comparisons/compare2/crawler_ideal` | sensor comparison runs (journal step 14) | walk 0.25, tracking 0.24, falls 2%, speed 0.00 m/s, 1253 s training |
| `exp1_crawler/3_comparisons/compare2/evolved_ideal` | sensor comparison runs (journal step 14) | walk 0.30, tracking 0.34, falls 0%, speed -0.00 m/s, 1394 s training |
| `exp1_crawler/3_comparisons/compare2/evolved_imuflow_h8` | sensor comparison runs (journal step 14) | walk 0.26, tracking 0.34, falls 0%, speed 0.00 m/s, 1356 s training |
| `exp1_crawler/4_skills/generalist` | RL generalist (0.256), beaten by the distilled student | walk 0.26, tracking 0.31, falls 0%, speed 0.00 m/s, 658 s training |
| `exp1_crawler/4_skills/spec_blocks` | blocks specialist teacher | walk 0.19, tracking 0.30, falls 0%, speed 0.00 m/s, 1966 s training |
| `exp1_crawler/4_skills/spec_stairs` | stairs specialist teacher | walk 0.22, tracking 0.30, falls 0%, speed 0.00 m/s, 1167 s training |
| `exp1_crawler/4_skills/student_polished` | RL polish after distillation fell back to 0.235 | walk 0.23, tracking 0.32, falls 0%, speed -0.00 m/s, 716 s training |
| `exp1_crawler/5_stairs/A1_mg90s` | stairs tests A1/A2/B/C; note: the height scan of test C saw nothing (bug found 2026-10-09) | walk 0.18, tracking 0.30, falls 0%, speed -0.00 m/s, 1285 s training |
| `exp1_crawler/5_stairs/A2_mg92b` | stairs tests A1/A2/B/C; note: the height scan of test C saw nothing (bug found 2026-10-09) | walk 0.19, tracking 0.31, falls 0%, speed -0.00 m/s, 1292 s training |
| `exp1_crawler/5_stairs/B_foot_lift` | stairs tests A1/A2/B/C; note: the height scan of test C saw nothing (bug found 2026-10-09) | walk 0.19, tracking 0.32, falls 0%, speed 0.00 m/s, 1351 s training |
| `exp1_crawler/5_stairs/C_height_scan` | stairs tests A1/A2/B/C; note: the height scan of test C saw nothing (bug found 2026-10-09) | walk 0.27, tracking 0.34, falls 0%, speed -0.00 m/s, 1372 s training |
| `exp1_crawler/5_stairs/genomes` | stairs tests A1/A2/B/C; note: the height scan of test C saw nothing (bug found 2026-10-09) |  |
| `exp1_crawler/6_navigation/nav/goal_v1` | first navigator (8.9 goals/min), superseded by poc/nav_camera | 8.9 goals/min, falls 0%, speed 0.00 m/s, 1443 s training |
| `exp1_crawler/6_navigation/poc/hier_rough` | hierarchical navigator on a bumpy floor (6.0 goals/min) | 6.0 goals/min, falls 0%, speed 0.00 m/s, 1183 s training |
| `exp1_crawler/6_navigation/poc/nav_control` | navigation control run (8.8 goals/min) | 8.8 goals/min, falls 0%, speed 0.00 m/s, 1106 s training |
| `exp1_crawler/6_navigation/poc/nav_strict` | stricter collision penalty (9.0 goals/min); nav_camera continues from it | 9.0 goals/min, falls 0%, speed 0.00 m/s, 1178 s training |
| `exp1_crawler/6_navigation/poc/student_v2_polished` | RL polish of the student (worse than the student) | walk 0.26, tracking 0.32, falls 0%, speed -0.00 m/s, 802 s training |
| `exp1_crawler/7_closest_point/pocket_p1_mlp_warm` | ring v1 (parked at the wall) and the pocket experiment (not solved) | 3.5 goals/min, falls 0%, within 5 cm 37%, speed -0.00 m/s, 712 s training |
| `exp1_crawler/7_closest_point/pocket_p2_hier_gru` | ring v1 (parked at the wall) and the pocket experiment (not solved) | 1.7 goals/min, falls 0%, within 5 cm 21%, speed -0.00 m/s, 1602 s training |
| `exp1_crawler/7_closest_point/pocket_p3_hier_mlp` | ring v1 (parked at the wall) and the pocket experiment (not solved) | 2.0 goals/min, falls 0%, within 5 cm 18%, speed -0.00 m/s, 1371 s training |
| `exp1_crawler/7_closest_point/pocket_zeroshot_ring_v2` | ring v1 (parked at the wall) and the pocket experiment (not solved) | 3.7 goals/min, falls 0%, within 5 cm 32%, speed -0.00 m/s, 0 s training |
| `exp1_crawler/7_closest_point/ring_v1` | ring v1 (parked at the wall) and the pocket experiment (not solved) | 4.2 goals/min, falls 0%, speed -0.00 m/s, 416 s training |
| `exp1_crawler/7_closest_point/ring_v1_eval` | ring v1 (parked at the wall) and the pocket experiment (not solved) | 4.1 goals/min, falls 0%, within 5 cm 38%, speed 0.00 m/s, 0 s training |
| `exp1_crawler/7_closest_point/zeroshot_nav_camera` | ring v1 (parked at the wall) and the pocket experiment (not solved) | 4.4 goals/min, falls 0%, within 5 cm 53%, speed -0.00 m/s, 0 s training |
| `exp1_crawler/8_detour/zeroshot_nav_camera` | detour zero-shot baseline (1 % at 180 deg) | 46.8 goals/min, falls 0%, inside goal 29%, speed 0.02 m/s, 1 s training |
| `exp2_dog/3_flexible_back/both_k8` | flexible back: every variant less stable than the stiff body (journal step 23) | walk 0.26, tracking 0.33, falls 24%, CoT 3.70, speed 0.00 m/s, 1210 s training |
| `exp2_dog/3_flexible_back/eval` | flexible back: every variant less stable than the stiff body (journal step 23) |  |
| `exp2_dog/3_flexible_back/pitch_k8` | flexible back: every variant less stable than the stiff body (journal step 23) | walk 0.29, tracking 0.32, falls 21%, CoT 3.78, speed -0.00 m/s, 1186 s training |
| `exp2_dog/3_flexible_back/roll_k16` | flexible back: every variant less stable than the stiff body (journal step 23) | walk 0.29, tracking 0.33, falls 21%, CoT 3.71, speed -0.00 m/s, 1269 s training |
| `exp2_dog/3_flexible_back/roll_k4` | flexible back: every variant less stable than the stiff body (journal step 23) | walk 0.29, tracking 0.35, falls 22%, CoT 3.72, speed 0.00 m/s, 1304 s training |
| `exp2_dog/3_flexible_back/roll_k8` | flexible back: every variant less stable than the stiff body (journal step 23) | walk 0.28, tracking 0.34, falls 21%, CoT 3.66, speed 0.00 m/s, 1161 s training |
| `exp2_dog/3_flexible_back/stiff` | flexible back: every variant less stable than the stiff body (journal step 23) | walk 0.29, tracking 0.35, falls 19%, CoT 3.66, speed 0.00 m/s, 1070 s training |
| `exp2_dog/3_flexible_back/zeroshot_roll_k8` | flexible back: every variant less stable than the stiff body (journal step 23) | walk 0.30, tracking 0.37, falls 14%, CoT 3.84, speed -0.00 m/s, 0 s training |
| `exp2_dog/4_stairs_seeing/blind` | blind control for seeing_v3: 16.0 % falls | walk 0.32, tracking 0.36, falls 16%, CoT 2.95, speed -0.00 m/s, 4653 s training |
| `exp2_dog/4_stairs_seeing/eval_range75` | per-type falls of range75 (stairs/blocks are the problem) | walk 0.31, tracking 0.39, falls 19%, CoT 3.01, speed -0.00 m/s, 0 s training |
| `exp2_dog/4_stairs_seeing/range50` | stairs with 50 % joint range: 21 % falls | walk 0.31, tracking 0.37, falls 21%, CoT 2.50, speed -0.00 m/s, 4451 s training |
| `exp2_dog/4_stairs_seeing/range75` | stairs with 75 % joint range (blind): 18 % falls; start of blind/seeing | walk 0.29, tracking 0.34, falls 18%, CoT 3.01, speed 0.00 m/s, 4519 s training |
| `exp2_dog/4_stairs_seeing/zeroshot_calm_v2` | calm_v2 on the dog course without training: 28 % falls | walk 0.30, tracking 0.39, falls 28%, CoT 2.61, speed -0.00 m/s, 0 s training |

## Removed

| Old path | Why | Result |
|---|---|---|
| `runs/_check_crawler` | scratch / temporary check |  |
| `runs/random_robots` | scratch / temporary check |  |
| `runs/distill_check` | scratch / temporary check | walk 0.27, tracking 0.31, falls 0%, speed -0.00 m/s, 178 s training |
| `runs/crawler_q4_steer_v1` | superseded by crawler_q4_steer (symmetric command ranges) | tracking 0.85, falls 0%, speed 0.00 m/s, 2262 s training |
| `runs/evo_rough_001/g00_000` | individual of the failed evolution run 1 | tracking 0.31, falls 0%, speed -0.00 m/s, 1120 s training |
| `runs/evo_rough_001/g00_001` | individual of the failed evolution run 1 | tracking 0.35, falls 0%, speed -0.00 m/s, 970 s training |
| `runs/evo_rough_001/g00_002` | individual of the failed evolution run 1 | tracking 0.34, falls 0%, speed 0.00 m/s, 1156 s training |
| `runs/evo_rough_001/g00_003` | individual of the failed evolution run 1 | tracking 0.32, falls 0%, speed 0.00 m/s, 1287 s training |
| `runs/evo_rough_001/g00_004` | individual of the failed evolution run 1 | tracking 0.33, falls 0%, speed 0.00 m/s, 1230 s training |
| `runs/evo_rough_001/g00_005` | individual of the failed evolution run 1 | tracking 0.30, falls 1%, speed 0.00 m/s, 1363 s training |
| `runs/evo_rough_001/g00_006` | individual of the failed evolution run 1 | tracking 0.24, falls 0%, speed 0.00 m/s, 1765 s training |
| `runs/evo_rough_001/g00_007` | individual of the failed evolution run 1 | tracking 0.20, falls 0%, speed -0.00 m/s, 1246 s training |
| `runs/evo_rough_001/g01_008` | individual of the failed evolution run 1 | tracking 0.35, falls 0%, speed -0.00 m/s, 551 s training |
| `runs/evo_rough_001/g01_009` | individual of the failed evolution run 1 | tracking 0.33, falls 1%, speed -0.00 m/s, 567 s training |
| `runs/evo_rough_001/g01_010` | individual of the failed evolution run 1 | tracking 0.34, falls 0%, speed -0.00 m/s, 556 s training |
| `runs/evo_rough_001/g01_011` | individual of the failed evolution run 1 | tracking 0.32, falls 0%, speed -0.00 m/s, 1866 s training |
| `runs/evo_rough_001/g02_012` | individual of the failed evolution run 1 | tracking 0.31, falls 0%, speed -0.00 m/s, 1068 s training |
| `runs/evo_rough_001/g02_013` | individual of the failed evolution run 1 | tracking 0.26, falls 10%, speed 0.00 m/s, 498 s training |
| `runs/evo_rough_001/g02_014` | individual of the failed evolution run 1 | tracking 0.27, falls 10%, speed -0.00 m/s, 1079 s training |
| `runs/evo_rough_001/g02_015` | individual of the failed evolution run 1 | tracking 0.35, falls 0%, speed 0.00 m/s, 576 s training |
| `runs/evo_rough_001/g03_016` | individual of the failed evolution run 1 | tracking 0.34, falls 0%, speed 0.00 m/s, 577 s training |
| `runs/evo_rough_001/g03_017` | individual of the failed evolution run 1 | tracking 0.27, falls 4%, speed 0.00 m/s, 530 s training |
| `runs/evo_rough_001/g03_018` | individual of the failed evolution run 1 | tracking 0.33, falls 0%, speed -0.00 m/s, 1016 s training |
| `runs/evo_rough_001/g03_019` | individual of the failed evolution run 1 | tracking 0.35, falls 0%, speed -0.00 m/s, 547 s training |
| `runs/evo_rough_001/g04_020` | individual of the failed evolution run 1 | tracking 0.34, falls 0%, speed -0.00 m/s, 553 s training |
| `runs/evo_rough_001/g04_021` | individual of the failed evolution run 1 | tracking 0.35, falls 0%, speed 0.00 m/s, 542 s training |
| `runs/evo_rough_001/g04_022` | individual of the failed evolution run 1 | tracking 0.27, falls 9%, speed -0.00 m/s, 1008 s training |
| `runs/evo_rough_001/g04_023` | individual of the failed evolution run 1 | tracking 0.35, falls 0%, speed 0.00 m/s, 565 s training |
| `runs/evo_rough_001/g05_024` | individual of the failed evolution run 1 | tracking 0.34, falls 0%, speed -0.00 m/s, 550 s training |
| `runs/evo_rough_001/g05_025` | individual of the failed evolution run 1 | tracking 0.29, falls 2%, speed -0.00 m/s, 516 s training |
| `runs/evo_rough_001/g05_026` | individual of the failed evolution run 1 | tracking 0.34, falls 0%, speed -0.00 m/s, 555 s training |
| `runs/evo_rough_001/g05_027` | individual of the failed evolution run 1 | tracking 0.33, falls 0%, speed -0.00 m/s, 818 s training |
| `runs/evo_rough_002/g00_000` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.25, tracking 0.26, falls 1%, speed -0.00 m/s, 555 s training |
| `runs/evo_rough_002/g00_001` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.17, tracking 0.39, falls 0%, speed -0.00 m/s, 739 s training |
| `runs/evo_rough_002/g00_002` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.22, tracking 0.32, falls 0%, speed 0.00 m/s, 609 s training |
| `runs/evo_rough_002/g00_003` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.00, tracking 0.34, falls 0%, speed 0.00 m/s, 511 s training |
| `runs/evo_rough_002/g00_004` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.01, tracking 0.34, falls 0%, speed -0.00 m/s, 684 s training |
| `runs/evo_rough_002/g00_005` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.01, tracking 0.35, falls 0%, speed 0.00 m/s, 568 s training |
| `runs/evo_rough_002/g00_006` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.24, tracking 0.34, falls 0%, speed -0.00 m/s, 691 s training |
| `runs/evo_rough_002/g00_007` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.10, tracking 0.28, falls 0%, speed -0.00 m/s, 487 s training |
| `runs/evo_rough_002/g01_008` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.27, tracking 0.28, falls 1%, speed 0.00 m/s, 318 s training |
| `runs/evo_rough_002/g01_009` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.22, tracking 0.33, falls 0%, speed 0.00 m/s, 352 s training |
| `runs/evo_rough_002/g01_010` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.24, tracking 0.23, falls 4%, speed -0.00 m/s, 290 s training |
| `runs/evo_rough_002/g01_011` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.27, tracking 0.31, falls 1%, speed -0.00 m/s, 539 s training |
| `runs/evo_rough_002/g02_012` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.29, tracking 0.34, falls 0%, speed 0.00 m/s, 285 s training |
| `runs/evo_rough_002/g02_013` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.05, tracking 0.35, falls 1%, speed -0.00 m/s, 581 s training |
| `runs/evo_rough_002/g02_014` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.28, tracking 0.30, falls 0%, speed -0.00 m/s, 606 s training |
| `runs/evo_rough_002/g02_015` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.27, tracking 0.30, falls 1%, speed 0.00 m/s, 320 s training |
| `runs/evo_rough_002/g03_016` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.27, tracking 0.31, falls 0%, speed 0.00 m/s, 709 s training |
| `runs/evo_rough_002/g03_017` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.22, tracking 0.24, falls 7%, speed -0.00 m/s, 636 s training |
| `runs/evo_rough_002/g03_018` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.31, tracking 0.36, falls 0%, speed 0.00 m/s, 308 s training |
| `runs/evo_rough_002/g03_019` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.25, tracking 0.29, falls 5%, speed -0.00 m/s, 634 s training |
| `runs/evo_rough_002/g04_020` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.31, tracking 0.35, falls 0%, speed -0.00 m/s, 288 s training |
| `runs/evo_rough_002/g04_021` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.31, tracking 0.36, falls 0%, speed 0.00 m/s, 306 s training |
| `runs/evo_rough_002/g04_022` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.28, tracking 0.34, falls 0%, speed -0.00 m/s, 305 s training |
| `runs/evo_rough_002/g04_023` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.28, tracking 0.36, falls 0%, speed -0.00 m/s, 324 s training |
| `runs/evo_rough_002/g05_024` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.28, tracking 0.32, falls 0%, speed 0.00 m/s, 713 s training |
| `runs/evo_rough_002/g05_026` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.29, tracking 0.32, falls 0%, speed -0.00 m/s, 282 s training |
| `runs/evo_rough_002/g05_027` | other individuals of evolution run 2 (leaderboard.md keeps their scores) | walk 0.30, tracking 0.35, falls 0%, speed -0.00 m/s, 313 s training |
| `runs/dog_walk/_debug` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md |  |
| `runs/dog_walk/_eval_rough2_flat_push` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.32, tracking 0.39, falls 5%, speed -0.00 m/s, 1 s training |
| `runs/dog_walk/_eval_rough4_border` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.32, tracking 0.40, falls 18%, speed 0.00 m/s, 0 s training |
| `runs/dog_walk/_eval_sym3_nopush` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.34, tracking 0.39, falls 2%, speed 0.00 m/s, 0 s training |
| `runs/dog_walk/_eval_sym3_push` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.33, tracking 0.38, falls 4%, speed 0.01 m/s, 0 s training |
| `runs/dog_walk/_mirror` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md |  |
| `runs/dog_walk/_smoke` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.04, tracking 0.29, falls 2%, speed 0.00 m/s, 5 s training |
| `runs/dog_walk/calm_both` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.28, tracking 0.34, falls 20%, CoT 3.10, speed -0.00 m/s, 1094 s training |
| `runs/dog_walk/calm_rate` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.29, tracking 0.36, falls 21%, CoT 3.23, speed -0.00 m/s, 1021 s training |
| `runs/dog_walk/calm_stand` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.26, tracking 0.27, falls 26%, CoT 3.42, speed 0.00 m/s, 1072 s training |
| `runs/dog_walk/energy_x10` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.29, tracking 0.36, falls 21%, CoT 3.26, speed -0.00 m/s, 1015 s training |
| `runs/dog_walk/energy_x3` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.30, tracking 0.36, falls 19%, CoT 3.74, speed 0.00 m/s, 1097 s training |
| `runs/dog_walk/puppy_print_v1` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.28, tracking 0.31, falls 1%, speed -0.01 m/s, 2003 s training |
| `runs/dog_walk/puppy_print_v2` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.25, tracking 0.23, falls 1%, speed -0.00 m/s, 2229 s training |
| `runs/dog_walk/puppy_rough_v1` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.23, tracking 0.27, falls 46%, speed 0.00 m/s, 2760 s training |
| `runs/dog_walk/puppy_rough_v2` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.23, tracking 0.27, falls 46%, speed -0.01 m/s, 2879 s training |
| `runs/dog_walk/puppy_rough_v3` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.27, tracking 0.30, falls 29%, speed -0.00 m/s, 3250 s training |
| `runs/dog_walk/puppy_rough_v4` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.29, tracking 0.35, falls 19%, speed -0.00 m/s, 2196 s training |
| `runs/dog_walk/puppy_rough_v5` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.30, tracking 0.34, falls 16%, CoT 3.57, speed -0.00 m/s, 2276 s training |
| `runs/dog_walk/puppy_smooth_v1` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.37, tracking 0.38, falls 2%, speed 0.00 m/s, 3719 s training |
| `runs/dog_walk/puppy_sym_v1` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.21, tracking 0.21, falls 4%, speed 0.00 m/s, 3018 s training |
| `runs/dog_walk/puppy_sym_v2` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.27, tracking 0.28, falls 4%, speed -0.00 m/s, 2276 s training |
| `runs/dog_walk/puppy_sym_v3` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.31, tracking 0.33, falls 3%, speed -0.01 m/s, 2253 s training |
| `runs/dog_walk/puppy_trot_v1` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.16, tracking 0.21, falls 4%, speed 0.00 m/s, 1658 s training |
| `runs/dog_walk/puppy_trot_v2` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md |  |
| `runs/dog_walk/puppy_trot_v3` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md |  |
| `runs/dog_walk/puppy_trot_v4` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.23, tracking 0.27, falls 5%, speed -0.00 m/s, 1511 s training |
| `runs/dog_walk/puppy_walk_v1` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.19, tracking 0.21, falls 100%, speed -0.00 m/s, 1099 s training |
| `runs/dog_walk/puppy_walk_v2` | intermediate dog walking versions (journal step 22-24): see the chain in INDEX.md | walk 0.46, tracking 0.44, falls 0%, speed 0.00 m/s, 981 s training |
| `runs/dog_stairs/_scan` | scratch |  |
| `runs/dog_stairs/seeing` | seeing v1: scan in every history frame, curriculum collapsed | walk 0.23, tracking 0.26, falls 33%, CoT 2.64, speed -0.00 m/s, 14610 s training |
| `runs/dog_stairs/seeing_v2` | seeing v2: the scan read -1 everywhere (bug) | walk 0.31, tracking 0.36, falls 16%, CoT 2.80, speed 0.00 m/s, 4986 s training |
| `runs/dog_jump/v1` | jump v1: kneeling and somersault exploits |  |

## Old -> new paths

| Old | New |
|---|---|
| `runs/crawler_q4` | `runs/exp1_crawler/1_walking/crawler_q4` |
| `runs/crawler_q4_steer` | `runs/exp1_crawler/1_walking/crawler_q4_steer` |
| `runs/crawler_q4_rough` | `runs/exp1_crawler/1_walking/crawler_q4_rough` |
| `runs/crawler_q4_steer_v1` | `runs/exp1_crawler/1_walking/crawler_q4_steer_v1` |
| `runs/evo_rough_001` | `runs/exp1_crawler/2_evolution/evo_rough_001` |
| `runs/evo_rough_002` | `runs/exp1_crawler/2_evolution/evo_rough_002` |
| `runs/compare` | `runs/exp1_crawler/3_comparisons/compare` |
| `runs/compare2` | `runs/exp1_crawler/3_comparisons/compare2` |
| `runs/skills` | `runs/exp1_crawler/4_skills` |
| `runs/stairs` | `runs/exp1_crawler/5_stairs` |
| `runs/nav` | `runs/exp1_crawler/6_navigation/nav` |
| `runs/poc` | `runs/exp1_crawler/6_navigation/poc` |
| `runs/closest` | `runs/exp1_crawler/7_closest_point` |
| `runs/detour` | `runs/exp1_crawler/8_detour` |
| `runs/dog_designs` | `runs/exp2_dog/1_design` |
| `runs/dog_capability` | `runs/exp2_dog/1_design/capability` |
| `runs/dog_walk` | `runs/exp2_dog/2_walking` |
| `runs/dog_spine` | `runs/exp2_dog/3_flexible_back` |
| `runs/dog_stairs` | `runs/exp2_dog/4_stairs_seeing` |
| `runs/dog_jump` | `runs/exp2_dog/5_jumping` |

## Added after the reorganization

| Run | What | Result |
|---|---|---|
| `exp2_dog/6_natural_gait/walk_trot` | walk/trot gait from seeing_v3 (warm start kept the hopping) | falls 4.5%, walk 0.38 |
| `exp2_dog/9_gait_body/B_springy` | springy ankle + soft paws, knees as before | falls 0.6%, walk 0.47, a clean trot |
| `exp2_dog/9_gait_body/D_spot_springy` | Spot knees + springy ankle + soft paws (the owner's choice) | falls 1.7%, walk 0.36 |
| `exp2_dog/9_gait_body/{A,C,E,F}_*` | leg variant comparison (stiff ankle / hard paws hop) | see journal step 29 |
| `exp2_dog/10_printable_dog/D_print_s2` | **best dog: the printable design (Spot knees, printed TPU ankle, soft sole, 2.45 kg)** | falls 0.3%, walk 0.46, CoT 4.16 |
| `exp2_dog/10_printable_dog/D_print_s1` | same recipe, other seed | falls 3.0%, walk 0.30 |
| `exp2_dog/11_imitation/amp_w6m_slow_s1` | **imitation of a real dog (AMP style 6, mirrored + slowed recordings) on D_print_s2; use with heading hold 2** | falls 0.4%, walk 0.40, CoT 3.60, walks at 0.1 m/s on CoT 1.6, 3-4x more dog-like |
| `exp2_dog/11_imitation/amp_w6m_slow_s2` | same recipe, seed 2 | falls 1.8% (blocks 7%), CoT 3.36 |
| `exp2_dog/11_imitation/amp_w6m_s{1,2}` | style 6, mirrored recordings (no slow copies): most dog-like, stand still at 0.1 m/s | falls 1.7 / 2.3%, CoT 3.24 / 3.12 |
| `exp2_dog/11_imitation/amp_w3`, `amp_w6` | round 1 (style 3 / 6, recordings not mirrored) | falls 1.1 / 1.4% |
| `exp2_dog/11_imitation/free_ctrl` | control: same fine-tune without the style reward | falls 0.3%, like D_print_s2 |
