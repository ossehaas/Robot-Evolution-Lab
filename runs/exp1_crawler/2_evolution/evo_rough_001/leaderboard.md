# Evolution run `evo_rough_001`

28 robots evaluated, 0 failed.

| Rank | Robot | Gen | Fitness | Speed +x (cm/s) | Falls | Servos | Mass (g) | Topology | Parents | Warm start |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [g01_008](g01_008/robot/preview.png) | 1 | 0.348 | -0.00 | 0.00 | 6 | 186 | `pp-p` | g00_001 | yes |
| 2 | [g02_015](g02_015/robot/preview.png) | 2 | 0.347 | 0.00 | 0.00 | 6 | 186 | `pp-p` | g01_008 | yes |
| 3 | [g03_019](g03_019/robot/preview.png) | 3 | 0.346 | -0.02 | 0.00 | 6 | 186 | `pp-p` | g01_008, g02_015 | yes |
| 4 | [g04_023](g04_023/robot/preview.png) | 4 | 0.345 | 0.00 | 0.00 | 6 | 190 | `pp-p` | g01_008 | yes |
| 5 | [g00_001](g00_001/robot/preview.png) | 0 | 0.345 | -0.02 | 0.00 | 6 | 186 | `pp-p` | random | no |
| 6 | [g04_021](g04_021/robot/preview.png) | 4 | 0.345 | 0.01 | 0.00 | 6 | 186 | `pp-p` | g01_008 | yes |
| 7 | [g00_002](g00_002/robot/preview.png) | 0 | 0.342 | 0.00 | 0.00 | 8 | 228 | `y-yy-p` | random | no |
| 8 | [g05_024](g05_024/robot/preview.png) | 5 | 0.341 | -0.04 | 0.00 | 6 | 187 | `pp-p` | g01_008, g02_015 | yes |
| 9 | [g05_026](g05_026/robot/preview.png) | 5 | 0.341 | -0.00 | 0.00 | 6 | 186 | `pp-p` | g01_008 | yes |
| 10 | [g01_010](g01_010/robot/preview.png) | 1 | 0.341 | -0.00 | 0.00 | 6 | 186 | `pp-p` | g00_001 | yes |
| 11 | [g03_016](g03_016/robot/preview.png) | 3 | 0.339 | 0.00 | 0.00 | 6 | 186 | `pp-p` | g02_015 | yes |
| 12 | [g04_020](g04_020/robot/preview.png) | 4 | 0.336 | -0.03 | 0.00 | 6 | 186 | `pp-p` | g01_008, g02_015 | yes |
| 13 | [g01_009](g01_009/robot/preview.png) | 1 | 0.328 | -0.02 | 0.01 | 6 | 185 | `pp-p` | g00_001 | yes |
| 14 | [g00_004](g00_004/robot/preview.png) | 0 | 0.327 | 0.01 | 0.00 | 8 | 196 | `py-yp` | random | no |
| 15 | [g05_027](g05_027/robot/preview.png) | 5 | 0.325 | -0.01 | 0.00 | 4 | 148 | `pp` | g01_008, g02_015 | no |
| 16 | [g01_011](g01_011/robot/preview.png) | 1 | 0.325 | -0.00 | 0.00 | 10 | 254 | `y-yy-pp` | g00_002 | no |
| 17 | [g03_018](g03_018/robot/preview.png) | 3 | 0.324 | -0.01 | 0.00 | 6 | 181 | `ppp` | g01_008, g02_015 | no |
| 18 | [g00_003](g00_003/robot/preview.png) | 0 | 0.315 | 0.01 | 0.00 | 8 | 191 | `y-ppy` | random | no |
| 19 | [g02_012](g02_012/robot/preview.png) | 2 | 0.310 | -0.02 | 0.00 | 6 | 180 | `pp-y` | g00_001, g01_008 | no |
| 20 | [g00_000](g00_000/robot/preview.png) | 0 | 0.309 | -0.02 | 0.00 | 8 | 180 | `yp-yp` | seed | no |
| 21 | [g00_005](g00_005/robot/preview.png) | 0 | 0.294 | 0.05 | 0.01 | 12 | 243 | `yp-py-pp` | random | no |
| 22 | [g05_025](g05_025/robot/preview.png) | 5 | 0.287 | -0.01 | 0.02 | 6 | 186 | `pp-p` | g01_008, g02_015 | yes |
| 23 | [g03_017](g03_017/robot/preview.png) | 3 | 0.262 | 0.02 | 0.04 | 6 | 187 | `pp-p` | g01_008, g02_015 | yes |
| 24 | [g04_022](g04_022/robot/preview.png) | 4 | 0.255 | -0.00 | 0.09 | 6 | 189 | `py-p` | g01_008 | no |
| 25 | [g02_014](g02_014/robot/preview.png) | 2 | 0.252 | -0.02 | 0.10 | 8 | 223 | `py-p-p` | g01_008 | no |

Play the best robot:

```
python -m core.evolution.play --run_dir runs\evo_rough_001\g01_008
```
