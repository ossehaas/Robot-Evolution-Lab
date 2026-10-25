# Evolution run `evo_rough_002`

28 robots evaluated, 0 failed.

| Rank | Robot | Gen | Fitness | Speed +x (cm/s) | Falls | Servos | Mass (g) | Topology | Parents | Warm start |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [g05_025](g05_025/robot/preview.png) | 5 | 0.311 | 0.12 | 0.00 | 6 | 162 | `yp-p` | g04_020 | yes |
| 2 | [g04_020](g04_020/robot/preview.png) | 4 | 0.307 | -0.18 | 0.00 | 6 | 162 | `yp-p` | g02_012 | yes |
| 3 | [g04_021](g04_021/robot/preview.png) | 4 | 0.306 | 0.04 | 0.00 | 6 | 163 | `yp-p` | g03_018, g02_012 | yes |
| 4 | [g03_018](g03_018/robot/preview.png) | 3 | 0.306 | 0.11 | 0.00 | 6 | 163 | `yp-p` | g02_012 | yes |
| 5 | [g05_027](g05_027/robot/preview.png) | 5 | 0.300 | -0.01 | 0.00 | 6 | 159 | `yp-p` | g04_021, g04_020 | yes |
| 6 | [g02_012](g02_012/robot/preview.png) | 2 | 0.293 | 0.10 | 0.00 | 6 | 164 | `yp-p` | g01_011 | yes |
| 7 | [g05_026](g05_026/robot/preview.png) | 5 | 0.288 | -0.05 | 0.00 | 6 | 163 | `yp-p` | g04_021, g04_020 | yes |
| 8 | [g04_022](g04_022/robot/preview.png) | 4 | 0.283 | -0.12 | 0.00 | 6 | 163 | `yp-p` | g03_018 | yes |
| 9 | [g05_024](g05_024/robot/preview.png) | 5 | 0.276 | 0.14 | 0.00 | 8 | 188 | `ypp-p` | g04_021, g04_020 | no |
| 10 | [g04_023](g04_023/robot/preview.png) | 4 | 0.276 | -0.29 | 0.00 | 6 | 162 | `yp-p` | g03_018, g02_012 | yes |
| 11 | [g02_014](g02_014/robot/preview.png) | 2 | 0.274 | -0.07 | 0.00 | 6 | 163 | `yp-p` | g01_008 | no |
| 12 | [g03_016](g03_016/robot/preview.png) | 3 | 0.274 | 0.16 | 0.00 | 8 | 190 | `ypp-p` | g02_012 | no |
| 13 | [g02_015](g02_015/robot/preview.png) | 2 | 0.272 | 0.06 | 0.01 | 8 | 182 | `yp-yp` | g01_008 | yes |
| 14 | [g01_011](g01_011/robot/preview.png) | 1 | 0.271 | -0.09 | 0.01 | 6 | 154 | `yp-p` | g00_000 | no |
| 15 | [g01_008](g01_008/robot/preview.png) | 1 | 0.264 | 0.02 | 0.01 | 8 | 186 | `yp-yp` | g00_000 | yes |
| 16 | [g00_000](g00_000/robot/preview.png) | 0 | 0.251 | -0.03 | 0.01 | 8 | 180 | `yp-yp` | seed | no |
| 17 | [g00_006](g00_006/robot/preview.png) | 0 | 0.244 | -0.09 | 0.00 | 10 | 269 | `pyp-p-p` | random | no |
| 18 | [g03_019](g03_019/robot/preview.png) | 3 | 0.238 | -0.01 | 0.05 | 8 | 194 | `p-yp-p` | g02_012, g02_014 | no |
| 19 | [g01_010](g01_010/robot/preview.png) | 1 | 0.232 | -0.32 | 0.04 | 8 | 180 | `yp-yp` | g00_000 | yes |
| 20 | [g01_009](g01_009/robot/preview.png) | 1 | 0.222 | 0.02 | 0.00 | 10 | 268 | `pyp-p-p` | g00_006 | yes |
| 21 | [g00_002](g00_002/robot/preview.png) | 0 | 0.221 | 0.01 | 0.00 | 6 | 172 | `p-y-p` | random | no |
| 22 | [g03_017](g03_017/robot/preview.png) | 3 | 0.208 | -0.13 | 0.07 | 6 | 156 | `yyp` | g02_012 | no |
| 23 | [g00_001](g00_001/robot/preview.png) | 0 | 0.175 | -0.12 | 0.00 | 8 | 206 | `ppp-y` | random | no |
| 24 | [g00_007](g00_007/robot/preview.png) | 0 | 0.099 | -0.03 | 0.00 | 2 | 149 | `p` | random | no |
| 25 | [g02_013](g02_013/robot/preview.png) | 2 | 0.053 | -0.04 | 0.01 | 4 | 127 | `pp` | g01_011 | no |

Play the best robot:

```
python -m core.evolution.play --run_dir runs\evo_rough_002\g05_025
```
