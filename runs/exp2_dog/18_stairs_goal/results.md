# Stairs as a goal task: the climbing limit (step 40)

Every dog walks a straight staircase of 5 risers up (floor -> landing) or down (landing -> floor), driven by
the navigation layer at 20 cm/s. clean = reached the goal without any fall. Falls on the stairs are soft (get
up and go on); falling down the stairs (> 1 step lower while fallen, or off an open side) is a hard fail.

## Limits: highest rise [cm] done clean by >= 90 % (every lower rise too)

| dog | up t25 | up t20 | up t15 | up t12 | up t9 | down t25 | down t20 | down t15 | down t12 | down t9 | sweet spot t15 up / down |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dog v1 (zero-shot) | 2.0 | 2.0 | 2.0 | 3.0 | 2.0 | 2.0 | 2.0 | 2.0 | 1.0 | 1.0 | 1 / 1 cm |
| original walker (zero-shot, cannot get up) | 3.0 | 2.0 | 4.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 1.0 | 1.0 | 2 / 1 cm |
| anchor_s1 | 3.0 | 2.0 | 3.0 | 3.0 | 2.0 | 2.0 | 2.0 | 1.0 | 1.0 | 1.0 | 3 / 1 cm |
| anchor_s2 | 3.0 | 3.0 | 3.0 | 3.0 | 2.0 | 2.0 | 2.0 | 2.0 | 1.0 | 1.0 | 2 / 2 cm |
| free_s1 | 5.0 | 4.0 | 3.0 | 3.0 | 2.0 | 3.0 | 2.0 | 2.0 | 1.0 | 1.0 | 3 / 1 cm |
| free_s2 | 5.0 | 4.0 | 4.0 | 3.0 | 2.0 | 3.0 | 3.0 | 2.0 | 1.0 | 1.0 | 3 / 2 cm |

Slope of each cell: atan(rise / tread), e.g. 6 cm on 15 cm = 22 deg, 8 cm on 12 cm = 34 deg, 10 cm on 9 cm = 48 deg.

## dog v1 (zero-shot)

up: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (8.9) | 100 (8.2) | 100 (8.6) | 100 (8.4) | 100 (7.7) |
| 2 cm | 100 (12.1) | 100 (10.9) | 98 (9.7) | 98 (9.1) | 100 (8.0) |
| 3 cm | 78 (14.1) | 82 (11.7) | 88 (9.9) | 95 (9.3) | 68 (8.8) |
| 4 cm | 72 (14.6) | 60 (12.8) | 75 (10.6) | 60 (10.9) | 22 (11.9) |
| 5 cm | 68 (19.7) | 62 (17.4) | 57 (12.6) | 5 (20.8) | 0 |
| 6 cm | 8 (22.6) | 0 (28.9) | 0 | 0 | 0 |
| 7 cm | 0 | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

up, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 99 % | 0 % | 0 % | 0 % | 5.0 | 100 / 98 % |
| 3 cm | 82 % | 0 % | 0 % | 0 % | 4.6 | 97 / 67 % |
| 4 cm | 59 % | 8 % | 0 % | 0 % | 3.9 | 84 / 32 % |
| 5 cm | 42 % | 20 % | 2 % | 0 % | 2.9 | 57 / 20 % |
| 6 cm | 2 % | 42 % | 6 % | 0 % | 1.5 | 0 / 3 % |
| 7 cm | 0 % | 57 % | 2 % | 2 % | 0.8 | 0 / 0 % |
| 8 cm | 0 % | 68 % | 2 % | 2 % | 0.5 | 0 / 0 % |
| 9 cm | 0 % | 76 % | 0 % | 5 % | 0.3 | 0 / 0 % |
| 10 cm | 0 % | 76 % | 0 % | 2 % | 0.3 | 0 / 0 % |

down: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (7.0) | 100 (6.2) | 100 (5.5) | 100 (5.0) | 100 (4.7) |
| 2 cm | 100 (7.7) | 100 (7.1) | 95 (7.0) | 38 (8.2) | 5 (13.4) |
| 3 cm | 88 (9.8) | 38 (10.7) | 12 (17.4) | 2 (28.8) | 0 |
| 4 cm | 28 (22.4) | 12 (19.4) | 0 | 0 | 0 |
| 5 cm | 10 (21.1) | 0 | 0 | 0 | 0 |
| 6 cm | 0 | 0 | 0 | 0 | 0 |
| 7 cm | 2 (28.3) | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

down, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 68 % | 0 % | 0 % | 0 % | 3.6 | 61 / 74 % |
| 3 cm | 28 % | 0 % | 0 % | 0 % | 1.7 | 26 / 30 % |
| 4 cm | 8 % | 0 % | 0 % | 0 % | 0.6 | 12 / 4 % |
| 5 cm | 2 % | 2 % | 1 % | 1 % | 0.3 | 4 / 0 % |
| 6 cm | 0 % | 2 % | 1 % | 0 % | 0.1 | 0 / 0 % |
| 7 cm | 0 % | 2 % | 2 % | 0 % | 0.1 | 1 / 0 % |
| 8 cm | 0 % | 3 % | 2 % | 1 % | 0.0 | 0 / 0 % |
| 9 cm | 0 % | 2 % | 2 % | 0 % | 0.1 | 0 / 0 % |
| 10 cm | 0 % | 1 % | 1 % | 0 % | 0.0 | 0 / 0 % |

## original walker (zero-shot, cannot get up)

up: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (10.0) | 100 (9.6) | 100 (9.3) | 100 (8.7) | 100 (8.0) |
| 2 cm | 100 (12.1) | 100 (11.3) | 100 (9.7) | 100 (9.1) | 98 (8.3) |
| 3 cm | 98 (13.8) | 88 (12.5) | 92 (9.7) | 90 (10.0) | 75 (9.3) |
| 4 cm | 80 (15.5) | 75 (13.6) | 92 (10.6) | 65 (11.1) | 10 (13.7) |
| 5 cm | 75 (19.3) | 50 (17.6) | 40 (13.6) | 0 (16.2) | 0 |
| 6 cm | 5 (25.1) | 2 (24.9) | 0 (22.6) | 0 | 0 |
| 7 cm | 5 (25.1) | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

up, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 99 % |
| 3 cm | 90 % | 2 % | 0 % | 0 % | 4.8 | 97 / 80 % |
| 4 cm | 74 % | 15 % | 2 % | 1 % | 4.3 | 74 / 55 % |
| 5 cm | 46 % | 36 % | 4 % | 5 % | 3.2 | 40 / 26 % |
| 6 cm | 11 % | 62 % | 8 % | 9 % | 2.0 | 2 / 1 % |
| 7 cm | 1 % | 64 % | 5 % | 16 % | 1.1 | 1 / 1 % |
| 8 cm | 0 % | 71 % | 2 % | 16 % | 0.6 | 0 / 0 % |
| 9 cm | 0 % | 77 % | 0 % | 14 % | 0.4 | 0 / 0 % |
| 10 cm | 0 % | 82 % | 0 % | 14 % | 0.3 | 0 / 0 % |

down: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (7.5) | 100 (6.8) | 100 (6.0) | 98 (5.6) | 100 (5.4) |
| 2 cm | 100 (8.8) | 98 (7.9) | 92 (7.9) | 38 (16.8) | 5 (13.7) |
| 3 cm | 80 (11.3) | 32 (15.4) | 2 (23.0) | 0 | 0 |
| 4 cm | 10 (21.0) | 0 | 0 | 0 | 0 |
| 5 cm | 0 | 0 | 0 | 0 | 0 |
| 6 cm | 0 | 0 | 0 | 0 | 0 |
| 7 cm | 0 | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

down, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 99 % |
| 2 cm | 66 % | 0 % | 0 % | 0 % | 3.6 | 71 / 62 % |
| 3 cm | 23 % | 3 % | 0 % | 3 % | 1.5 | 26 / 20 % |
| 4 cm | 2 % | 4 % | 0 % | 2 % | 0.4 | 3 / 1 % |
| 5 cm | 0 % | 2 % | 0 % | 2 % | 0.1 | 0 / 0 % |
| 6 cm | 0 % | 4 % | 0 % | 1 % | 0.0 | 0 / 0 % |
| 7 cm | 0 % | 6 % | 2 % | 2 % | 0.1 | 0 / 0 % |
| 8 cm | 0 % | 4 % | 2 % | 2 % | 0.1 | 0 / 0 % |
| 9 cm | 0 % | 4 % | 0 % | 1 % | 0.0 | 0 / 0 % |
| 10 cm | 0 % | 3 % | 2 % | 2 % | 0.0 | 0 / 0 % |

## anchor_s1

up: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (8.8) | 100 (8.1) | 100 (8.6) | 100 (8.5) | 100 (7.7) |
| 2 cm | 100 (11.6) | 100 (10.2) | 100 (9.1) | 100 (8.6) | 98 (7.8) |
| 3 cm | 98 (12.2) | 82 (11.1) | 100 (9.4) | 95 (9.0) | 80 (8.6) |
| 4 cm | 88 (13.4) | 80 (11.8) | 85 (10.5) | 82 (10.7) | 0 (12.5) |
| 5 cm | 88 (17.8) | 60 (14.4) | 52 (12.2) | 0 | 0 |
| 6 cm | 28 (25.6) | 22 (27.4) | 0 | 0 | 0 |
| 7 cm | 0 | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

up, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 99 % |
| 3 cm | 91 % | 0 % | 0 % | 0 % | 4.8 | 100 / 82 % |
| 4 cm | 69 % | 13 % | 2 % | 0 % | 4.1 | 74 / 60 % |
| 5 cm | 42 % | 26 % | 2 % | 2 % | 2.7 | 54 / 26 % |
| 6 cm | 10 % | 43 % | 6 % | 4 % | 1.7 | 14 / 6 % |
| 7 cm | 0 % | 48 % | 2 % | 8 % | 0.8 | 0 / 0 % |
| 8 cm | 0 % | 64 % | 0 % | 12 % | 0.3 | 0 / 0 % |
| 9 cm | 0 % | 77 % | 0 % | 18 % | 0.3 | 0 / 0 % |
| 10 cm | 0 % | 76 % | 0 % | 11 % | 0.3 | 0 / 0 % |

down: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (6.8) | 100 (6.1) | 100 (5.4) | 100 (4.9) | 100 (4.5) |
| 2 cm | 100 (7.6) | 100 (7.0) | 90 (6.7) | 45 (7.6) | 8 (25.4) |
| 3 cm | 90 (9.5) | 48 (12.0) | 0 | 0 | 0 |
| 4 cm | 32 (19.8) | 0 | 0 | 0 | 0 |
| 5 cm | 0 | 0 | 0 | 0 | 0 |
| 6 cm | 0 | 0 | 0 | 0 | 0 |
| 7 cm | 0 | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

down, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 68 % | 0 % | 0 % | 0 % | 3.7 | 70 / 67 % |
| 3 cm | 28 % | 0 % | 0 % | 0 % | 1.6 | 31 / 24 % |
| 4 cm | 6 % | 1 % | 0 % | 0 % | 0.5 | 11 / 2 % |
| 5 cm | 0 % | 2 % | 0 % | 0 % | 0.1 | 0 / 0 % |
| 6 cm | 0 % | 1 % | 0 % | 1 % | 0.0 | 0 / 0 % |
| 7 cm | 0 % | 2 % | 0 % | 0 % | 0.0 | 0 / 0 % |
| 8 cm | 0 % | 1 % | 0 % | 0 % | 0.0 | 0 / 0 % |
| 9 cm | 0 % | 0 % | 0 % | 0 % | 0.0 | 0 / 0 % |
| 10 cm | 0 % | 1 % | 0 % | 0 % | 0.0 | 0 / 0 % |

## anchor_s2

up: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (8.9) | 100 (8.3) | 100 (8.8) | 100 (8.5) | 100 (7.7) |
| 2 cm | 100 (11.5) | 100 (10.5) | 100 (9.1) | 98 (8.4) | 100 (7.8) |
| 3 cm | 95 (12.1) | 92 (11.3) | 92 (9.6) | 92 (9.3) | 80 (8.5) |
| 4 cm | 85 (14.1) | 88 (12.1) | 85 (10.5) | 78 (11.3) | 2 (13.2) |
| 5 cm | 70 (17.9) | 57 (15.1) | 48 (12.7) | 2 (14.2) | 0 |
| 6 cm | 22 (26.6) | 18 (25.0) | 0 | 0 | 0 |
| 7 cm | 2 (24.5) | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

up, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 99 % |
| 3 cm | 90 % | 0 % | 0 % | 0 % | 4.9 | 100 / 81 % |
| 4 cm | 69 % | 10 % | 2 % | 1 % | 4.0 | 79 / 56 % |
| 5 cm | 38 % | 24 % | 3 % | 0 % | 2.7 | 54 / 17 % |
| 6 cm | 11 % | 39 % | 2 % | 2 % | 1.8 | 14 / 2 % |
| 7 cm | 0 % | 44 % | 1 % | 2 % | 1.0 | 0 / 1 % |
| 8 cm | 0 % | 52 % | 2 % | 4 % | 0.5 | 0 / 0 % |
| 9 cm | 0 % | 54 % | 0 % | 2 % | 0.4 | 0 / 0 % |
| 10 cm | 0 % | 60 % | 0 % | 2 % | 0.3 | 0 / 0 % |

down: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (6.9) | 100 (6.2) | 100 (5.5) | 100 (5.0) | 100 (4.7) |
| 2 cm | 100 (7.7) | 100 (7.1) | 100 (6.9) | 78 (8.8) | 5 (21.2) |
| 3 cm | 90 (9.6) | 35 (16.8) | 8 (22.9) | 0 | 0 |
| 4 cm | 35 (20.9) | 2 (23.5) | 0 | 0 | 0 |
| 5 cm | 2 (20.3) | 0 | 0 | 0 | 0 |
| 6 cm | 0 | 0 | 0 | 0 | 0 |
| 7 cm | 0 | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

down, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 76 % | 0 % | 0 % | 0 % | 4.1 | 74 / 79 % |
| 3 cm | 26 % | 0 % | 0 % | 0 % | 1.6 | 24 / 29 % |
| 4 cm | 8 % | 2 % | 1 % | 0 % | 0.6 | 9 / 6 % |
| 5 cm | 0 % | 3 % | 2 % | 0 % | 0.2 | 0 / 1 % |
| 6 cm | 0 % | 4 % | 0 % | 3 % | 0.0 | 0 / 0 % |
| 7 cm | 0 % | 3 % | 1 % | 0 % | 0.1 | 0 / 0 % |
| 8 cm | 0 % | 3 % | 2 % | 0 % | 0.1 | 0 / 0 % |
| 9 cm | 0 % | 2 % | 0 % | 0 % | 0.0 | 0 / 0 % |
| 10 cm | 0 % | 3 % | 2 % | 0 % | 0.1 | 0 / 0 % |

## free_s1

up: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (8.5) | 100 (7.9) | 100 (8.5) | 100 (8.2) | 100 (7.4) |
| 2 cm | 100 (11.3) | 100 (10.1) | 100 (9.1) | 100 (8.6) | 100 (7.5) |
| 3 cm | 100 (11.9) | 100 (10.8) | 100 (9.3) | 98 (8.9) | 80 (8.8) |
| 4 cm | 95 (13.4) | 98 (11.9) | 85 (10.4) | 75 (10.6) | 2 (10.0) |
| 5 cm | 95 (15.8) | 88 (14.8) | 70 (13.1) | 0 (16.0) | 0 |
| 6 cm | 38 (26.9) | 38 (22.6) | 0 | 0 | 0 |
| 7 cm | 5 (26.5) | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

up, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 3 cm | 96 % | 0 % | 0 % | 0 % | 4.9 | 100 / 91 % |
| 4 cm | 72 % | 12 % | 1 % | 1 % | 4.0 | 80 / 62 % |
| 5 cm | 51 % | 23 % | 2 % | 2 % | 3.1 | 59 / 42 % |
| 6 cm | 17 % | 45 % | 1 % | 3 % | 2.0 | 15 / 15 % |
| 7 cm | 1 % | 54 % | 3 % | 8 % | 0.7 | 1 / 1 % |
| 8 cm | 0 % | 76 % | 0 % | 13 % | 0.3 | 0 / 0 % |
| 9 cm | 0 % | 72 % | 0 % | 10 % | 0.1 | 0 / 0 % |
| 10 cm | 0 % | 78 % | 0 % | 10 % | 0.1 | 0 / 0 % |

down: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (6.5) | 100 (6.0) | 100 (5.3) | 100 (4.7) | 100 (4.6) |
| 2 cm | 100 (7.4) | 100 (6.6) | 98 (6.5) | 80 (7.6) | 12 (16.7) |
| 3 cm | 92 (8.4) | 82 (9.3) | 30 (17.7) | 22 (18.1) | 10 (22.9) |
| 4 cm | 65 (15.0) | 30 (15.6) | 12 (21.5) | 0 | 0 |
| 5 cm | 52 (16.6) | 22 (19.9) | 0 | 0 | 0 |
| 6 cm | 22 (23.2) | 0 | 0 | 0 | 0 |
| 7 cm | 0 (18.7) | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

down, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 78 % | 0 % | 0 % | 0 % | 4.1 | 78 / 78 % |
| 3 cm | 48 % | 0 % | 0 % | 0 % | 2.7 | 49 / 46 % |
| 4 cm | 22 % | 2 % | 2 % | 0 % | 1.4 | 25 / 18 % |
| 5 cm | 15 % | 3 % | 2 % | 0 % | 1.0 | 18 / 12 % |
| 6 cm | 5 % | 6 % | 2 % | 1 % | 0.6 | 8 / 1 % |
| 7 cm | 0 % | 8 % | 3 % | 0 % | 0.2 | 0 / 0 % |
| 8 cm | 0 % | 3 % | 2 % | 0 % | 0.1 | 0 / 0 % |
| 9 cm | 0 % | 0 % | 0 % | 0 % | 0.0 | 0 / 0 % |
| 10 cm | 0 % | 2 % | 1 % | 0 % | 0.0 | 0 / 0 % |

## free_s2

up: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (8.4) | 100 (7.6) | 100 (8.5) | 100 (8.2) | 100 (7.6) |
| 2 cm | 100 (11.7) | 100 (11.0) | 100 (9.4) | 100 (8.9) | 100 (7.7) |
| 3 cm | 100 (12.0) | 100 (11.0) | 100 (9.5) | 100 (9.1) | 85 (8.4) |
| 4 cm | 98 (13.1) | 100 (11.4) | 98 (10.0) | 80 (10.3) | 5 (12.9) |
| 5 cm | 98 (16.0) | 78 (12.9) | 68 (12.2) | 0 | 0 |
| 6 cm | 60 (24.6) | 20 (23.4) | 0 | 0 | 0 |
| 7 cm | 8 (25.9) | 0 | 0 | 0 | 0 |
| 8 cm | 0 | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

up, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 3 cm | 97 % | 0 % | 0 % | 0 % | 4.9 | 100 / 94 % |
| 4 cm | 77 % | 12 % | 1 % | 0 % | 4.2 | 80 / 72 % |
| 5 cm | 50 % | 27 % | 0 % | 4 % | 3.1 | 57 / 40 % |
| 6 cm | 16 % | 45 % | 4 % | 6 % | 1.8 | 17 / 15 % |
| 7 cm | 2 % | 67 % | 0 % | 4 % | 0.6 | 2 / 1 % |
| 8 cm | 0 % | 76 % | 0 % | 2 % | 0.2 | 0 / 0 % |
| 9 cm | 0 % | 90 % | 0 % | 16 % | 0.1 | 0 / 0 % |
| 10 cm | 0 % | 87 % | 0 % | 6 % | 0.1 | 0 / 0 % |

down: clean % per rise (rows) and tread (columns); in brackets median time to the goal [s]

| rise | 25 cm | 20 cm | 15 cm | 12 cm | 9 cm |
|---|---|---|---|---|---|
| 1 cm | 100 (6.3) | 100 (5.6) | 100 (5.1) | 100 (4.6) | 100 (4.3) |
| 2 cm | 100 (7.1) | 98 (6.7) | 100 (6.0) | 82 (6.9) | 38 (18.6) |
| 3 cm | 98 (8.6) | 92 (10.0) | 70 (13.7) | 52 (13.7) | 18 (27.6) |
| 4 cm | 75 (14.9) | 52 (14.6) | 40 (15.9) | 8 (19.0) | 0 |
| 5 cm | 75 (17.4) | 35 (16.6) | 0 | 0 | 0 |
| 6 cm | 45 (22.5) | 2 (19.7) | 0 | 0 | 0 |
| 7 cm | 5 (18.6) | 0 (23.1) | 0 | 0 | 0 |
| 8 cm | 0 (23.8) | 0 | 0 | 0 | 0 |
| 9 cm | 0 | 0 | 0 | 0 | 0 |
| 10 cm | 0 | 0 | 0 | 0 | 0 |

down, per rise over all treads: reached / any fall / fell down the stairs / stayed down / steps made good (of 5); open sides vs walls: clean

| rise | reached | any fall | down the stairs | stayed down | steps | clean open / walls |
|---|---|---|---|---|---|---|
| 1 cm | 100 % | 0 % | 0 % | 0 % | 5.0 | 100 / 100 % |
| 2 cm | 85 % | 2 % | 0 % | 0 % | 4.3 | 84 / 83 % |
| 3 cm | 66 % | 0 % | 0 % | 0 % | 3.5 | 61 / 71 % |
| 4 cm | 38 % | 2 % | 0 % | 0 % | 2.3 | 33 / 37 % |
| 5 cm | 22 % | 4 % | 1 % | 0 % | 1.5 | 28 / 16 % |
| 6 cm | 10 % | 5 % | 2 % | 0 % | 0.9 | 10 / 9 % |
| 7 cm | 3 % | 14 % | 4 % | 0 % | 0.6 | 1 / 1 % |
| 8 cm | 0 % | 10 % | 3 % | 2 % | 0.2 | 0 / 0 % |
| 9 cm | 0 % | 10 % | 4 % | 3 % | 0.1 | 0 / 0 % |
| 10 cm | 0 % | 8 % | 2 % | 2 % | 0.1 | 0 / 0 % |

## Nothing forgotten? Walking (falls, walk score; steps climbed on the hardest pyramid stairs)

| dog | forward | course | stairs straight | steps up (rows 6-9) | power forward |
|---|---|---|---|---|---|
| dog v1 (zero-shot) | 0.4 % (0.45) | 1.0 % (0.43) | 1.2 % (0.44) | 3.9 | 12.4 W |
| anchor_s1 | 0.8 % (0.44) | 1.2 % (0.42) | 2.0 % (0.44) | 3.7 | 12.7 W |
| anchor_s2 | 1.0 % (0.44) | 1.0 % (0.43) | 1.4 % (0.43) | 3.1 | 12.6 W |
| free_s1 | 1.4 % (0.45) | 0.9 % (0.44) | 1.2 % (0.45) | 3.8 | 12.7 W |
| free_s2 | 1.0 % (0.46) | 0.7 % (0.43) | 1.8 % (0.45) | 3.6 | 12.9 W |

## One shove while walking at 20 cm/s: stays up / falls and gets up / stays down

| dog | 0.3 m/s | 0.5 m/s | 0.7 m/s | 0.9 m/s | 1.1 m/s | 1.3 m/s | 1.5 m/s |
|---|---|---|---|---|---|---|---|
| bc_s2 | 100 / 0 / 0 % | 86 / 14 / 0 % | 65 / 35 / 0 % | 53 / 47 / 0 % | 29 / 70 / 1 % | 3 / 95 / 2 % | 1 / 97 / 2 % |
| anchor_s1 | 99 / 1 / 0 % | 76 / 24 / 0 % | 65 / 35 / 0 % | 56 / 44 / 0 % | 39 / 61 / 0 % | 14 / 81 / 6 % | 0 / 92 / 8 % |
| anchor_s2 | 100 / 0 / 0 % | 92 / 8 / 0 % | 68 / 32 / 0 % | 64 / 36 / 0 % | 30 / 67 / 3 % | 3 / 90 / 7 % | 0 / 92 / 8 % |
| free_s1 | 99 / 1 / 0 % | 86 / 14 / 0 % | 68 / 32 / 0 % | 61 / 39 / 0 % | 34 / 64 / 2 % | 7 / 91 / 2 % | 0 / 95 / 5 % |
| free_s2 | 99 / 1 / 0 % | 77 / 23 / 0 % | 69 / 31 / 0 % | 64 / 36 / 0 % | 39 / 61 / 0 % | 20 / 80 / 0 % | 2 / 97 / 1 % |

## Getting up: share standing within 8 s (median time)

| dog | standing | dropped upright | on its side | on its back |
|---|---|---|---|---|
| bc_s2 | 100 % (1.46 s) | 100 % (1.16 s) | 100 % (1.54 s) | 100 % (1.6 s) |
| anchor_s1 | 100 % (1.86 s) | 99 % (1.08 s) | 100 % (1.66 s) | 100 % (1.82 s) |
| anchor_s2 | 100 % (2.4 s) | 100 % (1.22 s) | 100 % (1.54 s) | 100 % (1.64 s) |
| free_s1 | 100 % (2.44 s) | 100 % (1.12 s) | 100 % (1.58 s) | 100 % (1.66 s) |
| free_s2 | 100 % (1.48 s) | 100 % (1.34 s) | 100 % (1.56 s) | 100 % (1.62 s) |

Best dog on the stairs: free_s2 (clean share 0.378; all: {'anchor_s1': 0.305, 'anchor_s2': 0.30625, 'free_s1': 0.34975, 'free_s2': 0.37799999999999995}; dog v1 0.292)

Videos (dog v1 / best anchor / best free, tread 15 cm, open sides): `videos/down_r2_t15.mp4`, `videos/up_r3_t15.mp4`, `videos/up_r4_t15.mp4`, `videos/up_r6_t15.mp4`
