@echo off
rem M3 first run (2026-10-04): reach goals through a pillar field with an 8x8 distance sensor. Evolved body, SG90.
cd /d %~dp0..\..\..\..
call lab.cmd -m core.evolution.train --genome runs\evo_rough_002\g05_025\genome.json --out runs\nav\goal_v1 --task goal --terrain obstacles --iterations 2000 --num_envs 2048 --sensors imu --obs_history 4 > runs\nav\goal_v1.log 2>&1
echo done > runs\nav\DONE_goal_v1
