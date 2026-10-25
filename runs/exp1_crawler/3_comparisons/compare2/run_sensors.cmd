@echo off
rem Rerun of the two sensor variants after fixing the observation-history bug (2026-10-03).
cd /d %~dp0..\..\..\..
set LAB=%CD%\lab.cmd
set COMMON=--task command --terrain rough --pretrain_flat 500 --iterations 1000 --num_envs 2048
call "%LAB%" -m core.evolution.train --genome runs\evo_rough_002\g05_025\genome.json --out runs\compare2\evolved_imu_h8 %COMMON% --sensors imu --obs_history 8 > runs\compare2\evolved_imu_h8.log 2>&1
call "%LAB%" -m core.evolution.train --genome runs\evo_rough_002\g05_025\genome.json --out runs\compare2\evolved_imuflow_h8 %COMMON% --sensors imu_flow --obs_history 8 > runs\compare2\evolved_imuflow_h8.log 2>&1
echo done > runs\compare2\DONE_SENSORS
