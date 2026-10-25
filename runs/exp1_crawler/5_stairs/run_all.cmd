@echo off
rem Stairs tests (2026-10-03): what limits stair climbing? Baseline = runs\skills\spec_stairs (SG90).
cd /d %~dp0..\..\..
set LAB=%CD%\lab.cmd
set COMMON=--task command --terrain rough --terrain_types stairs_climb,stairs_descend --pretrain_flat 500 --iterations 1500 --sensors imu --obs_history 8 --num_envs 2048
call "%LAB%" -m core.evolution.train --genome runs\stairs\genomes\evolved_mg90s.json --out runs\stairs\A1_mg90s %COMMON% > runs\stairs\A1_mg90s.log 2>&1
call "%LAB%" -m core.evolution.train --genome runs\stairs\genomes\evolved_mg92b.json --out runs\stairs\A2_mg92b %COMMON% > runs\stairs\A2_mg92b.log 2>&1
call "%LAB%" -m core.evolution.train --genome runs\evo_rough_002\g05_025\genome.json --out runs\stairs\B_foot_lift %COMMON% --foot_clearance > runs\stairs\B_foot_lift.log 2>&1
call "%LAB%" -m core.evolution.train --genome runs\evo_rough_002\g05_025\genome.json --out runs\stairs\C_height_scan %COMMON% --height_scan > runs\stairs\C_height_scan.log 2>&1
echo done > runs\stairs\DONE
