@echo off
rem POC finish (2026-10-04), batch 2: hierarchical navigator (steers the frozen distilled walker), rough + pillars,
rem and a camera-like goal signal. All with collision_weight 6.
cd /d %~dp0..\..\..\..
set LAB=%CD%\lab.cmd
set GENOME=runs\evo_rough_002\g05_025\genome.json
set WALKER=runs\poc\student_v2\policy.pt
set COMMON=--task goal --num_envs 2048 --sensors imu --obs_history 4 --env_override collision_weight=6.0
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\poc\hier_flat --terrain obstacles --iterations 1000 --low_level_policy %WALKER% %COMMON% > runs\poc\hier_flat.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\poc\hier_rough --terrain obstacles_rough --iterations 1500 --low_level_policy %WALKER% %COMMON% > runs\poc\hier_rough.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\poc\e2e_rough --terrain obstacles_rough --iterations 1500 --init_checkpoint runs\poc\nav_strict\model_final.pt %COMMON% > runs\poc\e2e_rough.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\poc\nav_camera --terrain obstacles --iterations 1000 --init_checkpoint runs\poc\nav_strict\model_final.pt %COMMON% --env_override goal_signal="camera" > runs\poc\nav_camera.log 2>&1
echo done > runs\poc\DONE_batch2
