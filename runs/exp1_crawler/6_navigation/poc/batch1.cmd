@echo off
rem POC finish (2026-10-04), batch 1: combining skills (fixed polish) + cleaner obstacle avoidance.
cd /d %~dp0..\..\..\..
set LAB=%CD%\lab.cmd
set GENOME=runs\evo_rough_002\g05_025\genome.json
call "%LAB%" -m core.evolution.distill --genome %GENOME% --out runs\poc\student_v2 --teacher stairs=runs\skills\spec_stairs\model_final.pt:stairs_climb,stairs_descend --teacher blocks=runs\skills\spec_blocks\model_final.pt:blocks --fallback runs\skills\generalist\model_final.pt --iterations 300 --sensors imu --obs_history 8 > runs\poc\student_v2.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\poc\student_v2_polished --task command --terrain rough --iterations 500 --critic_warmup 100 --init_checkpoint runs\poc\student_v2\model_final.pt --sensors imu --obs_history 8 > runs\poc\student_v2_polished.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\poc\nav_control --task goal --terrain obstacles --iterations 1500 --init_checkpoint runs\nav\goal_v1\model_final.pt --sensors imu --obs_history 4 > runs\poc\nav_control.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\poc\nav_strict --task goal --terrain obstacles --iterations 1500 --init_checkpoint runs\nav\goal_v1\model_final.pt --sensors imu --obs_history 4 --env_override collision_weight=6.0 > runs\poc\nav_strict.log 2>&1
echo done > runs\poc\DONE_batch1
