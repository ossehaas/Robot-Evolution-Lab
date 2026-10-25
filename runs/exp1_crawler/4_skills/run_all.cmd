@echo off
rem Roadmap M2c: specialists -> distilled student -> RL polish. Evolved body g05_025, realistic IMU + 8-frame memory.
rem Started 2026-10-03.
cd /d %~dp0..\..\..
set LAB=%CD%\lab.cmd
set GENOME=runs\evo_rough_002\g05_025\genome.json
set SENSES=--sensors imu --obs_history 8 --num_envs 2048
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\skills\spec_stairs --task command --terrain rough --terrain_types stairs_climb,stairs_descend --pretrain_flat 500 --iterations 1500 %SENSES% > runs\skills\spec_stairs.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\skills\spec_blocks --task command --terrain rough --terrain_types blocks --pretrain_flat 500 --iterations 1500 %SENSES% > runs\skills\spec_blocks.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\skills\generalist --task command --terrain rough --iterations 500 --init_checkpoint runs\compare2\evolved_imu_h8\model_final.pt %SENSES% > runs\skills\generalist.log 2>&1
call "%LAB%" -m core.evolution.distill --genome %GENOME% --out runs\skills\student --teacher stairs=runs\skills\spec_stairs\model_final.pt:stairs_climb,stairs_descend --teacher blocks=runs\skills\spec_blocks\model_final.pt:blocks --fallback runs\skills\generalist\model_final.pt --iterations 300 %SENSES% > runs\skills\student.log 2>&1
call "%LAB%" -m core.evolution.train --genome %GENOME% --out runs\skills\student_polished --task command --terrain rough --iterations 500 --init_checkpoint runs\skills\student\model_final.pt %SENSES% > runs\skills\student_polished.log 2>&1
echo done > runs\skills\DONE
