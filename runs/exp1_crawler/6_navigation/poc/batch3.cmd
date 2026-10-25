@echo off
rem POC finish, rerun of the camera goal-signal run (quoting fix).
cd /d %~dp0..\..\..\..
call lab.cmd -m core.evolution.train --genome runs\evo_rough_002\g05_025\genome.json --out runs\poc\nav_camera --task goal --terrain obstacles --iterations 1000 --num_envs 2048 --sensors imu --obs_history 4 --init_checkpoint runs\poc\nav_strict\model_final.pt --env_override collision_weight=6.0 --env_override goal_signal=camera > runs\poc\nav_camera.log 2>&1
echo done > runs\poc\DONE_batch3
