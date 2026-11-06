@echo off
rem Like lab.cmd, but with Isaac Lab 3.0.0 GA (IsaacLab-3.0.0\, a git worktree of IsaacLab\ with its own environment).
rem   lab3 tools/dog/benchmark.py runs/exp2_dog/17_recover_bc/bc_s2 course --physics newton --out x.jsonl
setlocal
set PYTHONIOENCODING=utf-8
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%~dp0IsaacLab-3.0.0\.venv\Scripts\python.exe" %*
