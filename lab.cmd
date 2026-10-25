@echo off
rem Runs Python from the Isaac Lab environment with this repository on the import path.
rem   lab -m core.evolution.train --genome prototypes\crawler_q4\genome.json --out runs\crawler_q4
setlocal
set PYTHONIOENCODING=utf-8
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
"%~dp0IsaacLab\.venv\Scripts\python.exe" %*
