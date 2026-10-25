#!/usr/bin/env bash
# Runs Python from the Isaac Lab environment with this repository on the import path (Linux / macOS; lab.cmd on Windows).
#   ./lab.sh -m core.evolution.train --genome prototypes/crawler_q4/genome.json --out runs/crawler_q4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$HERE/IsaacLab/.venv/bin/python" "$@"
