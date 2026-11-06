#!/usr/bin/env bash
# Like lab.sh, but with Isaac Lab 3.0.0 GA (IsaacLab-3.0.0/, a git worktree of IsaacLab/ with its own environment).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$HERE/IsaacLab-3.0.0/.venv/bin/python" "$@"
