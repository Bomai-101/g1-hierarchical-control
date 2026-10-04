#!/usr/bin/env bash
# Existing local Isaac installation; process-local paths, no dependency changes.
set -euo pipefail
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_lab="/home/omai/robotics/reference_projects/IsaacLab"
task_env="/home/omai/miniforge3/envs/g1-walk-isaac"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$task_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$task_lab/source/isaaclab:$task_lab/source/isaaclab_assets:$task_lab/source/isaaclab_tasks:$task_lab/source/isaaclab_rl:$task_lab/source/isaaclab_mimic${PYTHONPATH:+:$PYTHONPATH}"
cd "$task_repo"
exec "$task_env/bin/python" -u "$@"
