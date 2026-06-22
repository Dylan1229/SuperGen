#!/usr/bin/env bash
# No-reference VBench scoring driver wrapper for SuperGen.
#
# Scores one or more directories of mp4 videos with VBench. The directories are
# passed through to run_vbench_eval_for_dir.py either as CLI args to this script
# or via the VBENCH_DIRS env var.
#
# Configuration (override via environment variables):
#   PY                    python interpreter (default: the supergen-vbench venv)
#   CUDA_VISIBLE_DEVICES  GPU to use (default: 0)
#   VBENCH_REPO           VBench source checkout (default set in the .py driver)
#   VBENCH_CACHE_DIR      where VBench downloads/loads backbones
#                         (default set in the .py driver: /mnt/data/.../.cache/vbench)
#   VBENCH_DIRS           os.pathsep-separated dirs to score (if not passed as args)
#
# Usage:
#   bash run_vbench_eval.sh /path/to/mp4_dir [more_dirs ...]
#   VBENCH_DIRS=/dir1:/dir2 bash run_vbench_eval.sh
#   CUDA_VISIBLE_DEVICES=1 bash run_vbench_eval.sh /path/to/mp4_dir
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1

# Default python = the dedicated VBench venv.
PY="${PY:-/mnt/data/fanjiang/venvs/supergen-vbench/bin/python}"

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using python=$PY"
echo "Starting evaluation at $(date)"

"${PY}" "${SCRIPT_DIR}/run_vbench_eval_for_dir.py" "$@"
