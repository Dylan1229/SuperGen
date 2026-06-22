#!/bin/bash
# =============================================================================
# GPU strong-scaling  ->  paper Fig.14 / GPU_scalability_comparison.pdf
# =============================================================================
# Sweeps TP_SIZE (number of GPUs for tile parallelism) and records latency.
#
# !!! This cluster has 2x H200, so only TP in {1,2} run here. The paper swept
#     {1,2,4,8} on 8x H100 -- run this on a >=8-GPU node with GPUS="1 2 4 8"
#     to reproduce the full curve.
#   Usage: RES=2k bash scripts/CogvideoI2V/scaling_gpus.sh
#          RES=4k GPUS="1 2 4 8" bash scripts/CogvideoI2V/scaling_gpus.sh   # big node
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
RES="${RES:-2k}"; PROMPT_SET="${PROMPT_SET:-single}"; CACHE="${CACHE:-0}"
GEN="$SUPERGEN_ROOT/scripts/CogvideoI2V/generate.sh"
have="$(gpu_count)"
GPUS="${GPUS:-1 2}"

for tp in $GPUS; do
  if [ "$tp" -gt "$have" ]; then
    echo "[skip] TP=$tp needs $tp GPUs but only $have visible (run on a larger node)"; continue
  fi
  echo "=== scaling: TP=$tp (RES=$RES) ==="
  RES="$RES" TP_SIZE="$tp" PROMPT_SET="$PROMPT_SET" CACHE="$CACHE" \
    TAG="scaling/${RES}/tp${tp}" bash "$GEN"
done
echo "=== scaling done (output/cogvideo/scaling/${RES}/) ==="
