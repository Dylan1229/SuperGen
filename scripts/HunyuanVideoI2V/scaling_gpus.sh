#!/bin/bash
# =============================================================================
# HunyuanVideo GPU strong-scaling  ->  paper Fig.14 / GPU_scalability_comparison.pdf
# =============================================================================
# !!! 2x H200 here => only TP in {1,2}. Paper used {1,2,4,8} on 8x H100.
#     On a >=8-GPU node:  RES=4k GPUS="1 2 4 8" bash scripts/HunyuanVideoI2V/scaling_gpus.sh
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
RES="${RES:-2k}"; PROMPT_SET="${PROMPT_SET:-single}"; CACHE="${CACHE:-0}"
GEN="$SUPERGEN_ROOT/scripts/HunyuanVideoI2V/generate.sh"
have="$(gpu_count)"
GPUS="${GPUS:-1 2}"

for tp in $GPUS; do
  if [ "$tp" -gt "$have" ]; then
    echo "[skip] TP=$tp needs $tp GPUs but only $have visible (run on a larger node)"; continue
  fi
  echo "=== hunyuan scaling: TP=$tp (RES=$RES) ==="
  RES="$RES" TP_SIZE="$tp" PROMPT_SET="$PROMPT_SET" CACHE="$CACHE" \
    TAG="scaling/${RES}/tp${tp}" bash "$GEN"
done
echo "=== hunyuan scaling done (output/hunyuan/scaling/${RES}/) ==="
