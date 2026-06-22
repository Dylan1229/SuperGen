#!/bin/bash
# =============================================================================
# HunyuanVideo cache threshold x scale grid  ->  paper Fig.16 cache ablation
# =============================================================================
# HunyuanVideo base cache threshold is 0.05 (vs 0.09 for CogVideoX).
#   Usage: RES=2k PROMPT_SET=10 bash scripts/HunyuanVideoI2V/ablation_cache_grid.sh
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
RES="${RES:-2k}"; PROMPT_SET="${PROMPT_SET:-10}"; TP_SIZE="${TP_SIZE:-2}"
GEN="$SUPERGEN_ROOT/scripts/HunyuanVideoI2V/generate.sh"

threshs=(${THRESHS:-0.03 0.05 0.07 0.09 0.11})
scales=(${SCALES:-0.1 0.2 0.3 0.4 0.5})

for t in "${threshs[@]}"; do
  for s in "${scales[@]}"; do
    echo "=== hunyuan cache grid: thresh=$t scale=$s (RES=$RES) ==="
    RES="$RES" TP_SIZE="$TP_SIZE" PROMPT_SET="$PROMPT_SET" CACHE=1 \
      CACHE_THRESH="$t" CACHE_SCALE="$s" \
      TAG="ablation_cache/${RES}/thresh_${t}_scale_${s}" \
      bash "$GEN"
  done
done
echo "=== hunyuan cache grid done (output/hunyuan/ablation_cache/${RES}/) ==="
