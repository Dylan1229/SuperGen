#!/bin/bash
# =============================================================================
# Cache threshold x scale grid  ->  paper Fig.16 / cache_ablation_heatmaps.pdf
# =============================================================================
# Sweeps cache_thresh x static_tile_cache_scale_factor with caching enabled.
# Read latency from run_log.log and VBench quality from the eval harness, then
# fill the heatmap cells in plots/plot_main.ipynb (cache_ablation_heatmaps).
#   Usage: RES=2k PROMPT_SET=10 bash scripts/CogvideoI2V/ablation_cache_grid.sh
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
RES="${RES:-2k}"; PROMPT_SET="${PROMPT_SET:-10}"; TP_SIZE="${TP_SIZE:-2}"
GEN="$SUPERGEN_ROOT/scripts/CogvideoI2V/generate.sh"

threshs=(${THRESHS:-0.05 0.07 0.09 0.11 0.13})
scales=(${SCALES:-0.1 0.2 0.3 0.4 0.5})

for t in "${threshs[@]}"; do
  for s in "${scales[@]}"; do
    echo "=== cache grid: thresh=$t scale=$s (RES=$RES) ==="
    RES="$RES" TP_SIZE="$TP_SIZE" PROMPT_SET="$PROMPT_SET" CACHE=1 \
      CACHE_THRESH="$t" CACHE_SCALE="$s" \
      TAG="ablation_cache/${RES}/thresh_${t}_scale_${s}" \
      bash "$GEN"
  done
done
echo "=== cache grid done (output/cogvideo/ablation_cache/${RES}/) ==="
