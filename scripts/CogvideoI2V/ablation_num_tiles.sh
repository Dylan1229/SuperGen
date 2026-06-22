#!/bin/bash
# =============================================================================
# Tile-count ablation  ->  paper Fig.17 / tiles_effect(_with_tile_sizes).pdf
# =============================================================================
# Sweeps the tile latent size (smaller tiles => more tiles => faster but lower
# quality). Each setting is one generate.sh run; read latencies from the
# per-sample run_log.log and VBench scores from the evaluation harness.
#   Usage: RES=2k PROMPT_SET=10 bash scripts/CogvideoI2V/ablation_num_tiles.sh
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
RES="${RES:-2k}"; PROMPT_SET="${PROMPT_SET:-10}"; TP_SIZE="${TP_SIZE:-2}"
GEN="$SUPERGEN_ROOT/scripts/CogvideoI2V/generate.sh"

# tile latent (W H) combos, matching the paper's tiles_effect figure
if [ "$RES" = "4k" ]; then
  widths=(160 120 160 96);  heights=(90 90 54 54)   # ~9,12,15,25 tiles @ 4K
else
  widths=(160 80 40);       heights=(90 90 90)      # 4,8,16 tiles @ 2K
fi

for j in "${!widths[@]}"; do
  W="${widths[$j]}"; H="${heights[$j]}"
  echo "=== num_tiles ablation: tile ${W}x${H} (RES=$RES) ==="
  RES="$RES" TP_SIZE="$TP_SIZE" PROMPT_SET="$PROMPT_SET" CACHE=0 \
    TILE_W="$W" TILE_H="$H" TAG="ablation_tiles/${RES}/${W}x${H}" \
    bash "$GEN"
done
echo "=== num_tiles ablation done (output/cogvideo/ablation_tiles/${RES}/) ==="
