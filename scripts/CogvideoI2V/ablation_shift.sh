#!/bin/bash
# =============================================================================
# Tile-shift schedule ablation  ->  paper Table 8 (tile shifting)
# =============================================================================
# Compares shifting tiles every step / every k steps / never. "every1" (shift
# at every refinement step) is the default used everywhere else.
#   Usage: RES=2k PROMPT_SET=10 bash scripts/CogvideoI2V/ablation_shift.sh
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
RES="${RES:-2k}"; PROMPT_SET="${PROMPT_SET:-10}"; TP_SIZE="${TP_SIZE:-2}"
GEN="$SUPERGEN_ROOT/scripts/CogvideoI2V/generate.sh"

names=(every1 every3 every5 every9 every15 none)
scheds=("$(seq -s, 0 1 44)" "$(seq -s, 0 3 44)" "$(seq -s, 0 5 44)" \
        "$(seq -s, 0 9 44)" "$(seq -s, 0 15 44)" "0")

for j in "${!names[@]}"; do
  echo "=== shift ablation: ${names[$j]} (RES=$RES) ==="
  RES="$RES" TP_SIZE="$TP_SIZE" PROMPT_SET="$PROMPT_SET" CACHE=0 \
    SHIFT="${scheds[$j]}" TAG="ablation_shift/${RES}/${names[$j]}" \
    bash "$GEN"
done
echo "=== shift ablation done (output/cogvideo/ablation_shift/${RES}/) ==="
