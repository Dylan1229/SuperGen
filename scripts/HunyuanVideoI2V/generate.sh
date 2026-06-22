#!/bin/bash
# =============================================================================
# SuperGen — HunyuanVideo I2V generation / timing driver (parametrized)
# =============================================================================
# Mirror of CogvideoI2V/generate.sh for the HunyuanVideo backend. Runs from
# HunyuanVideoI2V/ so MODEL_BASE=./ckpts resolves to the downloaded weights.
# Per-sample stdout (with the stage timers the plot notebooks parse) -> run_log.log.
#
# Examples:
#   RES=2k TP_SIZE=1 PROMPT_SET=single CACHE=0 bash scripts/HunyuanVideoI2V/generate.sh
#   RES=4k TP_SIZE=2 PROMPT_SET=20 CACHE=1 bash scripts/HunyuanVideoI2V/generate.sh
#
# Env knobs (defaults in []):
#   RES[2k] TP_SIZE[2] PROMPT_SET[single] CACHE[0] REDISTRIBUTE[0]
#   STEPS[50] VIDEO_LENGTH[41] LOOP_STEP[16] UPSCALE_RES_STEPS[45]
#   CACHE_THRESH[0.05] CACHE_SCALE[1.0]  SHIFT  TAG  MASTER_PORT[33334]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"

RES="${RES:-2k}"; TP_SIZE="${TP_SIZE:-2}"; PROMPT_SET="${PROMPT_SET:-single}"
CACHE="${CACHE:-0}"; export ENABLE_REDISTRIBUTE="${REDISTRIBUTE:-0}"
STEPS="${STEPS:-50}"; VIDEO_LENGTH="${VIDEO_LENGTH:-41}"; LOOP_STEP="${LOOP_STEP:-16}"
UPSCALE_RES_STEPS="${UPSCALE_RES_STEPS:-45}"
CACHE_THRESH="${CACHE_THRESH:-0.05}"; CACHE_SCALE="${CACHE_SCALE:-1.0}"
MASTER_PORT="${MASTER_PORT:-33334}"

set_resolution "$RES"; check_tp "$TP_SIZE"
SHIFT="${SHIFT:-$SHIFT_ALL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TP_SIZE-1)))}"

# shellcheck disable=SC1091
source "$HUN_VENV/bin/activate"
cd "$SUPERGEN_ROOT/HunyuanVideoI2V"      # MODEL_BASE=./ckpts

cache_tag="nocache"; [ "$CACHE" = "1" ] && cache_tag="cache"
TAG="${TAG:-${RES}_tp${TP_SIZE}_${cache_tag}_${PROMPT_SET}}"
exp_root="$SUPERGEN_ROOT/output/hunyuan/${TAG}"
low_root="$SUPERGEN_ROOT/output/hunyuan/_stage1/${RES}"
mkdir -p "$exp_root" "$low_root"
echo "[generate] hunyuan RES=$RES TP=$TP_SIZE set=$PROMPT_SET cache=$CACHE redistribute=$ENABLE_REDISTRIBUTE -> $exp_root"

while IFS=$'\t' read -r image_name prompt; do
  [ -z "$image_name" ] && continue
  img="$IMG_DIR/$image_name"
  [ -f "$img" ] || { echo "[skip] missing image: $img"; continue; }

  out_dir="$exp_root/$prompt"; mkdir -p "$out_dir"
  mkdir -p "$low_root/$prompt"
  low_latents="$low_root/$prompt/stage1_lowres_latents.pt"
  export LOW_RES_SAVE_PATH="$low_latents"
  log="$out_dir/run_log.log"

  extra=()
  [ "$CACHE" = "1" ] && extra+=( --enable-intra-tile-cache --cache-thresh "$CACHE_THRESH" \
                                 --enable-region-aware-cache --static-tile-cache-scale-factor "$CACHE_SCALE" )

  echo "[run] $prompt -> $log"
  torchrun --nproc_per_node="$TP_SIZE" --master_port "$MASTER_PORT" sample_image2video.py \
      --model "$HUN_MODEL_ARCH" \
      --prompt "$prompt" \
      --i2v-mode \
      --i2v-image-path "$img" \
      --i2v-resolution 720p \
      --infer-steps "$STEPS" \
      --video-length "$VIDEO_LENGTH" \
      --video-size "$TARGET_H" "$TARGET_W" \
      --flow-reverse --flow-shift 7.0 --i2v-stability \
      --seed 0 --embedded-cfg-scale 6.0 \
      --output-dir "$out_dir" --save-path "$out_dir" \
      --two-stage-generation \
      --upscale-factor "$UPSCALE" \
      --upscale-res-steps "$UPSCALE_RES_STEPS" \
      --save-intermediate \
      --loop-step "$LOOP_STEP" \
      --shift-timesteps "$SHIFT" \
      --load-prev-latents-path "$low_latents" \
      "${extra[@]}" \
      > "$log" 2>&1 && echo "[ok]   $prompt" || echo "[FAIL] $prompt (see $log)"
done < <(emit_prompt_set "$PROMPT_SET")
echo "[generate] done -> $exp_root"
