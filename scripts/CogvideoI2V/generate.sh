#!/bin/bash
# =============================================================================
# SuperGen — CogVideoX-1.5 I2V generation / timing driver (parametrized)
# =============================================================================
# One driver for every CogVideoX experiment. Knobs are passed as env vars.
# Per-sample stdout (which carries the "Total running time" / "First/Second
# Stage Running time" lines the plot notebooks parse) is written to
# output/cogvideo/<TAG>/<prompt>/run_log.log .
#
# Examples:
#   # single-GPU baseline, 2K, one sample  (paper Fig.11 "Origin")
#   RES=2k TP_SIZE=1 PROMPT_SET=single CACHE=0 bash scripts/CogvideoI2V/generate.sh
#
#   # 2-GPU tiled + cache, 4K, 20-prompt quality set
#   RES=4k TP_SIZE=2 PROMPT_SET=20 CACHE=1 bash scripts/CogvideoI2V/generate.sh
#
# Env knobs (defaults in []):
#   RES[2k] 720p|1080p|2k|4k     TP_SIZE[2]       PROMPT_SET[single] single|10|20|30|full
#   CACHE[0] 0|1                 REDISTRIBUTE[0]  STEPS[50]          LOOP_STEP[16]
#   UPSCALE_RES_STEPS[45]        CACHE_THRESH[0.02] CACHE_SCALE[0.5]
#   TILE_W TILE_H (optional tile-latent size override)  SHIFT (optional schedule)
#   TAG (output subdir; auto-derived if unset)          MASTER_PORT[33333]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"

RES="${RES:-2k}"
TP_SIZE="${TP_SIZE:-2}"
PROMPT_SET="${PROMPT_SET:-single}"
CACHE="${CACHE:-0}"
export ENABLE_REDISTRIBUTE="${REDISTRIBUTE:-0}"
STEPS="${STEPS:-50}"
LOOP_STEP="${LOOP_STEP:-16}"
UPSCALE_RES_STEPS="${UPSCALE_RES_STEPS:-45}"
CACHE_THRESH="${CACHE_THRESH:-0.02}"
CACHE_SCALE="${CACHE_SCALE:-0.5}"
MASTER_PORT="${MASTER_PORT:-33333}"

set_resolution "$RES"
check_tp "$TP_SIZE"
SHIFT="${SHIFT:-$SHIFT_ALL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TP_SIZE-1)))}"

# activate the CogVideoX env, run from repo root
# shellcheck disable=SC1091
source "$COG_VENV/bin/activate"
cd "$SUPERGEN_ROOT"

cache_tag="nocache"; [ "$CACHE" = "1" ] && cache_tag="cache"
TAG="${TAG:-${RES}_tp${TP_SIZE}_${cache_tag}_${PROMPT_SET}}"
exp_root="output/cogvideo/${TAG}"
low_root="output/cogvideo/_stage1/${RES}"          # stage-1 latents reused across runs
mkdir -p "$exp_root" "$low_root"
echo "[generate] cog RES=$RES TP=$TP_SIZE set=$PROMPT_SET cache=$CACHE redistribute=$ENABLE_REDISTRIBUTE -> $exp_root"

while IFS=$'\t' read -r image_name prompt; do
  [ -z "$image_name" ] && continue
  img="$IMG_DIR/$image_name"
  if [ ! -f "$img" ]; then echo "[skip] missing image: $img"; continue; fi

  out_dir="$exp_root/$prompt"; mkdir -p "$out_dir"
  mkdir -p "$low_root/$prompt"
  low_latents="$low_root/$prompt/stage1_lowres_latents.pt"
  export LOW_RES_SAVE_PATH="$low_latents"
  log="$out_dir/run_log.log"

  extra=()
  [ "$CACHE" = "1" ] && extra+=( --enable_intra_tile_cache --cache_thresh "$CACHE_THRESH" \
                                 --enable_region_aware_cache --static_tile_cache_scale_factor "$CACHE_SCALE" )
  # tile-size override for the number-of-tiles ablation: window_size = (H, W) in latent space
  [ -n "$TILE_W" ] && export TILE_LAT_OVERRIDE="${TILE_H}x${TILE_W}"

  echo "[run] $prompt -> $log"
  torchrun --nproc_per_node="$TP_SIZE" --master_port "$MASTER_PORT" CogvideoI2V/pipeline.py \
      --prompt "$prompt" \
      --model_path "$COG_MODEL" \
      --generate_type "i2v" \
      --image_or_video_path "$img" \
      --width "$TARGET_W" --height "$TARGET_H" \
      --num_inference_steps "$STEPS" \
      --guidance_scale 6.0 --num_frames 41 --fps 8 --seed 0 \
      --output_path "$out_dir/${image_name%.*}_final.mp4" \
      --upscale_factor "$UPSCALE" \
      --upscale_res_steps "$UPSCALE_RES_STEPS" \
      --low_res_latents_path "$low_latents" \
      --loop_step "$LOOP_STEP" \
      --shift_timesteps "$SHIFT" \
      "${extra[@]}" \
      > "$log" 2>&1 && echo "[ok]   $prompt" || echo "[FAIL] $prompt (see $log)"
done < <(emit_prompt_set "$PROMPT_SET")
echo "[generate] done -> $exp_root"
