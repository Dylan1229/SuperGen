#!/bin/bash
# =============================================================================
# SuperGen reproduction — shared configuration & helpers
# =============================================================================
# New cluster: 2x NVIDIA H200 NVL (144 GB each), CUDA 13 driver.
# (The paper used 8x H100-80GB, so 4/8-GPU runs need a larger node; see
#  scaling_gpus.sh and REPRODUCE.md.)
#
# Source this from an experiment script, e.g. from scripts/<Model>/<script>.sh:
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib.sh"
#
# Every path/venv below can be overridden by exporting it before running.
# =============================================================================

# --- repo root (this file lives in scripts/) ---------------------------------
SUPERGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUPERGEN_ROOT

# --- uv-created virtualenvs ---------------------------------------------------
COG_VENV="${COG_VENV:-/mnt/data/fanjiang/venvs/supergen-cogvideo}"   # py3.10, torch 2.5.1
HUN_VENV="${HUN_VENV:-/mnt/data/fanjiang/venvs/supergen-hunyuan}"    # py3.11, torch 2.4.0 + flash-attn

# --- HuggingFace cache (CogVideoX resolves its model by repo id from here) ---
export HF_HOME="${HF_HOME:-/mnt/data/fanjiang/.cache/huggingface}"

# --- models ------------------------------------------------------------------
COG_MODEL="${COG_MODEL:-THUDM/CogVideoX1.5-5b-i2v}"
HUN_MODEL_ARCH="${HUN_MODEL_ARCH:-HYVideo-T/2}"
# HunyuanVideo weights live in HunyuanVideoI2V/ckpts (MODEL_BASE default).

# --- input images (355 VBench-I2V crops per resolution) ----------------------
IMG_2K="${IMG_2K:-$SUPERGEN_ROOT/input_image/2k_1440x2560}"
IMG_4K="${IMG_4K:-$SUPERGEN_ROOT/input_image/4k_2160x3840}"
PROMPTS_DIR="$SUPERGEN_ROOT/scripts/prompts"

# --- full tile-shift schedule (shift at every refinement step 0..44) ---------
SHIFT_ALL="$(seq -s, 0 44)"

# --- resolution presets: sets TARGET_H TARGET_W UPSCALE IMG_DIR ---------------
set_resolution() {
  case "$1" in
    720p)  TARGET_H=720;  TARGET_W=1280; UPSCALE=1; IMG_DIR="$IMG_2K";;
    1080p) TARGET_H=1080; TARGET_W=1920; UPSCALE=2; IMG_DIR="$IMG_2K";;
    2k)    TARGET_H=1440; TARGET_W=2560; UPSCALE=2; IMG_DIR="$IMG_2K";;
    4k)    TARGET_H=2160; TARGET_W=3840; UPSCALE=3; IMG_DIR="$IMG_4K";;
    *) echo "[lib] unknown RES '$1' (use 720p|1080p|2k|4k)" >&2; exit 1;;
  esac
}

# --- emit "image_name<TAB>prompt" lines for a prompt set ---------------------
# arg: single | 10 | 20 | 30 | full   (the JSONs live in scripts/prompts/)
emit_prompt_set() {
  local set="$1"
  if [ "$set" = "single" ]; then
    printf '%s\t%s\n' \
      "a bar with chairs and a television on the wall.jpg" \
      "a bar with chairs and a television on the wall"
    return
  fi
  /usr/bin/python3 - "$PROMPTS_DIR/vbench2_i2v_${set}.json" <<'PY'
import json, sys
for d in json.load(open(sys.argv[1])):
    print(f"{d['image_name']}\t{d['prompt_en']}")
PY
}

# --- GPU helpers -------------------------------------------------------------
gpu_count() { nvidia-smi -L 2>/dev/null | wc -l; }

check_tp() {
  local tp="$1" have; have="$(gpu_count)"
  if [ "$tp" -gt "$have" ]; then
    echo "[lib] WARNING: TP_SIZE=$tp but only $have GPU(s) visible." >&2
    echo "[lib]          This cluster has 2x H200; the paper's $tp-GPU point needs a $tp-GPU node." >&2
  fi
}
