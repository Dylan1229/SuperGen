#!/bin/bash

set -x 
model="cogvideo"

# Multi GPU configuration, TP_SIZE is the number of GPUs to use
export TP_SIZE="2"
export CUDA_VISIBLE_DEVICES="0,1"  # Use the GPU index you want


# Set target resolution 
# 720p resolution:
# target_height=720
# target_width=1280

# 1080p resolution:
# target_height=1080
# target_width=1920

# 2K resolution:
target_height=1440
target_width=2560
upscale_factor=2

# 4K resolution:
# target_height=2160
# target_width=3840
# upscale_factor=3

# Hyperparameters
upscale_res_steps=45
total_steps=50
loop_step=16
cache_thresh=0.02
shift_timesteps='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44'
static_tile_cache_scale_factor=0.5

timestamp=$(date +%Y%m%d_%H%M%S)

images=(
"a bar with chairs and a television on the wall.jpg"
)

version="01"
test_dir="./output/${model}"

# Loop through each prompt and image pair
for i in "${!images[@]}"; do
    image_name="${images[$i]}"
    prompt="${image_name%.jpg}"

    base_dir="${test_dir}/${prompt%.*}"

    # Create output directory for this sample
    output_dir="${base_dir}/${version}_${timestamp}"
    mkdir -p "${output_dir}"
    LOGFILE="${output_dir}/run_log.log"

    low_res_latents_path="${base_dir}/stage1_lowres_latents.pt"
    export LOW_RES_SAVE_PATH="${low_res_latents_path}"

    # Whether to enable redistribution, 0 for no redistribution, 1 for redistribution
    export ENABLE_REDISTRIBUTE=0

    # Run the pipeline
    exec >"$LOGFILE" 2>&1
    torchrun --nproc_per_node=${TP_SIZE} --master_port 33333 CogvideoI2V/pipeline.py \
            --prompt "$prompt" \
            --model_path "THUDM/CogVideoX1.5-5b-i2v" \
            --generate_type "i2v" \
            --image_or_video_path "input_image/2k_1440x2560/${image_name}" \
            --width ${target_width} \
            --height ${target_height}  \
            --num_inference_steps ${total_steps} \
            --guidance_scale 6.0 \
            --num_frames 41 \
            --fps 8 \
            --seed 0 \
            --output_path "${output_dir}/${image_name%.*}_final.mp4" \
            --upscale_factor ${upscale_factor} \
            --upscale_res_steps ${upscale_res_steps} \
            --low_res_latents_path "$low_res_latents_path" \
            --loop_step ${loop_step} \
            --shift_timesteps ${shift_timesteps} \

done
    # The following parameters are for cache acceleration
            # --enable_region_aware_cache \
            # --static_tile_cache_scale_factor ${static_tile_cache_scale_factor} \
            # --cache_thresh ${cache_thresh} \
            # --enable_intra_tile_cache \
