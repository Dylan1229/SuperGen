#!/bin/bash

set -x 
cd HunyuanVideoI2V

# TP_SIZE is the number of GPUs to use
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
# Hyperparameters
infer_steps=50
video_length=41
upscale_res_steps=45
loop_step=16
cache_thresh=0.05
shift_timesteps='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44'
static_tile_cache_scale_factor=0.5

timestamp=$(date +%Y%m%d_%H%M%S)

images=(
"a bar with chairs and a television on the wall.jpg"
)

version="01"
model="hunyuan"
test_dir="../output/${model}"

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

    export ENABLE_REDISTRIBUTE=0

    # Run the pipeline
    exec >"$LOGFILE" 2>&1
    torchrun --nproc_per_node=${TP_SIZE} --master_port 33334 sample_image2video.py \
        --model HYVideo-T/2 \
        --prompt "$prompt" \
        --i2v-mode \
        --i2v-image-path "../input_image/2k_1440x2560/${image_name}" \
        --i2v-resolution 720p \
        --infer-steps ${infer_steps} \
        --video-length ${video_length} \
        --video-size ${target_height} ${target_width} \
        --flow-reverse \
        --flow-shift 7.0 \
        --i2v-stability \
        --seed 0 \
        --embedded-cfg-scale 6.0 \
        --output-dir "${output_dir}" \
        --save-path "${output_dir}" \
        --two-stage-generation \
        --upscale-factor ${upscale_factor} \
        --upscale-res-steps ${upscale_res_steps} \
        --save-intermediate \
        --loop-step ${loop_step} \
        --shift-timesteps ${shift_timesteps} \
        --load-prev-latents-path "${low_res_latents_path}" \

done

        # --enable-intra-tile-cache \
        # --cache-thresh ${cache_thresh} \
        # --enable-region-aware-cache \
        # --static-tile-cache-scale-factor 1.0 \