# Efficient Ultra-High-Resolution Video Generation with Tiling and Sketching

A unified framework for **multi-GPU distributed ultra-high-resolution video generation** (up to 4K and beyond), supporting multiple state-of-the-art video diffusion models. The system uses a two-stage generation pipeline with sliding-window tiling to enable efficient ultra-high-resolution video synthesis on commodity GPUs.

## Key Features

- **Two-stage generation**: First generates at low resolution, then upscales using tiled distributed denoising
- **Sliding-window tiling**: Distributes overlapping tiles across multiple GPUs with load balancing
- **Cache acceleration**: Identifies static (low-motion) tiles and skips redundant computation
- **Multi-model support**: Works with both CogVideoX-1.5 I2V and HunyuanVideo I2V

## Supported Models

| Model | Type | Recommended Resolution |
|-------|------|----------------------|
| CogVideoX-1.5 I2V | Image-to-Video | 768x1360 (base) |
| HunyuanVideo I2V | Image-to-Video | 720p (base) |

## Repository Structure

```
.
├── CogvideoI2V/                  # CogVideoX pipeline
│   ├── pipeline.py               # Entry point & CLI
│   ├── pipeline_cogvideox_i2v_TVG.py  # Tiled video generation pipeline
│   ├── modules/                  # Custom transformer & scheduler
│   └── requirements.txt
├── HunyuanVideoI2V/              # HunyuanVideo pipeline
│   ├── sample_image2video.py     # Entry point & CLI
│   ├── hyvideo/
│   │   ├── inference.py          # Main inference logic
│   │   ├── config.py             # Argument parsing
│   │   ├── modules/              # Transformer, attention, embeddings
│   │   ├── diffusion/            # Schedulers & diffusion pipeline
│   │   ├── vae/                  # Video VAE
│   │   ├── text_encoder/         # Text encoder wrapper
│   │   └── utils/                # Data utils, LoRA loading, helpers
│   └── requirements.txt
├── utils/                        # Shared core infrastructure
│   ├── distributed.py            # DistributedManager for multi-GPU tiling
│   ├── tile_utils.py             # Tiled latent tensors & noise aggregation
│   └── tile_std_tracker.py       # Motion-aware tile cache tracker
├── scripts/                      # Launch scripts
│   ├── CogvideoI2V/
│   │   └── run_cogvideo_i2v_distributed.sh
│   └── HunyuanVideoI2V/
│       └── run_hunyuan_i2v_distributed.sh
└── input_image/                  # Sample input images (2K, 4K)
```

## Installation

The two model backends require **separate Python environments** due to different dependency versions.

### 1. CogVideoX-1.5 I2V (Python 3.10)

```bash
# Create and activate virtual environment
python3.10 -m venv /path/to/envs/cogvideo
source /path/to/envs/cogvideo/bin/activate

# Install dependencies
cd CogvideoI2V
pip install -r requirements.txt
cd ..
```

### 2. HunyuanVideo I2V (Python 3.11)

```bash
# Create and activate virtual environment
python3.11 -m venv /path/to/envs/hunyuan
source /path/to/envs/hunyuan/bin/activate

cd HunyuanVideoI2V

# Install PyTorch (CUDA 12.4)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Install flash-attention v2 for acceleration
pip install ninja
pip install flash-attn==2.6.3

cd ..
```

## Pretrained Models

Download the pretrained model weights before running inference:

| Model | Download |
|-------|----------|
| CogVideoX-1.5 I2V | `THUDM/CogVideoX1.5-5b-i2v` from Hugging Face |
| HunyuanVideo I2V | Place checkpoints under `HunyuanVideoI2V/ckpts/` following the model's official instructions |

## Usage

### CogVideoX-1.5 I2V — Distributed Inference

```bash
source /path/to/envs/cogvideo/bin/activate

# Edit the script to configure resolution, GPU count, and input images
bash scripts/CogvideoI2V/run_cogvideo_i2v_distributed.sh
```

Or run directly with `torchrun`:

```bash
export TP_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=${TP_SIZE} --master_port 33333 CogvideoI2V/pipeline.py \
    --prompt "your prompt here" \
    --model_path "THUDM/CogVideoX1.5-5b-i2v" \
    --generate_type "i2v" \
    --image_or_video_path "input_image/2k_1440x2560/your_image.jpg" \
    --width 2560 \
    --height 1440 \
    --num_inference_steps 50 \
    --guidance_scale 6.0 \
    --num_frames 41 \
    --fps 8 \
    --seed 0 \
    --output_path "./output/final.mp4" \
    --upscale_factor 2 \
    --upscale_res_steps 45 \
    --loop_step 16 \
    --shift_timesteps '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44'
```

### HunyuanVideo I2V — Distributed Inference

```bash
source /path/to/envs/hunyuan/bin/activate

# Edit the script to configure resolution, GPU count, and input images
bash scripts/HunyuanVideoI2V/run_hunyuan_i2v_distributed.sh
```

Or run directly with `torchrun`:

```bash
export TP_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

cd HunyuanVideoI2V

torchrun --nproc_per_node=${TP_SIZE} --master_port 33334 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "your prompt here" \
    --i2v-mode \
    --i2v-image-path "../input_image/2k_1440x2560/your_image.jpg" \
    --i2v-resolution 720p \
    --infer-steps 50 \
    --video-length 41 \
    --video-size 1440 2560 \
    --flow-reverse \
    --flow-shift 7.0 \
    --i2v-stability \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --save-path "./output" \
    --two-stage-generation \
    --upscale-factor 2 \
    --upscale-res-steps 45 \
    --loop-step 16 \
    --shift-timesteps '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44'
```

## Configuration Guide

### Resolution Presets

| Target | Height | Width | upscale_factor |
|--------|--------|-------|----------------|
| 720p   | 720    | 1280  | 1 (no upscale) |
| 1080p  | 1080   | 1920  | 2 |
| 2K     | 1440   | 2560  | 2 |
| 4K     | 2160   | 3840  | 3 |

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TP_SIZE` | Number of GPUs for tensor parallelism | 2 |
| `upscale_factor` | Resolution multiplier between Stage 1 and Stage 2 | 2 |
| `upscale_res_steps` | Timestep at which to switch from low-res to high-res | 45 |
| `loop_step` | Sliding window offset step for tile traversal | 16 |
| `shift_timesteps` | Comma-separated timesteps at which to shift tile positions | - |
| `num_inference_steps` / `infer-steps` | Total denoising steps | 50 |

### Cache Acceleration (Optional)

To enable cache acceleration, add these flags:

**CogVideoX:**
```bash
    --enable_region_aware_cache \
    --static_tile_cache_scale_factor 0.5 \
    --cache_thresh 0.02 \
    --enable_intra_tile_cache
```

**HunyuanVideo:**
```bash
    --enable-intra-tile-cache \
    --cache-thresh 0.05 \
    --enable-region-aware-cache \
    --static-tile-cache-scale-factor 1.0
```

### Workload Redistribution (Optional)

Set the environment variable to enable dynamic workload redistribution when cache skipping creates load imbalance:

```bash
export ENABLE_REDISTRIBUTE=1
```

### Pre-saved Low-Resolution Latents

Stage 1 latents are automatically saved to disk. On subsequent runs with the same input, Stage 1 is skipped and the saved latents are reused. The save path is controlled by the `LOW_RES_SAVE_PATH` environment variable in the launch scripts.
