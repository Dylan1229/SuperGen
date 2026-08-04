# SuperGen

A unified framework for multi-GPU distributed ultra-high-resolution video generation, supporting multiple state-of-the-art video generation models.

## Supported Models

| Model | Reference |
|-------|-----------|
| CogVideoX-1.5 I2V | [Github](https://github.com/zai-org/CogVideo) |
| HunyuanVideo I2V | [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) |
| Wan 2.1 I2V (14B) | [GitHub](https://github.com/Wan-Video/Wan2.1) |
| Wan 2.1 T2V (14B) | [GitHub](https://github.com/Wan-Video/Wan2.1) |

## Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/Dylan1229/SuperGen.git
cd SuperGen
```

### Environment Setup

#### 1. CogVideoX-1.5 I2V

Follow the [official instructions](https://github.com/zai-org/CogVideo/tree/main?tab=readme-ov-file#quick-start), or use `uv` + `venv`:

```bash
# Create venv environment
uv venv /path/to/your/venv_env/CogVideox --python 3.10 --seed

# Activate the environment
source /path/to/your/venv_env/CogVideox/bin/activate

# Install dependencies
cd CogvideoI2V
uv pip install -r requirements.txt
```

#### 2. HunyuanVideo I2V

Follow the [official installation guide](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V?tab=readme-ov-file#installation-guide-for-linux), or use `uv` + `venv`:

```bash
# Create venv environment
uv venv /path/to/your/venv_env/HunyuanVideo --python 3.11.9 --seed

# Activate the environment
source /path/to/your/venv_env/HunyuanVideo/bin/activate

cd HunyuanVideoI2V

# Install PyTorch and other dependencies
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install pip dependencies
uv pip install -r requirements.txt

# Install flash-attention v2 for acceleration
uv pip install ninja
uv pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

```

#### 3. Wan 2.1 (I2V and T2V)

Wan runs against its own upstream runtime, which is a **separate checkout** rather than a pip
package. Clone it anywhere and point `WAN_REPO` at it; if unset, a sibling directory named
`Wan2.1` is used.

```bash
git clone https://github.com/Wan-Video/Wan2.1.git       # beside this repo, or anywhere
export WAN_REPO=/path/to/Wan2.1

python -m venv ~/envs/wan && source ~/envs/wan/bin/activate
pip install -r $WAN_REPO/requirements.txt
```

Sequence parallelism (optional, and only for T2V) additionally needs `xfuser`:

```bash
pip install xfuser==0.4.3
```

### Downloaded Pretrained Models

| Model | Download Link |
|-------|---------------|
| CogVideoX-1.5 I2V | [Hugging Face](https://huggingface.co/zai-org/CogVideoX1.5-5B) |
| HunyuanVideo I2V | [Instructions](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/main/ckpts/README.md) |
| Wan 2.1 I2V (14B, 720P) | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |
| Wan 2.1 T2V (14B) | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |

## Usage

For CogVideoX-1.5 and HunyuanVideo, see [scripts/README.md](scripts/README.md).

### Wan 2.1

Both entry points take the target resolution and an upscale factor; the first stage always runs at
the backbone's native 720p and the second stage denoises the upscaled canvas tile by tile.

```bash
# I2V, 2K on 4 GPUs
torchrun --nproc_per_node=4 WanI2V/pipeline.py \
    --prompt "a mountain range with a sky background" \
    --image_path inputs/2k_1440x2560/"a mountain range with a sky background".jpg \
    --ckpt_dir ~/ckpts/Wan2.1-I2V-14B-720P \
    --height 1440 --width 2560 --upscale_factor 2 \
    --output_path out_2k.mp4

# T2V, 4K on 4 GPUs, with the region-aware cache
torchrun --nproc_per_node=4 WanI2V/pipeline_t2v.py \
    --prompt "the parthenon in acropolis, greece" \
    --task t2v-14B --ckpt_dir ~/ckpts/Wan2.1-T2V-14B \
    --height 2160 --width 3840 --upscale_factor 3 \
    --enable_cache --cache_thresh 0.20 \
    --output_path out_4k.mp4
```

`--ulysses_size` composes sequence parallelism with tile parallelism on T2V. The degree must divide
the model's head count (40 for T2V-14B), and `ulysses_size * ring_size` must equal the world size.

### Cache thresholds are per backbone

The gate compares residual magnitudes, and those differ by up to 3.9x between backbones, so a
threshold tuned on one does not transfer. Measured operating points:

| Backbone | `--cache_thresh` |
|---|---|
| CogVideoX-1.5 | 0.20 |
| HunyuanVideo | 0.06 |
| Wan 2.1 I2V | 0.15 |
| Wan 2.1 T2V | 0.20 |

### Tests

The `test_*.py` files beside each pipeline are runnable checks of the pieces that are easy to get
silently wrong -- tiled VAE encode/decode error, canvas-absolute RoPE, tile-parallel equivalence
against single-GPU, and the cache gate. They need the corresponding checkpoint.

```bash
WAN_REPO=/path/to/Wan2.1 python WanI2V/test_tile_parallel.py
```

## BibTeX
If you find [SuperGen](https://arxiv.org/abs/2508.17756) useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{ye2025supergenefficientultrahighresolutionvideo,
      title={SuperGen: An Efficient Ultra-high-resolution Video Generation System with Sketching and Tiling}, 
      author={Fanjiang Ye and Zepeng Zhao and Yi Mu and Jucheng Shen and Renjie Li and Kaijian Wang and Saurabh Agarwal and Myungjin Lee and Triston Cao and Aditya Akella and Arvind Krishnamurthy and T. S. Eugene Ng and Zhengzhong Tu and Yuke Wang},
      year={2025},
      eprint={2508.17756},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.17756}, 
}
```
