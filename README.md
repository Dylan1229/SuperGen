# SuperGen

A unified framework for multi-GPU distributed ultra-high-resolution video generation, supporting multiple state-of-the-art video generation models.

## Supported Models

| Model | Reference |
|-------|-----------|
| CogVideoX-1.5 I2V | [Github](https://github.com/zai-org/CogVideo) |
| HunyuanVideo I2V | [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) |

## Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/your-org/SuperGen.git
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

### Downloaded Pretrained Models

| Model | Download Link |
|-------|---------------|
| CogVideoX-1.5 I2V | [Hugging Face](https://huggingface.co/zai-org/CogVideoX1.5-5B) |
| HunyuanVideo I2V | [Instructions](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V/blob/main/ckpts/README.md) |

## Usage

See this [README.md](scripts/README.md)

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
