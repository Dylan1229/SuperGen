## Multi-GPU Distributed Inference

### CogvideoX-1.5 I2V

```bash
# Activate the environment
source /path/to/your/venv_env/CogVideox/bin/activate
cd SuperGen
# Run the script
bash scripts/CogVideoI2V/run_cogvideo_i2v_distributed.sh
```
**Note:** You can change whether use cache acceleration by enabling hyperparameters in the script. To enable redistribution, set `ENABLE_REDISTRIBUTE=1` in the script.

### HunyuanVideo I2V

```bash
# Activate the environment
source /path/to/your/venv_env/HunyuanVideo/bin/activate
cd SuperGen
# Run the script
bash scripts/HunyuanVideoI2V/run_hunyuan_i2v_distributed.sh
```