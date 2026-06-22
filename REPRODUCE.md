# Reproducing the SuperGen experiments (restored environment)

This branch (`reproduce`) restores the environment used for the paper
**“Efficient Ultra-High-Resolution Video Generation with Tiling and Sketching”**
(SuperGen, arXiv:2508.17756) on the **current cluster**. Everything below is
already set up and smoke-tested; you can run experiments directly.

The clean, anonymized open-source code lives on the `open-source` branch. This
branch adds back the experiment launch scripts, prompt sets, input images,
quality-evaluation harness, and figure notebooks that the open-source cut
stripped out.

---

## ⚠️ Hardware caveat (read first)

| | This cluster | Paper |
|---|---|---|
| GPUs | **2× NVIDIA H200 NVL (144 GB)** | 8× H100-80GB (NVLink) |
| CUDA driver | 13.0 | 12.2 |

Consequences:
- **4-/8-GPU runs cannot be reproduced here** — only `TP_SIZE ∈ {1, 2}`. The
  `scaling_gpus.sh` scripts run the points that fit and skip the rest; run them
  with `GPUS="1 2 4 8"` on a ≥8-GPU node for the full strong-scaling curve.
- **Absolute latencies won't match the paper** (H200 ≠ H100). Speedup *ratios*,
  quality numbers, and all ablations/plots reproduce faithfully.

---

## 1. Python environments (uv venvs)

Five isolated venvs under `/mnt/data/fanjiang/venvs/` (the launch scripts
activate the right one automatically — you only need these paths for manual runs):

| venv | Python | Purpose | Key packages |
|---|---|---|---|
| `supergen-cogvideo` | 3.10 | CogVideoX-1.5 generation | torch 2.5.1+cu124, diffusers 0.33.1, transformers 4.46.2 |
| `supergen-hunyuan`  | 3.11 | HunyuanVideo generation | torch 2.4.0+cu124, diffusers 0.31.0, **flash-attn 2.6.3** |
| `supergen-metrics`  | 3.10 | PSNR/SSIM/LPIPS/FVD | torch 2.6.0+cu124, lpips, decord, opencv |
| `supergen-vbench`   | 3.10 | VBench scoring | torch 2.6.0+cu124 (+ VBench repo via `sys.path`) |
| `supergen-plots`    | 3.10 | figure notebooks | matplotlib, numpy, pandas, seaborn, jupyter |

Rebuild any of them with `uv` if needed (set `export UV_CACHE_DIR=/mnt/data/fanjiang/.cache/uv`).

---

## 2. Models (already downloaded)

| Model | Location | Notes |
|---|---|---|
| CogVideoX-1.5-5B I2V | HF cache `$HF_HOME` (`/mnt/data/fanjiang/.cache/huggingface`) | `THUDM/CogVideoX1.5-5b-i2v`, resolved by repo id |
| HunyuanVideo I2V (13B) | `HunyuanVideoI2V/ckpts/` (51 GB) | DiT+VAE (`tencent/HunyuanVideo-I2V`), `text_encoder_i2v`=LLaVA-LLaMA3, `text_encoder_2`=CLIP-L |

`HunyuanVideoI2V/ckpts/` is git-ignored. The CogVideoX weights live in the HF cache.

---

## 3. Prompts & input images

- **Prompt sets** — `scripts/prompts/vbench2_i2v_{10,20,30,full}.json`.
  The 10/20/30-prompt balanced subsets are what the paper used (10 for timing,
  30 for 2K quality, 20 for 4K quality); `full` is the 1118-prompt VBench-I2V pool.
- **Input images** — `input_image/{2k_1440x2560,4k_2160x3840}/` (355 VBench-I2V
  crops each). They cover every prompt in the 10/20/30 sets.

---

## 4. Running generation / timing experiments

Everything is driven by one parametrized script per backend,
`scripts/<Backend>/generate.sh`, configured through env vars (shared config &
helpers live in `scripts/lib.sh`). The ablation/scaling scripts are thin wrappers
that call `generate.sh` with different settings.

```bash
# Quick smoke run: one 2K sample on both GPUs (CogVideoX)
RES=2k TP_SIZE=2 PROMPT_SET=single CACHE=0 bash scripts/CogvideoI2V/generate.sh

# 4K, 2 GPUs, cache on, over the 20-prompt quality set (HunyuanVideo)
RES=4k TP_SIZE=2 PROMPT_SET=20 CACHE=1 bash scripts/HunyuanVideoI2V/generate.sh
```

`generate.sh` knobs (defaults in `[]`): `RES[2k]` `720p|1080p|2k|4k`,
`TP_SIZE[2]`, `PROMPT_SET[single]` `single|10|20|30|full`, `CACHE[0]`,
`REDISTRIBUTE[0]` (cache-guided workload rebalance), `STEPS[50]`, `LOOP_STEP[16]`,
`CACHE_THRESH`, `CACHE_SCALE`, `SHIFT`, `TAG`.

**Script → paper figure map:**

| Experiment | Command | Paper figure/table |
|---|---|---|
| Single-GPU baseline | `TP_SIZE=1 … generate.sh` | Fig. 11 “Origin” / `e2e_performance_comparison.pdf` |
| Multi-GPU tiled | `TP_SIZE=2 … generate.sh` | Fig. 11 “+Parallelism” |
| + cache | `CACHE=1 … generate.sh` | Fig. 11/12, `latency_comparison_4GPUs.pdf` |
| GPU strong-scaling | `scripts/<B>/scaling_gpus.sh` | Fig. 14 `GPU_scalability_comparison.pdf` |
| Cache threshold×scale grid | `scripts/<B>/ablation_cache_grid.sh` | Fig. 16 `cache_ablation_heatmaps.pdf` |
| Number-of-tiles | `scripts/CogvideoI2V/ablation_num_tiles.sh` | Fig. 17 `tiles_effect*.pdf` |
| Tile-shift schedule | `scripts/CogvideoI2V/ablation_shift.sh` | Table 8 |
| Quality gen, 2K | `RES=2k PROMPT_SET=30 … generate.sh` | Table 3 (VBench) |
| Quality gen, 4K | `RES=4k PROMPT_SET=20 … generate.sh` | Table 3 (VBench) |

Outputs:
- Per-sample logs `output/<backend>/<TAG>/<prompt>/run_log.log` — these carry the
  `Total running time` / `First/Second Stage Running time` lines that become the
  latency numbers in the figures.
- Stage-1 (low-res) latents are cached under `output/<backend>/_stage1/<RES>/` and
  reused across runs of the same input (matches the paper's "pre-saved latents").
- The number-of-tiles ablation overrides the latent tile size via the
  `TILE_LAT_OVERRIDE` env var (read in `utils/tile_utils.py:SlidingWindowConfig`).

---

## 5. Quality evaluation

Full details in **`evaluation/README.md`**. Two tracks:

**Frame fidelity (cache vs no-cache): PSNR / SSIM / LPIPS / FVD**
```bash
cd evaluation/common_metrics_on_video_quality
/mnt/data/fanjiang/venvs/supergen-metrics/bin/python evaluate_pairs.py \
  --baseline_dir /path/to/no_cache_videos --test_dir /path/to/cache_videos \
  --output ../exp_result/VisualRetention/run.json --device cuda --with_fvd
```

**No-reference quality: VBench** (subject/background consistency, motion
smoothness, aesthetic, imaging)
```bash
cd evaluation/vbench
CUDA_VISIBLE_DEVICES=0 \
/mnt/data/fanjiang/venvs/supergen-vbench/bin/python run_vbench_eval_for_dir.py /path/to/mp4_dir
```
VBench downloads its scoring backbones into `$VBENCH_CACHE_DIR` on the first run
(needs internet once). The original generated-video corpora are **not** on this
cluster, so you supply the video directories (generate them with §4).

---

## 6. Regenerating the paper figures

Full details in **`plots/README.md`**. All figure data is hardcoded inline in the
notebooks (they read no external files), so they regenerate as-is:
```bash
cd plots
/mnt/data/fanjiang/venvs/supergen-plots/bin/jupyter nbconvert --to notebook \
  --execute --ExecutePreprocessor.timeout=600 --output /tmp/out.ipynb plot_main.ipynb
```
→ PDFs land in `plots/figures/`. `plot_main.ipynb` is authoritative;
`plot_legacy.ipynb` additionally emits `breakdown_stages.pdf` and
`techniques_breakdown.pdf`.

**To refresh numbers after re-running experiments:** read latency from the
`run_log.log` files (§4) and VBench scores from §5, then edit the inline
arrays/dicts in the relevant notebook cell and re-run the notebook.

---

## 7. Known gaps / TODO

- 4-/8-GPU scaling points require a ≥8-GPU node (this cluster has 2).
- VBench scoring backbones download lazily on first use (one-time internet need).
- `plots/method_analysis/*.py` (noise-L1 figures) need a *profiled* generation run
  that emits `noise_prediction_profile/*.json` — not produced by the default runs.
- The original `exp_result/` generated videos are gone; eval consumes whatever you
  regenerate via §4.
