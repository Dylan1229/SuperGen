# SuperGen video-quality evaluation

This directory holds the two video-quality evaluation tracks used to produce the
paper's quality numbers, restored for the H200 cluster.

```
evaluation/
├── common_metrics_on_video_quality/   # frame-fidelity: PSNR / SSIM / LPIPS / FVD (self-contained, bundled weights)
│   ├── evaluate_pairs.py              # real entrypoint: baseline_dir vs test_dir -> JSON
│   ├── run_all_visual_retention.sh    # configurable driver (env vars)
│   ├── demo.py                        # synthetic self-test
│   ├── fvd/                           # styleganv + videogpt i3d weights (bundled .pt)
│   └── lpips/weights/                 # LPIPS linear-calibration weights (bundled .pth)
└── vbench/                            # no-reference: VBench quality dimensions
    ├── run_vbench_eval_for_dir.py     # scores a dir of mp4s -> per-video + per-dir JSON
    └── run_vbench_eval.sh             # configurable wrapper
```

## Venvs (created on this cluster)

| venv | python | purpose |
|------|--------|---------|
| `/mnt/data/fanjiang/venvs/supergen-metrics` | 3.10 | frame-fidelity (PSNR/SSIM/LPIPS/FVD) |
| `/mnt/data/fanjiang/venvs/supergen-vbench`  | 3.10 | no-reference VBench scoring |

Both use `torch 2.6.0+cu124 / torchvision 0.21.0+cu124` (H200-compatible) and
`numpy<2`. The existing `supergen-cogvideo` / `supergen-hunyuan` venvs are for
*generating* videos and are unrelated to evaluation.

> NOTE: The original generated video corpora are **not** present on this cluster.
> You supply your own directories of `.mp4` files to both tracks
> (`--baseline_dir` / `--test_dir` for frame-fidelity; the positional dir args
> for VBench).

---

## Track 1 — Frame fidelity (PSNR / SSIM / LPIPS / FVD)

Measures how close a *test* corpus is to a *baseline* corpus, video-by-video.
`evaluate_pairs.py` walks `--baseline_dir`, finds every leaf sub-directory that
contains videos (or the root itself if it holds videos directly), and matches
each to the same-named sub-directory under `--test_dir`. One video per directory
is compared (the first by sorted name). Frames are aligned by count (shortest)
and resolution (smallest H/W, bilinear) before scoring.

This track is fully self-contained: the FVD i3d backbones
(`fvd/styleganv/i3d_torchscript.pt`, `fvd/videogpt/i3d_pretrained_400.pt`) and
the LPIPS calibration weights (`lpips/weights/v0.{0,1}/*.pth`) are bundled here.
(LPIPS additionally downloads the torchvision AlexNet trunk to
`~/.cache/torch/hub` on first run — needs network once.)

### Direct command (recommended)

```bash
cd /mnt/data/fanjiang/repo/SuperGen/evaluation/common_metrics_on_video_quality

/mnt/data/fanjiang/venvs/supergen-metrics/bin/python evaluate_pairs.py \
  --baseline_dir /path/to/baseline_videos \
  --test_dir     /path/to/test_videos \
  --output       /mnt/data/fanjiang/repo/SuperGen/evaluation/exp_result/VisualRetention/my_run.json \
  --device cuda \
  --with_fvd        # optional: also compute FVD (styleganv); omit to skip
  # --only_final    # optional: single final-clip metric instead of per-clip curve
```

The output JSON contains `summary.average` (mean PSNR/SSIM/LPIPS, and FVD if
`--with_fvd`) plus a per-video `items` list.

### Via the driver (env-configurable)

```bash
cd /mnt/data/fanjiang/repo/SuperGen/evaluation/common_metrics_on_video_quality

BASELINE_DIR=/path/to/baseline_videos \
TEST_DIR=/path/to/test_videos \
OUTPUT_JSON=/mnt/data/fanjiang/repo/SuperGen/evaluation/exp_result/VisualRetention/my_run.json \
WITH_FVD=1 \
DEVICE=cuda \
./run_all_visual_retention.sh
```

Driver env vars: `PY` (default = metrics venv), `OUT_ROOT`
(default `evaluation/exp_result/VisualRetention`), `BASELINE_DIR`, `TEST_DIR`,
`OUTPUT_JSON`, `WITH_FVD` (0/1), `ONLY_FINAL` (0/1), `DEVICE`.

### Self-test (no real videos needed)

```bash
cd /mnt/data/fanjiang/repo/SuperGen/evaluation/common_metrics_on_video_quality
CUDA_VISIBLE_DEVICES=0 /mnt/data/fanjiang/venvs/supergen-metrics/bin/python demo.py
```

This runs PSNR/SSIM/LPIPS + styleganv FVD on synthetic all-zero vs all-one
tensors and prints a JSON result — use it to confirm the venv + bundled weights
load correctly.

---

## Track 2 — No-reference quality (VBench)

Scores a directory of generated `.mp4`s with five VBench *no-reference*
dimensions (no baseline needed):

```
subject_consistency, background_consistency, motion_smoothness,
aesthetic_quality, imaging_quality
```

It writes, for every video, a `<prefix>_<video>_result.json` next to the video,
and one `<prefix>_result.json` (per-dir averages) at the directory root.
`<prefix>` is derived from the path components after `exp_result/` (or the last
three path components if `exp_result` is not in the path).

VBench source: `/mnt/data/fanjiang/repo/4k-video-generation-wan2.1-i2v/VBench`.
The `vbench` package is **not** pip-installed (its `setup.py` hard-rejects
CUDA != 11.6-12.1, and we run cu124 for the H200s). Instead the driver prepends
`$VBENCH_REPO` to `sys.path` at import time, so `import vbench` resolves to the
checkout directly. Only VBench's *runtime dependencies* are installed into the
`supergen-vbench` venv (see "Dependency notes" below).

### Direct command

```bash
cd /mnt/data/fanjiang/repo/SuperGen/evaluation/vbench

CUDA_VISIBLE_DEVICES=0 \
VBENCH_CACHE_DIR=/mnt/data/fanjiang/.cache/vbench \
/mnt/data/fanjiang/venvs/supergen-vbench/bin/python run_vbench_eval_for_dir.py \
  /path/to/mp4_dir [/path/to/another_mp4_dir ...]
```

### Via the wrapper

```bash
cd /mnt/data/fanjiang/repo/SuperGen/evaluation/vbench
CUDA_VISIBLE_DEVICES=0 ./run_vbench_eval.sh /path/to/mp4_dir [more_dirs ...]
# or supply dirs via env instead of args:
VBENCH_DIRS=/dir1:/dir2 ./run_vbench_eval.sh
```

Driver env vars:
- `VBENCH_REPO` — VBench checkout (default = the cluster clone above).
- `VBENCH_CACHE_DIR` — where VBench downloads/loads its scoring backbones
  (default `/mnt/data/fanjiang/.cache/vbench`).
- `VBENCH_DIRS` — `os.pathsep`-separated dirs to score (if not passed as args).
- `CUDA_VISIBLE_DEVICES` — GPU index (default `0`).
- `PY` — python interpreter (default = vbench venv).

### IMPORTANT: VBench scoring backbones are NOT downloaded yet

The VBench clone's `pretrained/` directory contains only *pointer scripts*
(`model_path.txt`, `download.sh`), **not** the actual weights. At scoring time
VBench downloads its backbones into `$VBENCH_CACHE_DIR` (default
`~/.cache/vbench`) on first use. The five dimensions above require:

| dimension | backbone | source |
|-----------|----------|--------|
| subject_consistency | DINO (ViT-S/16) | torch.hub (`facebookresearch/dino`) |
| background_consistency | CLIP ViT-B/32 | `$VBENCH_CACHE_DIR/clip_model/ViT-B-32.pt` |
| aesthetic_quality | CLIP ViT-L/14 + LAION aesthetic head | `$VBENCH_CACHE_DIR/clip_model/ViT-L-14.pt`, `~/.cache/aesthetic_model/` |
| motion_smoothness | AMT-S | `$VBENCH_CACHE_DIR/amt_model/amt-s.pth` |
| imaging_quality | MUSIQ (pyiqa) | downloaded by `pyiqa` on first use |

VBench downloads these automatically (needs internet on the first scoring run).
The relevant URLs are in
`/mnt/data/fanjiang/repo/4k-video-generation-wan2.1-i2v/VBench/pretrained/*/`
and in `vbench/utils.py` if you need to pre-stage them offline.

### Dependency notes (how the venvs were built)

- **`supergen-metrics`**: `torch torchvision` from the cu124 index, then
  `numpy<2 opencv-python lpips einops scipy scikit-image tqdm decord`. `decord`
  installs cleanly on Python 3.10 (no `eva-decord` fallback needed); the
  OpenCV-based reader in `evaluate_pairs.py` is the fallback if decord is ever
  unavailable.
- **`supergen-vbench`**: `torch torchvision` from cu124, then VBench's
  `requirements.txt` **with two deviations**:
  1. We do **not** run `pip install -e VBench`. `setup.py`'s
     `check_torch_version()` raises `RuntimeError: Unsupported CUDA version:
     12.4` (it only allows 11.6-12.1). The driver injects `$VBENCH_REPO` onto
     `sys.path` instead, which is sufficient for `import vbench`.
  2. The `transformers==4.33.2` pin was dropped (incompatible with torch 2.6 and
     not needed by the five no-reference dimensions used here). `transformers`
     floats to a current version; `numpy` stays pinned at `1.26.4`.
  `detectron2` (a commented git dependency in `requirements.txt`) is **not**
  installed — it is only needed by VBench dimensions we do not run.

Verified: `import vbench` succeeds, `VBench_full_info.json` and
`vbench2_beta_i2v/` are locatable, and `run_vbench_eval_for_dir.py` imports
VBench successfully. Full end-to-end scoring was not run because no videos and
no scoring backbones are present yet (see the IMPORTANT note above).

---

## What produces the paper's quality numbers

The paper's video-quality evaluation has two complementary halves. **Frame
fidelity** (Track 1) quantifies how much visual content is *retained* relative
to a full-quality baseline: for each method/setting we point `evaluate_pairs.py`
at the baseline corpus and the corresponding test corpus, and report the mean
PSNR / SSIM / LPIPS (and FVD when enabled) from `summary.average` of the output
JSON — higher PSNR/SSIM and lower LPIPS/FVD mean the accelerated/cached output
stays closer to the baseline. **No-reference quality** (Track 2) reports the
intrinsic perceptual quality of a corpus *without* any baseline, via VBench's
subject/background consistency, motion smoothness, aesthetic quality and imaging
quality — averaged across all videos in a directory (the `<prefix>_result.json`
file). Together, Track 1 shows fidelity is preserved and Track 2 shows the
generated videos are independently high-quality; both are run per method/setting
over the matching directories of `.mp4`s.
