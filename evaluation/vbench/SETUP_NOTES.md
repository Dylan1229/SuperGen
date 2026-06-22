# VBench evaluation — setup notes (new cluster)

VBench scoring (`run_vbench_eval_for_dir.py`) was validated end-to-end on this
cluster (2026-06-22). Two fixes were needed beyond the earlier import-level check:

1. **`setuptools < 81` is required.** setuptools 81 removed `pkg_resources`,
   which VBench's `background_consistency` dimension imports; with setuptools 82
   that dimension fails with `No module named 'pkg_resources'`. The venv is pinned:
   ```bash
   uv pip install --python /mnt/data/fanjiang/venvs/supergen-vbench/bin/python "setuptools<81"
   ```
   (currently 80.10.2). Re-apply if the venv is rebuilt or setuptools is upgraded.

2. **Output-path fix** (already applied in `run_vbench_eval_for_dir.py`):
   `build_prefix_from_exp_dir` previously kept the filesystem root anchor (`/`)
   when the input dir had no `exp_result` path segment, producing an *absolute*
   result filename and crashing with `PermissionError: '/..._result.json'`. The
   anchor is now excluded so results are written inside the scored directory.

## Backbones (downloaded on first use)
VBench fetches its scoring backbones into `$VBENCH_CACHE_DIR`
(default `/mnt/data/fanjiang/.cache/vbench`): DINO (subject_consistency),
CLIP (background_consistency), AMT (motion_smoothness), LAION aesthetic predictor
(aesthetic_quality), MUSIQ (imaging_quality). One-time internet access required.

## Validated run (two 2K smoke videos)

| dimension | CogVideoX-2K | HunyuanVideo-2K |
|---|---|---|
| subject_consistency | 0.965 | 0.993 |
| background_consistency | 0.904 | 0.988 |
| motion_smoothness | 0.974 | 0.997 |
| aesthetic_quality | 0.567 | 0.616 |
| imaging_quality (0–100) | 77.74 | 79.27 |

Run command:
```bash
CUDA_VISIBLE_DEVICES=0 VBENCH_CACHE_DIR=/mnt/data/fanjiang/.cache/vbench \
  /mnt/data/fanjiang/venvs/supergen-vbench/bin/python \
  evaluation/vbench/run_vbench_eval_for_dir.py <dir-of-mp4s>
```

Per-video and per-directory averages are written as `*_result.json` inside the
scored directory.
