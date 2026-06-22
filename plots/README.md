# SuperGen paper figures

This directory regenerates the paper's figure PDFs from self-contained Jupyter
notebooks. **All figure data is hardcoded inline in the notebook cells** — the
notebooks read no external CSV/JSON/npy files, so they are fully portable and
regenerate the PDFs as-is once the plotting dependencies are installed.

To update numbers after re-running experiments, edit the inline Python arrays /
dicts in the relevant notebook cell (e.g. `cogvideo_e2e_data`, `*_latency`,
heatmap matrices) and re-run that notebook.

## Layout

```
plots/
├── plot_main.ipynb          # AUTHORITATIVE, newest (from plot-OSDI26.ipynb)
├── plot_legacy.ipynb        # older; also produces breakdown_stages + techniques_breakdown (from plot.ipynb)
├── plot_eval_repo.ipynb     # eval-repo variant (subset; from yimu-evaluate-supergen .../Plot_for_paper/plot.ipynb)
├── figures/                 # generated figure PDFs (committed output of the run below)
├── method_analysis/         # method-analysis scripts (require a profiled run; see below)
│   ├── plot_L1_noise_across_timestep.py
│   └── plot_L1_noise_tile_to_tile.py
└── README.md
```

## Plotting environment

A dedicated uv venv (Python 3.10) holds only the plotting deps — separate from
the model/eval venvs.

```bash
export UV_CACHE_DIR=/mnt/data/fanjiang/.cache/uv
/home/fanjiang/.local/bin/uv venv --python 3.10 /mnt/data/fanjiang/venvs/supergen-plots
/home/fanjiang/.local/bin/uv pip install --python /mnt/data/fanjiang/venvs/supergen-plots/bin/python \
    jupyter nbconvert ipykernel matplotlib numpy pandas seaborn
```

Pinned versions used to produce the committed figures:
Python 3.10.12, matplotlib 3.10.9, numpy 2.2.6, pandas 2.3.3, seaborn 0.13.2,
nbconvert 7.17.1, ipykernel 7.3.0.

## Regenerating the figures

Run from inside `plots/` so the notebooks' relative `savefig()` paths land in
this directory. The notebooks save PDFs to the *current working directory*, so
the move step below relocates them into `figures/`.

```bash
cd /mnt/data/fanjiang/repo/SuperGen/plots
export UV_CACHE_DIR=/mnt/data/fanjiang/.cache/uv
JUP=/mnt/data/fanjiang/venvs/supergen-plots/bin/jupyter

# Execute each notebook end-to-end (writes PDFs into the CWD = plots/)
$JUP nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output /tmp/exec_main.ipynb      plot_main.ipynb
$JUP nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output /tmp/exec_legacy.ipynb    plot_legacy.ipynb
$JUP nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output /tmp/exec_eval_repo.ipynb plot_eval_repo.ipynb

# Collect the generated PDFs into figures/
mkdir -p figures && mv *.pdf figures/
```

Note: `plot_main`, `plot_legacy`, and `plot_eval_repo` share several output
*filenames* (e.g. `e2e_performance_comparison.pdf`). When run in sequence the
last one wins for a shared name. The committed `figures/` were assembled so that
**`plot_main` (authoritative) provides every figure it produces**, plus the
three figures only `plot_legacy` produces (`breakdown_stages.pdf`,
`techniques_breakdown.pdf`, `tiles_effect_comparison_with_sizes.pdf`). If you
re-run with the simple `mv *.pdf figures/` above, the shared-name figures will
instead reflect whichever notebook ran last (`plot_eval_repo`); to reproduce the
authoritative set, run `plot_main` last (or run each notebook in its own temp
directory and copy the desired files).

## Figure map: PDF → producing notebook → paper experiment

| Figure PDF | Produced by | Paper figure / experiment |
|---|---|---|
| `e2e_performance_comparison.pdf` | plot_main, plot_legacy, plot_eval_repo | End-to-end speedup |
| `GPU_scalability_comparison.pdf` | plot_main, plot_legacy, plot_eval_repo | Latency vs. number of GPUs |
| `latency_comparison_4GPUs.pdf` | plot_main, plot_legacy, plot_eval_repo | Per-resolution latency with/without cache at 4 GPUs |
| `multimodel_multiGPU_latency_4k_comparison.pdf` | plot_main, plot_legacy, plot_eval_repo | Multi-model, multi-GPU 4K latency comparison |
| `cache_ablation_heatmaps.pdf` | plot_main, plot_legacy, plot_eval_repo | Cache ablation: threshold × scale grid |
| `tiles_effect_comparison.pdf` | plot_main, plot_legacy, plot_eval_repo | Quality-vs-latency as a function of number of tiles |
| `tiles_effect_with_tile_sizes.pdf` | plot_main | Quality-vs-latency vs. number of tiles (annotated with tile sizes) |
| `tiles_effect_comparison_with_sizes.pdf` | plot_legacy | Quality-vs-latency vs. number of tiles (legacy variant w/ tile sizes) |
| `techniques_breakdown.pdf` | plot_legacy | Cumulative optimization waterfall |
| `breakdown_stages.pdf` | plot_legacy | Stage 1 / Upscale / Stage 2 latency split |

Per-notebook output inventory (isolated run):

- **plot_main** (7 PDFs): `cache_ablation_heatmaps`, `e2e_performance_comparison`,
  `GPU_scalability_comparison`, `latency_comparison_4GPUs`,
  `multimodel_multiGPU_latency_4k_comparison`, `tiles_effect_comparison`,
  `tiles_effect_with_tile_sizes`.
- **plot_legacy** (9 PDFs): all of plot_main's *except* `tiles_effect_with_tile_sizes`,
  plus the three legacy-only figures `breakdown_stages`, `techniques_breakdown`,
  `tiles_effect_comparison_with_sizes`.
- **plot_eval_repo** (6 PDFs): `cache_ablation_heatmaps`, `e2e_performance_comparison`,
  `GPU_scalability_comparison`, `latency_comparison_4GPUs`,
  `multimodel_multiGPU_latency_4k_comparison`, `tiles_effect_comparison`.

> Note: although `techniques_breakdown` is associated with the OSDI26 figure set,
> only `plot_legacy` actually emits `techniques_breakdown.pdf`; `plot_main` does
> not have a `savefig` for it.

## Minimal fixes applied during restoration

Both notebooks below were fixed *minimally* so they execute headlessly; only
non-plotting / placeholder lines were touched, no figure data was altered.

- **plot_main.ipynb** — the final cell (illustrative two-stage-generation
  *pseudocode*, no `savefig`) executed `model = original_pipeline`, where
  `original_pipeline` is undefined, raising `NameError`. That single executable
  line was commented out; the rest of the pseudocode (function `def`s, never
  called) is left intact for reference.
- **plot_eval_repo.ipynb** — the E2E-latency cell had an unfilled placeholder
  `'W/ Parallelism + Cache': XXX` for CogVideo 2K, raising `NameError`. It was
  filled with `280`, the authoritative value used by the same datapoint in both
  `plot_main` and `plot_legacy` (all surrounding values match exactly).

## method_analysis/ scripts (NOT run here)

`plot_L1_noise_across_timestep.py` and `plot_L1_noise_tile_to_tile.py` produce
the method-analysis noise plots. **They are not part of the inline-data figure
pipeline**: they read profile JSONs that are produced by a *profiled generation
run* and are **not present in this repo**:

- `Across_timesteps_l1_distances_50steps.json`
- `tile_to_tile_l1_distances_50steps.json`

The previously hardcoded `/home/ubuntu/Yimu/...` paths were replaced with a
configurable `NOISE_PROFILE_DIR` constant at the top of each script
(default `/mnt/data/fanjiang/repo/SuperGen/plots/noise_prediction_profile/`,
overridable via the `NOISE_PROFILE_DIR` environment variable). To run them, first
perform a profiled generation to emit the two JSONs into `NOISE_PROFILE_DIR`,
then:

```bash
cd /mnt/data/fanjiang/repo/SuperGen/plots/method_analysis
export NOISE_PROFILE_DIR=/path/to/noise_prediction_profile   # optional override
/mnt/data/fanjiang/venvs/supergen-plots/bin/python plot_L1_noise_across_timestep.py
/mnt/data/fanjiang/venvs/supergen-plots/bin/python plot_L1_noise_tile_to_tile.py
```

Outputs (PNGs) are written under `${NOISE_PROFILE_DIR}/plots/`.
