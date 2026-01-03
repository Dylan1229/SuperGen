### Prompt selection overview

- **Source**: prompts were selected from `vbench2_i2v_full_info.json`.
- **Goal**: create subsets of 10, 20, and 30 prompts with near-equal totals across all 10 dimensions. Exact equality across dimensions is infeasible because items are tagged in only three combinations:
  - **camera_motion** only
  - **subject group**: `i2v_subject`, `subject_consistency`, `motion_smoothness`, `dynamic_degree`, plus `aesthetic_quality`, `imaging_quality`, `temporal_flickering`
  - **background group**: `i2v_background`, `background_consistency`, plus `aesthetic_quality`, `imaging_quality`, `temporal_flickering`
- **Method**: partition items by these combinations, pick counts that minimize the spread across dimension totals, sample randomly without replacement within each group, then shuffle. Because of overlap in tags, quality-related dimensions may appear more often.
- **Files**: `vbench2_i2v_10.json`, `vbench2_i2v_20.json`, `vbench2_i2v_30.json`.

### Per-dimension counts

Counts below refer to each subset file (the full dataset is not included).

- **vbench2_i2v_10.json** (10 items, 10 unique prompts)
  - **camera_motion**: 5
  - **i2v_background**: 2
  - **background_consistency**: 2
  - **aesthetic_quality**: 5
  - **imaging_quality**: 5
  - **temporal_flickering**: 5
  - **i2v_subject**: 3
  - **subject_consistency**: 3
  - **motion_smoothness**: 3
  - **dynamic_degree**: 3

- **vbench2_i2v_20.json** (20 items, 20 unique prompts)
  - **camera_motion**: 10
  - **i2v_background**: 5
  - **background_consistency**: 5
  - **aesthetic_quality**: 10
  - **imaging_quality**: 10
  - **temporal_flickering**: 10
  - **i2v_subject**: 5
  - **subject_consistency**: 5
  - **motion_smoothness**: 5
  - **dynamic_degree**: 5

- **vbench2_i2v_30.json** (30 items, 30 unique prompts)
  - **camera_motion**: 15
  - **i2v_background**: 7
  - **background_consistency**: 7
  - **aesthetic_quality**: 15
  - **imaging_quality**: 15
  - **temporal_flickering**: 15
  - **i2v_subject**: 8
  - **subject_consistency**: 8
  - **motion_smoothness**: 8
  - **dynamic_degree**: 8


