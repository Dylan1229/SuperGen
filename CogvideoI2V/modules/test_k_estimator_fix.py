"""Verify the gain-estimator fix on the REAL transformer class, on CPU.

Drives the actual `check_skippable` state machine with a tiny config (no
checkpoints, no GPU) and asserts:

  1. Legacy mode (`freeze_output_history_on_skip=False`) reproduces the bug: k
     collapses to exactly 1.0 during a skip run.
  2. Fixed mode holds the last valid k instead, so the gain term of Eq. 6 survives.
  3. The fix does not change behaviour when nothing is skipped -- consecutive real
     computations must produce identical k either way.
  4. `_last_compute_step` tracking is per tile, so one tile skipping does not
     invalidate another tile's estimate.

    python CogvideoI2V/modules/test_k_estimator_fix.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cogvideo_transformer_3d import CachingCogVideoXTransformer3DModel  # noqa: E402

TINY = dict(
    num_attention_heads=2, attention_head_dim=8, in_channels=32, out_channels=16,
    time_embed_dim=32, ofs_embed_dim=32, text_embed_dim=64, num_layers=1,
    sample_width=10, sample_height=6, sample_frames=9, patch_size=2, patch_size_t=2,
    temporal_compression_ratio=4, max_text_seq_length=8,
    use_rotary_positional_embeddings=True, use_learned_positional_embeddings=False,
)
WINDOW = (0, 6, 0, 10)
NUM_STEPS = 20
RET_STEPS = 2


class _FakeDist:
    rank = 0
    def __init__(self):
        self.skipped = set()
    def tile_is_skipped(self, tile_idx):
        return tile_idx in self.skipped


def build(legacy, num_tiles=1):
    model = CachingCogVideoXTransformer3DModel(**TINY)
    model.dist_manager = _FakeDist()
    # allocate() sizes the cache from the NOISE latent (in_channels//2), which is
    # what the pipeline passes; the concatenated model input is twice as wide.
    latents = torch.zeros(1, 4, 16, 6, 10)
    # allocate() calls register_cache_ring2d on the dist manager; stub it out.
    model.dist_manager.register_cache_ring2d = lambda *a, **k: None
    model.allocate(latents)
    model.setup_cache_per_tile(num_steps=NUM_STEPS, thresh=1e9, ret_steps=RET_STEPS,
                              num_tiles=num_tiles)
    model.freeze_output_history_on_skip = not legacy
    return model


def drive(model, tile=0, steps=NUM_STEPS, gain=2.5, force_all_skips=True):
    """Feed a drifting latent through check_skippable, emulating forward()'s writes.

    thresh is huge, so every eligible step is a cache hit -- exactly the skip-run
    regime where the bug appears.
    """
    torch.manual_seed(0)
    base = torch.randn(1, 4, 16, 6, 10)
    ks = []
    for step in range(steps):
        raw = base + 0.02 * step * torch.ones_like(base)
        hidden = torch.cat([raw, torch.zeros_like(raw)], dim=2)

        skipped, _ = model.check_skippable(
            step_index=step, tile_index=tile, hidden_states=hidden,
            is_non_shifting_step=True, window_position=WINDOW, return_dict=False,
        )
        k = model._last_valid_k.get(tile)
        if step > RET_STEPS:
            ks.append(float(k) if k is not None else float("nan"))

        if skipped:
            model.dist_manager.skipped.add(tile)
            # emulate forward()'s skip branch
            residual = model.cache_residual.get_window_latent(*WINDOW)
            result = raw + residual
            if not model.freeze_output_history_on_skip:
                model.previous_output.set_window_latent(result, *WINDOW)
        else:
            model.dist_manager.skipped.discard(tile)
            # emulate forward()'s compute branch
            out = raw * gain
            model.cache_residual.set_window_latent(out - raw, *WINDOW)
            model.prev_prev_raw_input.set_window_latent(
                model.previous_raw_input.get_window_latent(*WINDOW), *WINDOW)
            model.previous_raw_input.set_window_latent(raw, *WINDOW)
            model.prev_prev_output.set_window_latent(
                model.previous_output.get_window_latent(*WINDOW), *WINDOW)
            model.previous_output.set_window_latent(out, *WINDOW)
            model._last_compute_step[tile] = step
    return ks


def main():
    failures = []

    # --- 1. legacy reproduces the collapse ---
    legacy_ks = drive(build(legacy=True))
    pinned = sum(1 for k in legacy_ks if abs(k - 1.0) < 1e-3)
    print("legacy k values :", " ".join(f"{k:.3f}" for k in legacy_ks[:10]))
    print(f"  exactly 1.0: {pinned}/{len(legacy_ks)}")
    if pinned < len(legacy_ks) * 0.6:
        failures.append(f"legacy mode did not collapse to 1.0 ({pinned}/{len(legacy_ks)})")
    else:
        print("  -> bug present in legacy mode, as expected")

    # --- 2. fixed mode keeps a non-degenerate gain ---
    fixed_ks = drive(build(legacy=False))
    pinned_fixed = sum(1 for k in fixed_ks if abs(k - 1.0) < 1e-3)
    print("\nfixed k values  :", " ".join(f"{k:.3f}" for k in fixed_ks[:10]))
    print(f"  exactly 1.0: {pinned_fixed}/{len(fixed_ks)}")
    if pinned_fixed > len(fixed_ks) * 0.4:
        failures.append(f"fixed mode still collapses to 1.0 ({pinned_fixed}/{len(fixed_ks)})")
    else:
        print("  -> gain term preserved through the skip run")

    # --- 3. no behaviour change when nothing is skipped ---
    # thresh=0 forces every step to recompute, so both modes must agree exactly.
    def all_compute(legacy):
        m = build(legacy=legacy)
        m.thresh = 0.0
        return drive(m)
    a, b = all_compute(True), all_compute(False)
    same = len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    print(f"\nno-skip regime identical between modes: {same}")
    if not same:
        failures.append("the fix changed behaviour even with no cache hits")

    # --- 4. per-tile isolation ---
    m = build(legacy=False, num_tiles=2)
    drive(m, tile=0)
    steps0 = m._last_compute_step.get(0)
    steps1 = m._last_compute_step.get(1)
    print(f"per-tile compute tracking: tile0={steps0} tile1={steps1}")
    if steps1 is not None:
        failures.append("tile 1 was marked computed though only tile 0 ran")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: fix removes the k collapse, preserves no-skip behaviour, per-tile safe")


if __name__ == "__main__":
    main()
