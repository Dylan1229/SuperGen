"""Equivalence test: our ported TeaCache gate vs the upstream decision sequence.

Runs on CPU with a tiny transformer config -- no checkpoints, no GPU. Verifies:

1. The gate reproduces upstream's skip/recompute sequence exactly, given the same
   timestep embeddings, thresholds, and coefficients.
2. Higher thresholds skip more (monotonicity), matching upstream's documented
   0.1 < 0.2 < 0.3 speedup ordering.
3. First and last steps always recompute.
4. In `token` storage mode, a window shift invalidates the cached residual.

    python CogvideoI2V/modules/test_teacache_gate.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.teacache_transformer_3d import (  # noqa: E402
    COEFFICIENTS_DICT,
    DEFAULT_COEFFICIENT_KEY,
    TeaCacheCogVideoXTransformer3DModel,
)

NUM_STEPS = 50
TINY_CONFIG = dict(
    num_attention_heads=2,
    attention_head_dim=8,
    in_channels=32,
    out_channels=16,
    time_embed_dim=32,
    ofs_embed_dim=32,
    text_embed_dim=64,
    num_layers=1,
    sample_width=10,
    sample_height=6,
    sample_frames=9,
    patch_size=2,
    patch_size_t=2,
    temporal_compression_ratio=4,
    max_text_seq_length=8,
    use_rotary_positional_embeddings=True,
    use_learned_positional_embeddings=False,
)


class _FakeDistManager:
    rank = 0

    def tile_is_skipped(self, tile_idx):
        return False


def upstream_decisions(embs, thresh, coefficients, num_steps):
    """Verbatim transcription of the upstream gate (teacache_sample_video.py)."""
    decisions = []
    accumulated = 0.0
    previous = None
    cnt = 0
    for emb in embs:
        if cnt == 0 or cnt == num_steps - 1:
            should_calc = True
            accumulated = 0.0
        else:
            rescale_func = np.poly1d(coefficients)
            accumulated += rescale_func(
                ((emb - previous).abs().mean() / previous.abs().mean()).cpu().item()
            )
            if accumulated < thresh:
                should_calc = False
            else:
                should_calc = True
                accumulated = 0.0
        previous = emb
        cnt += 1
        if cnt == num_steps:
            cnt = 0
        decisions.append(should_calc)
    return decisions


def ported_decisions(model, embs, thresh, storage="latent", window_position=(0, 6, 0, 10)):
    """Drive the real check_skippable, feeding embeddings via a stubbed projector."""
    model.setup_teacache(rel_l1_thresh=thresh, storage=storage)
    model.num_steps = NUM_STEPS
    model.enable_cache = True

    hidden = torch.zeros(1, 4, 32, 6, 10)
    decisions = []
    for step, emb in enumerate(embs):
        model._timestep_embedding = lambda *a, _e=emb, **k: _e
        skipped, _ = model.check_skippable(
            step_index=step,
            tile_index=0,
            hidden_states=hidden,
            is_non_shifting_step=True,
            window_position=window_position,
            timestep=torch.tensor([step], dtype=torch.long),
        )
        decisions.append(not skipped)
    return decisions


def main():
    torch.manual_seed(0)
    coefficients = COEFFICIENTS_DICT[DEFAULT_COEFFICIENT_KEY]

    model = TeaCacheCogVideoXTransformer3DModel(**TINY_CONFIG)
    model.dist_manager = _FakeDistManager()

    # A realistic embedding trajectory: smooth drift plus small noise, so the
    # relative-L1 signal varies across steps the way it does in a real run.
    base = torch.linspace(1.0, 0.05, NUM_STEPS).unsqueeze(1).repeat(1, 32)
    embs = [base[i] + 0.01 * torch.randn(32) for i in range(NUM_STEPS)]

    failures = []

    # --- 1. exact equivalence with upstream, across all documented thresholds ---
    for thresh in (0.1, 0.2, 0.3):
        want = upstream_decisions(embs, thresh, coefficients, NUM_STEPS)
        got = ported_decisions(model, embs, thresh)
        if want != got:
            diff = [i for i, (a, b) in enumerate(zip(want, got)) if a != b]
            failures.append(f"thresh={thresh}: decisions differ at steps {diff}")
        else:
            n_skip = sum(1 for d in got if not d)
            print(f"  thresh={thresh}: identical to upstream, {n_skip}/{NUM_STEPS} steps skipped")

    # --- 2. monotonicity: higher threshold => at least as many skips ---
    skips = []
    for thresh in (0.1, 0.2, 0.3):
        got = ported_decisions(model, embs, thresh)
        skips.append(sum(1 for d in got if not d))
    if not (skips[0] <= skips[1] <= skips[2]):
        failures.append(f"skip counts not monotonic in threshold: {skips}")
    else:
        print(f"  monotonic in threshold: skips={skips} for 0.1/0.2/0.3")

    # --- 3. edge steps always recompute ---
    got = ported_decisions(model, embs, 0.3)
    if not got[0]:
        failures.append("first step was skipped; upstream always computes it")
    if not got[-1]:
        failures.append("last step was skipped; upstream always computes it")
    if got[0] and got[-1]:
        print("  first and last steps always recomputed")

    # --- 4. token storage: a window shift must invalidate the residual ---
    model.setup_teacache(rel_l1_thresh=0.3, storage="token")
    model.num_steps = NUM_STEPS
    hidden = torch.zeros(1, 4, 32, 6, 10)
    # Seed a residual as if a real forward had cached it at this window.
    model._tc_residual[0] = torch.zeros(1)
    model._tc_residual_encoder[0] = torch.zeros(1)
    model._tc_residual_position[0] = (0, 6, 0, 10)
    # Same window: the gate may skip. Shifted window: it must not.
    shifted_forced_recompute = None
    for step, emb in enumerate(embs[:12]):
        model._timestep_embedding = lambda *a, _e=emb, **k: _e
        pos = (0, 6, 0, 10) if step < 6 else (1, 7, 2, 12)
        skipped, _ = model.check_skippable(
            step_index=step,
            tile_index=0,
            hidden_states=hidden,
            is_non_shifting_step=True,
            window_position=pos,
            timestep=torch.tensor([step], dtype=torch.long),
        )
        if step >= 6 and skipped:
            shifted_forced_recompute = False
    if shifted_forced_recompute is False:
        failures.append("token storage reused a residual across a window shift")
    elif model.teacache_stats["shift_invalidated"] == 0:
        failures.append("token storage never registered a shift invalidation (test ineffective)")
    else:
        print(
            f"  token storage invalidates on shift "
            f"({model.teacache_stats['shift_invalidated']} invalidations)"
        )

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: ported gate matches upstream TeaCache")


if __name__ == "__main__":
    main()
