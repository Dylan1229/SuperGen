"""Minimal reproduction: the gain estimator k degenerates to exactly 1.0 during
consecutive cache hits, which blinds the error predictor.

Root cause (CachingCogVideoXTransformer3DModel):

  On a cache HIT, `forward` takes the early-return branch and writes
      previous_output <- raw_input + cache_residual
  reusing a residual that is NOT refreshed while skipping. So after one or more
  consecutive hits:

      prev_output      = raw_input_{t-1} + R
      prev_prev_output = raw_input_{t-2} + R        (same frozen R)

  and therefore

      output_change     = |prev_output - prev_prev_output|
                        = |raw_input_{t-1} - raw_input_{t-2}|
      prev_input_change = |prev_raw_input - prev_prev_raw_input|
                        = |raw_input_{t-1} - raw_input_{t-2}|      <- identical

      k = output_change / prev_input_change == 1.0

  k is supposed to estimate the local gain dO/dI (paper Eq. 6's g_c). Once it is
  pinned at 1.0 the predictor reduces to pred_change = raw_input_change /
  output_norm, losing the sensitivity term entirely. Because output_norm shrinks
  monotonically over the trajectory, pred_change grows only slowly, the
  accumulated error creeps under the threshold, and the cache keeps skipping --
  producing the long skip runs (up to 15 consecutive steps at tau=0.40) that
  collapse quality.

Confirmed on real logs (720p, tau=0.40, "a mountain range with a sky background"):
  k immediately after a cache HIT : 31/34 are exactly 1.000
  k immediately after a cache MISS:  0/5  are exactly 1.000

    python CogvideoI2V/modules/test_k_estimator_bug.py
"""
import sys

import torch


def simulate(consecutive_skips, refresh_output_on_skip):
    """Replay the state updates for a hit-run and report the resulting k values.

    refresh_output_on_skip=False reproduces current behaviour.
    True models the fix: keep the *computed* output as the reference for the gain
    estimate instead of the reconstructed one.
    """
    torch.manual_seed(0)
    shape = (1, 4, 16, 8, 10)

    # A latent trajectory that changes by a roughly constant amount per step.
    inputs = [torch.randn(shape)]
    for _ in range(consecutive_skips + 3):
        inputs.append(inputs[-1] + 0.02 * torch.randn(shape))

    # The true transform has gain != 1 so a correct estimator should not return 1.
    true_gain = 2.5
    def true_output(x):
        return x * true_gain

    residual = true_output(inputs[0]) - inputs[0]   # frozen at the last compute

    prev_raw, prev_prev_raw = inputs[1], inputs[0]
    prev_out = true_output(inputs[1])
    prev_prev_out = true_output(inputs[0])

    ks = []
    for t in range(2, 2 + consecutive_skips):
        raw = inputs[t]
        out_change = (prev_out - prev_prev_out).abs().mean()
        in_change = (prev_raw - prev_prev_raw).abs().mean()
        k = (out_change / in_change).item()
        ks.append(k)

        # --- state updates on a cache hit ---
        reconstructed = raw + residual
        prev_prev_raw, prev_raw = prev_raw, raw
        prev_prev_out = prev_out
        prev_out = true_output(raw) if refresh_output_on_skip else reconstructed

    return ks


def main():
    failures = []

    print("Current behaviour (previous_output := raw_input + frozen residual):")
    ks = simulate(consecutive_skips=8, refresh_output_on_skip=False)
    print("  k over a run of 8 skips:", " ".join(f"{k:.4f}" for k in ks))
    # The first two values still see one real output, so the degeneracy only
    # locks in from the third skip onward -- which is exactly when long skip runs
    # start doing damage.
    steady = ks[2:]
    pinned = sum(1 for k in steady if abs(k - 1.0) < 1e-4)
    print(f"  exactly 1.0 from the 3rd skip onward: {pinned}/{len(steady)}")
    if pinned < len(steady):
        failures.append("expected k pinned at 1.0 in steady state -- not reproduced")
    else:
        print("  -> REPRODUCED: the gain term is destroyed (true gain was 2.5)")

    print("\nWith the output reference kept from the last real computation:")
    ks_fixed = simulate(consecutive_skips=8, refresh_output_on_skip=True)
    print("  k over a run of 8 skips:", " ".join(f"{k:.4f}" for k in ks_fixed))
    near_true = sum(1 for k in ks_fixed if abs(k - 2.5) < 0.3)
    print(f"  within 0.3 of the true gain 2.5: {near_true}/{len(ks_fixed)}")
    if near_true == 0:
        failures.append("the modelled fix did not recover the true gain either")

    print()
    if failures:
        print("INCONCLUSIVE")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: bug reproduced, and the gain is recoverable when the output")
    print("      reference is not overwritten by the reconstructed value.")


if __name__ == "__main__":
    main()
