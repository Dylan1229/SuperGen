"""Verify the Wan canvas-absolute RoPE against upstream `rope_apply`. CPU, no weights.

  1. EQUIVALENCE: offset=(0,0,0), mode='extend' must match upstream bit-for-bit.
  2. BUG: upstream gives identical output for two different tile positions.
  3. FIX: offset-aware version distinguishes them.
  4. UNITARITY: RoPE must stay a pure rotation (|freq|==1) in both modes, else the
     complex multiply changes token magnitudes. This is the failure mode a naive
     linear interpolation would introduce.
  5. WRAP: shifted windows past the canvas edge stay in range and finite.

    python WanI2V/test_global_rope_wan.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_rope_wan import _slice_axis, rope_apply_at_offset  # noqa: E402


# --- upstream implementation, transcribed verbatim from wan/modules/model.py ----
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply_upstream(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


HEAD_DIM = 128           # Wan2.1 14B: dim 5120 / 40 heads
NUM_HEADS = 4            # small for the test
F, H, W = 3, 8, 12       # one tile, in patches
CANVAS = (3, 24, 36)     # 3x3 tiles
TRAINED = (3, 16, 28)


def build_freqs():
    c = HEAD_DIM // 2
    d_t, d_h, d_w = c - 2 * (c // 3), c // 3, c // 3
    return torch.cat([
        rope_params(1024, d_t * 2),
        rope_params(1024, d_h * 2),
        rope_params(1024, d_w * 2),
    ], dim=1)


def main():
    torch.manual_seed(0)
    failures = []
    freqs = build_freqs()
    grid = torch.tensor([[F, H, W]])
    x = torch.randn(1, F * H * W + 5, NUM_HEADS, HEAD_DIM)

    # --- 1. equivalence at the origin ---
    ref = rope_apply_upstream(x, grid, freqs)
    got = rope_apply_at_offset(x, grid, freqs, offsets=[(0, 0, 0)], mode="extend")
    same = torch.equal(ref, got)
    md = (ref - got).abs().max().item()
    print(f"[1] equivalence with upstream at offset (0,0,0): {same} (max diff {md:.3e})")
    if not same:
        failures.append(f"not identical to upstream at the origin (max diff {md:.3e})")

    # --- 2. upstream cannot distinguish tiles ---
    up_a = rope_apply_upstream(x, grid, freqs)
    up_b = rope_apply_upstream(x, grid, freqs)   # a "different tile", same call
    print(f"[2] upstream output identical for two tile positions: {torch.equal(up_a, up_b)}")
    if not torch.equal(up_a, up_b):
        failures.append("could not reproduce the upstream position-blindness")

    # --- 3. the fix distinguishes them ---
    t0 = rope_apply_at_offset(x, grid, freqs, offsets=[(0, 0, 0)], canvas=CANVAS,
                              mode="extend")
    t4 = rope_apply_at_offset(x, grid, freqs, offsets=[(0, 8, 12)], canvas=CANVAS,
                              mode="extend")
    t8 = rope_apply_at_offset(x, grid, freqs, offsets=[(0, 16, 24)], canvas=CANVAS,
                              mode="extend")
    d04 = (t0 - t4).abs().max().item()
    d08 = (t0 - t8).abs().max().item()
    print(f"[3] tile(0,0) vs tile(8,12): {d04:.4f}; vs tile(16,24): {d08:.4f}")
    if d04 < 1e-6 or d08 < 1e-6:
        failures.append("offset-aware RoPE still identical across tiles")
    else:
        print("    -> tiles positionally distinguishable")

    # --- 4. unitarity: RoPE must remain a pure rotation ---
    c = HEAD_DIM // 2
    parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    for mode in ("extend", "interp"):
        sl = _slice_axis(parts[1], 8, H, CANVAS[1], mode, TRAINED[1])
        mag = sl.abs()
        ok = bool(torch.allclose(mag, torch.ones_like(mag), atol=1e-9))
        print(f"[4] mode={mode:6s} |freq| == 1: {ok} "
              f"(min {mag.min():.9f} max {mag.max():.9f})")
        if not ok:
            failures.append(f"mode={mode} produced non-unit RoPE magnitudes "
                            f"(min {mag.min():.6f}, max {mag.max():.6f})")

    # --- 4b. interp really compresses positions ---
    ext = _slice_axis(parts[1], 16, H, CANVAS[1], "extend", TRAINED[1])
    itp = _slice_axis(parts[1], 16, H, CANVAS[1], "interp", TRAINED[1])
    ext0 = _slice_axis(parts[1], 0, H, CANVAS[1], "extend", TRAINED[1])
    itp0 = _slice_axis(parts[1], 0, H, CANVAS[1], "interp", TRAINED[1])
    s_ext = (ext - ext0).abs().mean().item()
    s_itp = (itp - itp0).abs().mean().item()
    print(f"[4b] far-tile displacement: extend={s_ext:.4f} interp={s_itp:.4f}")
    if not s_itp < s_ext:
        failures.append("interp did not compress positions relative to extend")

    # --- 5. wrap past the canvas edge ---
    wrapped = rope_apply_at_offset(x, grid, freqs, offsets=[(0, 22, 33)],
                                   canvas=CANVAS, mode="extend")
    finite = bool(torch.isfinite(wrapped).all())
    print(f"[5] window crossing the canvas edge is finite: {finite}")
    if not finite:
        failures.append("wrapped window produced non-finite output")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: Wan canvas-absolute RoPE generalises upstream and stays unitary")


def test_ntk():
    """NTK mode must scale the frequency basis without breaking RoPE's invariants.

    NTK acts on theta, not on positions, so it cannot be a reindex of the existing
    table -- the table is recomputed. Three things must hold, and each has a silent
    failure mode:

      * unit modulus: RoPE is applied as a complex multiply, so a table that is not a
        pure rotation rescales the activations instead of rotating them.
      * alpha == 1 is the identity: otherwise 720p, where canvas == trained, would
        quietly stop matching the base model.
      * canvas == trained reproduces upstream exactly: the guard that the new mode is
        a generalisation rather than a different algorithm.
    """
    import os
    import sys
    import torch
    # main() sets these up; do it here too so this test is runnable on its own.
    for pth in (os.path.dirname(os.path.abspath(__file__)),
                os.environ.get("WAN_REPO", "/home/ubuntu/repo/Wan2.1")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    from global_rope_wan import _ntk_alpha, _ntk_table, rope_apply_at_offset
    from wan.modules.model import rope_params, rope_apply

    # 4K: latent 270x480 -> patch grid 135x240 against a trained 45x80.
    assert _ntk_alpha(135, 45) == 9.0, _ntk_alpha(135, 45)
    assert _ntk_alpha(240, 80) == 9.0
    assert _ntk_alpha(45, 45) == 1.0, "no scaling when the canvas is the trained size"

    table = rope_params(1024, 128)
    scaled = _ntk_table(table, _ntk_alpha(135, 45))
    assert torch.allclose(scaled.abs(), torch.ones_like(scaled.abs()), atol=1e-6), \
        "NTK table is not a pure rotation"
    assert (scaled - table).abs().max() > 1e-3, "NTK table is indistinguishable from base"
    assert torch.equal(_ntk_table(table, 1.0), table), "alpha=1 must be the identity"

    x = torch.randn(1, 11 * 45 * 80, 8, 128)
    gs = torch.tensor([[11, 45, 80]])
    ref = rope_apply(x.clone(), gs, table)
    got = rope_apply_at_offset(x.clone(), gs, table, offsets=[(0, 0, 0)],
                               canvas=(11, 45, 80), trained=(11, 45, 80), mode="ntk")
    err = (ref - got).abs().max().item()
    assert err == 0.0, f"ntk at canvas==trained must match upstream exactly, got {err}"
    print("PASS: Wan NTK RoPE scales theta, stays unitary, and generalises upstream")


    test_ntk()


if __name__ == "__main__":
    main()
    test_ntk()
