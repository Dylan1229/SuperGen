"""Verify Hunyuan canvas-absolute RoPE. CPU, no checkpoints, no weights.

  1. EQUIVALENCE: offset (0,0,0) in 'extend' mode must reproduce the legacy call
     `get_nd_rotary_pos_embed(rope_dim_list, rope_sizes)` bit-for-bit, so the fix
     is a strict generalisation of current behaviour.
  2. BUG: the legacy call is position-blind -- identical for every tile.
  3. FIX: different offsets give different RoPE.
  4. CONSISTENCY: a tile's RoPE equals the matching slice of a whole-canvas field.
  5. NTK: mode='ntk' stays finite, differs from extend, and leaves the temporal
     axis untouched (CineScale sets alpha[0]=1).
  6. WRAP + CACHE.

    python hyvideo/modules/test_tiled_rope.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HYI2V = os.path.dirname(os.path.dirname(_HERE))              # HunyuanVideoI2V/
sys.path.insert(0, _HYI2V)
sys.path.insert(0, os.path.dirname(_HYI2V))                   # SuperGen/, for utils.*

# Import the two modules directly rather than through hyvideo.modules.__init__,
# which pulls in the whole transformer stack (and DeepSpeed) just to reach RoPE.
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_posemb = _load("_posemb_layers", os.path.join(_HERE, "posemb_layers.py"))
get_nd_rotary_pos_embed = _posemb.get_nd_rotary_pos_embed
# tiled_rope does `from .posemb_layers import ...`, so give it a package-free view
sys.modules["hyvideo.modules.posemb_layers"] = _posemb
_tiled = _load("_tiled_rope", os.path.join(_HERE, "tiled_rope.py"))
HunyuanTiledRope = _tiled.HunyuanTiledRope

# HunyuanVideo: hidden 3072 / 24 heads -> head_dim 128, rope_dim_list [16,56,56]
ROPE_DIM = [16, 56, 56]
TILE = (8, 45, 80)        # one 720p window in patches (frames, h, w)
CANVAS_2K = (8, 90, 160)  # 2x2 tiles
CANVAS_4K = (8, 135, 240) # 3x3 tiles
THETA = 256.0


def main():
    failures = []

    # --- 1. equivalence at the origin ---
    ref = get_nd_rotary_pos_embed(ROPE_DIM, TILE, theta=THETA, use_real=True)
    rope = HunyuanTiledRope(ROPE_DIM, TILE, CANVAS_4K, mode="extend", theta=THETA)
    got = rope.get((0, 0, 0))
    same = torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    print(f"[1] equivalence with legacy call at offset (0,0,0): {same}")
    print(f"    shapes ref={tuple(ref[0].shape)} got={tuple(got[0].shape)}")
    if not same:
        failures.append(f"not identical at origin (max diff "
                        f"{(ref[0]-got[0]).abs().max().item():.3e})")

    # --- 2. legacy is position-blind ---
    legacy = [get_nd_rotary_pos_embed(ROPE_DIM, TILE, theta=THETA, use_real=True)
              for _ in range(3)]
    blind = all(torch.equal(legacy[0][0], l[0]) for l in legacy[1:])
    print(f"[2] legacy call identical across tiles (the bug): {blind}")
    if not blind:
        failures.append("could not reproduce the position-blind baseline")

    # --- 3. the fix distinguishes tiles ---
    offs = [(0, 0, 0), (0, 45, 80), (0, 90, 160)]
    outs = [rope.get(o) for o in offs]
    d01 = (outs[0][0] - outs[1][0]).abs().max().item()
    d02 = (outs[0][0] - outs[2][0]).abs().max().item()
    print(f"[3] tile(0,0) vs (45,80): {d01:.4f}; vs (90,160): {d02:.4f}")
    if d01 < 1e-6 or d02 < 1e-6:
        failures.append("offsets still produce identical RoPE")
    else:
        print("    -> tiles positionally distinguishable")

    # --- 4. a tile equals the matching slice of the canvas field ---
    full = get_nd_rotary_pos_embed(ROPE_DIM, CANVAS_4K, theta=THETA, use_real=True)
    F_, H_, W_ = CANVAS_4K
    tf, th, tw = TILE
    oh, ow = 45, 80
    full_cos = full[0].view(F_, H_, W_, -1)
    tile_cos = rope.get((0, oh, ow))[0].view(tf, th, tw, -1)
    sub = full_cos[:tf, oh:oh + th, ow:ow + tw, :]
    consistent = torch.allclose(sub, tile_cos, atol=1e-6)
    print(f"[4] tile == slice of whole-canvas field: {consistent}")
    if not consistent:
        failures.append(f"tile is not a canvas slice (max diff "
                        f"{(sub-tile_cos).abs().max().item():.3e})")

    # --- 5. ntk mode ---
    ntk = HunyuanTiledRope(ROPE_DIM, TILE, CANVAS_4K, trained_sizes=TILE,
                           mode="ntk", theta=THETA)
    n0 = ntk.get((0, 0, 0))
    finite = bool(torch.isfinite(n0[0]).all() and torch.isfinite(n0[1]).all())
    shape_ok = n0[0].shape == ref[0].shape
    differs = not torch.allclose(n0[0], ref[0], atol=1e-6)
    # temporal axis must be untouched: first ROPE_DIM[0]//2 cos columns
    t_cols = ROPE_DIM[0] // 2
    temporal_same = torch.allclose(n0[0][:, :t_cols], ref[0][:, :t_cols], atol=1e-6)
    print(f"[5] ntk: finite={finite} shape_ok={shape_ok} differs_from_extend={differs}")
    print(f"    temporal axis unchanged by ntk: {temporal_same}")
    if not (finite and shape_ok):
        failures.append("ntk mode produced bad output")
    if not differs:
        failures.append("ntk mode identical to extend -- theta scaling had no effect")
    if not temporal_same:
        failures.append("ntk mode altered the temporal axis (should use alpha[0]=1)")

    # --- 6. wrap + cache ---
    w = rope.get((0, 130, 235))
    wrap_ok = bool(torch.isfinite(w[0]).all())
    a = rope.get((0, 45, 80))
    b = rope.get((0, 45, 80))
    cached = a[0] is b[0]
    print(f"[6] wrapped offset finite: {wrap_ok}; cache hit returns same object: {cached}")
    print(f"    cache: {rope.stats()}")
    if not wrap_ok:
        failures.append("wrapped offset produced non-finite RoPE")
    if not cached:
        failures.append("cache did not return the same object")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: Hunyuan canvas-absolute RoPE generalises the legacy call")


if __name__ == "__main__":
    main()
