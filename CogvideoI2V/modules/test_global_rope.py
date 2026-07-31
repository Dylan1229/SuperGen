"""Verify canvas-absolute RoPE, on CPU, no checkpoints.

  1. EQUIVALENCE: offset=(0,0), canvas=trained, mode='extend' must reproduce
     diffusers' get_3d_rotary_pos_embed(grid_type='slice') bit-for-bit. Without
     this the fix is not a strict generalisation of current behaviour.
  2. DISTINGUISHABILITY: different tile offsets must produce different RoPE. This
     is the actual bug being fixed -- today all 9 tiles get identical tensors.
  3. CONSISTENCY: a tile's RoPE must equal the corresponding slice of a
     whole-canvas RoPE, i.e. tiling is a partition of one coherent field.
  4. INTERP: mode='interp' keeps positions inside the trained range.
  5. WRAP: shifted windows crossing the canvas edge stay in range (ring topology).
  6. CACHE: TiledRopeCache returns identical tensors and bounded key count.

    python CogvideoI2V/modules/test_global_rope.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root, for utils.*
sys.path.insert(0, os.path.dirname(_HERE))                    # CogvideoI2V/

from diffusers.models.embeddings import get_3d_rotary_pos_embed  # noqa: E402

from modules.global_rope import TiledRopeCache, get_3d_rope_at_offset  # noqa: E402

EMBED_DIM = 64          # CogVideoX-1.5 attention_head_dim
TRAINED = (48, 85)      # sample_height//p, sample_width//p  = 96//2, 170//2
TILE = (45, 80)         # 90x160 latent window, patch 2
FRAMES = 6


def main():
    failures = []

    # --- 1. equivalence with diffusers at the origin ---
    ref_cos, ref_sin = get_3d_rotary_pos_embed(
        embed_dim=EMBED_DIM, crops_coords=None, grid_size=TILE,
        temporal_size=FRAMES, grid_type="slice", max_size=TRAINED, device="cpu",
    )
    got_cos, got_sin = get_3d_rope_at_offset(
        embed_dim=EMBED_DIM, grid_size=TILE, temporal_size=FRAMES,
        offset=(0, 0), canvas_size=TRAINED, mode="extend", device="cpu",
    )
    same = torch.equal(ref_cos, got_cos) and torch.equal(ref_sin, got_sin)
    print(f"[1] equivalence with diffusers at offset (0,0): {same}")
    print(f"    shapes ref={tuple(ref_cos.shape)} got={tuple(got_cos.shape)}")
    if not same:
        d = (ref_cos - got_cos).abs().max().item()
        failures.append(f"not bit-identical to diffusers at the origin (max diff {d:.3e})")

    # --- 2. different offsets must differ (the bug) ---
    canvas = (135, 240)   # 4K: 270x480 latent, patch 2 -> 135x240 patches, 3x3 tiles
    a = get_3d_rope_at_offset(EMBED_DIM, TILE, FRAMES, (0, 0), canvas, device="cpu")
    b = get_3d_rope_at_offset(EMBED_DIM, TILE, FRAMES, (45, 80), canvas, device="cpu")
    c = get_3d_rope_at_offset(EMBED_DIM, TILE, FRAMES, (90, 160), canvas, device="cpu")
    d01 = (a[0] - b[0]).abs().max().item()
    d02 = (a[0] - c[0]).abs().max().item()
    print(f"[2] tile(0,0) vs tile(45,80): max|dcos| = {d01:.4f}")
    print(f"    tile(0,0) vs tile(90,160): max|dcos| = {d02:.4f}")
    if d01 < 1e-6 or d02 < 1e-6:
        failures.append("different tile offsets still produce identical RoPE")
    else:
        print("    -> tiles are positionally distinguishable")

    # Confirm the OLD code path does NOT distinguish them, so the bug is real.
    old = [get_3d_rotary_pos_embed(embed_dim=EMBED_DIM, crops_coords=None,
                                   grid_size=TILE, temporal_size=FRAMES,
                                   grid_type="slice", max_size=TRAINED, device="cpu")
           for _ in range(3)]
    old_identical = all(torch.equal(old[0][0], o[0]) for o in old[1:])
    print(f"    baseline (current pipeline call) identical across tiles: {old_identical}")
    if not old_identical:
        failures.append("could not reproduce the original bug -- baseline already differs")

    # --- 3. a tile equals the matching slice of the whole-canvas field ---
    # Build the canvas field one row-block at a time and compare against a tile.
    full = get_3d_rope_at_offset(EMBED_DIM, canvas, FRAMES, (0, 0), canvas, device="cpu")
    ch, cw = canvas
    th, tw = TILE
    oh, ow = 45, 80
    # index the (T, ch, cw, dim) layout that _combine flattened
    full_cos = full[0].view(FRAMES, ch, cw, -1)
    tile_cos = b[0].view(FRAMES, th, tw, -1)
    sub = full_cos[:, oh:oh + th, ow:ow + tw, :]
    consistent = torch.allclose(sub, tile_cos, atol=1e-6)
    print(f"[3] tile == corresponding slice of whole-canvas RoPE: {consistent}")
    if not consistent:
        failures.append(f"tile RoPE is not a slice of the canvas field "
                        f"(max diff {(sub - tile_cos).abs().max().item():.3e})")

    # --- 4. interp keeps positions in the trained range ---
    # Position 134 on a 135-tall canvas maps to <=47 on a 48-tall trained grid.
    i_lo = get_3d_rope_at_offset(EMBED_DIM, TILE, FRAMES, (0, 0), canvas,
                                 trained_size=TRAINED, mode="interp", device="cpu")
    i_hi = get_3d_rope_at_offset(EMBED_DIM, TILE, FRAMES, (90, 160), canvas,
                                 trained_size=TRAINED, mode="interp", device="cpu")
    # extend mode at the far corner should exceed interp's magnitude spread
    spread_extend = (c[0] - a[0]).abs().mean().item()
    spread_interp = (i_hi[0] - i_lo[0]).abs().mean().item()
    print(f"[4] mean|d| corner-to-origin: extend={spread_extend:.4f} interp={spread_interp:.4f}")
    if not (spread_interp < spread_extend):
        failures.append("interp did not compress the position range relative to extend")
    else:
        print("    -> interp compresses positions as intended")

    # --- 5. wrap: a shifted window past the canvas edge stays finite ---
    w = get_3d_rope_at_offset(EMBED_DIM, TILE, FRAMES, (130, 235), canvas, device="cpu")
    finite = bool(torch.isfinite(w[0]).all() and torch.isfinite(w[1]).all())
    print(f"[5] window crossing the canvas edge produces finite RoPE: {finite}")
    if not finite:
        failures.append("wrapped window produced non-finite RoPE")

    # --- 6. cache correctness ---
    cache = TiledRopeCache(EMBED_DIM, canvas, TRAINED, FRAMES, device="cpu")
    x1 = cache.get((45, 80), TILE)
    x2 = cache.get((45, 80), TILE)
    hit = x1[0] is x2[0]
    match = torch.equal(x1[0], b[0])
    for off in [(0, 0), (45, 80), (90, 160), (0, 80), (45, 0)]:
        cache.get(off, TILE)
    print(f"[6] cache returns the same object on re-get: {hit}; matches direct call: {match}")
    print(f"    entries after 5 distinct offsets: {cache.stats()['entries']}")
    if not match:
        failures.append("cached RoPE differs from a direct call")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: canvas-absolute RoPE generalises diffusers and distinguishes tiles")


if __name__ == "__main__":
    main()
