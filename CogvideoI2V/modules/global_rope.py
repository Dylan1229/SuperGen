"""Canvas-absolute RoPE for tiled Stage-2 denoising (duplicate-object artifact fix).

The artifact
------------
At 4K the same object can appear as several small copies, one per tile. Two things
combine to cause it:

1. **Every tile gets the same positional encoding.** The pipeline builds RoPE with
   `_prepare_rotary_positional_embeddings(window_h, window_w, ...)` once per tile
   index, but the arguments do not contain the tile's position, so all tiles get a
   bit-identical tensor. Diffusers' `grid_type="slice"` path always crops the
   frequency table from the origin (`h_cos[:grid_size_h]`), i.e. every tile is told
   "you are the top-left corner of the canvas".
2. **Every tile gets the full global prompt.** So each tile independently tries to
   satisfy "a lion is roaring" inside its own window.

A tile that cannot tell where it is, and is asked to render the whole prompt, draws
the whole scene. With 9 tiles you can get 9 lions.

This module fixes (1): each tile receives RoPE sliced at its **absolute canvas
offset**, so tiles are positionally distinguishable and the model's learned notion
of "this is the upper-left / centre / lower-right of a frame" applies correctly.

Note (2) is a separate axis; per-tile prompt weighting is not attempted here.
Fixing (1) alone is the cheap, mechanically-correct half, and it is what the
NTK-RoPE family (CineScale) also manipulates.

Interpolation vs extension
--------------------------
Two ways to place a tile on the canvas, both provided:

- `mode="extend"` (default): keep the model's native frequency spacing and index it
  at absolute positions. Tile at latent offset (oh, ow) reads rows
  `[oh_grid : oh_grid + grid_h]` of the frequency table. Positions beyond the
  training range are extrapolated, which is what the `slice` path was designed for
  (`max_size` = the trained `sample_height/width`), but for a 3x upscale the canvas
  exceeds it -- so we grow the table and let RoPE extrapolate. This preserves local
  detail statistics and is the behaviour closest to "the tile knows where it is".

- `mode="interp"`: rescale absolute positions back into the trained range
  (NTK-style position interpolation). Tile positions become fractional, spacing
  shrinks by `canvas/trained`. Keeps every position in-distribution but slightly
  blurs high-frequency spatial detail.

Both are exposed so the ablation is a flag, not a rewrite.

Shifting
--------
The sliding window moves every step, so the offset changes per step. RoPE is
therefore keyed on `(offset_h, offset_w, grid_h, grid_w, frames)` and cached; a
canvas of 9 tiles x loop_step 16 has a small bounded key space.
"""

from typing import Dict, Optional, Tuple

import torch

from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _combine(freqs_t, freqs_h, freqs_w, temporal_size, grid_h, grid_w):
    """Broadcast per-axis frequencies into a flat (T*H*W, dim) table."""
    freqs_t = freqs_t[:, None, None, :].expand(-1, grid_h, grid_w, -1)
    freqs_h = freqs_h[None, :, None, :].expand(temporal_size, -1, grid_w, -1)
    freqs_w = freqs_w[None, None, :, :].expand(temporal_size, grid_h, -1, -1)
    freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
    return freqs.view(temporal_size * grid_h * grid_w, -1)


def _ntk_alpha(canvas: int, trained: int) -> float:
    """NTK factor that makes `canvas` positions fit the range `trained` covered.

    RoPE's slowest channel has wavelength ~ theta. Scaling theta by alpha stretches
    the low-frequency channels by roughly alpha**(1) while leaving the fastest
    channel (k=0, weight exactly 1.0) untouched. CineScale hard-codes alpha=20 for a
    ~3x canvas; the closed form for a scale factor s = canvas/trained is s**dim,
    which is unusable, so use the standard NTK-by-parts heuristic alpha = s**2 --
    empirically close to CineScale's 20 at s~3 (3**2 = 9; they use 20, i.e. more
    aggressive). Clamped to [1, 64].
    """
    if trained <= 0 or canvas <= trained:
        return 1.0
    s = canvas / trained
    return float(min(64.0, max(1.0, s ** 2)))


def get_3d_rope_at_offset(
    embed_dim: int,
    grid_size: Tuple[int, int],
    temporal_size: int,
    offset: Tuple[int, int] = (0, 0),
    canvas_size: Optional[Tuple[int, int]] = None,
    trained_size: Optional[Tuple[int, int]] = None,
    mode: str = "extend",
    theta: int = 10000,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE for a tile at an absolute position on the canvas.

    Mirrors `diffusers.get_3d_rotary_pos_embed(grid_type="slice")` exactly when
    `offset=(0, 0)` and `mode="extend"` with `canvas_size=trained_size`, so the
    fix is a strict generalisation of the current behaviour.

    Args:
        embed_dim: attention head dim (dim_t = embed_dim//4, dim_h = dim_w = 3/8).
        grid_size: (grid_h, grid_w) of THIS tile, in patch units.
        temporal_size: number of latent frames after temporal patching.
        offset: (offset_h, offset_w) of this tile on the canvas, in patch units.
        canvas_size: (canvas_h, canvas_w) of the WHOLE canvas, in patch units.
            Defaults to the tile size (which reproduces the unfixed behaviour).
        trained_size: (h, w) the model was trained at, in patch units. Required for
            `mode="interp"`.
        mode: "extend" (native spacing, absolute indexing, extrapolates past the
            trained range) or "interp" (positions rescaled into the trained range).
    """
    grid_h, grid_w = grid_size
    off_h, off_w = offset
    canvas_h, canvas_w = canvas_size if canvas_size is not None else (
        off_h + grid_h, off_w + grid_w)

    if off_h + grid_h > canvas_h or off_w + grid_w > canvas_w:
        # A shifted window can run past the canvas edge; the ring topology wraps it,
        # so wrap the positions the same way rather than indexing out of range.
        pass

    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    grid_t_pos = torch.arange(temporal_size, device=device, dtype=torch.float32)

    # Absolute canvas positions for this tile, wrapped like the latent ring.
    pos_h = (torch.arange(off_h, off_h + grid_h, device=device, dtype=torch.float32)
             % canvas_h)
    pos_w = (torch.arange(off_w, off_w + grid_w, device=device, dtype=torch.float32)
             % canvas_w)

    theta_h = theta_w = float(theta)

    if mode == "interp":
        # Naive position interpolation. MEASURED TO BE BAD: it multiplies EVERY
        # frequency channel by canvas/trained, so the fastest channel (k=0), which
        # is what distinguishes neighbouring patches, is compressed just as much as
        # the slow ones. At 4K that factor is 48/135 = 0.356, i.e. adjacent patches
        # end up 0.36 of a position apart instead of 1.0. The model cannot separate
        # them and the output destabilises: measured 5.2x the colour flicker of
        # baseline (global_chroma_std 12.90 vs 2.47, hf_temporal_energy 0.40 vs
        # 0.07). Kept only so the ablation can show why NTK is the right answer.
        if trained_size is None:
            raise ValueError("mode='interp' needs trained_size")
        tr_h, tr_w = trained_size
        if canvas_h > tr_h:
            pos_h = pos_h * (tr_h / canvas_h)
        if canvas_w > tr_w:
            pos_w = pos_w * (tr_w / canvas_w)
    elif mode == "ntk":
        # NTK-aware scaling, the mechanism CineScale uses (`set_ntk([1, 20, 20])`
        # -> `theta = theta * ntk_factor` in
        # diffsynth/models/wan_video_dit.py::precompute_freqs_cis_ntk).
        # Positions stay at native spacing; instead the frequency BASE is enlarged,
        # which stretches the slow (long-range) channels to cover the bigger canvas
        # while leaving the fast (local) channels essentially untouched -- the k=0
        # channel has weight exactly 1.0 for any theta. That is precisely the
        # property naive interpolation destroys.
        # Note the temporal axis keeps the original theta (CineScale's alpha[0]=1):
        # the frame count does not change when the canvas grows.
        if trained_size is None:
            raise ValueError("mode='ntk' needs trained_size")
        tr_h, tr_w = trained_size
        theta_h = theta * _ntk_alpha(canvas_h, tr_h)
        theta_w = theta * _ntk_alpha(canvas_w, tr_w)
    elif mode != "extend":
        raise ValueError(
            f"unknown rope mode {mode!r}; use 'extend', 'ntk' or 'interp'")

    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t_pos, theta=theta, use_real=True)
    freqs_h = get_1d_rotary_pos_embed(dim_h, pos_h, theta=theta_h, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, pos_w, theta=theta_w, use_real=True)

    t_cos, t_sin = freqs_t
    h_cos, h_sin = freqs_h
    w_cos, w_sin = freqs_w

    cos = _combine(t_cos, h_cos, w_cos, temporal_size, grid_h, grid_w)
    sin = _combine(t_sin, h_sin, w_sin, temporal_size, grid_h, grid_w)
    return cos, sin


class TiledRopeCache:
    """Per-tile canvas-absolute RoPE, cached on (offset, grid, frames).

    The sliding window moves every step, so offsets change; the key space is
    bounded by num_tiles x loop_step, which is small.
    """

    def __init__(
        self,
        embed_dim: int,
        canvas_size: Tuple[int, int],
        trained_size: Tuple[int, int],
        temporal_size: int,
        mode: str = "extend",
        device: Optional[torch.device] = None,
    ):
        self.embed_dim = embed_dim
        self.canvas_size = canvas_size
        self.trained_size = trained_size
        self.temporal_size = temporal_size
        self.mode = mode
        self.device = device
        self._cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        logger.info(
            f"TiledRopeCache: canvas={canvas_size} trained={trained_size} "
            f"mode={mode} temporal={temporal_size}"
        )

    def get(self, offset: Tuple[int, int], grid_size: Tuple[int, int]):
        key = (int(offset[0]), int(offset[1]), int(grid_size[0]), int(grid_size[1]))
        if key not in self._cache:
            self._cache[key] = get_3d_rope_at_offset(
                embed_dim=self.embed_dim,
                grid_size=grid_size,
                temporal_size=self.temporal_size,
                offset=offset,
                canvas_size=self.canvas_size,
                trained_size=self.trained_size,
                mode=self.mode,
                device=self.device,
            )
        return self._cache[key]

    def stats(self):
        return {"entries": len(self._cache), "mode": self.mode,
                "canvas": self.canvas_size, "trained": self.trained_size}
