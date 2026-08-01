"""Canvas-absolute RoPE for HunyuanVideo tiled Stage-2 denoising.

Same defect as CogVideoX and Wan: `pipeline_hunyuan_video.py` computes
`window_freqs_cos/sin` **once, outside the tile loop** (around line 1635) and
reuses the identical tensor for every tile (line 1886). Every tile is therefore
told it sits at the canvas origin, and since each tile also receives the full
global prompt, each one renders the whole scene -> duplicated objects at 2K/4K.

Hunyuan is the easiest of the three to fix, because `get_nd_rotary_pos_embed`
already accepts a `start`/`stop` pair and forwards it to `get_meshgrid_nd`
(posemb_layers.py:191-221). So a tile's RoPE is just the same call with its
absolute canvas offset as `start` instead of an implicit 0.

Mirrors CogvideoI2V/modules/global_rope.py so the two backbones expose the same
three modes:
  local  -- legacy, every tile identical (kept as the baseline / ablation arm)
  extend -- absolute canvas offset, native frequency spacing
  ntk    -- absolute offset + NTK theta scaling (CineScale's mechanism)
"""

import math
from typing import Dict, Optional, Tuple

import torch

try:
    from .posemb_layers import get_nd_rotary_pos_embed
except ImportError:  # loaded standalone (e.g. by the unit test)
    from posemb_layers import get_nd_rotary_pos_embed


def _ntk_alpha(canvas: int, trained: int) -> float:
    """NTK factor for a canvas larger than the trained extent (see global_rope.py)."""
    if trained <= 0 or canvas <= trained:
        return 1.0
    return float(min(64.0, max(1.0, (canvas / trained) ** 2)))


class HunyuanTiledRope:
    """Per-tile canvas-absolute RoPE, cached on (offset, size).

    The sliding window moves every step, so the key space is num_tiles x loop_step,
    which is small and bounded.
    """

    def __init__(
        self,
        rope_dim_list,
        rope_sizes: Tuple[int, int, int],       # (frames, h, w) of ONE tile, in patches
        canvas_sizes: Tuple[int, int, int],     # (frames, H, W) of the whole canvas
        trained_sizes: Optional[Tuple[int, int, int]] = None,
        theta: float = 256.0,
        mode: str = "extend",
        use_real: bool = True,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
    ):
        self.rope_dim_list = rope_dim_list
        self.rope_sizes = tuple(rope_sizes)
        self.canvas_sizes = tuple(canvas_sizes)
        self.trained_sizes = tuple(trained_sizes) if trained_sizes else None
        self.theta = theta
        self.mode = mode
        self.use_real = use_real
        self.theta_rescale_factor = theta_rescale_factor
        self.interpolation_factor = interpolation_factor
        self._cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

        # NTK scales theta per axis; the temporal axis is left alone because the
        # frame count does not grow with the canvas (CineScale uses alpha[0]=1).
        if mode == "ntk":
            if self.trained_sizes is None:
                raise ValueError("mode='ntk' needs trained_sizes")
            self._theta_per_axis = [
                theta,
                theta * _ntk_alpha(self.canvas_sizes[1], self.trained_sizes[1]),
                theta * _ntk_alpha(self.canvas_sizes[2], self.trained_sizes[2]),
            ]
        else:
            self._theta_per_axis = None

    def get(self, offset: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE for the tile whose top-left patch is at `offset` on the canvas."""
        off = tuple(int(o) for o in offset)
        if off in self._cache:
            return self._cache[off]

        # Wrap like the latent ring: a shifted window can pass the canvas edge.
        start = tuple(off[i] % self.canvas_sizes[i] for i in range(3))
        stop = tuple(start[i] + self.rope_sizes[i] for i in range(3))

        if self._theta_per_axis is None:
            freqs = get_nd_rotary_pos_embed(
                self.rope_dim_list, start, stop,
                theta=self.theta, use_real=self.use_real,
                theta_rescale_factor=self.theta_rescale_factor,
                interpolation_factor=self.interpolation_factor,
            )
        else:
            # get_nd_rotary_pos_embed takes ONE theta, so build the per-axis
            # frequencies separately and concatenate, matching its own layout
            # (it concatenates per-axis embeddings along the last dim).
            try:
                from .posemb_layers import get_1d_rotary_pos_embed, get_meshgrid_nd
            except ImportError:
                from posemb_layers import get_1d_rotary_pos_embed, get_meshgrid_nd
            grid = get_meshgrid_nd(start, stop, dim=len(self.rope_dim_list))
            cos_parts, sin_parts = [], []
            for i, dim_i in enumerate(self.rope_dim_list):
                c, s = get_1d_rotary_pos_embed(
                    dim_i, grid[i].reshape(-1), self._theta_per_axis[i],
                    use_real=True,
                    theta_rescale_factor=self.theta_rescale_factor,
                    interpolation_factor=self.interpolation_factor,
                )
                cos_parts.append(c)
                sin_parts.append(s)
            freqs = (torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1))

        self._cache[off] = freqs
        return freqs

    def stats(self):
        return {"entries": len(self._cache), "mode": self.mode,
                "canvas": self.canvas_sizes, "tile": self.rope_sizes}
