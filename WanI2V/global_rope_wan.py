"""Canvas-absolute RoPE for Wan2.1 tiled denoising (P1-5 + duplicate-object fix).

Wan2.1 has the same positional-encoding defect as CogVideoX when its denoiser is
driven tile-by-tile. In `wan/modules/model.py::rope_apply`:

    freqs_i = torch.cat([
        freqs[0][:f]...,     # temporal, fine
        freqs[1][:h]...,     # HEIGHT sliced from 0
        freqs[2][:w]...,     # WIDTH  sliced from 0
    ])

Every call slices the frequency table from the origin, so a tile at canvas offset
(oh, ow) is told it sits at (0, 0). Combined with each tile receiving the full
global prompt, each tile renders the whole scene -> duplicated small objects.

Wan differs from CogVideoX in two ways that matter here:
  * RoPE is **complex** (`torch.polar`), applied as a complex multiply, whereas
    CogVideoX uses a real (cos, sin) pair.
  * The table is precomputed once at `max_seq_len` and indexed per sample via
    `grid_sizes`, so the fix is a slice-offset rather than a re-derivation.

`rope_apply_at_offset` is a drop-in replacement for `rope_apply` that takes a
per-sample (offset_f, offset_h, offset_w) and slices the table there. With
offset=(0,0,0) it is numerically identical to upstream, so it is a strict
generalisation (asserted in test_global_rope_wan.py).
"""

from typing import Optional, Sequence, Tuple

import torch


def _slice_axis(table: torch.Tensor, offset: int, size: int, extent: Optional[int],
                mode: str, trained: Optional[int]) -> torch.Tensor:
    """Slice one axis of a precomputed RoPE table at an absolute offset.

    Args:
        table: (max_seq_len, dim) complex frequency table from `rope_params`.
        offset: absolute start position of this tile on the canvas, in patches.
        size: number of positions this tile needs.
        extent: canvas extent for wrapping (None = no wrap).
        mode: 'extend' (native spacing, absolute indexing) or
              'interp' (positions rescaled into the trained range).
        trained: trained extent, required for 'interp'.
    """
    if mode == "extend":
        pos = torch.arange(offset, offset + size, device=table.device)
        if extent is not None:
            pos = pos % extent
        pos = pos.clamp_(max=table.shape[0] - 1)
        return table[pos]

    if mode != "interp":
        raise ValueError(f"unknown rope mode {mode!r}")
    if trained is None or extent is None:
        raise ValueError("mode='interp' needs both trained and extent")

    # NTK-style position interpolation. The table is integer-indexed, so
    # interpolate between neighbouring rows instead of rounding, which would
    # quantise several tiles onto the same position.
    scale = min(1.0, trained / max(extent, 1))
    pos = (torch.arange(offset, offset + size, device=table.device,
                        dtype=torch.float64) % extent) * scale
    lo = pos.floor().long().clamp_(max=table.shape[0] - 1)
    hi = (lo + 1).clamp_(max=table.shape[0] - 1)
    frac = (pos - lo.to(pos.dtype)).unsqueeze(-1)
    # Interpolate on the unit circle: normalise after the linear blend so the
    # result stays a pure rotation (|freq| == 1), which the complex multiply needs.
    blended = table[lo] * (1.0 - frac) + table[hi] * frac
    return blended / blended.abs().clamp_min(1e-12)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_at_offset(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    offsets: Optional[Sequence[Tuple[int, int, int]]] = None,
    canvas: Optional[Tuple[int, int, int]] = None,
    trained: Optional[Tuple[int, int, int]] = None,
    mode: str = "extend",
) -> torch.Tensor:
    """Wan `rope_apply` with per-sample absolute canvas offsets.

    Args:
        x: (B, L, num_heads, head_dim) attention input.
        grid_sizes: (B, 3) int tensor of (f, h, w) per sample.
        freqs: (max_seq_len, head_dim//2) complex table from `rope_params`.
        offsets: per-sample (off_f, off_h, off_w) in patch units. None = all zeros,
            which reproduces upstream exactly.
        canvas: (F, H, W) canvas extent in patches, for wrapping shifted windows.
        trained: (F, H, W) extent the model was trained at; needed for mode='interp'.
        mode: 'extend' or 'interp'.
    """
    n, c = x.size(2), x.size(3) // 2
    # Same split as upstream: temporal gets the remainder, h and w get c//3 each.
    parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        off_f, off_h, off_w = (0, 0, 0) if offsets is None else offsets[i]
        cf, ch, cw = (None, None, None) if canvas is None else canvas
        tf, th, tw = (None, None, None) if trained is None else trained

        fr_f = _slice_axis(parts[0], off_f, f, cf, mode, tf)
        fr_h = _slice_axis(parts[1], off_h, h, ch, mode, th)
        fr_w = _slice_axis(parts[2], off_w, w, cw, mode, tw)

        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            fr_f.view(f, 1, 1, -1).expand(f, h, w, -1),
            fr_h.view(1, h, 1, -1).expand(f, h, w, -1),
            fr_w.view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


class WanTiledRope:
    """Holds the tiling geometry and produces per-tile offsets for rope_apply.

    Wan applies RoPE inside the model, so the tile offset has to travel with the
    forward call. Rather than thread a new argument through every block, patch
    `wan.modules.model.rope_apply` once and set `current_offset` before each tile
    -- the same monkey-patch shape TeaCache/AdaCache use upstream.
    """

    def __init__(self, canvas, trained=None, mode="extend"):
        self.canvas = tuple(canvas)
        self.trained = tuple(trained) if trained is not None else None
        self.mode = mode
        self.current_offset = (0, 0, 0)

    def install(self, model_module):
        """Replace `model_module.rope_apply` with an offset-aware version."""
        holder = self

        def patched(x, grid_sizes, freqs):
            offsets = [holder.current_offset] * x.size(0)
            return rope_apply_at_offset(
                x, grid_sizes, freqs, offsets=offsets,
                canvas=holder.canvas, trained=holder.trained, mode=holder.mode,
            )

        self._original = getattr(model_module, "rope_apply")
        model_module.rope_apply = patched
        return self

    def uninstall(self, model_module):
        if hasattr(self, "_original"):
            model_module.rope_apply = self._original

    def set_tile(self, offset_f: int, offset_h: int, offset_w: int):
        self.current_offset = (int(offset_f), int(offset_h), int(offset_w))
