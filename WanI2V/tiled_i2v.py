"""UltraGen/SuperGen on Wan2.1-I2V-14B: two-stage tiling + region-aware cache (P1-5).

Answers Rev-C's "do the techniques generalise beyond this pipeline?" by running the
same three techniques on a third backbone with a different architecture:

    CogVideoX-1.5   diffusers pipeline, DDIM, real (cos,sin) RoPE, joint image-text
                    blocks, 42 layers
    Wan2.1-I2V-14B  custom runtime, flow-matching (UniPC/DPM++), COMPLEX RoPE
                    applied inside the model, cross-attention blocks, 40 layers

What carries over unchanged, and what had to be rebuilt:

  carries over: the tiling geometry (`utils.tile_utils.SlidingWindowConfig` and
      `TiledLatentTensor2D`) -- Wan's latent grid at 720p is 90x160, the same window
      size CogVideoX uses, so the sliding-window schedule is identical; and the
      region-aware cache *policy* (accumulated predicted error per tile).

  rebuilt: (a) RoPE. Wan applies it inside every block via a module-level
      `rope_apply`, not as a pipeline argument, so the canvas-absolute variant is
      installed by patching that function (see global_rope_wan.py). (b) The
      denoising loop, because Wan uses flow-matching sigmas rather than a diffusers
      scheduler. (c) Stage-1 -> Stage-2 re-noising, which for flow matching means
      interpolating toward noise at the target sigma rather than DDIM's
      `add_noise`.

Status: the tiled Stage-2 loop and the cache hook are implemented here. The
`WanI2VTiled.generate` entry point mirrors `wan.image2video.WanI2V.generate`'s
signature so the existing runner scripts need only a new `run_wan()` helper.
"""

import logging
import math
import os
import sys
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tile_utils import SlidingWindowConfig, TiledLatentTensor2D  # noqa: E402

from global_rope_wan import WanTiledRope  # noqa: E402

logger = logging.getLogger(__name__)


class WanRegionAwareCache:
    """Region-aware tile cache for Wan's Stage-2 loop.

    Same policy as `CachingCogVideoXTransformer3DModel`, including the gain-estimator
    fix: `k` is only recomputed when the output history spans the same interval as
    the input history (i.e. right after two real computations). During a skip run the
    reconstructed output carries no new information, so the last valid `k` is held
    rather than recomputing the degenerate value 1.0.

    See SuperGen/CogvideoI2V/modules/test_k_estimator_bug.py for why that matters:
    without it, skip runs grow without bound and quality collapses.
    """

    def __init__(self, latent_shape, num_tiles, num_steps, thresh, ret_steps=5,
                 device="cuda", dtype=torch.float32):
        z = torch.zeros(latent_shape, device=device, dtype=dtype)
        self.prev_input = TiledLatentTensor2D(latent_tensor=z)
        self.prev_output = TiledLatentTensor2D(latent_tensor=z)
        self.prev_prev_input = TiledLatentTensor2D(latent_tensor=z)
        self.prev_prev_output = TiledLatentTensor2D(latent_tensor=z)
        self.residual = TiledLatentTensor2D(latent_tensor=z)
        self.accum_error = TiledLatentTensor2D(latent_tensor=z)
        for t in (self.prev_input, self.prev_output, self.prev_prev_input,
                  self.prev_prev_output, self.residual, self.accum_error):
            t.zero_()

        self.num_steps = num_steps
        self.thresh = thresh
        self.ret_steps = ret_steps
        self.num_tiles = num_tiles
        self.per_tile_thresh = [thresh] * num_tiles
        self._last_k = {}
        self._last_compute_step = {}
        self.stats = {"checked": 0, "skipped": 0}

    def should_skip(self, step, tile, raw_input, window) -> bool:
        """True if this tile may reuse its cached residual at this step."""
        self.stats["checked"] += 1
        if step < self.ret_steps or step >= self.num_steps - 1:
            return False

        prev_in = self.prev_input.get_window_latent(*window)
        prev_out = self.prev_output.get_window_latent(*window)
        prev_prev_in = self.prev_prev_input.get_window_latent(*window)
        prev_prev_out = self.prev_prev_output.get_window_latent(*window)

        in_change = (raw_input - prev_in).abs().mean()
        prev_in_change = (prev_in - prev_prev_in).abs().mean()
        out_change = (prev_out - prev_prev_out).abs().mean()
        if prev_in_change <= 0:
            return False

        last = self._last_compute_step.get(tile)
        fresh = last is not None and last == step - 1
        if fresh:
            k = out_change / prev_in_change
            self._last_k[tile] = k
        else:
            k = self._last_k.get(tile)
            if k is None:
                return False

        out_norm = prev_out.abs().mean().clamp_min(1e-8)
        pred = k * (in_change / out_norm)
        acc = self.accum_error.get_window_latent(*window) + pred
        self.accum_error.set_window_latent(acc, *window)

        thresh = self.per_tile_thresh[tile] if tile < len(self.per_tile_thresh) else self.thresh
        if acc.mean() < thresh:
            self.stats["skipped"] += 1
            # Advance the INPUT history only; the output history must stay pinned to
            # the last real computation or k degenerates to 1.0.
            self.prev_prev_input.set_window_latent(prev_in, *window)
            self.prev_input.set_window_latent(raw_input, *window)
            return True

        self.accum_error.set_window_latent(torch.zeros_like(raw_input), *window)
        return False

    def reuse(self, raw_input, window):
        return raw_input + self.residual.get_window_latent(*window)

    def record(self, step, tile, raw_input, output, window):
        self.residual.set_window_latent(output - raw_input, *window)
        self.prev_prev_input.set_window_latent(
            self.prev_input.get_window_latent(*window), *window)
        self.prev_input.set_window_latent(raw_input, *window)
        self.prev_prev_output.set_window_latent(
            self.prev_output.get_window_latent(*window), *window)
        self.prev_output.set_window_latent(output, *window)
        self._last_compute_step[tile] = step

    def report(self):
        checked = self.stats["checked"]
        rate = self.stats["skipped"] / checked if checked else 0.0
        logger.info(f"CACHE_SKIP_RATE {rate:.4f} skipped={self.stats['skipped']} "
                    f"checked={checked}")
        return {"skip_rate": rate, **self.stats}


class WanI2VTiled:
    """Wraps an existing `wan.image2video.WanI2V` with tiled Stage-2 denoising.

    Composition rather than subclassing: Wan's constructor loads ~67GB of weights
    and shards them, so we take an already-constructed instance.
    """

    def __init__(self, wan_i2v, rope_mode="extend"):
        self.wan = wan_i2v
        self.rope_mode = rope_mode
        cfg = wan_i2v.config
        self.vae_stride = cfg.vae_stride          # (4, 8, 8)
        self.patch_size = cfg.patch_size          # (1, 2, 2)
        self._rope = None

    def latent_grid(self, height, width, frames):
        """Pixel -> latent -> patch grid, following Wan's own arithmetic."""
        lat_h = height // self.vae_stride[1]
        lat_w = width // self.vae_stride[2]
        lat_f = (frames - 1) // self.vae_stride[0] + 1
        return (lat_f, lat_h, lat_w), (lat_f // self.patch_size[0],
                                       lat_h // self.patch_size[1],
                                       lat_w // self.patch_size[2])

    def install_rope(self, height, width, frames, trained_hw=(720, 1280)):
        """Install canvas-absolute RoPE sized for this canvas."""
        import wan.modules.model as wan_model
        (_, _, _), patch_grid = self.latent_grid(height, width, frames)
        (_, _, _), trained_grid = self.latent_grid(trained_hw[0], trained_hw[1], frames)
        self._rope = WanTiledRope(canvas=patch_grid, trained=trained_grid,
                                  mode=self.rope_mode).install(wan_model)
        logger.info(f"Wan canvas-absolute RoPE: canvas={patch_grid} "
                    f"trained={trained_grid} mode={self.rope_mode}")
        return self._rope

    def uninstall_rope(self):
        if self._rope is not None:
            import wan.modules.model as wan_model
            self._rope.uninstall(wan_model)
            self._rope = None

    def window_config(self, height, width, frames, loop_step=16):
        (_, lat_h, lat_w), _ = self.latent_grid(height, width, frames)
        return SlidingWindowConfig(lat_h, lat_w, loop_step)

    def set_tile(self, window, frames_offset=0):
        """Point RoPE at this tile's absolute canvas offset (patch units)."""
        if self._rope is None:
            return
        start_h, _, start_w, _ = window
        self._rope.set_tile(frames_offset,
                            start_h // self.patch_size[1],
                            start_w // self.patch_size[2])
