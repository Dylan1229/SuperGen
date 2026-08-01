"""Two-stage tiled Stage-2 denoising for Wan2.1-I2V-14B (P1-5).

Answers Rev-C's generality objection by running the same three techniques --
two-stage tiling, region-aware caching, canvas-absolute RoPE -- on a third backbone
whose architecture differs from CogVideoX/Hunyuan in ways that matter:

  * **flow matching**, not DDIM. There is no `add_noise`/`re_noise`; re-noising to a
    timestep means interpolating toward noise at that sigma:
        x_t = (1 - sigma_t) * x_0 + sigma_t * noise
    (`wan/utils/fm_solvers.py`). So the Stage-1 -> Stage-2 handoff is rebuilt, not
    reused.
  * **complex RoPE applied inside the model** via a module-level `rope_apply`, so
    per-tile positional encoding is installed by patching that function
    (`global_rope_wan.py`) rather than passed as an argument.
  * **CFG by two separate forward calls** (`wan/image2video.py:308-317`), not a
    batched pair, so a tile's cache decision has to hold across both calls.

What is reused unchanged: `utils.tile_utils.SlidingWindowConfig` and
`TiledLatentTensor2D` (Wan's 720p latent grid is 90x160 -- the same window size
CogVideoX uses, so the sliding-window schedule is identical), and the
region-aware cache policy including the gain-estimator fix.

Upscaling is done in **pixel space** (decode -> bicubic -> re-encode), matching the
CogVideoX path and FreeSwim (`inference.py:55`, cv2.resize + re-encode). Latent-space
upscaling was measured and rejected: it tears (ringing 0.244 vs 0.130), and
CineScale only gets away with it because it loads a LoRA trained under that
configuration. See results/P0-2_rope_artifact/upscale_mode/NOTES.md.
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tile_utils import SlidingWindowConfig, TiledLatentTensor2D  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from global_rope_wan import WanTiledRope  # noqa: E402
from tiled_i2v import WanRegionAwareCache  # noqa: E402

logger = logging.getLogger(__name__)


def flow_match_renoise(x0: torch.Tensor, noise: torch.Tensor, sigma: float) -> torch.Tensor:
    """Re-noise for flow matching: linear interpolation toward noise.

    DDIM's `add_noise` uses sqrt(alpha_bar) weighting; flow matching is linear in
    sigma, so the two are NOT interchangeable. Getting this wrong is silent -- the
    output just degrades -- which is why it is a named function with a test.
    """
    return (1.0 - sigma) * x0 + sigma * noise


class WanTiledStage2:
    """Runs Wan's Stage-2 denoising tile by tile over an upscaled latent.

    Deliberately composed around an existing `wan.image2video.WanI2V` rather than
    subclassing: its constructor shards ~67 GB of weights, and we only need to
    replace the denoising loop.
    """

    def __init__(self, wan_i2v, rope_mode: str = "local",
                 loop_step: int = 16, device: str = "cuda"):
        self.wan = wan_i2v
        self.cfg = wan_i2v.config
        self.rope_mode = rope_mode
        self.loop_step = loop_step
        self.device = device
        self.vae_stride = self.cfg.vae_stride       # (4, 8, 8)
        self.patch_size = self.cfg.patch_size       # (1, 2, 2)
        self._rope = None
        self.cache = None

    # ------------------------------------------------------------------ geometry
    def latent_shape(self, height: int, width: int, frames: int):
        return ((frames - 1) // self.vae_stride[0] + 1,
                height // self.vae_stride[1],
                width // self.vae_stride[2])

    def patch_grid(self, height: int, width: int, frames: int):
        lat_f, lat_h, lat_w = self.latent_shape(height, width, frames)
        return (lat_f // self.patch_size[0],
                lat_h // self.patch_size[1],
                lat_w // self.patch_size[2])

    # ------------------------------------------------------------------- setup
    def install_rope(self, height: int, width: int, frames: int,
                     trained_hw: Tuple[int, int] = (720, 1280)):
        """Install canvas-absolute RoPE for this canvas, if enabled."""
        if self.rope_mode == "local":
            return None
        import wan.modules.model as wan_model
        canvas = self.patch_grid(height, width, frames)
        trained = self.patch_grid(trained_hw[0], trained_hw[1], frames)
        self._rope = WanTiledRope(canvas=canvas, trained=trained,
                                  mode=self.rope_mode).install(wan_model)
        logger.info(f"Wan canvas-absolute RoPE: canvas={canvas} trained={trained} "
                    f"mode={self.rope_mode}")
        return self._rope

    def uninstall_rope(self):
        if self._rope is not None:
            import wan.modules.model as wan_model
            self._rope.uninstall(wan_model)
            self._rope = None

    def setup_cache(self, latent_shape, num_tiles, num_steps, thresh,
                    ret_steps: int = 5, dtype=torch.float32):
        self.cache = WanRegionAwareCache(
            latent_shape=latent_shape, num_tiles=num_tiles, num_steps=num_steps,
            thresh=thresh, ret_steps=ret_steps, device=self.device, dtype=dtype)
        logger.info(f"Wan region-aware cache: {num_tiles} tiles, thresh={thresh}, "
                    f"ret_steps={ret_steps}")
        return self.cache

    # ------------------------------------------------------------------- upscale
    def upscale_pixel(self, latent: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """decode -> bicubic in pixel space -> re-encode. Matches CogVideoX/FreeSwim.

        Wan's VAE wrapper takes and returns a LIST of [C, F, H, W] tensors, unlike
        diffusers' batched [B, C, F, H, W], so shapes are handled explicitly here.
        """
        import torch.nn.functional as F
        video = self.wan.vae.decode([latent])[0]        # [C, F, H, W] in [-1, 1]
        C, Fr, H, W = video.shape
        frames = video.permute(1, 0, 2, 3)              # [F, C, H, W]
        up = F.interpolate(frames.float(), size=(target_h, target_w),
                           mode="bicubic", align_corners=False)
        up = up.clamp(-1, 1).to(video.dtype).permute(1, 0, 2, 3)   # [C, F, h, w]
        return self.wan.vae.encode([up])[0]

    # ------------------------------------------------------------- denoise loop
    def denoise(
        self,
        latent: torch.Tensor,
        timesteps,
        sample_scheduler,
        arg_c: dict,
        arg_null: dict,
        guide_scale: float = 5.0,
        shift_timesteps: Optional[List[int]] = None,
        seed_g=None,
        enable_cache: bool = False,
        y_canvas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stage-2 loop: for each step, for each tile, predict and fuse.

        `latent` is the full upscaled canvas [C, F, H, W]. Tiles are cut from it
        with the same sliding-window schedule the other backbones use.

        `y_canvas` is Wan's I2V conditioning ([mask | encoded first frame]) at CANVAS
        size. It must be cut with the SAME window as the latent -- feeding the whole
        canvas `y` to a tile would both mismatch the sequence length and condition
        every tile on the entire first frame, which is exactly the "every tile draws
        the whole scene" failure we are trying to remove.
        """
        # Stage 1 may have left the DiT on the CPU (Wan's `offload_model` moves it
        # there between the cond/uncond calls). Bring it back before the tile loop,
        # or every forward fails on a device mismatch.
        self.wan.model.to(self.device)
        # Wan keeps fp32 parameters and relies on autocast to run in bf16
        # (image2video.py:258 wraps its whole loop in
        # `amp.autocast(dtype=self.param_dtype)`). Without the same context the very
        # first matmul fails on a BFloat16-vs-Float mismatch.
        self._param_dtype = getattr(self.wan, "param_dtype", torch.bfloat16)
        logger.info(f"Wan Stage-2 autocast dtype: {self._param_dtype}")

        C, Fr, H, W = latent.shape
        wcfg = SlidingWindowConfig(H, W, self.loop_step)
        wp = wcfg.get_window_params()
        win_h, win_w = wp["window_size"]
        num_tiles = wp["total_windows"]
        step_h, step_w = wp["step_size_h"], wp["step_size_w"]
        logger.info(f"Wan Stage-2: canvas {H}x{W}, {num_tiles} tiles of {win_h}x{win_w}, "
                    f"step=({step_h},{step_w})")

        # TiledLatentTensor2D wants [B, F, C, H, W]; Wan carries [C, F, H, W].
        canvas = TiledLatentTensor2D(
            latent_tensor=latent.permute(1, 0, 2, 3).unsqueeze(0).contiguous())
        y_tiler = None
        if y_canvas is not None:
            y_tiler = TiledLatentTensor2D(
                latent_tensor=y_canvas.permute(1, 0, 2, 3).unsqueeze(0).contiguous())

        if enable_cache and self.cache is None:
            self.setup_cache(canvas.torch_latent.shape, num_tiles,
                             len(timesteps), thresh=0.09, dtype=latent.dtype)

        shift_h = shift_w = 0
        autocast_ctx = torch.amp.autocast("cuda", dtype=self._param_dtype)
        for step_idx, t in enumerate(timesteps):
            if shift_timesteps is not None and step_idx in shift_timesteps:
                shift_h = (shift_h + 1) % max(self.loop_step, 1)
                shift_w = (shift_w + 1) % max(self.loop_step, 1)

            fused = torch.zeros_like(canvas.torch_latent)
            counts = torch.zeros_like(canvas.torch_latent)

            for tile_idx in range(num_tiles):
                row, col = divmod(tile_idx, wp["num_windows_w"])
                top = row * win_h + shift_h * step_h
                left = col * win_w + shift_w * step_w
                window = (top, top + win_h, left, left + win_w)

                tile = canvas.get_window_latent(*window)          # [1, F, C, h, w]
                raw = tile

                if self._rope is not None:
                    self._rope.set_tile(0, top // self.patch_size[1],
                                        left // self.patch_size[2])

                reuse = False
                if enable_cache and self.cache is not None:
                    reuse = self.cache.should_skip(step_idx, tile_idx, raw, window)

                if reuse:
                    out = self.cache.reuse(raw, window)
                else:
                    # Wan's model takes a list of [C, F, h, w].
                    model_in = [tile.squeeze(0).permute(1, 0, 2, 3).contiguous()]
                    ts = torch.stack([t]).to(self.device)
                    # Per-tile I2V conditioning, cut with the same window.
                    kw_c, kw_null = dict(arg_c), dict(arg_null)
                    if y_tiler is not None:
                        y_tile = y_tiler.get_window_latent(*window)
                        y_tile = y_tile.squeeze(0).permute(1, 0, 2, 3).contiguous()
                        kw_c["y"] = [y_tile]
                        kw_null["y"] = [y_tile]
                    with autocast_ctx, torch.no_grad():
                        pred_c = self.wan.model(model_in, t=ts, **kw_c)[0]
                        pred_u = self.wan.model(model_in, t=ts, **kw_null)[0]
                    pred = pred_u + guide_scale * (pred_c - pred_u)
                    out = pred.permute(1, 0, 2, 3).unsqueeze(0).to(canvas.torch_latent.dtype)
                    if enable_cache and self.cache is not None:
                        self.cache.record(step_idx, tile_idx, raw, out, window)

                # Accumulate; overlapping regions are averaged by `counts`.
                prev = fused_slice = None
                acc = TiledLatentTensor2D(latent_tensor=fused)
                cnt = TiledLatentTensor2D(latent_tensor=counts)
                acc.set_window_latent(acc.get_window_latent(*window) + out, *window)
                cnt.set_window_latent(cnt.get_window_latent(*window) + 1.0, *window)
                fused, counts = acc.torch_latent, cnt.torch_latent
                del prev, fused_slice

            noise_pred = fused / counts.clamp_min(1.0)
            # scheduler.step wants [B, C, F, H, W]
            npred = noise_pred.squeeze(0).permute(1, 0, 2, 3).unsqueeze(0)
            cur = canvas.torch_latent.squeeze(0).permute(1, 0, 2, 3).unsqueeze(0)
            stepped = sample_scheduler.step(npred, t, cur, return_dict=False,
                                            generator=seed_g)[0]
            canvas.torch_latent = stepped.squeeze(0).permute(1, 0, 2, 3) \
                                         .unsqueeze(0).contiguous()

        if enable_cache and self.cache is not None:
            self.cache.report()

        # back to Wan's [C, F, H, W]
        return canvas.torch_latent.squeeze(0).permute(1, 0, 2, 3).contiguous()
