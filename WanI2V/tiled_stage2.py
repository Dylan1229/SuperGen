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
                 loop_step: int = 16, device: str = "cuda",
                 dist_manager=None):
        self.wan = wan_i2v
        # Tile parallelism. `None` keeps the single-process path (world_size 1 is
        # also fine, but skipping the manager avoids requiring a process group at
        # all for quick single-GPU checks).
        self.dm = dist_manager
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

        Encoding is tiled spatially. Unlike diffusers' CogVideoX VAE, Wan's has no
        spatial tiling at all -- only temporal chunking (`vae.py` caches frames
        between chunks) -- so a whole 4K frame hits a single 80 GiB conv allocation
        and OOMs on an 80 GB H100.
        """
        import torch.nn.functional as F
        video = self.wan.vae.decode([latent])[0]        # [C, F, H, W] in [-1, 1]
        C, Fr, H, W = video.shape
        frames = video.permute(1, 0, 2, 3)              # [F, C, H, W]
        up = F.interpolate(frames.float(), size=(target_h, target_w),
                           mode="bicubic", align_corners=False)
        up = up.clamp(-1, 1).to(video.dtype).permute(1, 0, 2, 3)   # [C, F, h, w]
        del video, frames
        torch.cuda.empty_cache()
        return self.encode_tiled(up)

    def encode_tiled(self, pixels: torch.Tensor, max_pixels_per_tile: int = 1280 * 720,
                     overlap: int = 64) -> torch.Tensor:
        """VAE-encode a large frame in overlapping spatial tiles.

        Encoding is not a no-op on tile boundaries -- the encoder has receptive field
        -- so tiles overlap in pixel space and only the interior of each tile is
        kept, with the overlap discarded. `overlap` is 64 px = 8 latent units, past
        the convolutional receptive field.

        The convolutions are then equivalent to a whole-frame encode, but the encoder
        is NOT purely convolutional: `middle.1` is an AttentionBlock over all spatial
        positions at 1/8 resolution, and no finite overlap reproduces a global
        operator. Measured at 2K against the whole-frame result:

            with the attention:      max 4.271%  mean 0.221%   of |latent|
            attention stubbed out:   max 0.314%  mean 0.013%

        so essentially the entire residual is that one block, and it does not shrink
        usefully with overlap (256 px, 4x the compute, only gets max to 2.808%).

        Left as is deliberately. The alternative -- reassembling the post-downsample
        feature map so attention runs once over the full grid -- is a rewrite of the
        encoder's chunking, and the error it would remove is negligible where it
        lands: Stage 2 immediately re-noises this latent to sigma=0.9208, so the
        injected noise is ~700x the encode error (~1440x at 4K's sigma). The
        difference is gone after the first denoising step.

        Whole-frame encode is not an option at 4K regardless: Wan's VAE has no
        spatial tiling of its own (only temporal, `vae.py:516` splits time into
        1+4+4+...), and a full 4K frame asks for a single 80 GiB conv allocation.

        Below the threshold this is a single whole-frame call, so 720p and 2K keep the
        exact previous behaviour and only 4K takes the tiled path.
        """
        C, Fr, H, W = pixels.shape
        if H * W <= max_pixels_per_tile:
            return self.wan.vae.encode([pixels])[0]

        sh, sw = self.vae_stride[1], self.vae_stride[2]
        # Tile in halves/thirds along each axis so tile edges land on VAE-stride
        # multiples; a ragged last tile would misalign the latent write.
        n_h = max(1, -(-H // 1088))
        n_w = max(1, -(-W // 1920))
        th = ((H // n_h) // sh) * sh
        tw = ((W // n_w) // sw) * sw
        logger.info(f"VAE encode tiled: {H}x{W} -> {n_h}x{n_w} tiles of {th}x{tw} "
                    f"(+{overlap}px overlap); Wan's VAE has no spatial tiling")

        lat_f = (Fr - 1) // self.vae_stride[0] + 1
        out = None
        for i in range(n_h):
            for j in range(n_w):
                top = i * th
                left = j * tw
                bot = H if i == n_h - 1 else top + th
                right = W if j == n_w - 1 else left + tw
                # Pad outward for receptive field, snapped to the VAE stride.
                pt = max(0, top - overlap)
                pl = max(0, left - overlap)
                pb = min(H, bot + overlap)
                pr = min(W, right + overlap)
                pt -= pt % sh
                pl -= pl % sw
                pb += (-pb) % sh
                pr += (-pr) % sw
                pb, pr = min(pb, H), min(pr, W)

                enc = self.wan.vae.encode([pixels[:, :, pt:pb, pl:pr]])[0]
                if out is None:
                    out = torch.zeros((enc.shape[0], lat_f, H // sh, W // sw),
                                      device=enc.device, dtype=enc.dtype)
                # Drop the padded margin: keep only the region this tile owns.
                ct = (top - pt) // sh
                cl = (left - pl) // sw
                keep_h = (bot - top) // sh
                keep_w = (right - left) // sw
                out[:, :, top // sh:top // sh + keep_h,
                    left // sw:left // sw + keep_w] = \
                    enc[:, :, ct:ct + keep_h, cl:cl + keep_w]
                del enc
                torch.cuda.empty_cache()
        return out

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
        step_h, step_w = wp["latent_step_size_h"], wp["latent_step_size_w"]
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

        # ------------------------------------------------------- tile parallelism
        # 4K is 9 tiles and 2K is 4, so one rank per tile is the natural split and
        # it is what makes the runtime comparison against FreeSwim/CineScale
        # meaningful -- their single global sequence has no equivalent axis to shard
        # except sequence parallelism. The manager owns the canvas from here on:
        # `get_tile`/`update_tile` are shift-aware and `allgather_fused_noise`
        # rebuilds the whole canvas from every rank's tiles each step.
        if self.dm is not None:
            # y is the I2V conditioning; setup_config's second slot is exactly the
            # "conditioning latent sliced with the same window" role, so T2V (no y)
            # passes a zero tensor of the same shape rather than a special case.
            img_lat = (y_tiler.torch_latent if y_tiler is not None
                       else torch.zeros_like(canvas.torch_latent))
            self.dm.setup_config(
                canvas.torch_latent, img_lat, wcfg,
                noise_fusion_method="weighted_average",
                std_tracker_update_interval=5,
                # Wan issues cond and uncond as two separate forward calls and
                # combines them here, so the manager only ever sees a single fused
                # prediction per tile -- not a batched CFG pair.
                do_classifier_free_guidance=False,
            )
            return self._denoise_distributed(
                timesteps, sample_scheduler, arg_c, arg_null, guide_scale,
                shift_timesteps, seed_g, enable_cache, y_tiler is not None,
                win_h, win_w, step_h, step_w, autocast_dtype=self._param_dtype)

        shift_h = shift_w = 0
        autocast_ctx = torch.amp.autocast("cuda", dtype=self._param_dtype)
        for step_idx, t in enumerate(timesteps):
            if shift_timesteps is not None and step_idx in shift_timesteps:
                # Wrap on each axis's own closing cycle (window/step), not on a
                # shared loop_step: with window 90x160 those are 18 and 16, so a
                # single modulus would leave a band on the height axis that the
                # tile boundary never visits. Matches DistributedManager.shift().
                cyc_h = (win_h // step_h) if step_h else 1
                cyc_w = (win_w // step_w) if step_w else 1
                shift_h = (shift_h + 1) % max(cyc_h, 1)
                shift_w = (shift_w + 1) % max(cyc_w, 1)

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

    # ------------------------------------------------- distributed denoise loop
    def _denoise_distributed(
        self, timesteps, sample_scheduler, arg_c, arg_null, guide_scale,
        shift_timesteps, seed_g, enable_cache, has_y,
        win_h, win_w, step_h, step_w, autocast_dtype,
    ):
        """The same loop as `denoise`, with the tiles split across ranks.

        Mirrors the structure CogVideoX and Hunyuan already use
        (`pipeline_cogvideox_i2v_TVG.py:655-811`): per step, clear the fuser,
        communicate the shifted canvas, run only this rank's tiles, then allgather
        the fused prediction so every rank steps the scheduler on an identical
        canvas. Keeping the scheduler replicated rather than sharded is what makes
        the multi-GPU output bit-comparable to the single-GPU one.
        """
        import torch.distributed as dist
        dm = self.dm
        autocast_ctx = torch.amp.autocast("cuda", dtype=autocast_dtype)
        logger.info(f"[rank={dm.rank}] Wan Stage-2 tile parallelism: "
                    f"{dm.num_total_windows} tiles over {dm.world_size} ranks, "
                    f"local={dm.get_local_indices()}")

        for step_idx, t in enumerate(timesteps):
            dm.clear()
            if shift_timesteps is not None and step_idx in shift_timesteps:
                # The canvas has to be materialised on every rank before it is
                # re-cut, because a shift moves tile boundaries across the previous
                # owner's slice.
                dm.communicate("latent")
                dm.shift()

            for tile_idx in dm.get_local_indices():
                tile, y_tile = dm.get_tile(tile_idx)
                top, _, left, _ = dm.get_tile_boundary_for_idx(tile_idx)
                window = dm.get_tile_boundary_for_idx(tile_idx)

                if self._rope is not None:
                    self._rope.set_tile(0, top // self.patch_size[1],
                                        left // self.patch_size[2])

                reuse = False
                if enable_cache and self.cache is not None:
                    reuse = self.cache.should_skip(step_idx, tile_idx, tile, window)

                if reuse:
                    out = self.cache.reuse(tile, window)
                else:
                    model_in = [tile.squeeze(0).permute(1, 0, 2, 3).contiguous()]
                    ts = torch.stack([t]).to(self.device)
                    kw_c, kw_null = dict(arg_c), dict(arg_null)
                    if has_y:
                        yt = y_tile.squeeze(0).permute(1, 0, 2, 3).contiguous()
                        kw_c["y"] = [yt]
                        kw_null["y"] = [yt]
                    with autocast_ctx, torch.no_grad():
                        pred_c = self.wan.model(model_in, t=ts, **kw_c)[0]
                        pred_u = self.wan.model(model_in, t=ts, **kw_null)[0]
                    pred = pred_u + guide_scale * (pred_c - pred_u)
                    out = pred.permute(1, 0, 2, 3).unsqueeze(0).to(
                        dm.get_latents().dtype)
                    if enable_cache and self.cache is not None:
                        self.cache.record(step_idx, tile_idx, tile, out, window)

                dm.tile_noise_fuser_add(tile_idx, out, tile_weight=1.0)

            # Every rank ends the step holding the identical full-canvas prediction.
            noise_pred = dm.allgather_fused_noise()
            npred = noise_pred.squeeze(0).permute(1, 0, 2, 3).unsqueeze(0)
            cur = dm.get_latents().squeeze(0).permute(1, 0, 2, 3).unsqueeze(0)
            stepped = sample_scheduler.step(npred, t, cur, return_dict=False,
                                           generator=seed_g)[0]
            dm.set_latents(stepped.squeeze(0).permute(1, 0, 2, 3)
                                  .unsqueeze(0).contiguous())

        if enable_cache and self.cache is not None:
            self.cache.report()
        return dm.get_latents().squeeze(0).permute(1, 0, 2, 3).contiguous()

