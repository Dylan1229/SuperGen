"""AdaCache ported into the SuperGen CogVideoX Stage-2 denoiser (P0-1 baseline).

Upstream AdaCache (`AdaCache-DiT/AdaCache`) only ships an **Open-Sora** STDiT3
implementation (`opensora_base/opensora/models/stdit/stdit3.py`), so its released
runner cannot produce a row comparable to ours. This module ports its policy onto
CogVideoX's joint image-text blocks, keeping the rest of the pipeline untouched.

What AdaCache actually does (upstream `compute_next_step`)
---------------------------------------------------------
Per *block*, it caches the residuals **inside** the block and reuses them for the
next `rate` steps, where `rate` is looked up in a codebook:

    cache_diff = ||cache - res||_p / ||cache||_p     (p=1)
    cache_diff /= prev_rate                          (normalize across steps)
    cache_diff  = running mean over the layers that vote
    cache_diff *= moreg                               (optional motion term)
    rate        = codebook[first threshold cache_diff falls under]

So unlike TeaCache (a static, timestep-only schedule) AdaCache **is**
content-dependent: the metric is computed from the block residual. That makes it
the more interesting baseline for our region-aware cache.

Mapping onto CogVideoX
----------------------
Open-Sora STDiT3 has separate spatial-attn / temporal-attn / cross-attn+MLP
residuals. A CogVideoX block has two residual points, each split over image and
text tokens:

    attn:  hidden += gate_msa * attn_hidden      encoder += enc_gate_msa * attn_encoder
    ff:    hidden += gate_ff  * ff_image         encoder += enc_gate_ff  * ff_text

We therefore keep two caches per block -- `attn_cache` and `ff_cache` -- each
holding the (image, text) residual pair, which is the faithful analogue. Both
token streams must be cached: dropping the text residual silently changes the
conditioning path and is a common way to get a "fast but wrong" port.

Deliberate deviations, all recorded because they affect how the row reads
-------------------------------------------------------------------------
1. **Codebook is upstream's default**, unchanged
   (`{0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}`). Upstream
   calibrated it for Open-Sora's scheduler and depth (28 blocks); CogVideoX-1.5
   has 42 blocks and a different scheduler, so the rates are almost certainly
   miscalibrated here. Recalibrating would make this "an AdaCache variant we
   tuned" rather than the released policy -- so instead we expose
   `--adacache_rate_scale` to sweep the whole codebook multiplicatively, the same
   way we sweep TeaCache's single threshold. Report the sweep, not one point.
2. **MoReg is off by default.** Upstream's motion term assumes a `B (T S) C`
   layout with a known spatial stride; CogVideoX gives us `B (text+image) C`
   after patch-embed, so the frame stride is `S = h*w/patch^2` per latent frame.
   We compute it explicitly and gate MoReg behind `--adacache_moreg`, since its
   hyperparameters (0.385, 8, 1, 2) are Open-Sora-specific.
3. **Voting layers**: upstream lets a subset of blocks (`cache_loc`) vote on the
   metric. We default to a single mid-depth block, matching the spirit of
   upstream's "one selected block drives the schedule".
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from diffusers.utils import logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .cogvideo_transformer_3d import CachingCogVideoXTransformer3DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Upstream default codebook: metric threshold -> how many steps to reuse.
UPSTREAM_CODEBOOK = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}
UPSTREAM_MOREG_HYP = (0.385, 8, 1, 2)


class AdaCacheCogVideoXTransformer3DModel(CachingCogVideoXTransformer3DModel):
    """CogVideoX-1.5 I2V transformer whose Stage-2 reuse is gated by AdaCache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adacache_enabled = False
        self.codebook = dict(UPSTREAM_CODEBOOK)
        self.rate_scale = 1.0
        self.apply_moreg = False
        self.moreg_strides = [1]
        self.moreg_steps = (10, 90)
        self.moreg_hyp = UPSTREAM_MOREG_HYP
        self.mograd_mul = 1.0
        self.norm_ord = 1
        self.vote_blocks = None          # which blocks vote on the metric
        self._prev_moreg = 1.0
        # per-block residual caches: blk -> (image_residual, text_residual)
        self._attn_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._ff_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_rate = 3             # steps to reuse before recomputing
        self._next_compute_step = 0
        self._spatial_dim = None
        self.adacache_stats = {"checked": 0, "skipped": 0, "rates": []}

    # ------------------------------------------------------------------ setup
    def setup_adacache(
        self,
        rate_scale: float = 1.0,
        apply_moreg: bool = False,
        codebook: Optional[Dict[float, int]] = None,
        vote_block: Optional[int] = None,
    ):
        """Configure the AdaCache policy. rate_scale multiplies every codebook rate."""
        self.adacache_enabled = True
        self.codebook = dict(codebook or UPSTREAM_CODEBOOK)
        self.rate_scale = float(rate_scale)
        self.apply_moreg = bool(apply_moreg)
        num_layers = len(self.transformer_blocks)
        # Upstream drives the schedule from a selected block; use mid-depth.
        self.vote_blocks = [num_layers // 2 if vote_block is None else vote_block]
        self._prev_moreg = 1.0
        self._attn_cache.clear()
        self._ff_cache.clear()
        self._cache_rate = 3
        self._next_compute_step = 0
        self.adacache_stats = {"checked": 0, "skipped": 0, "rates": []}
        logger.info(
            f"AdaCache gate active: rate_scale={rate_scale} moreg={apply_moreg} "
            f"vote_blocks={self.vote_blocks} codebook={self.codebook}"
        )

    # ------------------------------------------------------------------- gate
    def _compute_next_rate(self, cache, res, step_index):
        """Upstream compute_next_step, transcribed for CogVideoX residuals."""
        p = self.norm_ord
        denom = cache.norm(p=p)
        if denom == 0:
            return self._cache_rate
        cache_diff = (cache - res).norm(p=p) / denom
        # normalize across steps by how long we just reused
        cache_diff = cache_diff / max(self._cache_rate, 1)

        if self.apply_moreg and self._spatial_dim:
            lo, hi = self.moreg_steps
            if lo <= step_index <= hi:
                s = self._spatial_dim
                moreg = 0.0
                for i in self.moreg_strides:
                    off = i * s
                    if res.shape[1] <= off:
                        continue
                    a, b = res[:, off:, :], res[:, :-off, :]
                    num = (a - b).norm(p=p)
                    den = a.norm(p=p) + b.norm(p=p)
                    if den > 0:
                        moreg += (num / den)
                moreg = moreg / max(len(self.moreg_strides), 1)
                moreg = ((1 / self.moreg_hyp[0] * moreg) ** self.moreg_hyp[1]) / self.moreg_hyp[2]
                moreg = float(moreg)
            else:
                moreg = 1.0
            mograd = self.mograd_mul * (moreg - self._prev_moreg) / max(self._cache_rate, 1)
            self._prev_moreg = moreg
            moreg = moreg + abs(mograd)
        else:
            moreg = 1.0

        cache_diff = float(cache_diff) * moreg

        thresholds = sorted(self.codebook)
        rate = self.codebook[thresholds[-1]]
        for t in thresholds:
            if cache_diff < t:
                rate = self.codebook[t]
                break
        # rate_scale is our sweep knob, standing in for upstream's per-model
        # codebook recalibration (which we deliberately do not perform).
        return max(1, int(round(rate * self.rate_scale)))

    def check_skippable(
        self,
        step_index: int,
        tile_index: int,
        hidden_states: torch.Tensor,
        is_non_shifting_step: bool,
        effective_cache_thresh: Optional[float] = None,
        return_dict: bool = False,
        window_position: Tuple = None,
        timestep: Union[int, float, torch.LongTensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    ) -> Tuple[bool, Any]:
        """AdaCache decides reuse by a step schedule set the last time it computed."""
        if not self.enable_cache:
            raise RuntimeError("check_skippable called with caching disabled")

        self.adacache_stats["checked"] += 1

        # Always compute the first and last step, as every cache method here does.
        if step_index == 0 or step_index >= self.num_steps - 1:
            self._next_compute_step = step_index
            return False, None

        # Token-space residuals are indexed relative to the window origin, so a
        # shifted window invalidates them -- same structural limitation as
        # TeaCache's token mode. At 720p (one window, shift step 0) this never
        # fires; at 2K with shifting it disables reuse, which is the honest
        # measurement, not a tuning failure.
        if not self._attn_cache:
            return False, None
        if window_position is not None and getattr(self, "_cache_position", None) not in (
            None, tuple(window_position)
        ):
            self._attn_cache.clear()
            self._ff_cache.clear()
            logger.info(f"AdaCache residuals invalidated by window shift (step {step_index})")
            return False, None

        if step_index >= self._next_compute_step:
            return False, None

        self.adacache_stats["skipped"] += 1
        logger.info(
            f"rank={self.dist_manager.rank} AdaCache hit, step {step_index} "
            f"skipped for tile {tile_index} (reusing until {self._next_compute_step})"
        )
        return True, None

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        tile_index: int = None,
        step_index: Optional[int] = None,
        is_non_shifting_step: bool = False,
        effective_cache_thresh: Optional[float] = None,
        return_dict: bool = True,
        window_position: Tuple = None,
    ):
        if not self.adacache_enabled:
            return super().forward(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                timestep=timestep, timestep_cond=timestep_cond, ofs=ofs,
                image_rotary_emb=image_rotary_emb, attention_kwargs=attention_kwargs,
                tile_index=tile_index, step_index=step_index,
                is_non_shifting_step=is_non_shifting_step,
                effective_cache_thresh=effective_cache_thresh,
                return_dict=return_dict, window_position=window_position,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_frames, channels, height, width = hidden_states.shape
        reuse = self.enable_cache and self.dist_manager.tile_is_skipped(tile_index)

        # 1. Time embedding
        t_emb = self.time_proj(timestep).to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.ofs_proj(ofs).to(dtype=hidden_states.dtype)
            emb = emb + self.ofs_embedding(ofs_emb)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        p, p_t = self.config.patch_size, self.config.patch_size_t
        # tokens per latent frame, for MoReg's frame stride
        self._spatial_dim = (height // p) * (width // p)

        # 3. Transformer blocks, with per-block residual reuse
        new_rate = None
        for blk_id, block in enumerate(self.transformer_blocks):
            if reuse and blk_id in self._attn_cache:
                # Reuse both residual points, image and text streams alike.
                a_img, a_txt = self._attn_cache[blk_id]
                f_img, f_txt = self._ff_cache[blk_id]
                hidden_states = hidden_states + a_img
                encoder_hidden_states = encoder_hidden_states + a_txt
                hidden_states = hidden_states + f_img
                encoder_hidden_states = encoder_hidden_states + f_txt
                continue

            h_before, e_before = hidden_states, encoder_hidden_states

            # -- attention half of the block --
            norm_h, norm_e, gate_msa, enc_gate_msa = block.norm1(
                hidden_states, encoder_hidden_states, emb
            )
            attn_h, attn_e = block.attn1(
                hidden_states=norm_h, encoder_hidden_states=norm_e,
                image_rotary_emb=image_rotary_emb, **(attention_kwargs or {}),
            )
            attn_res_img = gate_msa * attn_h
            attn_res_txt = enc_gate_msa * attn_e
            hidden_states = hidden_states + attn_res_img
            encoder_hidden_states = encoder_hidden_states + attn_res_txt

            # -- feed-forward half --
            norm_h, norm_e, gate_ff, enc_gate_ff = block.norm2(
                hidden_states, encoder_hidden_states, emb
            )
            norm_cat = torch.cat([norm_e, norm_h], dim=1)
            ff_out = block.ff(norm_cat)
            ff_res_img = gate_ff * ff_out[:, text_seq_length:]
            ff_res_txt = enc_gate_ff * ff_out[:, :text_seq_length]
            hidden_states = hidden_states + ff_res_img
            encoder_hidden_states = encoder_hidden_states + ff_res_txt

            # The voting block sets the next reuse length, from the change in its
            # attention residual -- upstream's metric, on CogVideoX's residual.
            if self.vote_blocks and blk_id in self.vote_blocks and blk_id in self._attn_cache:
                prev_img, _ = self._attn_cache[blk_id]
                if prev_img.shape == attn_res_img.shape:
                    new_rate = self._compute_next_rate(prev_img, attn_res_img, step_index or 0)

            self._attn_cache[blk_id] = (attn_res_img.detach(), attn_res_txt.detach())
            self._ff_cache[blk_id] = (ff_res_img.detach(), ff_res_txt.detach())

        if not reuse:
            if window_position is not None:
                self._cache_position = tuple(window_position)
            if new_rate is not None:
                self._cache_rate = new_rate
            self.adacache_stats["rates"].append(self._cache_rate)
            self._next_compute_step = (step_index or 0) + self._cache_rate
            logger.info(
                f"AdaCache computed step {step_index}, rate={self._cache_rate}, "
                f"next compute at {self._next_compute_step}"
            )

        if not self.config.use_rotary_positional_embeddings:
            hidden_states = self.norm_final(hidden_states)
        else:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    # ------------------------------------------------------------------ report
    def log_adacache_report(self):
        checked = self.adacache_stats["checked"]
        skipped = self.adacache_stats["skipped"]
        rate = (skipped / checked) if checked else 0.0
        rates = self.adacache_stats["rates"]
        avg_rate = sum(rates) / len(rates) if rates else 0.0
        logger.info(
            f"ADACACHE_SKIP_RATE {rate:.4f} skipped={skipped} checked={checked} "
            f"avg_rate={avg_rate:.2f} rates={rates[:20]}"
        )
        return {"skip_rate": rate, "skipped": skipped, "checked": checked, "avg_rate": avg_rate}
