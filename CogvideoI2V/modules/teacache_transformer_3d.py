"""TeaCache ported into the SuperGen Stage-2 tiled denoiser (P0-1 baseline).

Upstream TeaCache (`ali-vilab/TeaCache`, `TeaCache4CogVideoX1.5`) monkey-patches
`CogVideoXTransformer3DModel.forward` and caches a whole-canvas residual. That
runner cannot enter a same-model table against UltraGen because it bypasses the
two-stage tiling pipeline entirely.

This module instead subclasses `CachingCogVideoXTransformer3DModel` and overrides
only the *cache decision* (and, in `token` storage mode, the residual bookkeeping).
Tiling, sliding-window shifting, tile parallelism, the scheduler, and the Stage-2
loop are inherited unchanged, so a TeaCache row differs from an UltraGen row in
exactly one respect: which policy decides whether a tile recomputes.

Gate (faithful to upstream)
---------------------------
    acc += poly(||emb_t - emb_{t-1}||_1 / ||emb_{t-1}||_1)
    skip if acc < rel_l1_thresh, else recompute and reset acc

`emb` is the timestep (+ ofs) embedding, exactly as upstream uses for
CogVideoX-1.5. NOTE that `emb` is a function of the timestep alone -- it does not
depend on latent content. Consequently TeaCache's decision is *identical for
every tile* at a given step: it is a static skip schedule, invariant to prompt
and to region. Recording that is a result, not a bug; per-tile accumulators are
kept anyway so the divergence (or lack of it) can be logged and reported.

Storage modes
-------------
`latent` (default, policy-isolating)
    Reuse the inherited canvas-aligned `cache_residual` (see
    `utils.tile_utils.TiledLatentTensor2D`). Identical storage and identical skip
    mechanics to UltraGen, so a speed/quality delta is attributable purely to the
    gate. This is the row to put beside ours.

`token` (faithful-storage)
    Cache the image- and text-token residuals around the transformer blocks, as
    upstream does, and still run patch-embed and the final projection on a skip.
    Higher fidelity to the paper, but the residual lives in per-tile token space,
    which is not canvas-aligned -- so it MUST be invalidated whenever the sliding
    window shifts. Under the paper's default `--shift_timesteps 0..44` (shift
    every step) this yields ~zero reuse. That is the honest measurement of
    TeaCache's incompatibility with window shifting; report it as such rather
    than as a tuning failure.

Both modes ignore `effective_cache_thresh` (UltraGen's region-aware per-tile
threshold) and use the single upstream `rel_l1_thresh`.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.utils import logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .cogvideo_transformer_3d import CachingCogVideoXTransformer3DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Upstream rescaling polynomials (TeaCache4CogVideoX1.5/teacache_sample_video.py).
# Calibrated by the TeaCache authors on the whole canvas at native resolution; we
# reuse them unmodified rather than recalibrating, so the baseline stays "released
# TeaCache" and not a variant we tuned ourselves.
COEFFICIENTS_DICT = {
    "CogVideoX-2b": [-3.10658903e01, 2.54732368e01, -5.92380459e00, 1.75769064e00, -3.61568434e-03],
    "CogVideoX-5b": [-1.53880483e03, 8.43202495e02, -1.34363087e02, 7.97131516e00, -5.23162339e-02],
    "CogVideoX-5b-I2V": [-1.53880483e03, 8.43202495e02, -1.34363087e02, 7.97131516e00, -5.23162339e-02],
    "CogVideoX1.5-5B": [2.50210439e02, -1.65061612e02, 3.57804877e01, -7.81551492e-01, 3.58559703e-02],
    "CogVideoX1.5-5B-I2V": [1.22842302e02, -1.04088754e02, 2.62981677e01, -3.06009921e-01, 3.71213220e-02],
}

DEFAULT_COEFFICIENT_KEY = "CogVideoX1.5-5B-I2V"


class TeaCacheCogVideoXTransformer3DModel(CachingCogVideoXTransformer3DModel):
    """CogVideoX-1.5 I2V transformer whose Stage-2 tile cache is gated by TeaCache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacache_rel_l1_thresh = 0.0
        self.teacache_storage = "latent"
        self.teacache_coefficients = COEFFICIENTS_DICT[DEFAULT_COEFFICIENT_KEY]
        # per-tile gate state
        self._tc_accumulated: Dict[int, float] = {}
        self._tc_prev_emb: Dict[int, torch.Tensor] = {}
        # per-tile token-space residuals (token storage mode only)
        self._tc_residual: Dict[int, torch.Tensor] = {}
        self._tc_residual_encoder: Dict[int, torch.Tensor] = {}
        self._tc_residual_position: Dict[int, Tuple] = {}
        # bookkeeping for the results table
        self.teacache_stats = {"checked": 0, "skipped": 0, "shift_invalidated": 0}
        self._tc_decision_log: Dict[int, list] = {}

    # ------------------------------------------------------------------ setup
    def setup_teacache(
        self,
        rel_l1_thresh: float,
        storage: str = "latent",
        coefficients: Optional[list] = None,
        coefficient_key: str = DEFAULT_COEFFICIENT_KEY,
    ):
        if storage not in ("latent", "token"):
            raise ValueError(f"teacache storage must be 'latent' or 'token', got {storage!r}")
        self.teacache_rel_l1_thresh = rel_l1_thresh
        self.teacache_storage = storage
        if coefficients is not None:
            self.teacache_coefficients = coefficients
        else:
            if coefficient_key not in COEFFICIENTS_DICT:
                raise ValueError(
                    f"no upstream TeaCache coefficients for {coefficient_key!r}; "
                    f"known: {sorted(COEFFICIENTS_DICT)}"
                )
            self.teacache_coefficients = COEFFICIENTS_DICT[coefficient_key]
        self._tc_accumulated.clear()
        self._tc_prev_emb.clear()
        self._tc_residual.clear()
        self._tc_residual_encoder.clear()
        self._tc_residual_position.clear()
        self.teacache_stats = {"checked": 0, "skipped": 0, "shift_invalidated": 0}
        self._tc_decision_log.clear()
        logger.info(
            f"TeaCache gate active: rel_l1_thresh={rel_l1_thresh} storage={storage} "
            f"coefficients={coefficient_key}"
        )

    def _timestep_embedding(
        self,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor],
        ofs: Optional[Union[int, float, torch.LongTensor]],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Recompute the modulation embedding TeaCache gates on (upstream step 1)."""
        t_emb = self.time_proj(timestep).to(dtype=dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.ofs_proj(ofs).to(dtype=dtype)
            emb = emb + self.ofs_embedding(ofs_emb)
        return emb

    # ------------------------------------------------------------------- gate
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
        if not self.enable_cache:
            raise RuntimeError("check_skippable called with caching disabled")
        if timestep is None:
            raise RuntimeError(
                "TeaCache needs the current timestep to build its gate signal; "
                "the Stage-2 loop must pass timestep= into check_skippable"
            )

        self.teacache_stats["checked"] += 1
        emb = self._timestep_embedding(timestep, timestep_cond, ofs, hidden_states.dtype)
        prev_emb = self._tc_prev_emb.get(tile_index)

        # Upstream always computes the first and last step.
        is_edge_step = step_index == 0 or step_index >= self.num_steps - 1

        if is_edge_step or prev_emb is None:
            should_calc = True
            self._tc_accumulated[tile_index] = 0.0
        else:
            rescale = np.poly1d(self.teacache_coefficients)
            rel_l1 = ((emb - prev_emb).abs().mean() / prev_emb.abs().mean()).cpu().item()
            accumulated = self._tc_accumulated.get(tile_index, 0.0) + float(rescale(rel_l1))
            if accumulated < self.teacache_rel_l1_thresh:
                should_calc = False
                self._tc_accumulated[tile_index] = accumulated
            else:
                should_calc = True
                self._tc_accumulated[tile_index] = 0.0

        self._tc_prev_emb[tile_index] = emb

        # Token-space residuals are indexed by patch grid relative to the window
        # origin, so a shifted window invalidates them. Canvas-aligned latent
        # storage does not have this problem.
        if not should_calc and self.teacache_storage == "token":
            cached_position = self._tc_residual_position.get(tile_index)
            if tile_index not in self._tc_residual or cached_position != tuple(window_position):
                should_calc = True
                self._tc_accumulated[tile_index] = 0.0
                self.teacache_stats["shift_invalidated"] += 1
                logger.info(
                    f"TeaCache token residual invalidated by window shift "
                    f"(tile {tile_index}, step {step_index})"
                )

        self._tc_decision_log.setdefault(tile_index, []).append(
            (step_index, bool(not should_calc))
        )

        if should_calc:
            logger.info(
                f"rank={self.dist_manager.rank} TeaCache miss, step {step_index} "
                f"recomputes tile {tile_index}"
            )
            return False, None

        self.teacache_stats["skipped"] += 1
        logger.info(
            f"rank={self.dist_manager.rank} TeaCache hit, step {step_index} "
            f"skipped for tile {tile_index}"
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
        # Latent storage reuses UltraGen's canvas-aligned residual path verbatim,
        # so only the gate above differs between the two rows.
        if self.teacache_storage == "latent":
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                timestep_cond=timestep_cond,
                ofs=ofs,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                tile_index=tile_index,
                step_index=step_index,
                is_non_shifting_step=is_non_shifting_step,
                effective_cache_thresh=effective_cache_thresh,
                return_dict=return_dict,
                window_position=window_position,
            )

        # ---- token storage: cache the residual around the transformer blocks ----
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

        batch_size, num_frames, channels, height, width = hidden_states.shape
        skipped = self.enable_cache and self.dist_manager.tile_is_skipped(tile_index)

        # 1. Time embedding
        emb = self._timestep_embedding(timestep, timestep_cond, ofs, hidden_states.dtype)

        # 2. Patch embedding (upstream runs this even on a cache hit)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks, or the cached residual
        if skipped:
            hidden_states = hidden_states + self._tc_residual[tile_index]
            encoder_hidden_states = encoder_hidden_states + self._tc_residual_encoder[tile_index]
        else:
            ori_hidden_states = hidden_states.clone()
            ori_encoder_hidden_states = encoder_hidden_states.clone()
            for block in self.transformer_blocks:
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    )
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )
            if self.enable_cache:
                self._tc_residual[tile_index] = hidden_states - ori_hidden_states
                self._tc_residual_encoder[tile_index] = (
                    encoder_hidden_states - ori_encoder_hidden_states
                )
                self._tc_residual_position[tile_index] = tuple(window_position)

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
        p = self.config.patch_size
        p_t = self.config.patch_size_t

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
    def teacache_report(self) -> Dict[str, Any]:
        """Skip statistics plus whether the gate ever diverged between tiles."""
        per_tile = {
            tile: [step for step, was_skipped in log if was_skipped]
            for tile, log in self._tc_decision_log.items()
        }
        schedules = {tile: tuple(steps) for tile, steps in per_tile.items()}
        uniform = len(set(schedules.values())) <= 1
        checked = self.teacache_stats["checked"]
        return {
            **self.teacache_stats,
            "skip_rate": (self.teacache_stats["skipped"] / checked) if checked else 0.0,
            "num_tiles_logged": len(per_tile),
            "decision_uniform_across_tiles": uniform,
            "skipped_steps_per_tile": per_tile,
        }

    def log_teacache_report(self):
        report = self.teacache_report()
        logger.info(
            f"TEACACHE_SKIP_RATE {report['skip_rate']:.4f} "
            f"skipped={report['skipped']} checked={report['checked']} "
            f"shift_invalidated={report['shift_invalidated']} "
            f"uniform_across_tiles={report['decision_uniform_across_tiles']}"
        )
        return report
