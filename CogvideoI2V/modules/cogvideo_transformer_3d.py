import json
import os
from typing import Any, Dict, Optional, Tuple, Union, List, TYPE_CHECKING

import torch
import numpy as np
from scipy.spatial.distance import cosine

from diffusers.utils import logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_utils import is_torch_version
from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXTransformer3DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from utils.tile_utils import TiledLatentTensor2D

if TYPE_CHECKING:
    from utils.distributed import DistributedManager

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CachingCogVideoXTransformer3DModel(CogVideoXTransformer3DModel):
    dist_manager: Optional["DistributedManager"] = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize cache attributes for tile-based caching
        self.enable_cache = True
    
        self.num_steps = None
        self.thresh = None
        self.ret_steps = None
        
        # K history tracking
        self.k_history = {}
        self.enable_k_tracking = False
        # noise_pred profiling
        self.enable_noise_pred_profile = False
        self.noise_pred_history = {}
        # cache_residual profiling
        self.enable_cache_residual_profile = False
        self.cache_residual_history = {}
    
    def allocate(self, latents):
        self.previous_raw_input = TiledLatentTensor2D(latent_tensor=latents)
        self.previous_output = TiledLatentTensor2D(latent_tensor=latents)
        self.prev_prev_raw_input = TiledLatentTensor2D(latent_tensor=latents)
        self.prev_prev_output = TiledLatentTensor2D(latent_tensor=latents)
        self.cache_residual = TiledLatentTensor2D(latent_tensor=latents)
        self.accumulated_error = TiledLatentTensor2D(latent_tensor=latents)
        self.previous_raw_input.zero_()
        self.previous_output.zero_()
        self.prev_prev_raw_input.zero_()
        self.prev_prev_output.zero_()
        self.cache_residual.zero_()
        self.accumulated_error.zero_()

        self.dist_manager.register_cache_ring2d(
            self.previous_raw_input,
            self.previous_output,
            self.prev_prev_raw_input,
            self.prev_prev_output,
            self.cache_residual,
            self.accumulated_error,
        )

    def setup_cache_per_tile(
        self,
        num_steps: int,
        thresh: float,
        ret_steps: int,
        num_tiles: int,
    ):
        self.enable_cache = True
        self.num_steps = num_steps
        self.thresh = thresh
        self.ret_steps = ret_steps

        # Initialize per-tile cache attributes
        for tile_idx in range(num_tiles):

            # Initialize k history tracking for each tile
            if self.enable_k_tracking:
                self.k_history[tile_idx] = []
            
            # noise_pred profile init
            if self.enable_noise_pred_profile:
                self.noise_pred_history[tile_idx] = []
            
            # cache_residual profile init
            if self.enable_cache_residual_profile:
                self.cache_residual_history[tile_idx] = []

    def check_skippable(
        self,
        step_index: int,
        tile_index: int,
        hidden_states: torch.Tensor,
        is_non_shifting_step: bool,
        effective_cache_thresh: Optional[float] = None,
        return_dict: bool = False,
        window_position: Tuple = None
    ) -> Tuple[bool, Any]:
        if not self.enable_cache:
            raise RuntimeError
        
        batch_size, num_frames, channels, height, width = hidden_states.shape
        
        # Separate noise latent from the concatenated input for caching
        noise_latent_channels = channels // 2
        raw_input = hidden_states[:, :, :noise_latent_channels, :, :].clone()

        cnt = step_index
        current_thresh = effective_cache_thresh if effective_cache_thresh is not None else self.thresh
        if cnt < self.ret_steps or cnt >= self.num_steps - 1:
            should_calc = True
            # self.dist_manager.set_tensor_in_buffer("accumulated_error", tile_index, 0.0)
        elif step_index > 2:
            # prev_raw_input = self.previous_raw_input[tile_index]
            # prev_output = self.previous_output[tile_index]
            prev_raw_input = self.previous_raw_input.get_window_latent(*window_position)
            raw_input_change = (raw_input - prev_raw_input).abs().mean()
            
            prev_output = self.previous_output.get_window_latent(*window_position)
            prev_prev_output = self.prev_prev_output.get_window_latent(*window_position)
            output_change = (prev_output - prev_prev_output).abs().mean()

            prev_prev_input = self.prev_prev_raw_input.get_window_latent(*window_position)
            prev_input_change = (prev_raw_input - prev_prev_input).abs().mean()

            k = output_change / prev_input_change
            # _, _, _, h, w = k.shape
            # k = k[:,:,:,h//2-20:h//2+20, w//2-35:w//2+35].mean()

            output_norm = prev_output.abs().mean()
            pred_change = k * (raw_input_change / output_norm)
            logger.info(f"Tile {tile_index} at step {step_index}: k={k}, raw_input_change={raw_input_change}, output_norm={output_norm}")

            intermediate_accumulate_error = self.accumulated_error.get_window_latent(*window_position) + pred_change
            self.accumulated_error.set_window_latent(intermediate_accumulate_error, *window_position)
            # Only do reuse when non-shifting step and smaller than threshold.
            # if self.accumulated_error[tile_index] < self.thresh and is_non_shifting_step:
            # if sel.accumulated_error.set_window_latent(torch.zeros_like(intermediate_accumulate_error), *window_position)
            if intermediate_accumulate_error.mean() < current_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_error.set_window_latent(torch.zeros_like(raw_input), *window_position)

        # TODO(MX): also return the predicted output, so that we don't need to call `forward` again
        # If cache hit, return the cached result.
        if should_calc:
            logger.info(f"rank={self.dist_manager.rank} Cache miss, step {step_index} should recalculate tile {tile_index}")
            return False, None

        logger.info(f"rank={self.dist_manager.rank} Cache hit, step {step_index} is skipped for tile {tile_index}")
        self.prev_prev_raw_input.set_window_latent(prev_raw_input, *window_position)
        self.previous_raw_input.set_window_latent(raw_input, *window_position)
        self.prev_prev_output.set_window_latent(prev_output, *window_position)
        return True, None
    
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
        window_position: Tuple = None
    ):

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape
        
        # Separate noise latent from the concatenated input for caching
        noise_latent_channels = channels // 2
        raw_input = hidden_states[:, :, :noise_latent_channels, :, :].clone()

        if self.enable_cache:
            if self.dist_manager.tile_is_skipped(tile_index):
                cache_residual = self.cache_residual.get_window_latent(*window_position)
                result = raw_input + cache_residual
                self.previous_output.set_window_latent(result, *window_position)
                if not return_dict:
                    return (result,)
                return Transformer2DModelOutput(sample=result)
        
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
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
        
        if self.enable_cache:
            self.cache_residual.set_window_latent(output - raw_input, *window_position)
            # if self.enable_cache_residual_profile:
            #     cache_residual = output - raw_input
            #     self.record_cache_residual(tile_index, step_index, cache_residual)

            self.prev_prev_raw_input.set_window_latent(self.previous_raw_input.get_window_latent(*window_position), *window_position)
            self.previous_raw_input.set_window_latent(raw_input, *window_position)
            self.prev_prev_output.set_window_latent(self.previous_output.get_window_latent(*window_position), *window_position)
            self.previous_output.set_window_latent(output, *window_position)
        #-------------------Profiling logic-------------------
        # Note: here if we want to record all the k and cache residual, we need to do it in the shifting step because we may do reuse in the non-shifting step. (skip recording current k and residual).
        # But because shifting will cause the k and residual to be different, we need to do it in the non-shifting step.
        # Need to rewrite the profiling logics: only when stop caching + always record these parameters + non-shifting is correct.
        #-------------------Profiling logic-------------------
            # Record k value in history.
            # if self.enable_k_tracking:
            #     k_value = float(self.dist_manager.get_tensor_in_buffer("k", tile_index))
            #     self.k_history[tile_index].append([step_index, k_value])
        
            # if self.enable_cache_residual_profile:
            #     cache_residual = output - raw_input
            #     self.record_cache_residual(tile_index, step_index, cache_residual)
            
        if not return_dict:
            return (output,)
        
        return Transformer2DModelOutput(sample=output) 

    def save_k_history(self, filename: str = "k_history.json"):
        """Save k history to JSON file"""
        if not self.enable_k_tracking:
            logger.warning("K tracking is disabled, no history to save")
            return
        
        if not self.k_history:
            logger.warning("No k history available to save")
            return
            
        # Convert k_history to serializable format
        k_history_serializable = {}
        for tile_idx, history in self.k_history.items():
            k_history_serializable[str(tile_idx)] = history
        
        # Prepare data structure
        data = {
            "metadata": {
                "num_tiles": len(self.k_history),
                "num_steps": self.num_steps,
                "thresh": self.thresh,
                "ret_steps": self.ret_steps
            },
            "k_history": k_history_serializable
        }
        
        # Save to file
        filepath = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"K history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save k history: {e}")
            
    def record_noise_pred(self, tile_idx: int, step: int, noise_pred: torch.Tensor):
        if self.enable_noise_pred_profile:
            self.noise_pred_history.setdefault(tile_idx, []).append((step, noise_pred.detach().cpu()))

    def save_noise_pred_profile(self, filename: str = "noise_pred_profile.json"):

        if not self.enable_noise_pred_profile or not self.noise_pred_history:
            logger.warning("Noise pred profiling is disabled or empty.")
            return
        profile = {}
        for tile_idx, history in self.noise_pred_history.items():
            relative_l1_norms = []
            cos_sims = []
            prev_vec = None
            for step, tensor in history:
                arr = tensor.flatten().numpy()
                if prev_vec is not None:
                    # Calculate relative L1 norm: ||current - prev||_1 / ||prev||_1
                    diff = np.abs(arr - prev_vec)
                    relative_l1 = float(diff.mean() / np.abs(prev_vec).mean())
                    relative_l1_norms.append((step, relative_l1))
                    
                    # Calculate cosine similarity
                    cos_sim = 1 - cosine(arr, prev_vec)
                    cos_sims.append((step, float(cos_sim)))
                else:
                    # First step: no previous to compare with
                    relative_l1_norms.append((step, None))
                    cos_sims.append((step, None))
                prev_vec = arr
            profile[tile_idx] = {
                "relative_l1_norm": relative_l1_norms,
                "cosine_similarity": cos_sims
            }
        # Save as json
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Noise pred profile saved to {filepath}") 

    def record_cache_residual(self, tile_idx: int, step: int, cache_residual: torch.Tensor):
        if self.enable_cache_residual_profile:
            self.cache_residual_history.setdefault(tile_idx, []).append((step, cache_residual.detach().cpu().float()))

    def save_cache_residual_profile(self, filename: str = "cache_residual_profile.json"):
        if not self.enable_cache_residual_profile or not self.cache_residual_history:
            logger.warning("Cache residual profiling is disabled or empty.")
            return
        profile = {}
        for tile_idx, history in self.cache_residual_history.items():
            relative_l1_norms = []
            cos_sims = []
            prev_vec = None
            for step, tensor in history:
                arr = tensor.flatten().numpy()
                if prev_vec is not None:
                    diff = np.abs(arr - prev_vec)
                    relative_l1 = float(diff.mean() / np.abs(prev_vec).mean())
                    relative_l1_norms.append((step, relative_l1))
                    cos_sim = 1 - cosine(arr, prev_vec)
                    cos_sims.append((step, float(cos_sim)))
                else:
                    relative_l1_norms.append((step, None))
                    cos_sims.append((step, None))
                prev_vec = arr
            profile[tile_idx] = {
                "relative_l1_norm": relative_l1_norms,
                "cosine_similarity": cos_sims
            }
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Cache residual profile saved to {filepath}") 