# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
import json
import os, time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from packaging import version

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
import math

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...text_encoder import TextEncoder
from ...modules import HYVideoDiffusionTransformer
from ...utils.data_utils import black_image
from utils import SlidingWindowConfig
from utils.distributed import DistributedManager
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""

def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)
    )  # [D/2]

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    # logger.info(f"num_inference_steps: {num_inference_steps}, sigmas: {sigmas}, timesteps: {timesteps}")
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
        logger.info("timestep is not None")
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
        # logger.info("sigmas is not None")
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        # logger.info("num_inference_steps is not None")
    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`TextEncoder`]):
            Frozen text-encoder.
        text_encoder_2 ([`TextEncoder`]):
            Frozen text-encoder_2.
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    dist_manager: "DistributedManager" = None

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.args = args
        # ==========================================================================================

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        semantic_images=None
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
            else:
                scale_lora_layers(text_encoder.model, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, semantic_images=semantic_images, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    semantic_images=semantic_images,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(
                    uncond_tokens, text_encoder.tokenizer
                )

            # max_length = prompt_embeds.shape[1]
            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            if semantic_images is not None:
                uncond_image = [black_image(img.size[0], img.size[1]) for img in semantic_images]
            else:
                uncond_image = None

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type, semantic_images=uncond_image, device=device
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        if text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(text_encoder.model, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def decode_latents(self, latents, enable_tiling=True):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        if enable_tiling:
            self.vae.enable_tiling()
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if image.ndim == 4:
            image = image.cpu().permute(0, 2, 3, 1).float()
        else:
            image = image.cpu().float()
        return image

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        video_length,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver="88-4c-sd",
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 4 but is {video_length}."
                    )
            elif "888" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 8 but is {video_length}."
                    )

        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents, 
        height, 
        width,  
        video_length, # Full video latent frame count
        dtype,
        device, 
        generator, 
        latents: Optional[torch.Tensor] = None, # Optional external x_T
        img_latents: Optional[torch.Tensor] = None, 
        i2v_mode=False,
        i2v_condition_type=None,
        i2v_stability=True,
        semantic_images=None,
    ):
        shape_num_channels = self.vae.config.latent_channels
        if i2v_mode and i2v_condition_type == "latent_concat":
            shape_num_channels = (shape_num_channels - 1) // 2

        # Process semantic images if provided for i2v mode
        if i2v_mode and semantic_images is not None and img_latents is None:
            from torchvision import transforms
            
            # Calculate target size for the image
            closest_size = (height, width)
            resize_param = min(closest_size)
            center_crop_param = closest_size

            ref_image_transform = transforms.Compose([
                transforms.Resize(resize_param),
                transforms.CenterCrop(center_crop_param),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            semantic_image_pixel_values = [ref_image_transform(semantic_image) for semantic_image in semantic_images]
            semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                img_latents = self.vae.encode(semantic_image_pixel_values).latent_dist.mode()
                img_latents.mul_(self.vae.config.scaling_factor)  

        shape = (
            batch_size, shape_num_channels, video_length,
            int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Batch size must match length of generators list.")

        creation_device = device
        if isinstance(generator, torch.Generator):
            creation_device = generator.device
        elif isinstance(generator, list) and len(generator) > 0:
            creation_device = generator[0].device

        if i2v_mode and i2v_stability:
            if img_latents.shape[2] == 1:
                img_latents_ = img_latents.repeat(1, 1, video_length, 1, 1)

            if latents is None:
                # if img_latents.shape[2] == 1:
                #     img_latents = img_latents.repeat(1, 1, video_length, 1, 1)
                x0 = randn_tensor(shape, generator=generator, device=creation_device, dtype=dtype)

                t_stability = torch.tensor([0.999], device=creation_device, dtype=dtype) 
                prepared_latents = x0 * t_stability + img_latents_ * (1 - t_stability)
                prepared_latents = prepared_latents.to(dtype=dtype) 
            else:
                x0 = latents
                t_stability = torch.tensor([0.999], device=creation_device, dtype=dtype) 
                prepared_latents = x0 * t_stability + img_latents_ * (1 - t_stability)
                if prepared_latents.shape != shape:
                    raise ValueError(f"Provided latents shape {prepared_latents.shape} doesn't match {shape}.")
            if prepared_latents.shape != shape:
                raise ValueError(f"Provided latents shape {prepared_latents.shape} doesn't match {shape}.")
        
        elif latents is None: 
            prepared_latents = randn_tensor(shape, generator=generator, device=creation_device, dtype=dtype)
        else: 
            prepared_latents = latents.to(creation_device, dtype=dtype) 
            if prepared_latents.shape != shape:
                raise ValueError(f"Provided latents shape {prepared_latents.shape} doesn't match {shape}.")

        prepared_latents = prepared_latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            prepared_latents = prepared_latents * self.scheduler.init_noise_sigma
        return prepared_latents, img_latents


    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2  # B, C, F, H, W -> F, H, W
    
        # Compute latent sizes based on VAE type
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]
    
        # Compute rope sizes - use transformer instead of model
        if isinstance(self.transformer.config.patch_size, int):
            assert all(s % self.transformer.config.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.transformer.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.transformer.config.patch_size for s in latents_size]
        elif isinstance(self.transformer.config.patch_size, list):
            assert all(
                s % self.transformer.config.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.transformer.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.transformer.config.patch_size[idx] for idx, s in enumerate(latents_size)]
    
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # Pad time axis
    
        # 20250316 pftq: Add RIFLEx logic for > 192 frames
        L_test = rope_sizes[0]  # Latent frames
        L_train = 25  # Training length from HunyuanVideo
        actual_num_frames = video_length  # Use input video_length directly
    
        # Use transformer config instead of model
        head_dim = self.transformer.config.hidden_size // self.transformer.config.heads_num
        rope_dim_list = self.transformer.config.rope_dim_list or [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"
    
        if actual_num_frames > 192:
            k = 2+((actual_num_frames + 3) // (4 * L_train))
            k = max(4, min(8, k))
            logger.debug(f"actual_num_frames = {actual_num_frames} > 192, RIFLEx applied with k = {k}")
    
            # Compute positional grids for RIFLEx - use execution device
            device = self._execution_device
            axes_grids = [torch.arange(size, device=device, dtype=torch.float32) for size in rope_sizes]
            grid = torch.meshgrid(*axes_grids, indexing="ij")
            grid = torch.stack(grid, dim=0)  # [3, t, h, w]
            pos = grid.reshape(3, -1).t()  # [t * h * w, 3]
    
            # Apply RIFLEx to temporal dimension
            freqs = []
            for i in range(3):
                if i == 0:  # Temporal with RIFLEx
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=k,
                        L_test=L_test
                    )
                else:  # Spatial with default RoPE
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=None,
                        L_test=None
                    )
                freqs.append((freqs_cos, freqs_sin))
                logger.debug(f"freq[{i}] shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
        else:
            # 20250316 pftq: Original code for <= 192 frames
            logger.debug(f"actual_num_frames = {actual_num_frames} <= 192, using original RoPE")
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.args.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
    
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def _no_tiling_call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        i2v_mode: bool = False,
        i2v_condition_type: str = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        #print all the input arguments
        start_time_stage1 = time.time()
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `_no_tiling_call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `_no_tiling_call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        # Disable cache for _no_tiling_call__
        self.transformer.enable_cache = False
        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device
        device = torch.device(f"cuda:{self.dist_manager.rank}")

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            data_type=data_type,
            semantic_images=semantic_images
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        if freqs_cis is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(video_length, height, width)
            freqs_cis = (freqs_cos, freqs_sin)
            n_tokens = freqs_cos.shape[0]

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        logger.info(f"no_tiling_call, num_inference_steps: {num_inference_steps}, timesteps: {timesteps}, sigmas: {sigmas}")
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:
            video_length = video_length

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, img_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            video_length,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            img_latents=img_latents,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            semantic_images=semantic_images
        )

        if i2v_mode and i2v_condition_type == "latent_concat":
            if img_latents.shape[2] == 1:
                img_latents_concat = img_latents.repeat(1, 1, video_length, 1, 1)
            else:
                img_latents_concat = img_latents
            img_latents_concat[:, :, 1:, ...] = 0

            i2v_mask = torch.zeros(video_length)
            i2v_mask[0] = 1

            mask_concat = torch.ones(img_latents_concat.shape[0], 1, img_latents_concat.shape[2], img_latents_concat.shape[3],
                                     img_latents_concat.shape[4]).to(device=img_latents.device)
            mask_concat[:, :, 1:, ...] = 0

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)    
        
        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if i2v_mode and i2v_condition_type == "token_replace":
                    latents = torch.concat([img_latents, latents[:, :, 1:, :, :]], dim=2)

                # expand the latents if we are doing classifier free guidance
                if i2v_mode and i2v_condition_type == "latent_concat":
                    latent_model_input = torch.concat([latents, img_latents_concat, mask_concat], dim=1)
                else:
                    latent_model_input = latents

                latent_model_input = (
                    torch.cat([latent_model_input] * 2)
                    if self.do_classifier_free_guidance
                    else latent_model_input
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                t_expand = t.repeat(latent_model_input.shape[0]).to(device)
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=prompt_embeds,  # [2, 256, 4096]
                        text_mask=prompt_mask,  # [2, 256]
                        text_states_2=prompt_embeds_2,  # [2, 768]
                        freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                        freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        return_dict=True,
                    )[
                        "x"
                    ]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if i2v_mode and i2v_condition_type == "token_replace":
                    latents = self.scheduler.step(
                        noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False
                    )[0]
                    latents = torch.concat(
                        [img_latents, latents], dim=2
                    )
                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        end_time_stage1 = time.time()
        logger.info(f"Stage 1 time: {end_time_stage1 - start_time_stage1} seconds")
        logger.info(f"latents type:: {latents.dtype}, target_dtype:: {target_dtype}, vae_dtype:: {vae_dtype}, vae_autocast_enabled:: {vae_autocast_enabled}")
        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if enable_tiling:
                    self.vae.enable_tiling()
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]
                else:
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        logger.info(f"return_dict: {return_dict}, image shape: {image.shape}")
        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)

    def _update_cache_thresholds_from_std(self, base_thresh: float, alpha: float, mode: str):
        """Update cache thresholds based on std tracker"""
        new_thresholds = self.dist_manager.std_tracker_get_new_threshold(base_thresh, alpha, mode)
        self.tile_effective_cache_threshold_classifier = new_thresholds

    def __tilling_call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int, 
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None, 
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None, 
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None, 
        embedded_guidance_scale: Optional[float] = None,
        i2v_mode: bool = False,
        i2v_condition_type: str = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        shift_timesteps: Optional[List[int]] = None,
        upscale_factor: int = 2,
        loop_step: int = 8,
        noise_fusion_method: str = "weighted_average",
        tile_overlap: int = 0,  # Parameter for overlap support
        enable_intra_tile_cache: bool = False, # Parameter for cache
        cache_thresh: float = 0.05,
        enable_region_aware_cache: bool = False,
        static_tile_cache_scale_factor: float = 1.0,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            shift_timesteps (`List[int]`, *optional*):
                List of steps to perform shift. If None, shift will be performed at all steps.
            noise_fusion_method: Method to fuse overlapping noise predictions
                               - "weighted_average": weighted average based on contribution count
                               - "simple_average": simple average  
                               - "last_wins": last prediction overwrites (current behavior)
            tile_overlap: Number of pixels to overlap between adjacent tiles
            enable_intra_tile_cache (`bool`, *optional*, defaults to `False`):
                Whether to enable intra-tile cache.
            cache_thresh (`float`, *optional*, defaults to `0.1`):
                The threshold for cache.
            enable_region_aware_cache (`bool`, *optional*, defaults to `False`):
                Whether to enable region-aware cache optimization.
            static_tile_cache_scale_factor (`float`, *optional*, defaults to `1.0`):
                Scale factor for cache threshold.    
        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        start_time_stage2 = time.time()
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `_no_tiling_call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `_no_tiling_call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length, # Pixel video length
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device

        # Ensure transformer is on the correct device
        self.transformer.to(device)
        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            data_type=data_type,
            semantic_images=semantic_images
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # 4. Prepare rotary position embeddings for tiling 
        # Calculate rotary embeddings for the actual window size that will be used
        window_freqs_cos, window_freqs_sin = self.get_rotary_pos_embed(video_length, int(height / upscale_factor), int(width / upscale_factor))
        # window_n_tokens = window_freqs_cos.shape[0] * upscale_factor**2
        window_n_tokens = n_tokens
        logger.info(f"window_n_tokens: {window_n_tokens}, n_tokens: {n_tokens}")
        
        # 4. Prepare timesteps with correct n_tokens
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": window_n_tokens}
        )
        # logger.info(f"tiling_call, num_inference_steps: {num_inference_steps}, timesteps: {timesteps}, sigmas: {sigmas}")
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        # Convert video_length from pixel frames to latent frames for prepare_latents
        video_length = video_length 
        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:  
            video_length = video_length 

        # 5. Prepare initial latents (x_T)
        transformer_in_channels = self.transformer.config.in_channels
        latents, img_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            transformer_in_channels, 
            height,
            width,
            video_length, # Pass video_length
            prompt_embeds.dtype,
            self.device, # This is self._execution_device
            generator, 
            latents, # Pass external latents if provided
            img_latents, # img_latents is already on the target CUDA `device` due to earlier .to(device)
            i2v_mode, 
            i2v_condition_type, 
            i2v_stability,
            semantic_images=semantic_images)
        
        # Explicitly move initial_latents_xt to the target computation `device` (CUDA)
        latents = latents.to(device)
        
        num_denoised_latent_channels = latents.shape[1] 

        img_latents_cond_full, mask_concat_full = None, None
        if i2v_mode and i2v_condition_type == "latent_concat":
            assert img_latents is not None and img_latents.shape[1] == num_denoised_latent_channels, \
                "img_latents (VAE encoded) required for latent_concat with matching channels."
            # Ensure img_latents_cond_full has video_length frames for TiledLatentTensor2D
            img_latents_cond_full = img_latents.repeat(1,1,video_length,1,1) if img_latents.shape[2] == 1 else img_latents
            if img_latents_cond_full.shape[2] != video_length: # If original img_latents had >1 frame but not matching
                 img_latents_cond_full = img_latents_cond_full[:,:,:video_length,:,:] 
                 logger.warning(f"img_latents for concat was reshaped to {video_length} frames")

            # Ensure mask_concat_full is created on the target CUDA `device`
            mask_concat_full = torch.ones(latents.shape[0], 1, video_length, 
                                          latents.shape[3], latents.shape[4], 
                                          device=device, dtype=prompt_embeds.dtype)
            if video_length > 1: mask_concat_full[:, :, 1:, ...] = 0

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        ###### Define the sliding-window related params ######
        rank = self.dist_manager.rank
        ll_height, ll_width = latents.shape[3], latents.shape[4] 

        window_config = SlidingWindowConfig(ll_height, ll_width, loop_step)
        window_params = window_config.get_window_params()
        window_size = window_params['window_size']
        num_windows_h = window_params['num_windows_h']
        num_windows_w = window_params['num_windows_w']
        total_windows = window_params['total_windows']
        latent_step_size_h = window_params['latent_step_size_h']
        latent_step_size_w = window_params['latent_step_size_w']

        # Setup Cache
        self.transformer.setup_cache_per_tile(
            num_steps=num_inference_steps,
            thresh=cache_thresh,
            ret_steps=0,
            num_tiles=total_windows,
        )
        if enable_intra_tile_cache:
            self.transformer.enable_cache = True
            t = latents
            if self.do_classifier_free_guidance:
                t = torch.cat([latents, latents], dim=0)
            self.transformer.allocate(t)
            logger.info(f"[{rank=}]Intra-Tile Cache for {total_windows} tiles has been set up with threshold={cache_thresh} and ret_steps={5}.")
        else:
            self.transformer.enable_cache = False
        # Initialize region-aware caching
        if enable_intra_tile_cache and enable_region_aware_cache:
            self.tile_effective_cache_threshold_classifier = [cache_thresh] * total_windows
            # self.std_tracker = TileStdTracker(total_windows, update_interval=5)
            logger.info(f"[{rank=}]Region-aware cache initialized for {total_windows} tiles with base threshold {cache_thresh:.3f}")
        else:
            self.tile_effective_cache_threshold_classifier = None


        # tiled_latent_handler = TiledLatentTensor2D(latents) 
        # tiled_image_handler = TiledLatentTensor2D(img_latents)

        # tile_noise_fuser = TileNoiseAggregator2D(
        #     noise_shape=(batch_size * num_videos_per_prompt, transformer_in_channels, latents.shape[2], ll_height, ll_width),
        #     device=device,
        #     dtype=prompt_embeds.dtype,  # Use prompt_embeds.dtype for consistency
        #     fusion_method=noise_fusion_method
        # )

        self.dist_manager.setup_config(latents, img_latents, window_config, 
                                       noise_fusion_method, std_tracker_update_interval=5,
                                       do_classifier_free_guidance=self.do_classifier_free_guidance)

        enable_redistribute = os.getenv("ENABLE_REDISTRIBUTE")
        enable_redistribute = enable_redistribute is not None and enable_redistribute == "1"

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps): 
                if self.interrupt: continue
                
                # tile_noise_fuser.reset()
                self.dist_manager.clear()
                # Reset tile tracking for new timestep
                should_shift = shift_timesteps is not None and i in shift_timesteps
                if should_shift:
                    self.dist_manager.communicate("latent")
                    self.dist_manager.shift()

                if enable_intra_tile_cache:
                    for tile_idx in self.dist_manager.get_local_indices():
                        latents_for_view, image_latents_for_view = self.dist_manager.get_tile(tile_idx)
                        window_position = self.dist_manager.get_tile_boundary_for_idx(tile_idx)

                        if i2v_condition_type == "token_replace":
                            # latents_for_view_slice will be on `device` because latent_model_input (from latents_for_view) is                                    
                            latents_for_view = torch.cat([image_latents_for_view, latents_for_view[:,:,1:,:,:]], dim=2) 
                        if i2v_condition_type == "latent_concat":
                            latent_model_input = torch.cat([latents, image_latents_for_view, mask_concat_full], dim=1)
                        else:
                            latent_model_input = latents_for_view
                        
                        latent_model_input = (
                            torch.cat([latent_model_input] * 2)
                            if self.do_classifier_free_guidance
                            else latent_model_input
                        )
                        # logger.info(f"latent_model_input shape after classifier_free_guidance: {latent_model_input.shape}")
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        t_expand = t.repeat(latent_model_input.shape[0])
                        guid_p = (
                            torch.tensor(
                                [embedded_guidance_scale] * latent_model_input.shape[0],
                                dtype=torch.float32,
                                device=device,
                            ).to(target_dtype)
                            * 1000.0
                            if embedded_guidance_scale is not None
                            else None
                        )
                        # Get effective cache threshold for this tile
                        effective_cache_thresh = cache_thresh
                        if enable_intra_tile_cache and enable_region_aware_cache:
                            effective_cache_thresh = self.tile_effective_cache_threshold_classifier[tile_idx]
                        
                        # predict the noise model_output
                        with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                            can_be_cached, _ = self.transformer.check_skippable(i, tile_idx,
                                                        latent_model_input, 
                                                        (not should_shift),
                                                        effective_cache_thresh,
                                                        window_position=window_position)

                        if can_be_cached:
                            self.dist_manager.set_tensor_in_buffer("skipped", tile_idx, 1.0)
                            self.dist_manager.mark_tile_skipped(tile_idx)

                    if enable_redistribute:
                        # Caching is possible only non-shifting
                        # Note: We should communicate after `check_skippable` if caching is possible.
                        #       And we don't need to comm again if shifting (comm done before).
                        self.dist_manager.communicate()
                        self.dist_manager.redistribute_workload()

                for tile_idx in self.dist_manager.get_local_indices():
                    latents_for_view, image_latents_for_view = self.dist_manager.get_tile(tile_idx)
                    window_position = self.dist_manager.get_tile_boundary_for_idx(tile_idx)

                    if i2v_condition_type == "token_replace":
                        # latents_for_view_slice will be on `device` because latent_model_input (from latents_for_view) is                                    
                        latents_for_view = torch.cat([image_latents_for_view, latents_for_view[:,:,1:,:,:]], dim=2) 
                        
                    if i2v_condition_type == "latent_concat":
                        latent_model_input = torch.cat([latents, image_latents_for_view, mask_concat_full], dim=1)
                    else:
                        latent_model_input = latents_for_view

                    latent_model_input = (
                        torch.cat([latent_model_input] * 2)
                        if self.do_classifier_free_guidance
                        else latent_model_input
                    )
                    # logger.info(f"latent_model_input shape after classifier_free_guidance: {latent_model_input.shape}")
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    t_expand = t.repeat(latent_model_input.shape[0])
                    guid_p = (
                        torch.tensor(
                            [embedded_guidance_scale] * latent_model_input.shape[0],
                            dtype=torch.float32,
                            device=device,
                        ).to(target_dtype)
                        * 1000.0
                        if embedded_guidance_scale is not None
                        else None
                    )
                    # Get effective cache threshold for this tile
                    effective_cache_thresh = cache_thresh
                    if enable_intra_tile_cache and enable_region_aware_cache:
                        effective_cache_thresh = self.tile_effective_cache_threshold_classifier[tile_idx]
                    
                    # predict the noise model_output
                    with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                        noise_pred = self.transformer(latent_model_input, 
                                                    t_expand, 
                                                    prompt_embeds, 
                                                    prompt_mask, 
                                                    prompt_embeds_2, 
                                                    window_freqs_cos, 
                                                    window_freqs_sin, 
                                                    guid_p, 
                                                    tile_idx,
                                                    i,
                                                    (not should_shift),
                                                    effective_cache_thresh,
                                                    True,
                                                    window_position=window_position)["x"]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    
                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    # Update std tracker for region-aware caching
                    if enable_intra_tile_cache and enable_region_aware_cache:
                        self.dist_manager.std_tracker_update(tile_idx, noise_pred, i)

                    # Calculate tile weight (for overlap handling)
                    tile_weight = 1.0
                    # if tile_overlap > 0:
                    #     tile_weight = 1.0 

                    self.dist_manager.tile_noise_fuser_add(
                        tile_idx=tile_idx,
                        noise_pred=noise_pred,
                        tile_weight=tile_weight
                    )

                # fused_noise_pred = tile_noise_fuser.get_fused_noise()
                fused_noise_pred = self.dist_manager.allgather_fused_noise()
                latent = self.dist_manager.get_latents()
                image_latent = self.dist_manager.get_image_latents()

                if i2v_mode and i2v_condition_type == "token_replace":
                    latents_denoised = self.scheduler.step(
                        # fused_noise_pred[:, :, 1:, :, :], t, tiled_latent_handler.torch_latent[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False
                        fused_noise_pred[:, :, 1:, :, :], t, latent[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False
                    )[0]
                    latents_denoised = torch.concat(
                        # [tiled_image_handler.torch_latent, latents_denoised], dim=2
                        [image_latent, latents_denoised], dim=2
                    )
                else:
                    latents_denoised = self.scheduler.step(
                        # fused_noise_pred, t, tiled_latent_handler.torch_latent, **extra_step_kwargs, return_dict=False
                        fused_noise_pred, t, latent, **extra_step_kwargs, return_dict=False
                    )[0]

                # tiled_latent_handler.torch_latent = latents_denoised
                self.dist_manager.set_latents(latents_denoised)
                latent = latents_denoised
                # if self.dist_manager.is_first_rank and i % 5 == 0:
                #     self.save_video(latents=latent, save_path=os.path.join(kwargs["output_dir"], f"step_{i}.mp4"), generator=generator, fps=8)

                if enable_intra_tile_cache and enable_region_aware_cache and self.dist_manager.std_tracker_should_update(i):
                    self._update_cache_thresholds_from_std(base_thresh=cache_thresh, alpha=static_tile_cache_scale_factor, mode="top_selective")
                    # self.std_tracker.update_last_step(i)
                    self.dist_manager.std_tracker_update_last_step(i)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k_cb in callback_on_step_end_tensor_inputs: 
                        callback_kwargs[k_cb] = locals()[k_cb]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    # _latents_from_cb = callback_outputs.pop("latents", tiled_latent_handler.torch_latent)
                    _latents_from_cb = callback_outputs.pop("latents", latent)
                    if _latents_from_cb is not None:
                        # tiled_latent_handler.torch_latent = _latents_from_cb 
                        self.dist_manager.set_latents(_latents_from_cb)
                        latent = self.dist_manager.get_latents()
                        
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        # callback(step_idx, t, tiled_latent_handler.torch_latent) 
                        callback(step_idx, t, latent) 

        end_time_stage2 = time.time()

        logger.info(f"[{rank=}]Stage 2 time: {end_time_stage2 - start_time_stage2} seconds")
        torch.cuda.empty_cache()

        if not self.dist_manager.is_first_rank:
            image = None
            if not return_dict:
                return image
            return HunyuanVideoPipelineOutput(videos=image)

        latent = self.dist_manager.get_latents()
        # latents = tiled_latent_handler.torch_latent.clone().to(device=latents.device)
        latents = latent.to(device=latents.device)
        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if enable_tiling:
                    self.vae.enable_tiling()
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]
                else:
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float()

        if i2v_mode and i2v_condition_type == "latent_concat":
            image = image[:, :, 4:, :, :]

        # Offload all models
        self.maybe_free_model_hooks()

        logger.info(f"return_dict: {return_dict}, image shape: {image.shape}")
        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)

    @torch.no_grad()
    def two_stage_generation(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        i2v_mode: bool = False,
        i2v_condition_type: str = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        upscale_factor: int = 2,
        upscale_res_steps: int = 45,  # Stage 2 steps
        save_intermediate: bool = False,
        output_dir: Optional[str] = None,
        shift_timesteps: Optional[List[int]] = None,
        load_prev_latents_path: Optional[str] = None,
        loop_step: int = 8,
        enable_intra_tile_cache: bool = False, # Parameter for cache
        cache_thresh: float = 0.05,
        enable_region_aware_cache: bool = False,
        static_tile_cache_scale_factor: float = 1.0,
        **kwargs,
    ) -> Union[HunyuanVideoPipelineOutput, Tuple]:
        rank = self.dist_manager.rank
        logger.info(f"[{rank=}]=== Starting Two-Stage Generation ===")
        
        # Determine step counts for each stage
        stage1_inference_steps = num_inference_steps
        stage2_inference_steps = upscale_res_steps
        
        logger.info(f"[{rank=}]Stage 1: {stage1_inference_steps} steps at {height//upscale_factor}x{width//upscale_factor}")
        logger.info(f"[{rank=}]Stage 2: {stage2_inference_steps} steps at {height}x{width}")
        
        # Stage 1: Low Resolution Generation
        logger.info(f"[{rank=}]=== Stage 1: Low Resolution Generation ===")
        if self.dist_manager.is_first_rank:
            if load_prev_latents_path is None or not os.path.exists(load_prev_latents_path):
                # Call the regular _no_tiling_call__ method for low-res generation
                stage1_output = self._no_tiling_call__(
                    prompt=prompt,
                    height=height // upscale_factor,
                    width=width // upscale_factor,
                    video_length=video_length,
                    data_type=data_type,
                    num_inference_steps=stage1_inference_steps,  # Use Stage 1 specific steps
                    timesteps=timesteps,
                    sigmas=sigmas,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    eta=eta,
                    generator=generator,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_attention_mask=negative_attention_mask,
                    output_type="latent",  # Get latents instead of decoded video
                    return_dict=False,
                    cross_attention_kwargs=cross_attention_kwargs,
                    guidance_rescale=guidance_rescale,
                    clip_skip=clip_skip,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    freqs_cis=None,  # Let Stage 1 compute its own rotary embeddings for low resolution
                    vae_ver=vae_ver,
                    enable_tiling=enable_tiling,
                    n_tokens=None,
                    embedded_guidance_scale=embedded_guidance_scale,
                    i2v_mode=i2v_mode,
                    i2v_condition_type=i2v_condition_type,
                    i2v_stability=i2v_stability,
                    img_latents=None,
                    semantic_images=semantic_images,
                    **kwargs,
                )

                low_res_latents = stage1_output
                print(f"low_res_latents shape: {low_res_latents.shape}")
                if save_intermediate and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save low-resolution latents
                    save_path = os.path.join(output_dir, "stage1_lowres_latents.pt")
                    if load_prev_latents_path is not None:
                        save_path = load_prev_latents_path
                    torch.save(low_res_latents, save_path)
                    logger.info(f"Saved low-res latents to {save_path}")
                    
                    self.save_video(latents=low_res_latents, save_path=os.path.join(output_dir, "stage1_lowres_video.mp4"), generator=generator, fps=8)
            elif os.path.exists(load_prev_latents_path):
                low_res_latents = torch.load(load_prev_latents_path, weights_only=True)
                logger.info(f"[{rank=}]=== Stage 1 Generation skipped: Loaded low-res latents from {load_prev_latents_path}")
                self.save_video(latents=low_res_latents, save_path=os.path.join(output_dir, "stage1_lowres_video.mp4"), generator=generator, fps=8)
        else:
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            num_channels_latents = self.transformer.config.in_channels
            _video_length = video_length
            if "884" in vae_ver:
                _video_length = (_video_length - 1) // 4 + 1
            elif "888" in vae_ver:
                _video_length = (_video_length - 1) // 8 + 1
            else:  
                _video_length = _video_length 
            low_res_latents, _ = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height // upscale_factor,
                width // upscale_factor,
                _video_length,
                # self.text_encoder.dtype,
                # self.transformer.dtype,
                torch.float32,
                torch.device(f"cuda:{self.dist_manager.rank}"),
                generator,
                latents,
                img_latents=None,
                i2v_mode=i2v_mode,
                i2v_condition_type=i2v_condition_type,
                i2v_stability=i2v_stability,
                semantic_images=semantic_images,
            )
        
        dist.broadcast(low_res_latents, src=self.dist_manager.first_rank)
        
        # Upscale and Refine
        logger.info(f"[{rank=}]=== Upscale low-res latents to high-res ({height}x{width}) ===")
        
        # Calculate target latent dimensions
        target_latent_height = height // self.vae_scale_factor
        target_latent_width = width // self.vae_scale_factor
        start_upscale_latents_time = time.time()
        # Upscale the low-res latents in latent space 
        # TODO: if in pixel space?
        upscaled_latents = self._upscale_latents(
            low_res_latents, 
            target_height=target_latent_height,
            target_width=target_latent_width
        )
        logger.info(f"[{rank=}]Upscaled latents shape: {upscaled_latents.shape}")
        end_upscale_latents_time = time.time()
        logger.info(f"[{rank=}]Upscale latents time: {end_upscale_latents_time - start_upscale_latents_time} seconds")

        # Save upscaled clean latents
        # if save_intermediate and output_dir:
        #     torch.save(upscaled_latents, os.path.join(output_dir, "stage2_upscaled_clean_latents.pt"))
        #     logger.info(f"Saved upscaled clean latents to {output_dir}/stage2_upscaled_clean_latents.pt")
            
        #     self.save_video(upscaled_latents, os.path.join(output_dir, "stage2_upscaled_clean_video.mp4"), generator, fps=8)
        
        start_renoise_time = time.time()
        # Re-noise if we need to do Stage 2 refinement
        if upscale_res_steps > 0:
            # Get timesteps for re-noising based on Stage 2 step count
            scheduler_n_tokens = n_tokens
            extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
                self.scheduler.set_timesteps, {"n_tokens": scheduler_n_tokens}
            )
            timesteps_full, _ = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,  # Use Stage 2 specific steps
                self.device,
                timesteps,
                sigmas,
                **extra_set_timesteps_kwargs,
            )
            # logger.info(f"timesteps_full: {timesteps_full}")
            # Ensure timesteps_full is on the correct device and not a meta tensor
            if hasattr(timesteps_full, 'is_meta') and timesteps_full.is_meta:
                timesteps_full = timesteps_full.to(self.device)
            
            renoise_timestep_idx = num_inference_steps - upscale_res_steps
            to_timestep = timesteps_full[renoise_timestep_idx]
            logger.info(f"Re-noising to timestep at index {renoise_timestep_idx}: {to_timestep}")
            
            # TODO: add the from_timestep_idx parameter
            renoised_latents = self.scheduler.re_noise(
                upscaled_latents,
                to_timestep_idx=renoise_timestep_idx,
                noise=None,  # Let the scheduler generate noise
                generator=generator
            )
            logger.info(f"Re-noised latents for Stage 2 denoising, will denoise remaining {upscale_res_steps} steps")
            
            # if save_intermediate and output_dir:
            #     torch.save(renoised_latents.cpu(), os.path.join(output_dir, "stage2_renoised_latents.pt"))
            #     logger.info(f"Saved re-noised latents to {output_dir}/stage2_renoised_latents.pt")
                
            #     self.save_video(renoised_latents, os.path.join(output_dir, "stage2_renoised_video.mp4"), generator, fps=8)
        else:
            renoised_latents = upscaled_latents
            timesteps_full = None
            renoise_timestep_idx = None
        end_renoise_time = time.time()
        logger.info(f"Renoise time: {end_renoise_time - start_renoise_time} seconds")
        # Stage 2
        if upscale_res_steps > 0 and renoise_timestep_idx is not None:
            stage2_timesteps = timesteps_full[renoise_timestep_idx:] if timesteps_full is not None else None
            stage2_output = self.__tilling_call__(
                prompt=prompt,
                height=height,
                width=width,
                video_length=video_length,
                data_type=data_type,
                num_inference_steps=upscale_res_steps,  # Use only the remaining steps to denoise
                timesteps=stage2_timesteps,  # Use subset of timesteps (remaining steps)
                sigmas=None,  # Let scheduler calculate sigmas based on the provided timesteps
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                eta=eta,
                generator=generator,
                latents=renoised_latents,  # Use re-noised latents as init
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_attention_mask=negative_attention_mask,
                output_type="pil",
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                freqs_cis=freqs_cis,
                vae_ver=vae_ver,
                enable_tiling=enable_tiling,
                n_tokens=n_tokens,
                embedded_guidance_scale=embedded_guidance_scale,
                i2v_mode=i2v_mode,
                i2v_condition_type=i2v_condition_type,
                i2v_stability=i2v_stability,
                img_latents=None,
                semantic_images=semantic_images,
                upscale_factor=upscale_factor,
                shift_timesteps=shift_timesteps,
                loop_step=loop_step,
                enable_intra_tile_cache=enable_intra_tile_cache,
                cache_thresh=cache_thresh,
                enable_region_aware_cache=enable_region_aware_cache,
                static_tile_cache_scale_factor=static_tile_cache_scale_factor,
                output_dir=output_dir,
                **kwargs,
            )
            
            logger.info(f"[{rank=}]=== Two-Stage Generation Completed ===")
            return stage2_output
        else:
            # If no Stage 2 refinement, just decode the upscaled latents
            if output_type != "latent":
                # Decode following the pattern from _no_tiling_call__
                latents_to_decode = upscaled_latents
                expand_temporal_dim = False
                if len(latents_to_decode.shape) == 4:
                    if isinstance(self.vae, AutoencoderKLCausal3D):
                        latents_to_decode = latents_to_decode.unsqueeze(2)
                        expand_temporal_dim = True
                
                # Scale latents
                if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                    latents_to_decode = (
                        latents_to_decode / self.vae.config.scaling_factor
                        + self.vae.config.shift_factor
                    )
                else:
                    latents_to_decode = latents_to_decode / self.vae.config.scaling_factor
                
                # Decode with autocast
                target_dtype = PRECISION_TO_TYPE[self.args.precision]
                vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
                vae_autocast_enabled = (vae_dtype != torch.float32) and not self.args.disable_autocast
                
                # Ensure latents are in the correct dtype for VAE
                latents_to_decode = latents_to_decode.to(dtype=vae_dtype)
                
                with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                    if enable_tiling:
                        self.vae.enable_tiling()
                    video = self.vae.decode(
                        latents_to_decode, return_dict=False, generator=generator
                    )[0]
                
                if expand_temporal_dim or video.shape[2] == 1:
                    video = video.squeeze(2)
                
                video = (video / 2 + 0.5).clamp(0, 1)
                video = video.cpu().float()
            else:
                video = upscaled_latents
            
            logger.info(f"[{rank=}]=== Two-Stage Generation Completed (No Stage 2 refinement) ===")
            
            if not return_dict:
                return video
            return HunyuanVideoPipelineOutput(videos=video)    


    #TODO Try pixel space upscale
    def _upscale_latents(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Upscale latents using interpolation"""
        import torch.nn.functional as F
        
        # latents shape: [batch, channels, frames, height, width]
        batch_size, channels, num_frames, height, width = latents.shape
        
        # Permute to [batch, frames, channels, height, width] for easier processing
        latents = latents.permute(0, 2, 1, 3, 4)
        
        # Reshape to [batch * frames, channels, height, width] for interpolation
        latents_reshaped = latents.contiguous().view(batch_size * num_frames, channels, height, width)
        
        # Upscale using bicubic interpolation
        upscaled = F.interpolate(
            latents_reshaped,
            size=(target_height, target_width),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back to [batch, frames, channels, height, width]
        upscaled = upscaled.view(batch_size, num_frames, channels, target_height, target_width)
        
        # Permute back to [batch, channels, frames, height, width]
        upscaled = upscaled.permute(0, 2, 1, 3, 4)
        
        return upscaled

    def save_video(self, latents, save_path, generator, fps):
        out_dict = {}
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast

        expand_temporal_dim = False
        if len(latents.shape) == 4:
            if isinstance(self.vae, AutoencoderKLCausal3D):
                latents = latents.unsqueeze(2)
                expand_temporal_dim = True
        elif len(latents.shape) == 5:
            pass
        else:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )

        if (
            hasattr(self.vae.config, "shift_factor")
            and self.vae.config.shift_factor
        ):
            latents = (
                latents / self.vae.config.scaling_factor                    
                + self.vae.config.shift_factor
            )
        else:
            latents = latents / self.vae.config.scaling_factor

        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            # Default to enable tiling
            self.vae.enable_tiling()
            image = self.vae.decode(
                latents, return_dict=False, generator=generator
            )[0]

        if expand_temporal_dim:
            image = image.squeeze(2)
        elif len(image.shape) == 5 and image.shape[2] == 1:
            image = image.squeeze(2)

        # Adaptive normalization based on actual value range
        img_min, img_max = image.min(), image.max()
        
        # Check if image is in [-1, 1] range (typical for VAE output)
        if img_min >= -1.1 and img_max <= 1.1:
            # Standard VAE output range [-1, 1] -> [0, 1]
            image = (image / 2 + 0.5).clamp(0, 1)
        elif img_min >= 0 and img_max <= 1.1:
            # Already in [0, 1] range
            image = image.clamp(0, 1)
        else:
            # Custom range, normalize to [0, 1]
            image = (image - img_min) / (img_max - img_min)
            # logger.info(f"Applied custom normalization from [{img_min.item():.4f}, {img_max.item():.4f}] to [0,1]")
        
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float()
        
        final_min, final_max = image.min(), image.max()
        samples = HunyuanVideoPipelineOutput(videos=image)

        # Save as MP4
        from hyvideo.utils.file_utils import save_videos_grid
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i]
                # Ensure sample has correct dimensions for video saving
                if len(sample.shape) == 4:
                    # Add batch dimension if missing: (C, T, H, W) -> (1, C, T, H, W)
                    sample = sample.unsqueeze(0)
                    logger.info(f"Added batch dimension, new shape: {sample.shape}")
                
                # Expected format for save_videos_grid: (B, C, T, H, W)
                logger.info(f"Saving video with fps={fps} to: {save_path}")
                save_videos_grid(sample, save_path, fps=fps)