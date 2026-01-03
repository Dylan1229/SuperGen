import time
import os
import inspect
import math
import datetime

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from diffusers.schedulers import CogVideoXDPMScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image as PILImage

from utils import SlidingWindowConfig
from utils.distributed import DistributedManager

logger = logging.get_logger(__name__) 

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import CogVideoXImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> video = pipe(image, prompt, use_dynamic_cfg=True)
        >>> export_to_video(video.frames[0], "output.mp4", fps=8)
        ```
"""   
# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
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

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
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
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
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
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
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

class TiledCogVideoXImageToVideoPipeline(CogVideoXImageToVideoPipeline):
    dist_manager: Optional["DistributedManager"] = None

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        image = image.unsqueeze(2)  # [B, C, F, H, W]

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

        image_latents = torch.cat(image_latents, dim=0)

        # Convert to the right format: [B, F, C, H, W]
        image_latents = image_latents.to(dtype).permute(0, 2, 1, 3, 4)

        if not self.vae.config.invert_scale_latents:
            image_latents = self.vae_scaling_factor_image * image_latents
        else:
            # This is awkward but required because the CogVideoX team forgot to multiply the
            # scaling factor during training :)
            image_latents = 1 / self.vae_scaling_factor_image * image_latents

        padding_shape = (
            batch_size,
            num_frames - 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Select the first frame along the frame dimension
        if self.transformer.config.patch_size_t is not None:
            first_frame = image_latents[:, : image_latents.size(1) % self.transformer.config.patch_size_t, ...]
            image_latents = torch.cat([first_frame, image_latents], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents
    
    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p
        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t
            # Shape of freqs_cos and freqs_sin: [temporal_size * grid_height * grid_width, embed_dim=64]
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin


    def _update_cache_thresholds_from_std(self, base_thresh: float, alpha: float, mode: str):
        """Update cache thresholds based on std tracker"""
        new_thresholds = self.dist_manager.std_tracker_get_new_threshold(base_thresh, alpha, mode)
        self.tile_effective_cache_threshold_classifier = new_thresholds


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def tiling_call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        init_latent: torch.Tensor = None,
        use_skip_time: bool = False,
        denoise_to_step: int = None,
        return_latents: bool = False,
        shift_timesteps: Optional[List[int]] = None,
        loop_step: int = 8,
        noise_fusion_method: str = "weighted_average",
        tile_overlap: int = 0,  # Parameter for overlap support
        enable_intra_tile_cache: bool = False, # Parameter for cache
        cache_thresh: float = 0.05,
        # For profiling only, otherwise, set False.
        save_k_history: bool = False,
        k_history_filename: Optional[str] = None,
        enable_noise_pred_profile: bool = False,  
        enable_cache_residual_profile: bool = False,
        enable_region_aware_cache: bool = False,
        static_tile_cache_scale_factor: float = 1.0,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.
            init_latent (`torch.Tensor`, *optional*):
                Initial latent tensor to use for generation. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
            use_skip_time (`bool`, *optional*, defaults to `False`):
                Whether to skip the first few timesteps for the upscale stage.
            denoise_to_step (`int`, *optional*, defaults to `None`):
                The number of timesteps to skip for the upscale stage.
            return_latents (`bool`, *optional*, defaults to `False`):
                Whether to return the final latents alongside the primary output.
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
            save_k_history (`bool`, *optional*, defaults to `False`):
                Whether to save k value history to JSON file for analysis.
            k_history_filename (`str`, *optional*):
                Custom filename for k history JSON file. If None, auto-generated filename will be used.
            enable_noise_pred_profile (`bool`, *optional*, defaults to `False`):
                Whether to enable noise prediction profile.
            enable_cache_residual_profile (`bool`, *optional*, defaults to `False`):
                Whether to enable cache residual profile. 
        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        start_time_stage2 = time.time()
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        
        # 0. Default height and width to dit
        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        assert num_videos_per_prompt == 1

        # 1. Check inputs. Raise error if not correct.
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, 
            num_inference_steps, 
            device,
            timesteps)
        # If use_skip_time is True, skip the first few timesteps for the upscale stage.
        if use_skip_time:
            timesteps = timesteps[denoise_to_step:]

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
        
        latent_channels = self.transformer.config.in_channels // 2

        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            init_latent,
        )
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create ROPE if required 
        # image_rotary_emb = (
        #     self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        #     if self.transformer.config.use_rotary_positional_embeddings
        #     else None
        # )
        
        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 9. Prepare window parameters
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
            num_steps=num_inference_steps-denoise_to_step,  
            thresh=cache_thresh, 
            ret_steps=5,
            num_tiles=total_windows
        )
        if enable_intra_tile_cache:
            self.transformer.enable_cache = True
            t = latents
            if do_classifier_free_guidance:
                t = torch.cat([latents, latents], dim=0)
            self.transformer.allocate(t)
            logger.info(f"[rank={self.dist_manager.rank}]: Intra-Tile Cache for {total_windows} tiles has been set up with threshold={cache_thresh} and ret_steps={5}.")
        else:
            self.transformer.enable_cache = False
        # Initialize region-aware caching
        if enable_region_aware_cache:
            self.tile_effective_cache_threshold_classifier = [cache_thresh] * total_windows
            logger.info(f"[rank={self.dist_manager.rank}]: Region-aware cache initialized for {total_windows} tiles with base threshold {cache_thresh:.3f}")
        else:
            self.tile_effective_cache_threshold_classifier = None
        
        # Set up profiling
        self.transformer.enable_noise_pred_profile = enable_noise_pred_profile
        if enable_noise_pred_profile:
            for j in range(total_windows):
                self.transformer.noise_pred_history[j] = []
        
        self.transformer.enable_cache_residual_profile = enable_cache_residual_profile
        if enable_cache_residual_profile:
            for j in range(total_windows):
                self.transformer.cache_residual_history[j] = []

        # Local ROPE
        image_rotary_emb = [None for _ in range(total_windows)]
        for j in range(total_windows):
            image_rotary_emb[j] = (
            self._prepare_rotary_positional_embeddings(window_size[0] * self.vae_scale_factor_spatial, window_size[1] * self.vae_scale_factor_spatial, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
            )

        # NOTE(MX)
        self.dist_manager.setup_config(
            latents, 
            image_latents, 
            window_config, 
            noise_fusion_method, 
            std_tracker_update_interval=5,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        enable_redistribute = os.getenv("ENABLE_REDISTRIBUTE")
        enable_redistribute = enable_redistribute is not None and enable_redistribute == "1"

        # 10. Denoising loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # Reset noise accumulator for each timestep
                # tile_noise_fuser.reset()
                self.dist_manager.clear()

                should_shift = shift_timesteps is not None and i in shift_timesteps
                if should_shift:
                    self.dist_manager.communicate("latent")
                    self.dist_manager.shift()

                # We must check cachability even shifting, to update tracked status.
                if enable_intra_tile_cache:
                    # Check cachability of each tile
                    for tile_idx in self.dist_manager.get_local_indices():
                        latents_for_view, image_latents_for_view = self.dist_manager.get_tile(tile_idx)
                        start_h, end_h, start_w, end_w = self.dist_manager.get_tile_boundary_for_idx(tile_idx)
                        window_position = (start_h, end_h, start_w, end_w)

                        # If do classifier free guidance.
                        latent_model_input = torch.cat([latents_for_view] * 2) if do_classifier_free_guidance else latents_for_view
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        latent_image_input = torch.cat([image_latents_for_view] * 2) if do_classifier_free_guidance else image_latents_for_view
                        # Concatenate all latents over channels dimention
                        latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                        # TODO(MX): this latent might be reused for DiT

                        # Get effective cache threshold for this tile
                        effective_cache_thresh = cache_thresh
                        if enable_intra_tile_cache and enable_region_aware_cache:
                            effective_cache_thresh = self.tile_effective_cache_threshold_classifier[tile_idx]
                        # If we do shifting this round, then no caching.
                        # But we should always call `check_skippable` to update data, even though
                        # we know caching is impossible.
                        can_be_cached, _ = self.transformer.check_skippable(
                            step_index=i,
                            tile_index=tile_idx,
                            hidden_states=latent_model_input,
                            is_non_shifting_step=(not should_shift),
                            effective_cache_thresh=effective_cache_thresh,
                            return_dict=False,
                            window_position=window_position,
                        )

                        if can_be_cached:
                            self.dist_manager.set_tensor_in_buffer("skipped", tile_idx, 1.0)
                            self.dist_manager.mark_tile_skipped(tile_idx)
                
                    if enable_redistribute:
                        self.dist_manager.communicate()
                        self.dist_manager.redistribute_workload()
                    
                # Perform real DiT
                for tile_idx in self.dist_manager.get_local_indices():
                    latents_for_view, image_latents_for_view = self.dist_manager.get_tile(tile_idx)
                    start_h, end_h, start_w, end_w = self.dist_manager.get_tile_boundary_for_idx(tile_idx)
                    window_position = (start_h, end_h, start_w, end_w)
                    # If do classifier free guidance.
                    latent_model_input = torch.cat([latents_for_view] * 2) if do_classifier_free_guidance else latents_for_view
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_image_input = torch.cat([image_latents_for_view] * 2) if do_classifier_free_guidance else image_latents_for_view
                    # Concatenate all latents over channels dimention
                    latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                        
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # Get effective cache threshold for this tile
                    effective_cache_thresh = cache_thresh
                    if enable_intra_tile_cache and enable_region_aware_cache:
                        effective_cache_thresh = self.tile_effective_cache_threshold_classifier[tile_idx]
                    
                    # predict the noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input, 
                        encoder_hidden_states=prompt_embeds, 
                        timestep=timestep,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb[tile_idx],
                        attention_kwargs=attention_kwargs,
                        tile_index=tile_idx,
                        step_index=i,
                        is_non_shifting_step=(not should_shift),
                        effective_cache_thresh=effective_cache_thresh,
                        return_dict=False,
                        window_position=window_position,
                    )[0]

                    noise_pred = noise_pred.float()

                    # perform guidance: guidance_scale + 1 -> 1. (from t=0 to t=N)
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) 
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Update std tracker for region-aware caching
                    if enable_intra_tile_cache and enable_region_aware_cache:
                        self.dist_manager.std_tracker_update(tile_idx, noise_pred, i)

                    # Calculate tile weight (for overlap handling)
                    tile_weight = 1.0
                    # if tile_overlap > 0:
                    #     tile_weight = 1.0  # 

                    # Add tile noise prediction to accumulator
                    self.dist_manager.tile_noise_fuser_add(
                        tile_idx=tile_idx,
                        noise_pred=noise_pred,
                        tile_weight=tile_weight,
                    )
                        
                # NOTE(MX)
                fused_noise_pred = self.dist_manager.allgather_fused_noise()
                latent = self.dist_manager.get_latents()
                
                # Update latents using the fused noise prediction
                latents_denoised = self.scheduler.step(
                    model_output=fused_noise_pred,
                    timestep=t,
                    sample=latent,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                # tiled_latent_handler.torch_latent = latents_denoised
                self.dist_manager.set_latents(latents_denoised)

                if enable_intra_tile_cache and enable_region_aware_cache and self.dist_manager.std_tracker_should_update(i):
                    self._update_cache_thresholds_from_std(base_thresh=cache_thresh, alpha=static_tile_cache_scale_factor, mode="top_selective")
                    self.dist_manager.std_tracker_update_last_step(i)

                if callback_on_step_end is not None: # callback_on_step_end is None
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    # tiled_latent_handler.torch_latent = callback_outputs.pop("latents", tiled_latent_handler.torch_latent)
                    latent = self.dist_manager.get_latents()
                    latent = callback_outputs.pop("latents", latent)
                    self.dist_manager.set_latents(latent)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        end_time_stage2 = time.time()
        logger.info(f"[rank={self.dist_manager.rank}] Second Stage Running time: {end_time_stage2 - start_time_stage2} seconds")

        if not self.dist_manager.is_first_rank:
            return None

        latents = self.dist_manager.get_latents()
        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else: 
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if return_latents:
                return (video, latents)
            return (video,)

        output = CogVideoXPipelineOutput(frames=video)

        if return_latents:
            return output, latents

        return output

    @torch.no_grad()
    def _no_tiling_call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        return_latents: bool = False,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.
            return_latents (`bool`, *optional*, defaults to `False`):
                Whether to return the final latents alongside the primary output.
        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        start_time_stage1 = time.time()
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Disable cache for _no_tiling_call__
        self.transformer.enable_cache = False

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)


        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        end_time_stage1 = time.time()
        logger.info(f"First Stage Running time: {end_time_stage1 - start_time_stage1} seconds")
        self._current_timestep = None

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            video_latents = latents[:, additional_frames:]
            video = self.decode_latents(video_latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if return_latents:
                return (video, latents)
            return (video,)

        output = CogVideoXPipelineOutput(frames=video)

        if return_latents:
            return output, latents

        return output

    @torch.no_grad()
    def two_stage_generation(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        output_dir: Optional[str] = None,
        upscale_factor: int = 2,
        upscale_res_steps: int = 0,  # Stage 2
        save_intermediate: bool = False,
        shift_timesteps: Optional[List[int]] = None,
        low_res_latents_path: Optional[str] = None,
        loop_step: int = 8,
        enable_intra_tile_cache: bool = False,
        cache_thresh: float = 0.05,
        enable_region_aware_cache: bool = False,
        static_tile_cache_scale_factor: float = 1.0,
        save_k_history: bool = False,
        k_history_filename: Optional[str] = "k_history.json",
        enable_noise_pred_profile: bool = False,
        enable_cache_residual_profile: bool = False,

    ) -> Union[CogVideoXPipelineOutput, Tuple]:

        logger.info(f"[rank={self.dist_manager.rank}]: === Starting Two-Stage Generation ===")
        # Make low_res  divisible by 8
        low_res_height = height // upscale_factor
        low_res_width = width // upscale_factor
        logger.info(f"[rank={self.dist_manager.rank}]: Stage 1: {num_inference_steps} steps at {low_res_height}x{low_res_width}")
        logger.info(f"[rank={self.dist_manager.rank}]: Stage 2: {upscale_res_steps} steps at {height}x{width}")
                
        # Check if we can load pre-saved low-resolution latents
        if low_res_latents_path and os.path.exists(low_res_latents_path):
            low_res_latents = torch.load(low_res_latents_path, map_location=self._execution_device, weights_only=True)
            logger.info(f"[rank={self.dist_manager.rank}]: Loaded low-res latents with shape: {low_res_latents.shape}")
            
            # Generate stage1_result for intermediate saving if needed
            if self.dist_manager.is_first_rank and save_intermediate and output_dir:
                stage1_video = self.decode_latents(low_res_latents)
                stage1_result = self.video_processor.postprocess_video(video=stage1_video, output_type="np")
                
                os.makedirs(output_dir, exist_ok=True)
                export_to_video(stage1_result[0], os.path.join(output_dir, "stage1_lowres_video.mp4"), fps=8)
                logger.info(f"[rank={self.dist_manager.rank}]: Saved Stage 1 video to {output_dir}/stage1_lowres_video.mp4")
        else:
            # NOTE(MX)
            if self.dist_manager.is_first_rank:
                logger.info(f"[rank={self.dist_manager.rank}]: No low-resolution latents path provided or not found. Running Stage 1 generation.")
                # Stage 1: Low Resolution Generation 
                # Resize image for low resolution
                if isinstance(image, (PILImage.Image, list)):
                    # Convert to PIL if needed and resize to low resolution
                    if isinstance(image, list):
                        pil_image = image[0] if len(image) > 0 else image
                    else:
                        pil_image = image
                    # Resize to low resolution
                    low_res_image = pil_image.resize((low_res_width, low_res_height), PILImage.LANCZOS)
                    logger.info(f"Resized input image from {pil_image.size} to {low_res_image.size} for Stage 1")
                else:
                    low_res_image = image
                    logger.info(f"Using original image for Stage 1 (no resize needed)")
                # # Save the low-resolution image if intermediate saving 
                # if save_intermediate and output_dir:
                #     os.makedirs(output_dir, exist_ok=True)
                #     low_res_image_path = os.path.join(output_dir, "stage1_lowres_input_image.png")
                #     low_res_image.save(low_res_image_path)
                #     logger.info(f"Saved low-res input image to {low_res_image_path}")

                stage1_result, low_res_latents = self._no_tiling_call__(
                    image=low_res_image, 
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=low_res_height,
                    width=low_res_width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    use_dynamic_cfg=use_dynamic_cfg, # This is used for DPM scheduler, for DDIM scheduler, it should be False
                    num_videos_per_prompt=num_videos_per_prompt,
                    eta=eta,
                    generator=generator,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    output_type=output_type,
                    return_dict=return_dict,
                    attention_kwargs=attention_kwargs,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    max_sequence_length=max_sequence_length,
                    return_latents=True,
                )
                if save_intermediate and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save low-resolution latents
                    save_path = os.getenv("LOW_RES_SAVE_PATH")
                    save_path = os.path.join(output_dir, "stage1_lowres_latents.pt") if save_path is None else save_path
                    torch.save(low_res_latents, save_path)
                    logger.info(f"[rank={self.dist_manager.rank}]: Saved low-res latents to {save_path}")
                
                    export_to_video(stage1_result.frames[0], os.path.join(output_dir, "stage1_lowres_video.mp4"), fps=8)
                    logger.info(f"[rank={self.dist_manager.rank}]: Saved Stage 1 video to {output_dir}/stage1_lowres_video.mp4")
            else:
                # Other ranks should wait for low res gen and broadcast
                if prompt is not None and isinstance(prompt, str):
                    batch_size = 1
                elif prompt is not None and isinstance(prompt, list):
                    batch_size = len(prompt)
                else:
                    raise NotImplementedError
                latent_channels = self.transformer.config.in_channels // 2
                l_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
                shape = (
                    batch_size,
                    l_num_frames,
                    latent_channels,
                    low_res_height // self.vae_scale_factor_spatial,
                    low_res_width // self.vae_scale_factor_spatial,
                )
                if self.transformer.config.patch_size_t is not None:
                    shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]
                low_res_latents = randn_tensor(shape, generator=generator, device=self._execution_device, dtype=self.text_encoder.dtype)

        # NOTE(MX)
        rank = self.dist_manager.rank
        dist.broadcast(low_res_latents, src=self.dist_manager.first_rank)
        
        # Stage 2: Upscale and Refine
        logger.info(f"=== Stage 2: Upscale and Refine ({height}x{width}) ===")
        start_time_upsampling = time.time()
        # Upscale in pixel space
        upscaled_latents = self._upscale_video(
            low_res_latents, 
            target_height=height,
            target_width=width,
            generator=generator
        )
        end_time_upsampling = time.time()
        logger.info(f"[rank={self.dist_manager.rank}]: Upsampling Running time: {end_time_upsampling - start_time_upsampling} seconds")

        # Upscale in latent space
        # upscaled_latents = self._upscale_latents(
        #     low_res_latents, 
        #     target_height=height // self.vae_scale_factor_spatial,
        #     target_width=width // self.vae_scale_factor_spatial,
        #     # generator=generator
        # )
        
        # Save upscaled clean latents and video
        # if save_intermediate and output_dir:
        #     torch.save(upscaled_latents, os.path.join(output_dir, "stage2_upscaled_clean_latents.pt"))
        #     logger.info(f"Saved upscaled clean latents to {output_dir}/stage2_upscaled_clean_latents.pt")
        #     upscaled_video = self.decode_latents(upscaled_latents)
        #     upscaled_video_output = self.video_processor.postprocess_video(video=upscaled_video, output_type="np")
            
        #     export_to_video(upscaled_video_output[0], os.path.join(output_dir, "stage2_upscaled_clean_video.mp4"), fps=8)
        #     logger.info(f"Saved upscaled clean video to {output_dir}/stage2_upscaled_clean_video.mp4")
                
        # Re-noise
        if upscale_res_steps > 0:
            start_time_re_noise = time.time()
            timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device=upscaled_latents.device)
            start_timestep_idx = num_inference_steps - upscale_res_steps
            to_timestep = timesteps[start_timestep_idx]

            renoised_latents = self.scheduler.re_noise(
                upscaled_latents,
                from_timestep=0,
                to_timestep=to_timestep.item(),
            )
            logger.info(f"[rank={self.dist_manager.rank}]: Re-noised latents for Stage 2 denoising (to_timestep: {to_timestep.item()})")
            end_time_re_noise = time.time()
            logger.info(f"[rank={self.dist_manager.rank}]: Re-noising Running time: {end_time_re_noise - start_time_re_noise} seconds")
            # if save_intermediate and output_dir:
            #     torch.save(renoised_latents.cpu(), os.path.join(output_dir, "stage2_renoised_latents.pt"))
            #     logger.info(f"Saved re-noised latents to {output_dir}/stage2_renoised_latents.pt")
            #     renoised_video = self.decode_latents(renoised_latents)
            #     renoised_video_output = self.video_processor.postprocess_video(video=renoised_video, output_type="np")
            #     export_to_video(renoised_video_output[0], os.path.join(output_dir, "stage2_renoised_init_video.mp4"), fps=8)
            #     logger.info(f"Saved re-noised initial video to {output_dir}/stage2_renoised_init_video.mp4")
        else:
            renoised_latents = upscaled_latents

        stage2_result = self.tiling_call__(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            eta=eta,
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            use_skip_time=True,
            denoise_to_step=num_inference_steps - upscale_res_steps, # Correctly skip steps for stage 2
            init_latent=renoised_latents,
            shift_timesteps=shift_timesteps,
            loop_step=loop_step,
            enable_intra_tile_cache=enable_intra_tile_cache,
            cache_thresh=cache_thresh,
            # Profiling
            save_k_history=save_k_history,
            k_history_filename=k_history_filename,
            enable_noise_pred_profile=enable_noise_pred_profile,  
            enable_cache_residual_profile=enable_cache_residual_profile,
            enable_region_aware_cache=enable_region_aware_cache,
            static_tile_cache_scale_factor=static_tile_cache_scale_factor,
        )

        if self.dist_manager.is_first_rank:
            if enable_noise_pred_profile:
                self.transformer.save_noise_pred_profile(os.path.join(output_dir or '.', "noise_pred_profile.json"))
            if enable_cache_residual_profile:
                self.transformer.save_cache_residual_profile(os.path.join(output_dir or '.', "cache_residual_profile.json"))
        logger.info(f"[rank={self.dist_manager.rank}]: === Two-Stage Generation Completed ===")
        return stage2_result

    def _upscale_latents(self, latents: torch.Tensor, target_height: int, target_width: int,) -> torch.Tensor:
        """Upscale latents using interpolation in latent space"""
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

        if not self.vae.config.invert_scale_latents:
            upscaled = self.vae_scaling_factor_image * upscaled
        else:
            upscaled = 1 / self.vae_scaling_factor_image * upscaled

        return upscaled

    def _upscale_video(self, latents: torch.Tensor, target_height: int, target_width: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Upscale latents by decoding to video, upscaling in pixel space, and re-encoding."""
        import torch.nn.functional as F
        # 1. Decode latents to video
        # decode_latents expects latents to be [B, F, C, H, W] and returns video as [B, 3, F, H, W]
        video = self.decode_latents(latents)
        batch_size, _, num_frames, height, width = video.shape

        # 2. Upscale video in pixel space
        # video is [B, 3, F, H, W], permute to [B, F, 3, H, W] then reshape for interpolation
        video = video.permute(0, 2, 1, 3, 4) # [B, F, 3, H, W]
        video_reshaped = video.contiguous().view(batch_size * num_frames, 3, height, width)
        upscaled_video_reshaped = F.interpolate(
            video_reshaped,
            size=(target_height, target_width),
            mode='bicubic',
            align_corners=False
        )
        
        # Reshape back to [B, F, 3, H_high, W_high] and then permute for VAE
        upscaled_video = upscaled_video_reshaped.view(batch_size, num_frames, 3, target_height, target_width)
        upscaled_video = upscaled_video.permute(0, 2, 1, 3, 4) # [B, 3, F, H_high, W_high]
        
        # 3. Re-encode the upscaled video back to latents
        if isinstance(generator, list):
             # Process each item in the batch separately with its generator
             image_latents_list = [
                retrieve_latents(self.vae.encode(upscaled_video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
             ]
             image_latents = torch.cat(image_latents_list, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(upscaled_video), generator)

        # image_latents shape is [B, C_latent, F, H_latent, W_latent]
        # Convert to [B, F, C_latent, H_latent, W_latent]
        upscaled_latents = image_latents.permute(0, 2, 1, 3, 4)

        if not self.vae.config.invert_scale_latents:
            upscaled_latents = self.vae_scaling_factor_image * upscaled_latents
        else:
            upscaled_latents = 1 / self.vae_scaling_factor_image * upscaled_latents

        return upscaled_latents
    
