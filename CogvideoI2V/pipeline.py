import argparse
import logging
import time
import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal, Optional, List

import torch
import torch.distributed as dist

from diffusers import (
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline
)
from diffusers.utils import export_to_video, load_image, load_video

from modules import CustomCogVideoXDDIMScheduler, CachingCogVideoXTransformer3DModel
from pipeline_cogvideox_i2v_TVG import TiledCogVideoXImageToVideoPipeline
from utils.distributed import DistributedManager

logging.basicConfig(level=logging.INFO)
# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    upscale_factor: int = 2,
    upscale_res_steps: int = 30,
    shift_timesteps: Optional[List[int]] = None,
    low_res_latents_path: Optional[str] = None,
    loop_step: int = 8,
    enable_intra_tile_cache: bool = False,
    cache_thresh: float = 0.1,
    enable_region_aware_cache: bool = False,
    static_tile_cache_scale_factor: float = 1,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    - upscale_factor (int): Factor to upscale the low resolution initialization
    - upscale_res_steps (int): Number of steps in the upscale stage
    - shift_timesteps (Optional[List[int]]): A list of timesteps to perform shift. If None, no shifting will occur.
    - low_res_latents_path (Optional[str]): Path to pre-saved low-resolution latents. If provided and exists, Stage 1 will be skipped.
    - enable_intra_tile_cache (bool): Enable intra-tile cache
    - cache_thresh (float): Threshold for cache. 
    - enable_region_aware_cache (bool): Enable region-aware cache optimization that identifies static tiles.
    - static_tile_cache_scale_factor (float): Scale factor for cache threshold.
    """
    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m")
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution

    if generate_type == "i2v":
        pipe = TiledCogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        caching_transformer = CachingCogVideoXTransformer3DModel(**pipe.transformer.config)
        caching_transformer.load_state_dict(pipe.transformer.state_dict())
        caching_transformer.to(dtype)
        pipe.transformer = caching_transformer
        # NOTE(MX)
        dist_manager = DistributedManager("allgather", enable_intra_tile_cache)
        pipe.dist_manager = dist_manager
        pipe.transformer.dist_manager = dist_manager
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)
        
    pipe.scheduler = CustomCogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Set up generator for reproducibility
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    
    logging.info(f"== Finish Loading, start to generate video ==")
    # Generate video

    if not os.path.exists(low_res_latents_path):
        low_res_latents_path = None

    if generate_type == "i2v":
        output_result = pipe.two_stage_generation(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=False,  # This is used for DPM scheduler, for DDIM scheduler, it should be False
            generator=generator,
            upscale_factor=upscale_factor,
            upscale_res_steps=upscale_res_steps,  
            save_intermediate=True, 
            output_dir=os.path.dirname(output_path) if output_path else None,
            shift_timesteps=shift_timesteps,
            low_res_latents_path=low_res_latents_path,
            loop_step=loop_step,
            enable_intra_tile_cache=enable_intra_tile_cache,
            cache_thresh=cache_thresh,
            enable_region_aware_cache=enable_region_aware_cache,
            static_tile_cache_scale_factor=static_tile_cache_scale_factor,
        )
        if dist.get_rank() == 0:
            output = output_result.frames[0]
    elif generate_type == "t2v":
        output = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
    else:
        output = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

    # Save final video if we have output
    if dist.get_rank() == 0 and output is not None and output_path:
        export_to_video(output, output_path, fps=fps)
        logging.info(f"Saved final video to: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX1.5-5B", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--upscale_factor", type=int, default=2, help="The factor by which to upscale the low-resolution image")
    parser.add_argument("--upscale_res_steps", type=int, default=30, help="The number of steps to upscale the low-resolution image")
    parser.add_argument("--shift_timesteps", type=str, default=None, help="Comma-separated list of steps to perform shift (e.g., '0,1,2,3,4,40,41,42,43,44,45,46'). If not provided, no shifting will occur.")
    parser.add_argument("--low_res_latents_path", type=str, default=None, help="Path to pre-saved low-resolution latents. If provided and exists, Stage 1 will be skipped.")
    parser.add_argument("--loop_step", type=int, default=8, help="The loop step for sliding window tiling. Controls the step size for window shifting.")
    parser.add_argument("--enable_intra_tile_cache", action="store_true", help="Enable intra-tile caching")
    parser.add_argument("--cache_thresh", type=float, default=0.05, help="Threshold for cache")
    parser.add_argument("--enable_region_aware_cache", action="store_true", help="Enable region-aware cache optimization")
    parser.add_argument("--static_tile_cache_scale_factor", type=float, default=0.5, help="Scale factor for cache threshold when tile is considered most static")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    shift_timesteps = None
    if args.shift_timesteps:
        shift_timesteps = [int(step) for step in args.shift_timesteps.split(",")]
    
    # NOTE(MX)
    long_timeout = datetime.timedelta(minutes=30)
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=long_timeout)
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        # Single GPU mode, skip distributed init
        pass

    start_time = time.time()
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
        upscale_factor=args.upscale_factor,
        upscale_res_steps=args.upscale_res_steps,
        shift_timesteps=shift_timesteps,
        low_res_latents_path=args.low_res_latents_path,
        loop_step=args.loop_step,
        enable_intra_tile_cache=args.enable_intra_tile_cache,
        cache_thresh=args.cache_thresh,
        enable_region_aware_cache=args.enable_region_aware_cache,
        static_tile_cache_scale_factor=args.static_tile_cache_scale_factor,
    )
    end_time = time.time()
    logging.info(f"Total running time is {end_time - start_time:.2f} seconds")