import os
import time
import sys
from pathlib import Path
from loguru import logger
import datetime
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

import torch
import torch.distributed as dist


def main():
    args = parse_args()
    shift_timesteps = None
    if args.shift_timesteps:
        shift_timesteps = [int(step) for step in args.shift_timesteps.split(",")]
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    long_timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", timeout=long_timeout)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        i2v_mode=args.i2v_mode,
        i2v_resolution=args.i2v_resolution,
        i2v_image_path=args.i2v_image_path,
        i2v_condition_type=args.i2v_condition_type,
        i2v_stability=args.i2v_stability,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        # two-stage-generation
        two_stage_generation=args.two_stage_generation,
        upscale_factor=args.upscale_factor,
        upscale_res_steps=args.upscale_res_steps,
        save_intermediate=args.save_intermediate,
        output_dir=args.output_dir,
        shift_timesteps=shift_timesteps,
        load_prev_latents_path=args.load_prev_latents_path,
        loop_step=args.loop_step,
        enable_intra_tile_cache=args.enable_intra_tile_cache,
        cache_thresh=args.cache_thresh,
        enable_region_aware_cache=args.enable_region_aware_cache,
        static_tile_cache_scale_factor=args.static_tile_cache_scale_factor,
    )
    
    # Save samples
    if hunyuan_video_sampler.dist_manager.is_first_rank:
        samples = outputs['samples']
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            print(f"sample shape: {sample.shape}")
            cur_save_path = f"{save_path}/{args.prompt}_final.mp4"
            save_videos_grid(sample, cur_save_path, fps=8)
            logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
