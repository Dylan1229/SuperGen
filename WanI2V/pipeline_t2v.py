#!/usr/bin/env python
"""Two-stage tiled T2V on Wan2.1 -- the SuperGen/UltraGen stack, text-to-video.

Why a T2V entry point exists alongside the I2V one
--------------------------------------------------
FreeSwim, the only training-free high-resolution competitor with released runnable
code, is **T2V**: `inference.py` loads `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and runs
T2V -> upscale -> video2video. Its `WanVideoToVideoPipeline` has no image
conditioning path (`encoder_hidden_states_img` never appears in it). So a
same-task, same-backbone comparison against FreeSwim requires us to run T2V too;
comparing our I2V against its T2V would confound task with method.

T2V is also simpler than I2V here: `arg_c` is just `{context, seq_len}`
(`wan/text2video.py:230`), with none of the I2V `y` mask/first-frame conditioning
or CLIP image features, so the per-tile conditioning problem disappears.

Everything else is shared with `pipeline.py`: pixel-space upscaling, flow-match
re-noising, `WanTiledStage2` for the tiled Stage-2 loop, canvas-absolute RoPE, and
the region-aware cache.

    python WanI2V/pipeline_t2v.py \
        --prompt "a lion is roaring in the wild" \
        --task t2v-1.3B --ckpt_dir ~/ckpts/Wan2.1-T2V-1.3B \
        --width 2560 --height 1440 --upscale_factor 2 \
        --output_path out.mp4

Env: ~/envs/easycache.
"""
import argparse
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))              # SuperGen/
sys.path.insert(0, _HERE)
sys.path.insert(0, "/home/ubuntu/repo/Wan2.1")

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


from pipeline import broadcast_stage1  # noqa: E402  (same derived-shape logic)


def setup_distributed(device_id, enable_cache):
    """Join the process group if launched under torchrun; else stay single-process.

    Returns a `DistributedManager` or None. None means the single-GPU path, which
    stays supported because the serial timings in the paper's tables must not carry
    collective overhead, and because a 720p single-stage run has one tile.
    """
    import torch.distributed as dist
    if "RANK" not in os.environ:
        logger.info("not under torchrun: single-GPU path")
        return None
    torch.cuda.set_device(device_id)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    from utils.distributed import DistributedManager
    dm = DistributedManager("allgather", enable_cache=enable_cache)
    logger.info(f"[rank={dm.rank}/{dm.world_size}] tile parallelism enabled")
    return dm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--task", default="t2v-1.3B",
                    choices=["t2v-1.3B", "t2v-14B"],
                    help="1.3B matches what FreeSwim actually runs; 14B is the larger arm.")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ckpts/Wan2.1-T2V-1.3B"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--num_frames", type=int, default=41)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--output_path", default="./wan_t2v_out.mp4")
    ap.add_argument("--upscale_factor", type=int, default=1)
    ap.add_argument("--upscale_res_steps", type=int, default=35)
    ap.add_argument("--loop_step", type=int, default=16)
    ap.add_argument("--rope_mode", default="local", choices=["local", "extend", "ntk"])
    ap.add_argument("--enable_cache", action="store_true")
    ap.add_argument("--cache_thresh", type=float, default=0.09)
    ap.add_argument("--negative_prompt", default=None)
    ap.add_argument("--offload_model", action="store_true", default=True)
    ap.add_argument("--no_offload_model", dest="offload_model", action="store_false")
    ap.add_argument("--stage1_latents_path", default=None)
    a = ap.parse_args()

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video

    from tiled_stage2 import WanTiledStage2, flow_match_renoise

    t_start = time.time()
    cfg = WAN_CONFIGS[a.task]
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{device_id}")
    dist_manager = setup_distributed(device_id, a.enable_cache)

    s1_h, s1_w = a.height // a.upscale_factor, a.width // a.upscale_factor
    logger.info(f"Stage 1 at {s1_h}x{s1_w}; Stage 2 target {a.height}x{a.width} "
                f"(upscale_factor={a.upscale_factor}, task={a.task})")

    logger.info(f"loading {a.task} ...")
    pipe = wan.WanT2V(config=cfg, checkpoint_dir=a.ckpt_dir, device_id=device_id,
                      rank=0, t5_cpu=False)

    # ---------------------------------------------------------------- Stage 1
    if a.upscale_factor == 1:
        video = pipe.generate(
            a.prompt, size=(s1_w, s1_h), frame_num=a.num_frames, shift=a.shift,
            sampling_steps=a.num_inference_steps, guide_scale=a.guidance_scale,
            seed=a.seed, offload_model=a.offload_model,
            n_prompt=a.negative_prompt or "",
        )
        cache_video(tensor=video[None], save_file=a.output_path, fps=a.fps,
                    normalize=True, value_range=(-1, 1))
        logger.info(f"Total running time is {time.time() - t_start:.2f} seconds")
        return

    stage2 = WanTiledStage2(pipe, rope_mode=a.rope_mode, loop_step=a.loop_step,
                            device=f"cuda:{device_id}",
                            dist_manager=dist_manager)

    s1_latent = None
    if a.stage1_latents_path and os.path.isfile(a.stage1_latents_path):
        s1_latent = torch.load(a.stage1_latents_path, map_location=dev,
                               weights_only=False)
        logger.info(f"reused Stage-1 latents {tuple(s1_latent.shape)}")

    if s1_latent is None:
        # rank 0 only, then broadcast -- see pipeline.py's note.
        if dist_manager is None or dist_manager.is_first_rank:
            logger.info("Stage 1: generating the low-resolution guide")
            s1_video = pipe.generate(
                a.prompt, size=(s1_w, s1_h), frame_num=a.num_frames, shift=a.shift,
                sampling_steps=a.num_inference_steps, guide_scale=a.guidance_scale,
                seed=a.seed, offload_model=a.offload_model,
                n_prompt=a.negative_prompt or "",
            )
            s1_latent = pipe.vae.encode([s1_video])[0]
            if a.stage1_latents_path:
                os.makedirs(os.path.dirname(a.stage1_latents_path) or ".", exist_ok=True)
                torch.save(s1_latent, a.stage1_latents_path)
            out_dir = os.path.dirname(a.output_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            cache_video(tensor=s1_video[None],
                        save_file=os.path.join(out_dir, "stage1_lowres_video.mp4"),
                        fps=a.fps, normalize=True, value_range=(-1, 1))
            del s1_video
            torch.cuda.empty_cache()
        s1_latent = broadcast_stage1(s1_latent, dist_manager, cfg, a, device_id)

    # ---------------------------------------------------------------- Stage 2
    t_up = time.time()
    upscaled = stage2.upscale_pixel(s1_latent, a.height, a.width)
    logger.info(f"Upsampling Running time: {time.time() - t_up:.4f} seconds "
                f"-> latent {tuple(upscaled.shape)}")

    stage2.install_rope(a.height, a.width, a.num_frames)

    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(a.num_inference_steps, device=dev, shift=a.shift)
    start_idx = a.num_inference_steps - a.upscale_res_steps
    sigma_start = float(sched.sigmas[start_idx])
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    noise = torch.randn(upscaled.shape, generator=gen, device=upscaled.device,
                        dtype=upscaled.dtype)
    renoised = flow_match_renoise(upscaled, noise, sigma_start)
    logger.info(f"re-noised to sigma={sigma_start:.4f} (step {start_idx}), "
                f"keeping {(1 - sigma_start):.1%} of the Stage-1 signal")
    timesteps2 = sched.timesteps[start_idx:]

    # T2V conditioning is text only -- no `y`, no clip_fea. seq_len is per TILE.
    pipe.text_encoder.model.to(dev)
    ctx = pipe.text_encoder([a.prompt], dev)
    ctx_null = pipe.text_encoder([a.negative_prompt or cfg.sample_neg_prompt], dev)
    if a.offload_model:
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()

    lat_f = (a.num_frames - 1) // cfg.vae_stride[0] + 1
    win_lat_h, win_lat_w = 90, 160
    tile_seq_len = (lat_f // cfg.patch_size[0]) * (win_lat_h // cfg.patch_size[1]) \
                   * (win_lat_w // cfg.patch_size[2])
    arg_c = {"context": ctx, "seq_len": tile_seq_len}
    arg_null = {"context": ctx_null, "seq_len": tile_seq_len}
    logger.info(f"T2V per-tile seq_len={tile_seq_len}")

    t_s2 = time.time()
    final = stage2.denoise(
        renoised, timesteps2, sched, arg_c, arg_null,
        guide_scale=a.guidance_scale,
        shift_timesteps=list(range(a.upscale_res_steps)),
        seed_g=gen, enable_cache=a.enable_cache,
        y_canvas=None,          # T2V: no image conditioning to slice
    )
    logger.info(f"Second Stage Running time: {time.time() - t_s2} seconds")
    stage2.uninstall_rope()

    if a.offload_model:
        pipe.model.cpu()
        torch.cuda.empty_cache()
    if dist_manager is None or dist_manager.is_first_rank:
        video = pipe.vae.decode([final])[0]
        cache_video(tensor=video[None], save_file=a.output_path, fps=a.fps,
                    normalize=True, value_range=(-1, 1))
        logger.info(f"Saved final video to: {a.output_path}")
    logger.info(f"Total running time is {time.time() - t_start:.2f} seconds")
    if dist_manager is not None:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
