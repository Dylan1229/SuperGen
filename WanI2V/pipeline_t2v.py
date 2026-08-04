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
# The Wan runtime, which is a separate checkout. Overridable because its location is
# site-specific: WAN_REPO if set, else a sibling of this repo, else the path used on our host.
_WAN = os.environ.get("WAN_REPO") or os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "Wan2.1")
if not os.path.isdir(_WAN):
    _WAN = "/home/ubuntu/repo/Wan2.1"
sys.path.insert(0, _WAN)

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


from pipeline import (broadcast_stage1, broadcast_upscaled,  # noqa: E402
                      rank0_only_stage1)


def setup_sequence_parallel(ulysses_size, ring_size, cfg):
    """Initialise xfuser's SP groups, if requested. Returns whether SP is active.

    This composes with tile parallelism rather than replacing it: tiles partition the canvas and
    need no communication inside attention, while SP shards each tile's sequence and does. The two
    axes are orthogonal, which is the claim the parallelism table is meant to support.

    Ulysses shards the head dimension, so the degree must divide num_heads. Wan T2V-14B has 40, so
    2/4/5/8 are valid and 3/6/7 are not -- an invalid degree is rejected here because the failure
    downstream is a silently wrong tensor shape rather than an error.
    """
    if ulysses_size <= 1 and ring_size <= 1:
        return False
    import os
    import torch.distributed as dist
    if "RANK" not in os.environ:
        raise SystemExit("sequence parallelism needs torchrun (RANK is unset)")
    if not dist.is_initialized():
        # There is a circular dependency to break here: xfuser's group setup needs a process
        # group, but the process group was created inside setup_distributed(), which in turn needs
        # sp_size to know how to partition tiles. So SP creates the group itself when it is first,
        # and setup_distributed() then finds it already initialised.
        #
        # Getting this wrong is what killed every SP run: setup_sequence_parallel ran first, found
        # no group, and exited with "sequence parallelism needs torchrun" -- on all 24 rc=1 cells.
        import datetime
        import torch
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl",
                                timeout=datetime.timedelta(minutes=60))
    world = dist.get_world_size()
    if ulysses_size * ring_size != world:
        raise SystemExit(f"ulysses_size({ulysses_size}) * ring_size({ring_size}) != "
                         f"world_size({world})")
    heads = getattr(cfg, "num_heads", None)
    if heads and ulysses_size > 1 and heads % ulysses_size != 0:
        raise SystemExit(f"ulysses_size {ulysses_size} does not divide num_heads {heads}; "
                         f"valid degrees are {[d for d in range(2, heads + 1) if heads % d == 0]}")
    from xfuser.core.distributed import (init_distributed_environment,
                                         initialize_model_parallel)
    init_distributed_environment(rank=dist.get_rank(), world_size=world)
    initialize_model_parallel(sequence_parallel_degree=world,
                              ring_degree=ring_size,
                              ulysses_degree=ulysses_size)
    logger.info(f"sequence parallelism: ulysses={ulysses_size} ring={ring_size} "
                f"over world={world}, num_heads={heads}")
    return True


def setup_distributed(device_id, enable_cache, sp_size=1):
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
        # NCCL's 10-minute default is shorter than Stage 1. Stage 1 runs on rank 0
        # alone (50 steps at ~13 s/it is ~11 min at 4K), so ranks 1..N-1 sit in the
        # Stage-1 broadcast that whole time and the watchdog kills them mid-run.
        # CogVideoX and Hunyuan both already raise this (CogvideoI2V/pipeline.py:396).
        import datetime
        dist.init_process_group(backend="nccl",
                                timeout=datetime.timedelta(minutes=60))
    from utils.distributed import DistributedManager
    dm = DistributedManager("allgather", enable_cache=enable_cache, sp_size=sp_size)
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
    # Sequence parallelism, for the parallelism-scaling comparison. Wan supports it natively:
    # `WanT2V(use_usp=True)` swaps in xfuser's usp_attn_forward / usp_dit_forward
    # (wan/text2video.py:91-101), so this is the model's own SP rather than something we bolt on.
    #
    # Ulysses shards the HEAD dimension, so the degree must divide num_heads. Wan T2V-14B has 40
    # (wan/configs/wan_t2v_14B.py:24), giving valid degrees 2, 4, 5, 8. A degree that does not
    # divide it is rejected below rather than producing a quietly wrong result.
    ap.add_argument("--ulysses_size", type=int, default=1,
                    help="Ulysses SP degree; must divide num_heads (40 for T2V-14B)")
    ap.add_argument("--ring_size", type=int, default=1,
                    help="Ring-attention degree; ulysses_size * ring_size must equal world size")
    a = ap.parse_args()

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video

    from tiled_stage2 import WanTiledStage2, flow_match_renoise

    t_start = time.time()
    cfg = WAN_CONFIGS[a.task]
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{device_id}")
    # SP groups first: the manager needs the degree to partition tiles over GROUPS rather than
    # over ranks.
    use_usp = setup_sequence_parallel(a.ulysses_size, a.ring_size, cfg)
    sp_size = a.ulysses_size * a.ring_size if use_usp else 1
    dist_manager = setup_distributed(device_id, a.enable_cache, sp_size=sp_size)

    s1_h, s1_w = a.height // a.upscale_factor, a.width // a.upscale_factor
    logger.info(f"Stage 1 at {s1_h}x{s1_w}; Stage 2 target {a.height}x{a.width} "
                f"(upscale_factor={a.upscale_factor}, task={a.task})")

    logger.info(f"loading {a.task} ...")
    # rank must be the REAL rank when SP is on: Wan uses it to decide which rank holds the T5 and
    # which returns the video, and hard-coding 0 makes every rank think it is the writer.
    import torch.distributed as _dist
    _rank = _dist.get_rank() if _dist.is_initialized() else 0
    pipe = wan.WanT2V(config=cfg, checkpoint_dir=a.ckpt_dir, device_id=device_id,
                      rank=_rank, t5_cpu=False, use_usp=use_usp)

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
    t_s1 = time.time()
    reused_s1 = False
    if a.stage1_latents_path and os.path.isfile(a.stage1_latents_path):
        s1_latent = torch.load(a.stage1_latents_path, map_location=dev,
                               weights_only=False)
        reused_s1 = True
        logger.info(f"reused Stage-1 latents {tuple(s1_latent.shape)}")

    if s1_latent is None:
        # rank 0 only, then broadcast -- see pipeline.py's note.
        if dist_manager is None or dist_manager.is_first_rank:
            logger.info("Stage 1: generating the low-resolution guide")
            with rank0_only_stage1():
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

    # See pipeline.py: the third bar of the breakdown figure. REUSED means it was loaded, not run.
    logger.info(f"First Stage Running time: {time.time() - t_s1:.4f} seconds"
                f"{' (REUSED from cache, not generated)' if reused_s1 else ''}")

    # ---------------------------------------------------------------- Stage 2
    t_up = time.time()
    # rank 0 only, then broadcast. Every rank was running the identical
    # decode->bicubic->encode over the whole canvas, which at 4K is both N-times
    # wasted work and N concurrent multi-GB VAE passes on one node.
    if dist_manager is None or dist_manager.is_first_rank:
        # Park the 14B DiT on the CPU first. It is ~67 GB resident after Stage 1, and
        # the tiled VAE still needs several GB of its own; without this, decode dies
        # asking for its last 2 GB. `denoise` moves it back before the tile loop.
        if a.offload_model:
            pipe.model.cpu()
            torch.cuda.empty_cache()
        upscaled = stage2.upscale_pixel(s1_latent, a.height, a.width)
    else:
        upscaled = None
    upscaled = broadcast_upscaled(upscaled, dist_manager, cfg, a, device_id)
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
        seed_g=gen, enable_cache=a.enable_cache, cache_thresh=a.cache_thresh,
        y_canvas=None,          # T2V: no image conditioning to slice
    )
    logger.info(f"Second Stage Running time: {time.time() - t_s2} seconds")
    stage2.uninstall_rope()

    if a.offload_model:
        pipe.model.cpu()
        torch.cuda.empty_cache()
    if dist_manager is None or dist_manager.is_first_rank:
        # Tiled for the same reason as the re-encode; a whole 4K decode OOMs.
        video = stage2.decode_tiled(final)
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
