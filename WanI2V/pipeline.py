#!/usr/bin/env python
"""Two-stage tiled I2V on Wan2.1-I2V-14B -- the SuperGen/UltraGen stack, third backbone.

Mirrors `CogvideoI2V/pipeline.py`'s CLI so the shared runners in
`UltraGen-PPoPP-experiments/run/` need only a `run_wan()` helper.

Stage 1: Wan's own `generate()` at 720p, kept verbatim -- it is the untouched
         base model, which is what makes the 2K/4K rows attributable to our stack.
Stage 2: decode -> pixel-space bicubic upscale -> re-encode -> flow-match re-noise
         -> tiled denoising with canvas-absolute RoPE and the region-aware cache
         (`tiled_stage2.py`).

Pixel-space upscaling matches CogVideoX and FreeSwim; latent-space was measured to
tear (results/P0-2_rope_artifact/upscale_mode/NOTES.md).

    python WanI2V/pipeline.py \
        --prompt "a lion is roaring in the wild" \
        --image_path .../inputs/4k_2160x3840/a\\ lion....jpg \
        --ckpt_dir ~/ckpts/Wan2.1-I2V-14B-720P \
        --width 3840 --height 2160 --upscale_factor 3 \
        --upscale_res_steps 35 --rope_mode local \
        --output_path out.mp4

Env: ~/envs/easycache (torch 2.5.1; imports `wan` and its 7 configs).
"""
import argparse
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))              # SuperGen/
sys.path.insert(0, _HERE)
sys.path.insert(0, "/home/ubuntu/repo/Wan2.1")          # the Wan runtime

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ckpts/Wan2.1-I2V-14B-720P"))
    ap.add_argument("--task", default="i2v-14B")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--num_frames", type=int, default=41)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--output_path", default="./wan_out.mp4")
    # --- our stack ---
    ap.add_argument("--upscale_factor", type=int, default=1,
                    help="1 = single stage at --width/--height; 2 = 2K; 3 = 4K")
    ap.add_argument("--upscale_res_steps", type=int, default=35,
                    help="Stage-2 denoising steps of --num_inference_steps. 35/50 keeps "
                         "~23.6%% of the Stage-1 signal, matching CineScale/FreeSwim.")
    ap.add_argument("--loop_step", type=int, default=16)
    ap.add_argument("--rope_mode", default="local", choices=["local", "extend", "ntk"])
    ap.add_argument("--enable_cache", action="store_true")
    ap.add_argument("--cache_thresh", type=float, default=0.09)
    ap.add_argument("--negative_prompt", default=None)
    ap.add_argument("--offload_model", action="store_true", default=True,
                    help="Move the DiT to CPU between the cond/uncond calls. Wan's own "
                         "default; the 14B model plus 4K activations does not fit one "
                         "80GB GPU without it.")
    ap.add_argument("--no_offload_model", dest="offload_model", action="store_false")
    ap.add_argument("--stage1_latents_path", default=None,
                    help="Reuse/save Stage-1 latents here so compared methods share them")
    a = ap.parse_args()

    import wan
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    from wan.utils.utils import cache_video

    from tiled_stage2 import WanTiledStage2, flow_match_renoise

    t_start = time.time()
    cfg = WAN_CONFIGS[a.task]
    device_id = int(os.environ.get("LOCAL_RANK", 0))

    # Stage-1 resolution: the base model's native scale.
    s1_h, s1_w = a.height // a.upscale_factor, a.width // a.upscale_factor
    logger.info(f"Stage 1 at {s1_h}x{s1_w}; Stage 2 target {a.height}x{a.width} "
                f"(upscale_factor={a.upscale_factor})")

    logger.info("loading Wan2.1-I2V-14B ...")
    pipe = wan.WanI2V(config=cfg, checkpoint_dir=a.ckpt_dir, device_id=device_id,
                      rank=0, t5_cpu=False)
    img = Image.open(a.image_path).convert("RGB")
    # Wan's own generate() converts internally; Stage 2 needs the tensor form too,
    # in [-1, 1] CHW, matching what image2video.py feeds clip.visual and vae.encode.
    import torchvision.transforms.functional as TF
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(f"cuda:{device_id}")

    # ---------------------------------------------------------------- Stage 1
    if a.upscale_factor == 1:
        video = pipe.generate(
            a.prompt, img, max_area=s1_h * s1_w, frame_num=a.num_frames,
            shift=a.shift, sampling_steps=a.num_inference_steps,
            guide_scale=a.guidance_scale, seed=a.seed, offload_model=a.offload_model,
            n_prompt=a.negative_prompt or "",
        )
        cache_video(tensor=video[None], save_file=a.output_path, fps=a.fps,
                    normalize=True, value_range=(-1, 1))
        logger.info(f"Total running time is {time.time() - t_start:.2f} seconds")
        return

    stage2 = WanTiledStage2(pipe, rope_mode=a.rope_mode, loop_step=a.loop_step,
                            device=f"cuda:{device_id}")

    s1_latent = None
    if a.stage1_latents_path and os.path.isfile(a.stage1_latents_path):
        s1_latent = torch.load(a.stage1_latents_path, map_location=f"cuda:{device_id}",
                               weights_only=False)
        logger.info(f"reused Stage-1 latents from {a.stage1_latents_path} "
                    f"{tuple(s1_latent.shape)}")

    if s1_latent is None:
        logger.info("Stage 1: generating the low-resolution guide")
        s1_video = pipe.generate(
            a.prompt, img, max_area=s1_h * s1_w, frame_num=a.num_frames,
            shift=a.shift, sampling_steps=a.num_inference_steps,
            guide_scale=a.guidance_scale, seed=a.seed, offload_model=a.offload_model,
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

    # ---------------------------------------------------------------- Stage 2
    t_up = time.time()
    lat_h = a.height // cfg.vae_stride[1]
    lat_w = a.width // cfg.vae_stride[2]
    upscaled = stage2.upscale_pixel(s1_latent, a.height, a.width)
    logger.info(f"Upsampling Running time: {time.time() - t_up:.4f} seconds "
                f"-> latent {tuple(upscaled.shape)}")

    stage2.install_rope(a.height, a.width, a.num_frames)

    # Flow-matching schedule, then re-noise to where Stage 2 starts.
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(a.num_inference_steps, device=f"cuda:{device_id}", shift=a.shift)
    start_idx = a.num_inference_steps - a.upscale_res_steps
    sigma_start = float(sched.sigmas[start_idx])
    gen = torch.Generator(device=f"cuda:{device_id}").manual_seed(a.seed)
    noise = torch.randn(upscaled.shape, generator=gen, device=upscaled.device,
                        dtype=upscaled.dtype)
    renoised = flow_match_renoise(upscaled, noise, sigma_start)
    logger.info(f"re-noised to sigma={sigma_start:.4f} (step index {start_idx}), "
                f"keeping {(1 - sigma_start):.1%} of the Stage-1 signal")

    # Stage-2 timesteps only, and a scheduler whose state starts there.
    sched2 = FlowUniPCMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    sched2.set_timesteps(a.num_inference_steps, device=f"cuda:{device_id}", shift=a.shift)
    timesteps2 = sched2.timesteps[start_idx:]

    # Rebuild the I2V conditioning at the TARGET resolution. Wan's own loop builds
    # these inside generate() (image2video.py:236-295):
    #   context   -- T5 text embedding
    #   clip_fea  -- CLIP visual feature of the input image (resolution-independent)
    #   y         -- [mask | VAE(first frame padded with zeros)], CANVAS-sized, so it
    #                must be cut per tile exactly like the latent
    #   seq_len   -- token count the model expects, which is PER TILE here, not for
    #                the whole canvas
    dev = torch.device(f"cuda:{device_id}")
    # Stage 1 offloads T5 and CLIP to the CPU when offload_model is on
    # (image2video.py:228, :238), so bring them back before Stage 2 uses them.
    pipe.text_encoder.model.to(dev)
    pipe.clip.model.to(dev)
    ctx = pipe.text_encoder([a.prompt], dev)
    ctx_null = pipe.text_encoder([a.negative_prompt or cfg.sample_neg_prompt], dev)
    clip_context = pipe.clip.visual([img_tensor[:, None, :, :]])
    if a.offload_model:
        # Free them again; the tile loop only needs the DiT and the VAE.
        pipe.text_encoder.model.cpu()
        pipe.clip.model.cpu()
        torch.cuda.empty_cache()

    lat_f = (a.num_frames - 1) // cfg.vae_stride[0] + 1
    msk = torch.ones(1, a.num_frames, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

    import torch.nn.functional as _F
    y_cond = pipe.vae.encode([
        torch.concat([
            _F.interpolate(img_tensor[None].cpu(), size=(a.height, a.width),
                           mode="bicubic").transpose(0, 1),
            torch.zeros(3, a.num_frames - 1, a.height, a.width),
        ], dim=1).to(dev)
    ])[0]
    y_canvas = torch.concat([msk, y_cond])          # [C_y, F_lat, lat_h, lat_w]
    logger.info(f"I2V conditioning: y_canvas {tuple(y_canvas.shape)} "
                f"clip_fea {tuple(clip_context[0].shape)}")

    # Per-tile seq_len: one window of 90x160 latent at patch (1,2,2).
    win_lat_h, win_lat_w = 90, 160
    tile_seq_len = (lat_f // cfg.patch_size[0]) * (win_lat_h // cfg.patch_size[1]) \
                   * (win_lat_w // cfg.patch_size[2])
    arg_c = {"context": [ctx[0]], "clip_fea": clip_context, "seq_len": tile_seq_len}
    arg_null = {"context": ctx_null, "clip_fea": clip_context, "seq_len": tile_seq_len}
    logger.info(f"per-tile seq_len={tile_seq_len}")

    t_s2 = time.time()
    final = stage2.denoise(
        renoised, timesteps2, sched2, arg_c, arg_null,
        guide_scale=a.guidance_scale,
        shift_timesteps=list(range(a.upscale_res_steps)),
        seed_g=gen, enable_cache=a.enable_cache,
        y_canvas=y_canvas,
    )
    logger.info(f"Second Stage Running time: {time.time() - t_s2} seconds")
    stage2.uninstall_rope()

    # Free the DiT before decoding: a 2K/4K VAE decode needs several GB of its own,
    # and the 14B model is still resident from the tile loop.
    if a.offload_model:
        pipe.model.cpu()
        torch.cuda.empty_cache()
    video = pipe.vae.decode([final])[0]
    cache_video(tensor=video[None], save_file=a.output_path, fps=a.fps,
                normalize=True, value_range=(-1, 1))
    logger.info(f"Saved final video to: {a.output_path}")
    logger.info(f"Total running time is {time.time() - t_start:.2f} seconds")


if __name__ == "__main__":
    main()
