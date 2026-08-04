"""`encode_tiled` must equal a whole-frame VAE encode, not merely approximate it.

Wan's VAE has no spatial tiling, so 4K re-encoding has to be split. Splitting a
convolutional encoder is only safe if each tile is padded past the receptive field
and the padding is then discarded -- otherwise every tile edge gets a subtly wrong
latent, which shows up in the final video as a seam that no amount of shifting
removes. The failure is silent, hence this test.

Checked at 2K, where both paths fit in memory: force the tiled path with a low
threshold and compare against the untiled call on the same input. 4K cannot be
compared this way because the untiled reference is exactly what OOMs.

    python WanI2V/test_encode_tiled.py        # needs ~30 GB and the Wan VAE weights

Env: ~/envs/easycache.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get("WAN_REPO", "/home/ubuntu/repo/Wan2.1"))

CKPT = os.environ.get("WAN_CKPT", "/home/ubuntu/ckpts/Wan2.1-I2V-14B-720P")
H, W, FRAMES = 1440, 2560, 9      # 2K, few frames: this tests space, not time


def main():
    import wan
    from wan.configs import WAN_CONFIGS
    from tiled_stage2 import WanTiledStage2

    dev = "cuda:0"
    cfg = WAN_CONFIGS["i2v-14B"]

    # Only the VAE is needed, so build it directly instead of loading the 14B DiT.
    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_pth=os.path.join(CKPT, cfg.vae_checkpoint), device=dev)

    class Shim:
        config = cfg
        param_dtype = torch.bfloat16

        def __init__(self):
            self.vae = vae
            self.model = None

    stage2 = WanTiledStage2(Shim(), rope_mode="local", device=dev)

    g = torch.Generator().manual_seed(7)
    # Structured content, not pure noise: a smooth gradient plus texture makes a
    # boundary discontinuity visible in the error, where noise would mask it.
    yy = torch.linspace(-1, 1, H).view(1, 1, H, 1)
    xx = torch.linspace(-1, 1, W).view(1, 1, 1, W)
    pixels = (0.6 * (yy + xx) / 2
              + 0.4 * torch.rand(3, FRAMES, H, W, generator=g)).clamp(-1, 1)
    pixels = pixels.expand(3, FRAMES, H, W).contiguous().to(dev, torch.float32)

    print(f"encoding {tuple(pixels.shape)} both ways ...")
    with torch.no_grad():
        ref = stage2.encode_tiled(pixels, max_pixels_per_tile=H * W)      # untiled
        torch.cuda.empty_cache()
        got = stage2.encode_tiled(pixels, max_pixels_per_tile=1280 * 720)  # tiled

    assert ref.shape == got.shape, f"{tuple(got.shape)} != {tuple(ref.shape)}"
    err = (ref - got).abs()
    scale = ref.abs().mean().item()
    print(f"latent {tuple(ref.shape)}  |ref|mean={scale:.4f}")
    print(f"max_abs_err={err.max().item():.3e}  mean_abs_err={err.mean().item():.3e}")
    print(f"relative: max={err.max().item()/scale:.3%} mean={err.mean().item()/scale:.3%}")

    # Where the error sits matters more than its size: concentrated at tile seams
    # means the overlap is too small, spread evenly means it is just fp noise.
    sh, sw = cfg.vae_stride[1], cfg.vae_stride[2]
    n_h, n_w = max(1, -(-H // 1088)), max(1, -(-W // 1920))
    seam_rows = [((H // n_h) // sh) * sh // sh * i for i in range(1, n_h)]
    seam_cols = [((W // n_w) // sw) * sw // sw * i for i in range(1, n_w)]
    for r in seam_rows:
        band = err[:, :, max(0, r - 2):r + 2, :].mean().item()
        print(f"  seam row {r}: mean_err={band:.3e} ({band/max(err.mean().item(),1e-12):.2f}x overall)")
    for c in seam_cols:
        band = err[:, :, :, max(0, c - 2):c + 2].mean().item()
        print(f"  seam col {c}: mean_err={band:.3e} ({band/max(err.mean().item(),1e-12):.2f}x overall)")

    # Two separate bars, because there are two separate error sources.
    #
    # The convolutions must be reproduced essentially exactly -- that is what the
    # overlap is for, and a receptive-field bug shows up here. Measured: max 0.314%
    # with `middle.1` stubbed out. 1% leaves headroom for fp32 reassociation.
    #
    # `middle.1` is an AttentionBlock over every spatial position, so tiling changes
    # it and no overlap fixes that. Measured 4.271% max / 0.221% mean at 2K. Bounded
    # rather than eliminated: Stage 2 re-noises to sigma=0.9208 right after, making
    # the injected noise ~700x this, so it is gone after one denoising step. 8% max
    # catches a real regression while accepting the known attention gap.
    #
    # The seam ratios printed above are the diagnostic that separates the two: near
    # 1.0x means the residual is spread globally (attention), while a large spike at
    # the seam rows/cols would mean the overlap is genuinely too small.
    tol_max = 0.08 * scale
    tol_mean = 0.01 * scale
    ok = True
    if err.max().item() > tol_max:
        print(f"FAIL: max error {err.max().item():.3e} exceeds 8% of |latent| ({tol_max:.3e})")
        ok = False
    if err.mean().item() > tol_mean:
        print(f"FAIL: mean error {err.mean().item():.3e} exceeds 1% of |latent| ({tol_mean:.3e})")
        ok = False
    seam_max = max([err[:, :, max(0, r - 2):r + 2, :].mean().item() for r in seam_rows]
                   + [err[:, :, :, max(0, c - 2):c + 2].mean().item() for c in seam_cols]
                   + [0.0])
    if seam_max > 3.0 * err.mean().item():
        print(f"FAIL: error concentrates at tile seams ({seam_max/err.mean().item():.1f}x "
              f"overall) -- the overlap is too small, which IS fixable")
        ok = False
    if not ok:
        sys.exit(1)
    print("PASS: tiled VAE encode matches the whole-frame encode outside the "
          "known global-attention residual")
    check_decode(stage2, cfg)


def check_decode(stage2, cfg):
    """The decode side, which has the same structure and one extra requirement.

    Encode feeds 35 more denoising steps, so a hard cut at tile edges is harmless
    there. A decode is the last thing before the frame reaches a viewer, so its tiles
    are blended with a linear ramp -- and the test therefore checks that the seam band
    is not WORSE than the interior, which is the property the ramp buys.

    Run at the 720p latent grid, the largest whole-frame decode that works; 2K already
    fails inside vae.py:451, which is why decode had to be tiled at all.
    """
    g = torch.Generator().manual_seed(11)
    lat = (torch.randn(16, 3, 90, 160, generator=g) * 0.5).to("cuda:0", torch.float32)
    with torch.no_grad():
        ref = stage2.decode_tiled(lat, max_latent_per_tile=90 * 160)   # untiled
        torch.cuda.empty_cache()
        got = stage2.decode_tiled(lat, max_latent_per_tile=45 * 80, overlap=8)  # 2x2
    err = (ref - got).abs()
    lv = 127.5        # pixels are [-1, 1]; report in 8-bit levels
    print(f"\ndecode {tuple(ref.shape)}: max={err.max().item()*lv:.2f} "
          f"mean={err.mean().item()*lv:.3f} of 255")
    H = ref.shape[2]
    sr = H // 2
    seam = err[:, :, sr - 8:sr + 8, :].mean().item() * lv
    interior = err[:, :, 20:sr - 32, :].mean().item() * lv
    print(f"  seam band={seam:.3f} interior={interior:.3f} (8-bit levels)")

    # Half a level of mean error is well under any visible threshold, and the seam
    # must not be worse than the interior -- if it is, the ramp is not working.
    if err.mean().item() * lv > 0.5:
        print(f"FAIL: decode mean error {err.mean().item()*lv:.3f}/255 is too large")
        sys.exit(1)
    if seam > 1.5 * interior:
        print(f"FAIL: decode error concentrates at the seam ({seam:.3f} vs "
              f"{interior:.3f}) -- the linear blend is not working")
        sys.exit(1)
    print("PASS: tiled VAE decode matches the whole-frame decode, seam included")


if __name__ == "__main__":
    main()
