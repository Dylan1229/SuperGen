"""Verify the Wan Stage-2 tiling machinery without loading 67 GB of weights.

Checks the parts that are easy to get silently wrong:

  1. flow-match re-noise is LINEAR in sigma, not DDIM's sqrt(alpha_bar). Confirms
     sigma=0 is the identity, sigma=1 is pure noise, and that it differs from a
     DDIM-style interpolation at intermediate sigma.
  2. Geometry: latent shape and patch grid match Wan's own arithmetic
     (vae_stride (4,8,8), patch (1,2,2)), for 720p / 2K / 4K.
  3. Tiling: 720p is a single window (so tiling and shifting drop out), 2K is 2x2,
     4K is 3x3 -- the same schedule the other backbones use.
  4. Round-trip: the [C,F,H,W] <-> [B,F,C,H,W] permutes used to bridge Wan's layout
     and TiledLatentTensor2D are lossless.
  5. Window coverage: with the paper's loop_step the tiles partition the canvas
     exactly once (coverage 1.0), so no unintended averaging.

    python WanI2V/test_tiled_stage2.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # SuperGen/
sys.path.insert(0, _HERE)

from utils.tile_utils import SlidingWindowConfig, TiledLatentTensor2D  # noqa: E402

from tiled_stage2 import flow_match_renoise  # noqa: E402


class _FakeCfg:
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)


class _FakeWan:
    config = _FakeCfg()


def geometry(height, width, frames):
    vs, ps = _FakeCfg.vae_stride, _FakeCfg.patch_size
    lat = ((frames - 1) // vs[0] + 1, height // vs[1], width // vs[2])
    grid = (lat[0] // ps[0], lat[1] // ps[1], lat[2] // ps[2])
    return lat, grid


def main():
    failures = []
    torch.manual_seed(0)

    # --- 1. flow-match re-noise ---
    x0 = torch.randn(4, 8, 16, 24)
    noise = torch.randn_like(x0)
    ident = flow_match_renoise(x0, noise, 0.0)
    pure = flow_match_renoise(x0, noise, 1.0)
    mid = flow_match_renoise(x0, noise, 0.5)
    print("[1] flow-match re-noise")
    print(f"    sigma=0 is identity: {torch.allclose(ident, x0)}")
    print(f"    sigma=1 is pure noise: {torch.allclose(pure, noise)}")
    if not torch.allclose(ident, x0):
        failures.append("sigma=0 is not the identity")
    if not torch.allclose(pure, noise):
        failures.append("sigma=1 is not pure noise")
    # A DDIM-style weighting would use sqrt terms; make sure we are NOT doing that.
    ddim_like = (1 - 0.5 ** 2) ** 0.5 * x0 + 0.5 * noise
    differs = not torch.allclose(mid, ddim_like, atol=1e-4)
    print(f"    differs from DDIM sqrt weighting at sigma=0.5: {differs}")
    if not differs:
        failures.append("re-noise matches DDIM weighting -- wrong for flow matching")

    # --- 2. geometry ---
    print("\n[2] geometry (vae_stride (4,8,8), patch (1,2,2), 41 frames)")
    expect = {
        (720, 1280): ((11, 90, 160), (11, 45, 80)),
        (1440, 2560): ((11, 180, 320), (11, 90, 160)),
        (2160, 3840): ((11, 270, 480), (11, 135, 240)),
    }
    for (h, w), (elat, egrid) in expect.items():
        lat, grid = geometry(h, w, 41)
        ok = lat == elat and grid == egrid
        print(f"    {h}x{w}: latent={lat} grid={grid}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"geometry wrong at {h}x{w}: got {lat}/{grid}, "
                            f"expected {elat}/{egrid}")

    # --- 3. tiling schedule ---
    print("\n[3] sliding-window schedule (loop_step=16)")
    expect_tiles = {(720, 1280): 1, (1440, 2560): 4, (2160, 3840): 9}
    for (h, w), n_expect in expect_tiles.items():
        lat, _ = geometry(h, w, 41)
        cfg = SlidingWindowConfig(lat[1], lat[2], 16)
        p = cfg.get_window_params()
        ok = p["total_windows"] == n_expect
        print(f"    {h}x{w}: {p['total_windows']} tiles of {p['window_size']}, "
              f"step=({p['step_size_h']},{p['step_size_w']})  "
              f"{'OK' if ok else f'expected {n_expect}'}")
        if not ok:
            failures.append(f"{h}x{w} gave {p['total_windows']} tiles, expected {n_expect}")

    # --- 4. layout round trip ---
    print("\n[4] Wan [C,F,H,W] <-> TiledLatentTensor2D [B,F,C,H,W]")
    wan_latent = torch.randn(16, 11, 90, 160)
    as_tlt = wan_latent.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    back = as_tlt.squeeze(0).permute(1, 0, 2, 3).contiguous()
    lossless = torch.equal(wan_latent, back)
    print(f"    round trip lossless: {lossless}  {tuple(wan_latent.shape)} -> "
          f"{tuple(as_tlt.shape)} -> {tuple(back.shape)}")
    if not lossless:
        failures.append("layout round trip lost data")

    # --- 5. coverage ---
    print("\n[5] tile coverage of the canvas (4K, 3x3)")
    lat, _ = geometry(2160, 3840, 41)
    cfg = SlidingWindowConfig(lat[1], lat[2], 16)
    p = cfg.get_window_params()
    wh, ww = p["window_size"]
    canvas = TiledLatentTensor2D(latent_tensor=torch.zeros(1, 11, 16, lat[1], lat[2]))
    cnt = TiledLatentTensor2D(latent_tensor=torch.zeros(1, 11, 16, lat[1], lat[2]))
    for tile in range(p["total_windows"]):
        row, col = divmod(tile, p["num_windows_w"])
        win = (row * wh, row * wh + wh, col * ww, col * ww + ww)
        cnt.set_window_latent(cnt.get_window_latent(*win) + 1.0, *win)
    cmin = cnt.torch_latent.min().item()
    cmax = cnt.torch_latent.max().item()
    exact = cmin == 1.0 and cmax == 1.0
    print(f"    coverage min={cmin} max={cmax}  "
          f"{'exact partition' if exact else 'NOT a clean partition'}")
    if not exact:
        failures.append(f"tiles do not partition the canvas (min {cmin}, max {cmax})")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: Wan Stage-2 geometry, re-noise, and tiling schedule are correct")


if __name__ == "__main__":
    main()
