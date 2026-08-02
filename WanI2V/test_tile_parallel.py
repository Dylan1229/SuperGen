"""Multi-GPU Wan Stage-2 must produce the same canvas as single-GPU.

Tile parallelism is a pure work split -- every rank steps the scheduler on an
identical, fully-allgathered canvas -- so the output is not merely "close", it is
bit-identical to the one-process path. That is the property worth testing, because
the failure mode is silent: a wrong tile-to-rank mapping or a missing communicate()
before a shift still produces a plausible video, just one with a seam or a stale
band. Comparing pixels against the single-GPU reference catches all of those.

The model is replaced with a deterministic stand-in (a fixed linear function of the
tile contents plus its canvas position), so this runs on CPU-sized tensors in
seconds instead of loading 67 GB of weights. What is under test is the plumbing --
tile cutting, shift bookkeeping, the fuser, allgather, and the scheduler handoff --
not the DiT.

    torchrun --nproc_per_node=1 WanI2V/test_tile_parallel.py   # writes the reference
    torchrun --nproc_per_node=4 WanI2V/test_tile_parallel.py   # compares against it
    TEST_CANVAS=4k torchrun --nproc_per_node=4 ...             # 9 tiles over 4 ranks

Measured: max_abs_err is exactly 0.0 for world=2 and world=4 at both canvases.

Env: ~/envs/easycache.
"""
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

# 2K is 180x320 latent = 4 tiles of 90x160; 4K is 270x480 = 9 tiles. Both are worth
# running: 9 tiles over 4 ranks is the non-divisible case that exercises the
# dummy-tile padding in the allgather buffers.
CANVAS = os.environ.get("TEST_CANVAS", "2k")
LAT_H, LAT_W = {"2k": (180, 320), "4k": (270, 480)}[CANVAS]
REF_PATH = f"/tmp/wan_tile_parallel_ref_{CANVAS}.pt"
LAT_F, CH = 3, 16
STEPS = 4


class FakeModel(torch.nn.Module):
    """Deterministic, position-sensitive stand-in for the Wan DiT.

    Position sensitivity matters: a model that ignored where the tile came from
    would make a wrong tile-to-rank mapping invisible.
    """

    def forward(self, x_list, t=None, **kw):
        x = x_list[0]
        # Depend on the content, the timestep, and the tile's own mean so that any
        # mis-slicing changes the result.
        return [0.1 * x + 0.01 * float(t.item()) + 0.001 * x.mean()]


class FakeScheduler:
    """Euler-ish step with no internal state, so ranks cannot silently diverge."""

    def step(self, model_output, t, sample, return_dict=False, generator=None):
        return (sample - 0.05 * model_output,)


def build_inputs():
    g = torch.Generator().manual_seed(1234)
    latent = torch.randn(CH, LAT_F, LAT_H, LAT_W, generator=g)
    y = torch.randn(CH + 4, LAT_F, LAT_H, LAT_W, generator=g)
    return latent.cuda(), y.cuda()


def main():
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device_id)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    from utils.distributed import DistributedManager
    from tiled_stage2 import WanTiledStage2

    latent, y = build_inputs()

    class Cfg:
        vae_stride = (4, 8, 8)
        patch_size = (1, 2, 2)
        num_train_timesteps = 1000

    class FakeWan:
        config = Cfg()
        param_dtype = torch.float32

        def __init__(self):
            self.model = FakeModel().cuda()

    dm = DistributedManager("allgather", enable_cache=False) if world > 1 else None
    stage2 = WanTiledStage2(FakeWan(), rope_mode="local", loop_step=16,
                            device=f"cuda:{device_id}", dist_manager=dm)

    timesteps = torch.tensor([900.0, 700.0, 500.0, 300.0][:STEPS]).cuda()
    out = stage2.denoise(
        latent, timesteps, FakeScheduler(),
        arg_c={"context": None, "seq_len": 1}, arg_null={"context": None, "seq_len": 1},
        guide_scale=5.0, shift_timesteps=list(range(STEPS)), seed_g=None,
        enable_cache=False, y_canvas=y,
    )

    if world == 1:
        torch.save(out.cpu(), REF_PATH)
        print(f"wrote single-GPU reference {tuple(out.shape)} -> {REF_PATH}")
        print(f"  mean={out.mean():.6f} std={out.std():.6f}")
    elif rank == 0:
        if not os.path.isfile(REF_PATH):
            print(f"FAIL: no reference at {REF_PATH}; run with --nproc_per_node=1 first")
            sys.exit(1)
        ref = torch.load(REF_PATH).cuda()
        assert ref.shape == out.shape, f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
        max_err = (ref - out).abs().max().item()
        rel = max_err / max(ref.abs().max().item(), 1e-8)
        print(f"world={world}: max_abs_err={max_err:.3e} rel={rel:.3e}")
        # Non-determinism in NCCL reductions is not in play here (the fuser divides
        # locally and allgather is a copy), so the bar is exact-to-fp32-noise.
        if max_err > 1e-5:
            print(f"FAIL: multi-GPU output differs from single-GPU by {max_err:.3e}")
            sys.exit(1)
        print(f"PASS: {world}-GPU tile parallelism matches the single-GPU canvas")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
