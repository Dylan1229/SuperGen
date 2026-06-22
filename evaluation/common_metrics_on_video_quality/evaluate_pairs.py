#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
import numbers
import numpy as np
import torch

# 这些函数来自仓库
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

# ================== 工具函数 ==================
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

def list_leaf_dirs(root):
    """返回 root 下最底层子目录（里面有视频文件）的路径列表；若根目录直接放视频，也包含根目录本身。"""
    root = Path(root)
    leaf_dirs = []
    for p in root.rglob("*"):
        if p.is_dir():
            if any((child.is_file() and child.suffix.lower() in VIDEO_EXTS) for child in p.iterdir()):
                leaf_dirs.append(p)
    if any((child.is_file() and child.suffix.lower() in VIDEO_EXTS) for child in root.iterdir()):
        leaf_dirs.append(root)
    leaf_dirs = list(sorted(set(leaf_dirs)))
    return leaf_dirs

def pick_one_video(dirpath: Path):
    """从目录里选一个视频（若有多个，取按名字排序的第一个）。"""
    vids = sorted([p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    return vids[0] if vids else None

def load_video_to_tensor(path: Path):
    """
    读取视频为 torch 张量，形状 [T, C, H, W] 且值域 [0,1]。
    优先使用 decord，否则回退到 OpenCV。
    """
    try:
        import decord
        from decord import VideoReader, cpu
        vr = VideoReader(str(path), ctx=cpu(0))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [T,H,W,C], uint8
        arr = frames.astype(np.float32) / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))  # [T,C,H,W]
        return torch.from_numpy(arr)
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        ok = True
        while ok:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"Failed to read any frame from {path}")
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
        arr = np.transpose(arr, (0, 3, 1, 2))  # [T,C,H,W]
        return torch.from_numpy(arr)

def align_videos(t1: torch.Tensor, t2: torch.Tensor):
    """
    帧数按最短对齐；分辨率按最小 H/W 统一（必要时插值）。
    返回 [T,C,H,W], [T,C,H,W]
    """
    import torch.nn.functional as F

    # 对齐帧数
    T = min(t1.shape[0], t2.shape[0])
    t1, t2 = t1[:T], t2[:T]

    # 对齐尺寸
    H1, W1 = t1.shape[-2:]
    H2, W2 = t2.shape[-2:]
    H, W = min(H1, H2), min(W1, W2)

    if (H1, W1) != (H, W):
        t1 = F.interpolate(t1, size=(H, W), mode="bilinear", align_corners=False)
    if (H2, W2) != (H, W):
        t2 = F.interpolate(t2, size=(H, W), mode="bilinear", align_corners=False)

    return t1, t2

def to_batch(v: torch.Tensor):
    """[T,C,H,W] -> [1,T,C,H,W]"""
    return v.unsqueeze(0).contiguous()

def to_scalar(name, x):
    """
    将各种可能的返回类型规整成 float：
    - 标量(int/float)
    - torch 张量（0/1+ 维）：item() 或均值
    - numpy 标量/数组：float(...) 或均值
    - list/tuple：均值
    - dict：优先取 'average'/'mean'/'avg'/'final'/'value'/'score'，否则对可数值项取均值
    """
    # 纯标量
    if isinstance(x, numbers.Number):
        return float(x)

    # torch
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return float(x.item())
        return float(x.float().mean().item())

    # numpy
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return float(x)
        return float(np.asarray(x, dtype=np.float64).mean())

    # numpy 标量
    if "numpy" in type(x).__module__ and hasattr(x, "dtype"):
        try:
            return float(x)
        except Exception:
            pass

    # list/tuple
    if isinstance(x, (list, tuple)):
        vals = []
        for v in x:
            try:
                vals.append(to_scalar(name, v))
            except Exception:
                continue
        if not vals:
            raise TypeError(f"{name} returned empty/non-numeric list.")
        return float(np.mean(vals))

    # dict
    if isinstance(x, dict):
        for key in ["average", "mean", "avg", "final", "value", "score"]:
            if key in x and x[key] is not None:
                return to_scalar(f"{name}.{key}", x[key])
        cand = []
        for v in x.values():
            try:
                cand.append(to_scalar(name, v))
            except Exception:
                continue
        if cand:
            return float(np.mean(cand))

    raise TypeError(f"{name} returns unsupported type: {type(x)}")

# ================== 主流程 ==================
def evaluate_pair(baseline_dir, test_dir, output_json, compute_fvd=False, only_final=False, device_str=None):
    """
    对 baseline_dir 与 test_dir 中同名子目录（或同层级视频）进行比对。
    每个子目录取一个视频（若多个取排序第一个）。
    写出逐视频 psnr/ssim/lpips（可选 fvd），并在 summary.average 附上均值。
    """
    baseline_dir = Path(baseline_dir)
    test_dir = Path(test_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str) if device_str else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    results = []

    base_leafs = list_leaf_dirs(baseline_dir)
    if not base_leafs:
        base_leafs = [baseline_dir]

    matched = 0
    for bdir in base_leafs:
        # 用相对路径匹配
        rel = bdir.relative_to(baseline_dir)
        tdir = test_dir / rel
        if not tdir.exists():
            tdir = test_dir / bdir.name
            if not tdir.exists():
                continue

        bvid = pick_one_video(bdir)
        tvid = pick_one_video(tdir)
        if (bvid is None) or (tvid is None):
            continue

        try:
            v1 = load_video_to_tensor(bvid)  # [T,C,H,W] in [0,1], CPU
            v2 = load_video_to_tensor(tvid)
            v1, v2 = align_videos(v1, v2)

            # 保持在 CPU（与仓库示例 run_metric.py 一致）
            V1 = to_batch(v1)  # CPU tensor
            V2 = to_batch(v2)  # CPU tensor

            # 计算（PSNR/SSIM 在 CPU；LPIPS/FVD 传 device 参数）
            psnr_ret  = calculate_psnr(V1, V2, only_final=only_final)
            ssim_ret  = calculate_ssim(V1, V2, only_final=only_final)
            lpips_ret = calculate_lpips(V1, V2, device, only_final=only_final)

            psnr_val  = to_scalar("psnr",  psnr_ret)
            ssim_val  = to_scalar("ssim",  ssim_ret)
            lpips_val = to_scalar("lpips", lpips_ret)

            item = {
                "name": str(rel) if rel != Path(".") else bvid.name,
                "baseline_video": str(bvid),
                "test_video": str(tvid),
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
            }

            if compute_fvd:
                fvd_ret = calculate_fvd(V1, V2, device, method='styleganv', only_final=only_final)
                item["fvd"] = to_scalar("fvd", fvd_ret)

            results.append(item)
            matched += 1
            print(f"[OK] {item['name']} -> PSNR {psnr_val:.3f}, SSIM {ssim_val:.4f}, LPIPS {lpips_val:.4f}")

        except Exception as e:
            print(f"[ERR] {bvid} vs {tvid}: {e}")

    # 汇总平均
    def mean_safe(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return float(np.mean(xs)) if xs else float("nan")

    avg = {
        "psnr": mean_safe([r["psnr"] for r in results]),
        "ssim": mean_safe([r["ssim"] for r in results]),
        "lpips": mean_safe([r["lpips"] for r in results]),
    }
    if compute_fvd and results and "fvd" in results[0]:
        avg["fvd"] = mean_safe([r["fvd"] for r in results])

    out = {
        "summary": {
            "baseline_dir": str(baseline_dir),
            "test_dir": str(test_dir),
            "num_pairs": matched,
            "only_final": only_final,
            "device": str(device),
            "average": avg,
        },
        "items": results,
    }

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] {matched} pairs evaluated. JSON -> {output_json}")
    print(f"[AVERAGE] PSNR {avg['psnr']:.3f}, SSIM {avg['ssim']:.4f}, LPIPS {avg['lpips']:.4f}")
    if compute_fvd and "fvd" in avg:
        print(f"[AVERAGE] FVD {avg['fvd']:.3f}")

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate visual retention between baseline_dir and test_dir.")
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--only_final", action="store_true", help="Use only_final=True when calling metrics")
    ap.add_argument("--with_fvd", action="store_true", help="Additionally compute FVD")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_pair(
        baseline_dir=args.baseline_dir,
        test_dir=args.test_dir,
        output_json=args.output,
        compute_fvd=args.with_fvd,
        only_final=args.only_final,
        device_str=args.device,
    )