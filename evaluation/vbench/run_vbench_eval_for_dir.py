import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch

# ---------------------------------------------------------------------------
# No-reference VBench scoring driver for SuperGen.
#
# Scores one or more directories of mp4 videos with VBench's no-reference
# quality dimensions (subject/background consistency, motion smoothness,
# aesthetic quality, imaging quality) and writes per-video + per-dir JSONs.
#
# The directories to score are supplied AT RUN TIME (not hardcoded):
#   * positional CLI args:   python run_vbench_eval_for_dir.py DIR1 DIR2 ...
#   * or env var VBENCH_DIRS (os.pathsep / ":" separated list of directories).
#
# Environment variables:
#   VBENCH_REPO        path to the VBench clone (with backbones available).
#                      Default: the cluster clone under
#                      /mnt/data/fanjiang/repo/4k-video-generation-wan2.1-i2v/VBench
#   VBENCH_CACHE_DIR   where VBench downloads / loads its scoring backbones
#                      (clip / raft / aesthetic / amt / musiq). VBench itself
#                      defaults this to ~/.cache/vbench; here we default it onto
#                      the big disk so the first run downloads weights there.
#   VBENCH_DIRS        directories to score (alternative to positional args).
# ---------------------------------------------------------------------------

# Path to the VBench source checkout (overridable via env).
VBENCH_REPO = os.environ.get(
    "VBENCH_REPO",
    "/mnt/data/fanjiang/repo/4k-video-generation-wan2.1-i2v/VBench",
)
VBENCH_FULL_INFO_JSON = os.path.join(VBENCH_REPO, "vbench", "VBench_full_info.json")

# Where VBench caches/downloads its scoring backbones. VBench reads this env var
# (vbench/utils.py: CACHE_DIR = os.environ.get('VBENCH_CACHE_DIR')). Default it
# onto the big disk so backbones land there on first download.
os.environ.setdefault(
    "VBENCH_CACHE_DIR",
    "/mnt/data/fanjiang/.cache/vbench",
)

if VBENCH_REPO not in sys.path:
    sys.path.insert(0, VBENCH_REPO)
from vbench import VBench  # noqa: E402

# Optional hardcoded fallback list of directories to score. Left empty on
# purpose: dirs are normally supplied via CLI args or $VBENCH_DIRS (see main()).
DIRS: List[str] = []

# Metrics to compute
METRICS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


def build_prefix_from_exp_dir(exp_dir: str) -> str:
    # Expect exp_dir contains path segment 'exp_result'; use the components after it
    p = Path(exp_dir).resolve()
    parts = list(p.parts)
    if "exp_result" in parts:
        idx = parts.index("exp_result")
        suffix_parts = parts[idx + 1 :]
    else:
        # Fallback: use last 3 components
        suffix_parts = parts[-3:]
    prefix = "_".join(x.replace(" ", "_") for x in suffix_parts)
    return prefix


def list_mp4_files(root_dir: str) -> List[Path]:
    return sorted([p for p in Path(root_dir).rglob("*.mp4")])


def eval_single_video(mp4_path: Path, my_vbench: VBench) -> Dict[str, float]:
    out_dir = str(mp4_path.parent)
    # Unique name for this run
    name = f"results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    my_vbench.output_path = out_dir  # ensure outputs go beside the video
    my_vbench.evaluate(
        videos_path=str(mp4_path),
        name=name,
        dimension_list=METRICS,
        mode="custom_input",
    )
    # Find freshly created eval json
    eval_json = Path(out_dir) / f"{name}_eval_results.json"
    if not eval_json.exists():
        # fallback to newest *_eval_results.json
        candidates = sorted(Path(out_dir).glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"No eval result JSON found for {mp4_path}")
        eval_json = candidates[-1]
    with open(eval_json, "r") as f:
        data = json.load(f)
    values: Dict[str, float] = {}
    for m in METRICS:
        try:
            values[m] = data[m][1][0]["video_results"]
        except Exception:
            values[m] = None
    return values


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def evaluate_dir(exp_dir: str) -> None:
    exp_dir = str(Path(exp_dir).resolve())
    prefix = build_prefix_from_exp_dir(exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_vbench = VBench(device, VBENCH_FULL_INFO_JSON, exp_dir)

    mp4_files = list_mp4_files(exp_dir)
    if not mp4_files:
        print(f"No mp4 files found under {exp_dir}")
        return

    per_video_results: List[Dict[str, float]] = []

    for idx, mp4 in enumerate(mp4_files, 1):
        try:
            print(f"[{idx}/{len(mp4_files)}] Evaluating {mp4}")
            vals = eval_single_video(mp4, my_vbench)
            per_video_results.append(vals)
            # Save per-video result json in the video folder with specified naming
            video_name = mp4.stem.replace(" ", "_")
            out_name = f"{prefix}_{video_name}_result.json"
            out_path = mp4.parent / out_name
            write_json(out_path, vals)
            print(f"  -> Saved {out_path}")
        except Exception as e:
            print(f"  [ERROR] {mp4}: {e}")

    # Compute averages across all processed videos
    averages: Dict[str, float] = {}
    for m in METRICS:
        valid = [r[m] for r in per_video_results if r.get(m) is not None]
        averages[m] = (sum(valid) / len(valid)) if valid else None

    # Save overall result JSON at the exp_dir root with specified naming
    overall_path = Path(exp_dir) / f"{prefix}_result.json"
    write_json(overall_path, averages)
    print(f"Saved overall averages to {overall_path}")


def resolve_dirs(cli_dirs: List[str]) -> List[str]:
    """Directories to score come from CLI args, else $VBENCH_DIRS, else module DIRS."""
    if cli_dirs:
        return list(cli_dirs)
    env = os.environ.get("VBENCH_DIRS", "").strip()
    if env:
        return [d for d in env.split(os.pathsep) if d]
    return list(DIRS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score directories of mp4 videos with VBench no-reference metrics.",
    )
    ap.add_argument(
        "dirs",
        nargs="*",
        help="One or more directories of mp4s to score. "
             "If omitted, falls back to $VBENCH_DIRS (os.pathsep-separated).",
    )
    args = ap.parse_args()

    dirs = resolve_dirs(args.dirs)
    if not dirs:
        print("No directories to evaluate. Pass them as CLI args, e.g.:")
        print(f"  python {os.path.basename(__file__)} /path/to/mp4_dir [more_dirs ...]")
        print("or set VBENCH_DIRS=/dir1:/dir2 in the environment.")
        sys.exit(1)

    if not os.path.exists(VBENCH_FULL_INFO_JSON):
        print(f"[ERROR] VBench info JSON not found: {VBENCH_FULL_INFO_JSON}")
        print(f"        Check VBENCH_REPO (currently: {VBENCH_REPO}).")
        sys.exit(1)

    for idx, d in enumerate(dirs, 1):
        print(f"\n=== [{idx}/{len(dirs)}] Evaluating directory: {d} ===")
        if not Path(d).is_dir():
            print(f"  [WARN] Skip missing directory: {d}")
            continue
        evaluate_dir(d)
        print(f"=== Done: {d} ===")


if __name__ == "__main__":
    main()