#!/usr/bin/env bash
# Frame-fidelity (visual-retention) driver for SuperGen.
#
# Computes PSNR / SSIM / LPIPS (optionally FVD) between a BASELINE corpus and a
# TEST corpus of generated videos, by pairing same-named sub-directories (each
# directory contributes one video; if several, the first by sorted name).
#
# This is a thin wrapper around evaluate_pairs.py. The original generated video
# corpora are NOT shipped on this cluster, so you must point BASELINE_DIR /
# TEST_DIR at your own directories of mp4s.
#
# ---------------------------------------------------------------------------
# Configuration (override via environment variables):
#   PY            python interpreter (default: the supergen-metrics venv)
#   OUT_ROOT      where result JSONs are written
#                 (default: <evaluation>/exp_result/VisualRetention)
#   BASELINE_DIR  directory tree of reference / full-quality videos
#   TEST_DIR      directory tree of videos to score against the baseline
#   OUTPUT_JSON   output JSON path (default: ${OUT_ROOT}/visual_retention.json)
#   WITH_FVD      set to 1 to additionally compute FVD (styleganv)  (default: 0)
#   ONLY_FINAL    set to 1 to use only_final=True for the metrics    (default: 0)
#   DEVICE        cuda / cpu                                          (default: cuda)
# ---------------------------------------------------------------------------
set -euo pipefail

# Directory containing this script (= common_metrics_on_video_quality/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default python = the metrics venv created for SuperGen evaluation.
PY="${PY:-/mnt/data/fanjiang/venvs/supergen-metrics/bin/python}"

# Default output root lives under evaluation/exp_result/ (sibling of this dir).
OUT_ROOT="${OUT_ROOT:-/mnt/data/fanjiang/repo/SuperGen/evaluation/exp_result/VisualRetention}"
mkdir -p "${OUT_ROOT}"

WITH_FVD="${WITH_FVD:-0}"
ONLY_FINAL="${ONLY_FINAL:-0}"
DEVICE="${DEVICE:-cuda}"

extra_flags=()
[ "${WITH_FVD}" = "1" ]   && extra_flags+=(--with_fvd)
[ "${ONLY_FINAL}" = "1" ] && extra_flags+=(--only_final)
[ -n "${DEVICE}" ]        && extra_flags+=(--device "${DEVICE}")

# ---------------------------------------------------------------------------
# Single pair mode (the common case): supply BASELINE_DIR + TEST_DIR.
# ---------------------------------------------------------------------------
BASELINE_DIR="${BASELINE_DIR:-}"
TEST_DIR="${TEST_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUT_ROOT}/visual_retention.json}"

if [ -n "${BASELINE_DIR}" ] && [ -n "${TEST_DIR}" ]; then
  echo "[INFO] python   = ${PY}"
  echo "[INFO] baseline = ${BASELINE_DIR}"
  echo "[INFO] test     = ${TEST_DIR}"
  echo "[INFO] output   = ${OUTPUT_JSON}"
  "${PY}" evaluate_pairs.py \
    --baseline_dir "${BASELINE_DIR}" \
    --test_dir     "${TEST_DIR}" \
    --output       "${OUTPUT_JSON}" \
    "${extra_flags[@]}"
  echo "[OK] Done. JSON in ${OUTPUT_JSON}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Batch mode: uncomment / edit the example pairs below, or (preferred) just
# call this script repeatedly with BASELINE_DIR / TEST_DIR set per pair.
#
# Example (replace the paths with your own corpora):
#
#   "${PY}" evaluate_pairs.py \
#     --baseline_dir "${OUT_ROOT}/../corpora/cogvideo/original-720p" \
#     --test_dir     "${OUT_ROOT}/../corpora/cogvideo/cache-0.08-720p" \
#     --output       "${OUT_ROOT}/cogvideo_cache_0.08.json" \
#     "${extra_flags[@]}"
#
#   "${PY}" evaluate_pairs.py \
#     --baseline_dir "${OUT_ROOT}/../corpora/hunyuan/baseline/2k" \
#     --test_dir     "${OUT_ROOT}/../corpora/hunyuan/cache/2k" \
#     --output       "${OUT_ROOT}/hunyuan_2k.json" \
#     "${extra_flags[@]}"
# ---------------------------------------------------------------------------

echo "[ERROR] No BASELINE_DIR / TEST_DIR provided and no batch pairs are enabled." >&2
echo "        Set BASELINE_DIR and TEST_DIR, e.g.:" >&2
echo "          BASELINE_DIR=/path/to/baseline TEST_DIR=/path/to/test \\" >&2
echo "          OUTPUT_JSON=${OUT_ROOT}/my_run.json bash run_all_visual_retention.sh" >&2
echo "        Or uncomment a batch pair inside this script." >&2
exit 1
