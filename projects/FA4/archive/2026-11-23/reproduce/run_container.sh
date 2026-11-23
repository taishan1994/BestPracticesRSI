#!/bin/bash
# Usage: run_container.sh GPU_UUID SOURCE_REV OUTPUT_DIR -- command [args...]
set -eu
BASE=$(cd -- "$(dirname -- "$0")/.." && pwd)
GPU_UUID="$1"; REV="$2"; OUTPUT_DIR="$3"; shift 3
if [ "${1:-}" = -- ]; then shift; fi
case "$REV" in pr3109|pr3132) ;; *) echo 'revision must be pr3109 or pr3132' >&2; exit 2;; esac
if [ ! -d "$BASE/_sources/$REV/flash_attn" ]; then echo 'Run fetch_sources.py for this revision first.' >&2; exit 2; fi
OUTPUT_DIR=$(realpath -m -- "$OUTPUT_DIR")
case "$OUTPUT_DIR" in "$BASE"/_work/*) ;; "$BASE"|"$BASE"/*) echo "Use a new _work directory or a path outside the archive." >&2; exit 2;; esac
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
case "$OUTPUT_DIR" in "$BASE"|"$BASE/reports"|"$BASE/reports/"*) echo 'Use a new output directory, never archived evidence.' >&2; exit 2;; esac
IMAGE=${FA4_IMAGE:-local/fa4-sm120:1.0.0}
docker run --rm --privileged --network none --gpus "device=$GPU_UUID" --ipc private --shm-size 2g \
 -e CUDA_VISIBLE_DEVICES="$GPU_UUID" -e FA4_SOURCE_ROOT="/archive/_sources/$REV/flash_attn" \
 -v "$BASE:/archive:ro" -v "$OUTPUT_DIR:/out:rw" \
 -v /usr/local/cuda-13.1/nsight-compute-2025.4.1:/opt/ncu:ro "$IMAGE" "$@"
