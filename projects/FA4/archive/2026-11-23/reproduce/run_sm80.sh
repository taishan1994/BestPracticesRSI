#!/bin/bash
# Usage: run_sm80.sh timing|profile|analyze GPU_UUID NEW_OUTPUT_DIR
set -eu
MODE="$1"; GPU_UUID="$2"; OUTPUT_DIR="$3"
BASE=$(cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=$(realpath -m -- "$OUTPUT_DIR")
case "$OUTPUT_DIR" in "$BASE"/_work/*) ;; "$BASE"|"$BASE"/*) echo "Use a new _work directory or a path outside the archive." >&2; exit 2;; esac
mkdir -p -- "$OUTPUT_DIR/analysis" "$OUTPUT_DIR/reports"
cp "$BASE/cases/full285.json" "$OUTPUT_DIR/cases.json"
cp "$BASE/provenance/pr3109-source-manifest.json" "$OUTPUT_DIR/source-manifest.json"
run() { bash "$BASE/reproduce/run_container.sh" "$GPU_UUID" pr3109 "$OUTPUT_DIR" -- env FA4_GREEN_LIBRARY="/out/green-$MODE.so" "$@"; }
if [ "$MODE" = analyze ]; then
 run env FA4_RESULTS_ROOT=/out python /archive/reproduce/sm80/analyze.py
 exit
fi
case "$MODE" in timing|profile) ;; *) echo 'mode must be timing, profile or analyze' >&2; exit 2;; esac
run nvcc -arch=sm_120 -shared -Xcompiler=-fPIC --cudart=shared /archive/reproduce/sm80/green.cu -lcuda -o "/out/green-$MODE.so"
for PHASE in forward decode; do
 if [ "$MODE" = timing ]; then
  run python /archive/reproduce/sm80/experiment.py --kind "$PHASE" --tag "$PHASE-" --resume
 else
  run /opt/ncu/ncu --profile-from-start off --nvtx --section LaunchStats --section Occupancy --section InstructionStats --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum --force-overwrite -o "/out/reports/$PHASE" python /archive/reproduce/sm80/experiment.py --profile --kind "$PHASE" --tag "$PHASE-"
 fi
done
if [ "$MODE" = timing ]; then run python /archive/reproduce/sm80/graph_proof.py; fi
