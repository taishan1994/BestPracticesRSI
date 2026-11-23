# SM120 FA4 stall / stage instrumentation

## What is now wired

`comparisons/v3/benchmarks/worker.py` accepts `FA4_STAGE_NVTX=1`. The profiled
launch is then tagged as:

`profile_v3|fused-stage-map|QK>softmax>PV>correction`

The marker is deliberately a **fused source map**. SM120 FA4 emits one fused
kernel, so a host NVTX range cannot measure QK, softmax, PV, and correction
durations separately. Source anchors were added next to the QK/softmax/P
publish loop, PV loop, and correction/epilogue. A debug-only `clock64()` output
tensor is now threaded through the debug ABI. The first CTA writes kernel
entry, QK start/end, softmax completion, PV completion, correction completion,
and epilogue completion. The production launcher does not pass this tensor.

The debug launcher is `kernels/compiled_attention_debug.py` with
`--profile debug-timing`. Its timing buffer is a flat `int64` tensor and the
worker records the first CTA timestamps in the JSON receipt. QK/softmax/PV
durations are for the first n-block; correction and epilogue are CTA-level
boundaries.

## NCU stall collection

`comparisons/v3/benchmarks/collect_stall_metrics.py` discovers the exact
`smsp__warp_issue_stalled_*` names supported by the installed NCU, writes
`comparisons/v3/configs/ncu-stall-metrics.json`, and uses fail-closed behavior
when performance-counter access is unavailable. The current host has no
`bprsi-fa4-comparison-v3-gpu0` container, so no new stall values were invented.

Run it inside the privileged profiling container, then pass the generated
metric list to the existing `run_profiles.py` protocol. The resulting CSV
should report the stall counters beside duration, SM/tensor active, clock,
L2/DRAM, registers, shared memory, and tile readback.

## PC-sampling attribution

NCU `SourceCounters` was collected with the debug binary and its
`smsp__pcsamp_warps_issue_stalled_*` counters. The source export has no CUDA
line table in this CUTLASS DSL build, so the report uses the seven emitted
`CS2R SR_CLOCKLO` PCs as interval anchors and aggregates SASS samples between
successive anchors. This gives a reproducible stage attribution for the first
CTA, while the interval boundaries are explicitly recorded rather than guessed
from aggregate kernel percentages.

The resulting per-stage table is in
`fa4-sm120-pcsamp-stage-attribution.json`. It contains barrier, long
scoreboard, math, LG/MIO throttle, short scoreboard, and wait counts and their
percent of samples for QK, softmax, PV, correction, and epilogue.

## Current evidence status

The existing pingpong report remains valid for aggregate NCU fields. Its
QK/PV/softmax/correction `busy` and `stall reason` cells must remain `N/A` until
the stall capture and a device-level timing buffer are available.
