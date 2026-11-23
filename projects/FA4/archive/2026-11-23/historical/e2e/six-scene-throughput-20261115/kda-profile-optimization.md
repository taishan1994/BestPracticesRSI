# 2026-11-15 KDA/profile E2E optimization

This is a fresh graph-enabled six-scene candidate measurement against the audited
2026-11-11 FA4 service runs. It uses the same model weights, BF16 runtime,
concurrency 8, one warmup, one formal batch, and no speculative decoding.
The candidate source is `comparisons/v3/e2e/fa4_service_backend.py` and its
profile policy is `comparisons/v3/configs/serving-policy-dev.json`.

## Changes

- Packed prefill profile selection is now model-shape specific and applies only
to `fa4_rsi`: Qwen D256 uses `128x32`, 256 threads, Q-in-registers and GQA
packing; Muse D128 uses `64x64`, 128 threads and GQA packing.
- Qwen decode uses the numerically checked `64x64`, 128-thread, `pack_gqa=true`
recipe. The direct kernel microbenchmark was 0.0769 ms versus 0.1521 ms for
the previous non-packed recipe; the max output difference was 0.
- Muse decode uses `64x160`, 128 threads and GQA packing. The direct kernel
microbenchmark improved from the `64x128` recipe by about 3.3%; the max output
difference was 0.00048828125.
- The PR/reference path is unchanged.

## Fresh E2E results

Throughput is tokens/s. Ratios are candidate divided by the prior audited
`fa4_rsi` run for the same model and scene.

| Model | Scene | Previous | Candidate | Ratio |
|---|---|---:|---:|---:|
| Qwen3.8-27B | short-short | 86.155 | 101.082 | 1.173x |
| Qwen3.8-27B | short-long | 174.570 | 174.493 | 1.000x |
| Qwen3.8-27B | long-short | 61.408 | 61.234 | 0.997x |
| Qwen3.8-27B | long-long | 126.081 | 126.135 | 1.000x |
| Qwen3.8-27B | prefill-short | 6654.446 | 6664.715 | 1.002x |
| Qwen3.8-27B | prefill-long | 6548.670 | 6570.041 | 1.003x |
| Muse-Glimmer-30B | short-short | 72.893 | 90.017 | 1.235x |
| Muse-Glimmer-30B | short-long | 159.092 | 165.682 | 1.041x |
| Muse-Glimmer-30B | long-short | 62.030 | 61.983 | 0.999x |
| Muse-Glimmer-30B | long-long | 132.850 | 133.008 | 1.001x |
| Muse-Glimmer-30B | prefill-short | 6766.961 | 6739.504 | 0.996x |
| Muse-Glimmer-30B | prefill-long | 6459.008 | 6449.451 | 0.999x |

The Qwen candidate artifact is `/data/fa4-rsi-v3/e2e/six-scenes/six-qwen-20261115-prefill-profile-002/summary.csv`, with the decode follow-up in `six-qwen-20261115-decode-kda-001/summary.csv`. The Muse candidate artifact is `/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-decode-kda-001/summary.csv`.

The remaining long-context decode gap is therefore not explained by the
attention tile alone: the optimized kernel is visible in dispatch readback,
but full model throughput is nearly unchanged. The next profile target is the
model-side decode path (GDN/MLP/scheduler overlap and KV gather), not another
unqualified FA4 tile guess.
