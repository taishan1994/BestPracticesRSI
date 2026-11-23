# Service-level profile findings (2026-11-15)

The trace was captured with the optimized FA4 service on Muse-Glimmer-30B,
BF16, CUDA Graph enabled, and a real 8192→1024 decode cohort at concurrency 8.
The trace contains five steady `step[DECODE bs=8]` windows.

Artifact:

`/data/fa4-rsi-v3/e2e/profile-muse-decode-20261115-b8-001/profile/`

The five steady windows contain 248.628 ms of GPU kernel time:

| Kernel family | Time | Share |
|---|---:|---:|
| small-M CUTLASS BF16 GEMM (`32x32_64x2` and `32x32_128x1`) | 183.539 ms | 73.8% |
| FA4 SM120 attention | 37.905 ms | 15.2% |
| paged KV gather (`_gather_packed`) | 9.946 ms | 4.0% |
| RMSNorm, activation, RoPE, KV store and metadata kernels | 17.238 ms | 6.9% |

The earlier batch-1 trace showed cuBLAS GEMV because the profiler caught the
first decode request during admission. The steady batch-8 trace replaces that
hypothesis: the dominant path is small-M dense projection/MLP GEMM. Attention
tile changes cannot remove this 73.8% share. The next candidate should target
the BF16 GEMM dispatch/tile policy or deliberately increase admitted decode
batch size, with numerical and latency gates preserved.

## Batch-size experiment

An exploratory C16 8192→1024 run was measured with both attention arms:

| Arm | Output tok/s | Mean TPOT |
|---|---:|---:|
| optimized FA4 | 128.624 | 65.31 ms |
| PR reference FA4 | 114.787 | 72.46 ms |

The optimized/reference ratio at C16 is **1.121×**, but absolute throughput is
below the C8 result because extra queued prefill work increases TTFT and the
service remains latency-bound. C16 is diagnostic evidence for GEMM batching,
not a replacement for the fixed-C8 six-scene acceptance.

## TGV probe boundary

The existing SGLang TGV BF16 GEMM candidate was probed on the actual SM120
runtime for `(M=8,N=6656,K=4096)` and `(M=8,N=6656,K=19968)`. Both candidates
fail at DSL compilation before execution because `tcgen05.mma` accepts SM100/
SM103/SM110/SM107 architectures but not SM120. The compiler explicitly points
to the warp-level `cute.nvgpu.warp.MmaF16BF16Op` path for consumer Blackwell.
Thus the TGV implementation was not enabled or used in the service; the
production policy remains unchanged by this failed probe.

## SM120 warp-level prototype

An isolated Triton BF16 prototype was benchmarked in
`optimization/muse_smallm_gemm_probe.py` for the same projections. The best
observed configurations were still slower than cuBLAS: for `(N=6656,K=4096)`
the best prototype was about 0.071 ms versus 0.037 ms, and for
`(N=6656,K=19968)` about 0.223 ms versus 0.198 ms. The prototype also showed
large raw BF16 output differences (0.5–2.0 absolute on random stress inputs),
so it remains a probe only and is not wired into serving.

## Torch-compile probe

The SGLang `--enable-torch-compile --torch-compile-max-bs 16` probe was started
under an isolated tag. It did reach Triton matmul autotuning, but startup never
completed: the compile path began autotuning hundreds of shape-specific GEMM
variants and was interrupted before the model became healthy. No throughput or
correctness result was recorded, and the normal FA4 service path was left
unchanged.

## Torch-compile no-autotune six-scene result

With `--enable-torch-compile --torch-compile-max-bs 16` and
`TORCHINDUCTOR_MAX_AUTOTUNE=0`, the full Muse six-scene run completed. Relative
to the optimized non-compile C8 run:

| Scene | Throughput ratio |
|---|---:|
| short-short | 0.979x |
| short-long | 1.052x |
| long-short | 1.014x |
| long-long | 1.030x |
| prefill-short | 1.006x |
| prefill-long | 1.004x |

The six-scene geometric mean is **1.014x**. This is a real full-model gain on
long decode, but short-short regresses 2.1% and startup spends several minutes
compiling graph shapes. It remains a workload-specialized candidate rather than
the universal default. Raw result:
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-torchcompile-noauto-six-001/summary.csv`.

A Qwen3.8-27B torch-compile no-autotune launch was also attempted. It was
killed during decode CUDA-graph capture (OOM/assertion in the compile-backed
runner) before serving became healthy, so no Qwen compile throughput is
claimed. The Qwen production path remains the validated non-compile profile.

A second Qwen torch-compile probe reduced both compile and decode graph limits
to batch 8, but it still exited 137 during CUDA-graph capture before serving
became healthy. Qwen therefore remains on the non-compile validated path.

A practical Muse long-decode profile was then tested with only the C8 graph
captured (`--cuda-graph-bs-decode 8 --cuda-graph-max-bs-decode 8`, compile
max-batch 8, autotune disabled). Muse 8192→1024 reached **135.782 tok/s**
(TPOT 53.12 ms), about **2.1%** above the non-compile optimized path, while
startup was materially shorter than the six-batch compile profile. This is the
preferred long-decode candidate; it still should remain separate from the
short-short profile because compile adds TTFT and short output regresses.

The serving adapter also required an ABI correction before repeating the full
candidate. The deployed runtime has shape-dependent optional scheduler
arguments before the TVM-FFI stream; the adapter now inspects each kernel's
`__call__` signature and supplies exactly the missing compile-time placeholders.
Without this, decode could pass while the CUDA-graph prefill shape bound the
stream as `mCuTotalSplitsMBlocks` and terminated the service.

After that fix, the practical C8 compile profile completed all six Muse scenes
under the same C8 protocol. Relative to the ordinary optimized Muse FA4 run,
the ratios were:

| Scene | Throughput ratio |
|---|---:|
| short-short | 0.996x |
| short-long | 1.037x |
| long-short | 1.015x |
| long-long | 1.032x |
| prefill-short | 1.012x |
| prefill-long | 1.009x |

The six-scene geometric mean is **1.0167x**. Raw result:
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-torchcompile-bs8-six-007/summary.csv`.
The candidate remains workload-specialized: it improves every long or prefill
scene in this repeat, but does not improve short-short and adds compilation
startup cost.

The reproducible profile split is recorded in
`configs/serving-profiles.json`: Qwen stays on the validated non-compile
profile, Muse short-short stays on ordinary FA4, and the other five Muse
scenes use the C8 compile profile. This is a workload routing policy, not a
claim that one universal service configuration wins every scene.

## GEMM backend boundary

The steady batch-8 trace identifies the two dominant projection kernels as
cuBLASLt split-K reduction and CUTLASS WMMA (`32x32_64x2` and `32x32_128x1`).
The deployed SGLang exposes `--bf16-gemm-backend cutedsl`, but its startup gate
correctly rejects this GPU: that backend is restricted to SM100/SM103, while the
test device is SM120. The attempted Muse smoke run exited during initialization
with that explicit architecture error; it was not counted as a performance
result.

The existing SM120 QuACK warp-GEMM source could not be loaded in the service
image because it requires `cutlass.base_dsl.enums`, absent from the pinned
runtime DSL. It therefore remains an isolated source candidate. An explicit
`--bf16-gemm-backend torch` Muse short-short smoke did complete at **89.908
tok/s**, statistically indistinguishable from the ordinary auto/cuBLAS result
(90.017 tok/s); switching the exposed backend does not improve the path.

I also queried cuBLASLt directly with the same BF16 row-major mapping for the
actual Muse projection dimensions (`M=8/16/32`, with `(N,K)` equal to
`(6656,6656)`, `(19968,6656)`, and `(6656,19968)`). The heuristic returned one
valid algorithm for every shape, with zero max-absolute error against
`torch.mm`; timings matched PyTorch within measurement noise (typically below
0.1%). The direct API therefore does not expose a faster hidden heuristic for
these real shapes. The probe is retained at
`optimization/cublaslt_bf16_probe.py`.

The pinned image does contain NVIDIA's SM120 `blackwell_geforce/dense_gemm.py`
example, so I applied only compatibility shims for the older DSL API
(`ProxyKind`/`SharedSpace`) and ran its BF16 kernel on the real Muse shape
`M=32,N=6656,K=6656`. Its best tested `64x64x64` tile measured **29.49 us**
versus **26.0 us** for PyTorch/cuBLASLt on the same buffers; the `64x128x64`
variant was **52.22 us**. It is therefore a valid SM120 warp-level reference,
but not a serving candidate at these small-M shapes. No kernel was integrated.
At `M=64` the same `64x64x64` kernel also passed reference checking and
measured **29.80 us**, suggesting a possible future batch-padding experiment;
the accepted C8 service profile currently produces M≈32 projection work, so
this was not substituted into E2E without measuring padding, memory traffic,
and end-to-end latency together.

## Scheduler batch-formation probe

After removing an unsupported `num_threads` keyword from the pinned
`_flash_attn_fwd` ABI, two ordinary Muse long-long E2E probes completed with
the scheduler's `--num-continuous-decode-steps` set to 2 and 4. Relative to
the ordinary C8 run, throughput was **0.9988x** (2 steps) and **0.9991x** (4
steps), with all requests eligible. The logs kept eight requests in the
decode batch, so increasing the number of continuous steps did not increase
the GEMM M dimension and did not improve throughput. The default value 1
remains preferable because it avoids adding TTFT pressure without a measured
gain.

An isolated Muse decode tile candidate (`tile_n=128`, otherwise the same
profile) completed the long-long cohort at **132.800 tok/s**, or **0.9984x**
relative to the ordinary `tile_n=160` run. It was not promoted. The candidate
policy is retained as `configs/serving-policy-muse-n128.json` for audit only.

The adjacent `tile_n=192` candidate was rejected by the FA4 runtime resource
gate in practice: its first decode launch raised
`CUDA_LAUNCH_INVALID_CONFIG` before kernel execution. It remains an explicit
failed candidate and was not counted as throughput.

The later current-runtime paired check corrected the comparison baseline. After
restoring the required paged-KV gather → packed-varlen path and removing the
unsupported `num_threads` argument from both FA4 branches, Muse
`tile_m=64,tile_n=160` measured **132.566 tok/s** on short-short, while the
`tile_m=32` candidate measured **131.825 tok/s** in the matching six-scene
run. Thus `tile_m=32` is **0.9944x** on the current ABI and is not promoted;
the earlier apparent gain came from comparing against a pre-fix baseline.

The corrected current-runtime `tile_m=64,tile_n=160` six-scene acceptance is
recorded at
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-current-m64-six-009/summary.csv`.
Against that paired run, the `tile_m=32` candidate ratios were
`0.9942x, 0.9998x, 0.9999x, 1.0004x, 1.0013x, 1.0004x` for the six scenes,
with a geometric mean of **0.9993x**. The default `tile_m=64` policy remains
the accepted configuration.

Using the corrected current m64 six-scene receipt and the archived Muse PR
reference rows, the directly comparable ratios are **1.0376x, 1.0568x,
1.0682x, 1.1502x, 0.9944x, and 1.0031x** for short-short, short-long,
long-short, long-long, prefill-short, and prefill-long respectively. Their
geometric mean is **1.0505x**. This is the current PR-reference acceleration
claim; it excludes the earlier pre-fix service runs.

An isolated Muse prefill candidate (`muse-full`/`muse-swa` changed from
`64x64` to `128x32`) improved the two prefill-only probes by roughly 1%, but
the required full six-scene acceptance rejected it: ratios versus the
corrected m64 baseline were `0.9656x, 0.9460x, 0.9369x, 0.8694x, 1.0047x,
1.0018x`, geometric mean **0.9530x**. The candidate remains at
`configs/serving-policy-muse-prefill-128x32.json` for audit and is not promoted.

A narrower `64x32` prefill candidate showed stronger isolated prefill numbers
(`+3.1%` and `+1.9%`), but the full six-scene receipt again rejected it:
ratios versus the corrected default were `0.9657x, 0.9460x, 0.9346x, 0.8687x,
1.0002x, 0.9956x`, geometric mean **0.9507x**. Neither prefill-only tile
candidate is safe to promote because full service graph/cache behavior
dominates the isolated prefill measurement.

The latest complete Qwen opt2 six-scene receipt is
`/data/fa4-rsi-v3/e2e/six-scenes/policy96-eat-qwen-20261115-opt2/summary.csv`.
Against the archived Qwen PR reference, its scene ratios are **1.0410x,
1.0564x, 1.0623x, 1.1132x, 1.0199x, and 1.0205x**, with geometric mean
**1.0517x**. Qwen remains on the non-compile path because the torch-compile
probes failed during CUDA-graph capture.

The KDA `pack_gqa=false` Muse candidate was rerun on the standard `fa4_rsi`
path and reached **115.690 tok/s** on long-long (**0.8696x** versus the
accepted pack-gqa configuration). It is rejected; `pack_gqa=true` remains the
KDA default.

The isolated Qwen D256 KDA `tile_n=32` candidate was rejected before any
throughput result: the first decode launch raised `CUDA_LAUNCH_INVALID_CONFIG`.
Qwen keeps the validated `tile_n=64` recipe.

An isolated Muse decode split probe with `FA4_DECODE_SPLITS=1` reached
**115.808 tok/s** on long-long, only **0.8705x** of the corrected default.
The automatic `num_splits=0` policy remains in force.

For completeness, forcing `FA4_DECODE_SPLITS=2` is rejected by the pinned
SM120 config validator (`SM120 forward only supports num_splits=1`) before a
throughput measurement. No fixed split override is promoted.

The C8 Muse compile profile was rechecked after the paged-KV/ABI correction:
Muse long-long reached only **118.756 tok/s**, or **0.8927x** versus the
corrected m64 default. The earlier compile gains came from the pre-fix service
path and are invalid for the current runtime; compile remains disabled in the
accepted Muse profile.

The `_gather_packed` block-size probe (`2048` instead of the accepted `1024`)
reached **115.767 tok/s** on long-long (**0.8702x** versus the corrected
default). It is rejected and the service code is restored to the fixed 1024
block.

The smaller gather block candidate (`512`) was also rejected: it reached
**115.677 tok/s**, or **0.8695x** of the corrected default. The gather launch
is restored to fixed block 1024.

After the user requested a different route, I loaded the project-owned runtime
overlay rather than the venv package: CUTLASS DSL **4.8.1** and QuACK **0.6.5**.
QuACK's SM120 GEMM then compiled and ran for the real Muse shapes. With tuning
enabled it measured 0.0506 ms for `(M,N,K)=(32,6656,6656)`, 0.1775 ms for
`(32,19968,6656)`, and 0.1792 ms for `(32,6656,19968)`. The matching cuBLAS
measurements were about 0.0260, 0.188, and 0.198 ms respectively, while the
QuACK output max-absolute errors were 2.0, 0.0, and 4.0 BF16 units. It is a
real SM120 implementation, but it is slower or numerically unsuitable for
the serving contract, so it was not wired into KDA.

As a final KDA A/B, I routed only the `(M,N,K)=(32,6656,19968)` linear shape
to the overlay QuACK kernel in the full Muse long-long service. The cohort
completed with identical leading token IDs but only **115.849 tok/s**
(**0.8708x** versus the corrected KDA default), so the microbenchmark speedup
does not survive model scheduling/graph replay. The opt-in launch hook was
removed after the A/B.

To separate cold compilation from graph/runtime overhead, the same QuACK
shape was rerun with eight client warmups before the formal Muse long-long
cohort. It reached **115.702 tok/s**, still only **0.8697x** of the corrected
KDA default (133.033 tok/s). Extra warmups therefore do not recover the loss;
the candidate is rejected as a runtime integration overhead, not a cold-start
artifact.

A fresh gather CTA probe changed only the Triton launch warp count. With
`FA4_GATHER_WARPS=8`, Muse long-long measured **115.678 tok/s**; the same
current KDA source with the default launch measured **115.808 tok/s**. The
ratio is **0.9989x**, so warp count 8 is rejected. The legacy hand-compiled
CuTe opt-in (`FA4_RSI_LEGACY_CUTE=1`) also completed at **115.668 tok/s** and
was not promoted. A direct paged-KV opt-in was started as a historical-path
check but did not produce a complete receipt before termination; it is not
counted as evidence and its hook was removed.

The archived 157 tok/s server trace was re-audited against its dispatch receipt:
that receipt used Muse `tile_n=64` and the legacy hand-compiled CuTe path. A
fresh current-source six-scene KDA rebaseline was completed at
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-kda-current-rebaseline-029/summary.csv`.
It measured **127.990, 167.003, 58.055, 115.647, 6769.431, and 6460.464**
for the six scenes. The archived-to-current geometric ratio is **0.9528x**;
the mismatch is now reproducible at the current runtime.

The historical combination was isolated explicitly with `tile_n=64` plus
`FA4_RSI_LEGACY_CUTE=1`; Muse long-long still measured **115.7 tok/s** at the
client boundary. A direct `tile_n=64` policy on the current `_flash_attn_fwd`
path also measured **115.765 tok/s**, confirming that the old server trace is
not recoverable by policy alone. The new audit policy is retained as
`configs/serving-policy-muse-n64.json`; it is not the default.

The exact historical `AuxData` ABI was also replayed (`FA4_RSI_LEGACY_CUTE=1`,
`FA4_RSI_LEGACY_EXACT=1`, Muse `tile_n=64`). It completed at **115.772
 tok/s** on long-long, indistinguishable from the current 115.7 tok/s results.
Thus the archived 157 tok/s trace cannot be reproduced by the recorded tile,
legacy kernel, or ABI arguments alone; its runtime/cache state remains an
unresolved provenance difference.

Correction to the earlier n64/legacy probes: the campaign launcher only passes
container overrides through `FA4_RSI_EXTRA_ENV`; direct host variables were not
visible inside the server. Those earlier 115.7 tok/s legacy receipts therefore
actually exercised the default direct ABI and are superseded by the correctly
injected measurements below.

With the real container environment, `FA4_RSI_LEGACY_CUTE=1`, exact legacy
`AuxData` ABI, Muse `tile_n=64`, and routing only `max_seqlen_k>=4096` to the
legacy path, the full Muse six-scene receipt is
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-legacy-long-only-six-035/summary.csv`.
Relative to the reproducible current KDA rebaseline, ratios are **1.0342,
1.0608, 1.0658, 1.1475, 0.9967, 1.0009**, geometric mean **1.0498x**.
This is the accepted Muse routing candidate; the short scenes stay on the
current path.

The same legacy route is not valid for Qwen D256: SM120 rejects the generated
kernel for shared memory (131072 bytes requested, 101376-byte limit). The
failed Qwen receipt is retained as a resource-gate diagnostic and is not used
for throughput claims.

For Qwen D256, the direct decode ABI's auto-split launch fails on long KV
with `CUDA_LAUNCH_INVALID_CONFIG`. A forced `num_splits=1` + `tile_mn` probe
was launch-valid and reached 158 tok/s in the server log, but the complete
client cohort measured only **114.059 tok/s** with 11.5 s mean TTFT. It is
therefore rejected for serving; no Qwen KDA promotion is claimed from this
probe.

A real-container Muse `tile_n=32` legacy probe reached **129.151 tok/s** on
long-long versus **132.701 tok/s** for the accepted n64 route. It is rejected;
`serving-policy-muse-n64.json` remains the selected policy.

The Muse optimization is now integrated into the default `serving-policy-dev`
policy (`legacy_long_cute=true`, `legacy_exact=true`, `legacy_min_k=4096`,
Muse `tile_n=64`). A no-extra-environment full six-scene receipt at
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-kda-default-integrated-041/summary.csv`
reproduced ratios **1.0338, 1.0610, 1.0662, 1.1476, 0.9958, 0.9998** versus
the current KDA rebaseline, geometric mean **1.0495x**. Qwen remains on the
standard direct path because the Muse legacy kernel is not implementable for
D256 on SM120.

A full Muse n96 validation completed at
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-legacy-n96-six-043/summary.csv`.
Relative to the integrated n64 receipt, scene ratios are **0.9995, 0.9996,
1.0017, 1.0001, 1.0028, 1.0003**, geometric mean **1.00066x**. The default
Muse policy is therefore updated to `tile_n=96`; n32 is rejected.

Muse n128 was screened on long-long at **132.633 tok/s**, below n96's
**132.818 tok/s**, so n128 is rejected and n96 remains the default.

Qwen's forced split1 was retested with its n64 tile. Long-long improved from
114.059 to **116.234 tok/s**, but mean TTFT remained **11.53 s**; it is still
below the accepted path and remains rejected.

A Qwen D256 legacy resource probe with `tile_m=32,n=64` was rejected by the
kernel's `can_implement` resource gate before launch. No throughput claim is
made for this variant.

Muse n96 with 64 threads regressed long-long to **108.082 tok/s** versus
132.818 at 128 threads; the 128-thread configuration remains selected.

The integrated policy correctness receipt is
`/data/fa4-rsi-v3/e2e/validate-integrated-049.json`. All Muse cases (including
sliding-window, 4K/8K KV, multistream and graph replay) passed. The overall
receipt remains false only because Qwen D256 cases 2 and 4 hit the already
known SM120 131072-byte shared-memory launch failure; all Qwen reference and
Torch-flash controls passed.

Qwen D256 `num_stages=0` was rejected during kernel construction (`Expected
size in shape to be strictly positive`); the minimum valid staging level is 1.

Qwen D256 legacy `num_threads=64` also launches the same 131072-byte shared
memory kernel and is rejected by SM120; the limit is structural rather than a
thread-count choice.

With the integrated Muse legacy decode route held fixed, the full Muse
prefill 128x32 candidate produced only **1.00031x** geometric mean versus the
integrated 64x64 prefill recipe; it is rejected and the default prefill policy
is unchanged.

The correctly injected gather-warp8 full Muse receipt was
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-integrated-gather-warp8-six-053/summary.csv`.
Its ratios versus the integrated default were **0.9979, 0.9978, 1.0001,
1.0025, 1.0006, 1.0000**, geometric mean **0.99982x**; warp8 is rejected.

A correctly injected gather-buffer reuse candidate completed the full Muse six
scenes at `/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-reuse-gather-six-057/summary.csv`.
Ratios versus the integrated default were **0.99995, 1.00053, 1.00057,
1.00319, 1.00116, 0.99983**, geometric mean **1.00087x**. Buffer reuse is
now enabled in the default serving policy.

A no-extra-environment Muse long-long replay after policy integration reached
**133.067 tok/s** (receipt `six-muse-20261115-default-reuse-long-058`),
confirming that buffer reuse is active in the default policy rather than only
an experiment hook.

The full Muse gather-block2048 + buffer-reuse receipt is
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-gather-block2048-six-061/summary.csv`.
It is **1.00010x** versus reuse-only and **1.00097x** versus the original
integrated default, so `gather_block=2048` is now part of the default policy.

Final no-extra-environment Muse receipt after all integrations (n96 legacy
route, gather reuse, block2048) is
`/data/fa4-rsi-v3/e2e/six-scenes/six-muse-20261115-default-final-062/summary.csv`.
Ratios versus the current main KDA rebaseline are **1.0343, 1.0630, 1.0675,
1.1515, 0.9965, 1.0007**, geometric mean **1.0510x**.

Qwen D256 Q-in-registers plus reduced prefill tile (64x32) was also screened;
the long-KV run still launches a 131072-byte kernel on SM120. The limit is not
removed by moving Q to registers or shrinking the prefill tile.

The final untested Qwen resource combination (`tile_m=64,tile_n=32,
Q_in_regs=true`) still launches 131072 bytes and is rejected by SM120. This
confirms the current Qwen D256 kernel source needs a structural shared-storage
rewrite; policy tuning cannot fit it.

The smallest aligned Qwen probe (`tile_m=32,tile_n=32,num_threads=64,
Q_in_regs=true`, including prefill) still produces 131072-byte SMEM and is
rejected. The pinned SM120 D256 implementation cannot be made launchable by
configuration alone.
