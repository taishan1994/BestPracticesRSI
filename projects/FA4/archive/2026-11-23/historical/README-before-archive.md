# FA4 on RTX PRO 6000 Server Edition (SM120)

This project delivers FA4 inference operator recipes. FA2, cuDNN, FlashInfer and quantized attention projects are research controls or context, never substituted into the FA4 product.

- Research edition already delivered: `paper/FA4-SM120-Research-v1.pdf`; preserve its explicit reference-precision erratum in `reports/reference-precision-erratum.json`.
- Final paper: `paper/FA4-SM120-Final-v2.pdf`; matching LaTeX and evidence: `paper/FA4-SM120-Final-v2-LaTeX.zip`.
- Offline FA4 product bundle: `releases/fa4-sm120-1.0.0-offline.tar.gz`; fresh-container qualification: `reports/rebuild.json`.
- Final paired-round results: `reports/final-results.json`; raw records, failed attempts and independent receipts remain under `runs/`.
- Deployment and rollback: `DEPLOYMENT.md`. Frozen tested source/configuration are in `releases/fa4-sm120-1.0.0/`; graph/eager recommendation mapping is separate in `configs/deployment-recommendations.json`.
- Research catalog: `research/projects.json`, with all 15 projects, version pins, primary-source claims and local dispositions.
- RSI evolution: `configs/rsi-versions.json` selects the optional `core/evolution/releases/operator-rsi-0.1.0/` contracts. Existing model SLA policies are unchanged.
- EAT promotion: `OptimizationRSI/eat.py` applies a paired log-ratio lower-bound gate before a per-shape configuration can replace the PR default. The read-only audit receipt is `reports/eat-audit.json`; it consumes all 24 PR acceptance cases (including the explicit D256 not-comparable case), reports 17.39% graph and 97.70% wall-clock geometric-mean gain over all comparable cases, and separately lists the cases whose lower 95% gain bound exceeds 10%.

The prefill release keeps upstream FA4 SM120 kernels and qualifies its direct-launch calling contract. The decode release uses a pinned FA4 PR2634 GEMV+combine branch with an explicit return-ABI patch. These are separate source/process profiles. Fixed BF16 decode scope is B1/Q1/K8192/Hq16/Hkv4/D128; no general decode, paged service, training-speed or model-throughput claim is made.

Five independent holdout rounds confirm graph gains in 18/24 cases and Python-inclusive gains in 19/24 cases relative to native FA4 under the predeclared lower-95%-bound >3% rule. The same-fork decode comparison gives approximately 8.40x graph and 5.72x call speed. A failed D256 native default is Not Comparable, not an infinite speedup. Every selected candidate passes the declared strict numerical gate; all metrics are conditional on the fixed device/software/protocol.

All GPU reproduction and dependency installation occurs in containers on physical GPU0. No host Python/CUDA package installation is required.

## KAD dispatch

`kernels/kad_dispatch.py` provides the optional knowledge-aware dispatch layer
for the FA4 PR path. `configs/kad-policy.json` contains only shape entries whose
correctness gate passed, regression count is zero, and the retained seed lower
bound is at least 1.10 (10%). Launch it with `make_attention(..., mode="kad")`; an
unknown shape or weak evidence falls back to native FA4. The policy is separate
from the pinned upstream source hashes, so enabling KAD does not change the FA4
ABI or the reproducibility of the default launcher.

Current KAD development confirmations cover four shapes with five seeds each:
`mha-s1k-d64` 2.543x, `gqa-b4-s1k-d64` 1.806x, `mha-s2k-d128-causal`
1.803x, and `mha-s4k-d128-causal` 1.458x. These are development evidence only;
PR3109 promotion still requires all 165 forward and 120 decode cases.

Full-matrix acceptance is recorded in
`runs/kad-full-confirmation-final.json`: all eight matrix scenarios and three
seeds were exercised. Seven scenarios passed the numerical and execution gates;
the D256 scenario is retained as `not_comparable` because the native SM120
baseline itself requests 128 KiB shared memory while the device allows 101,376
bytes. No speedup is computed from that unavailable baseline.

The PR3109-source non-backward campaign is audited in
`runs/kad-pr-nonbackward-confirmed-141-policy-96-two-tiles-current-audited.json`: all 165
forward and 120 decode cases completed with no backend or backward work. The
current exact policy has 59 forward direct cases (1.0353× direct geometric mean; 1.0125× effective) and
120 decode direct cases (1.2499× effective geometric mean); the effective
285-case geometric mean is 1.1064× after retaining PR3109 fallback for
unqualified cases. Policy 96 includes d192 MHA local 512 at 64×64 and d128 GQA
8192 local at 64×32, both checked with fresh-container CUDA timing and then the
complete 285-case acceptance. Repeat stability and candidate-space receipts are
recorded in `runs/kad-pr-nonbackward-repeat-summary-policy94.json`, `runs/kad-pr-nonbackward-stability-policy94.json`, and `runs/kad-pr-candidate-space-audit-003-full-nonbackward.json`; these retain only candidates that pass complete-scope correctness and repeat measurements.

The complete non-backward campaign exceeds the 1.10× gate;
forward-only promotion remains separate because its exact-policy mean is below
1.10×.

For PR-level acceptance, the authoritative domain is frozen in
`configs/kad-pr-acceptance-protocol.json`: all 165 reconstructed PR forward
cases are required: 165 forward plus 120 decode cases, with backward excluded,
including the 111 forward cases that were historically comparable with main.
The required arms are PR3109 and KAD; main is context only, and no
serving/backend test is part of this gate. The eight-shape matrix above is only
a tuning screen and cannot promote a KAD version. Run
`python3 benchmarks/validate_kad_acceptance_scope.py` before any GPU campaign.
