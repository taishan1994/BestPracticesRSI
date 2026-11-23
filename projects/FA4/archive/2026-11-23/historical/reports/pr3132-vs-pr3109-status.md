# PR3132 vs PR3109：SM120 forward/decode 对比状态

## 版本关系

PR3132 (`e1f207e`) explicitly builds on PR3109 and adds SM120 FP8 forward. It is not PR3120; PR3120 only changes dependency floors.

PR3132 input/output contract:

- input: FP8 E4M3 or E5M2 Q/K/V;
- output: BF16 O and FP32 LSE;
- optional Q/K/V descales;
- forward only; no backward.

Therefore `PR3132 FP8 / PR3109 BF16` is a dtype-and-kernel comparison, not a same-precision regression comparison. The same-shape BF16 check must be reported separately.

## Upstream claimed matrix

The PR description claims, on RTX PRO 6000 (188 SM):

- 40 prefill cases: FP8/BF16 geometric mean 1.69x;
- 8 additional D192/D128 cases: 1.67x;
- 24 paged decode cases: 1.84x.

The PR reports CUDA-graph replay, median of 9, BF16/FP8/FP8/BF16 order, and under 6% run-to-run spread.

## Local validation

Source was cloned from `lucifer1004/flash-attention` branch `pr/sm12x-fp8` at `e1f207e` and forced through `PYTHONPATH=/workspace` to avoid the container's installed FA4 package.

Command:

```text
pytest -q tests/cute/test_flash_attn_sm120_fp8.py --disable-warnings --maxfail=0
```

The local run was interrupted after observing repeated failures; the first failure was:

```text
test_sm120_fp8_dense[e4m3-causal-gqa-hd128x128]
LSE error = 2.045e-03; gate is < 2e-03
```

The first two FP8 cases passed before this failure. This means the local container/source pair is not yet a clean acceptance environment for PR3132, so no local FP8 speedup number is promoted.

## Required comparison

The valid comparison has two reports:

1. Common BF16/FP16 forward + decode matrix: PR3132 branch must be numerically and performance compatible with PR3109; no FP8 path is involved.
2. FP8 subset: run PR3132 FP8 against PR3109 BF16 on the PR3132 40-prefill + 24-paged-decode shapes, with dtype, descales, page layout, graph protocol, and numerical reference recorded explicitly.

The published 1.69x/1.84x values are upstream claims until the local source/dependency environment passes the FP8 correctness gate.
