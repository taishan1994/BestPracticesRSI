# FA4 SM120 qualified deployment

Both product profiles execute FA4. FA2, cuDNN and FlashInfer are research controls only.

Use physical GPU0, RTX PRO 6000 Blackwell Server Edition. The image pins Torch, CUDA and FA4 source; launchers reject mismatched versions, source or GPU models. The source and wheel manifest is in `releases/fa4-sm120-1.0.0/SOURCE-MANIFEST.json`.

```bash
docker build --pull=false -t local/fa4-sm120:1.0.0 releases/fa4-sm120-1.0.0
docker run --rm --gpus '"device=0"' --network none --tmpfs /cache:exec --tmpfs /tmp:exec local/fa4-sm120:1.0.0 python examples/prefill.py
docker run --rm --gpus '"device=0"' --network none --tmpfs /cache:exec --tmpfs /tmp:exec local/fa4-sm120:1.0.0 python examples/decode.py
```

Commands are relative to projects/FA4. The fixed base image must already be available for an offline build; all nine pinned product wheels are included locally. No host package installation is required. Initial JIT compilation is outside steady-state timing.

Prefill: initialize `kernels/compiled_attention_v2.py:make_attention` once with `configs/release-policy.json`, under `torch.inference_mode()`. Inputs are [B,Q,H,D], matching dtype/device; qualified direct inputs must be aligned and contiguous. Unknown layouts/shapes use native FA4. The launcher is inference-only; use native FA4 for training and varlen. All 24 tested scenarios pass numerical validation; confirmed gains depend on graph or eager timing, as shown in the final report. Keep native FA4 for cases without a confirmed improvement. The D256 default fails resource qualification; the tested 64x64 tile is a supported special case with no valid default-relative speedup.

Decode: use `kernels/compiled_decode.py:make_decode` in a separate process, loading the bundled pinned PR2634 source first (see examples/decode.py). Only B1/Q1/K8192/Hq16/Hkv4/D128 BF16 is qualified. Return-ABI patch and source hashes are explicit. The actual 47-split GEMV+FA4 combine plan uses private FP32 scratch for each call; no KV quantization. Other shapes are rejected. Do not mix the official and fork flash_attn.cute trees in one interpreter.

Rollback: stop using the custom launcher and restore the pinned native FA4 public API. Preserve the D256 safe tile or reject its failed default. Restore source/configuration from the archive and inspect hashes before reuse. Never silently substitute another attention library. The supplied source archive, original measurements and RSI release remain available for audit.

Scope: one GPU, fixed inference operator workloads. No complete-model throughput, multi-GPU, paged service, general decode or training speed claim. The native forward/backward/varlen checks prove their declared correctness matrix only.
