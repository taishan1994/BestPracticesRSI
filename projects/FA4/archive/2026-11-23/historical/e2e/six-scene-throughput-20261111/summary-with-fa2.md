# Qwen/Muse 六场景 attention 吞吐对比（含真实 FA2）

固定并发 8；每点 8 个正式请求。prefill 输出 1，仅比较输入吞吐；decode 比较输出吞吐。FA2 使用独立 flash-attn 2.8.4 CUDA runtime。

| 模型 | 场景 | V405 FA4 | 官方 PR FA4 | FA2 | FlashInfer | Torch Flash | Triton | V405/PR | FA2/PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | short-short | 86.2 | 129.5 | 62.8 | 136.8 | 132.9 | 117.8 | 0.665× | 0.485× |
| Qwen3.8-27B | short-long | 174.6 | 174.5 | 109.3 | 190.7 | 182.7 | 190.4 | 1.001× | 0.627× |
| Qwen3.8-27B | long-short | 61.4 | 61.3 | 61.7 | 66.9 | 64.0 | 65.8 | 1.001× | 1.006× |
| Qwen3.8-27B | long-long | 126.1 | 126.1 | 126.7 | 155.3 | 139.4 | 153.6 | 1.000× | 1.005× |
| Qwen3.8-27B | prefill-short | 6654.4 | 6663.3 | 6768.9 | 6656.0 | 6672.7 | 6625.7 | 0.999× | 1.016× |
| Qwen3.8-27B | prefill-long | 6548.7 | 6555.3 | 6662.5 | 6522.7 | 6563.3 | 6376.6 | 0.999× | 1.016× |
| Muse-Glimmer-30B | short-short | 72.9 | 127.8 | 102.9 | 137.8 | 121.1 | 100.1 | 0.570× | 0.805× |
| Muse-Glimmer-30B | short-long | 159.1 | 167.0 | 156.3 | 188.8 | 158.5 | 188.4 | 0.952× | 0.936× |
| Muse-Glimmer-30B | long-short | 62.0 | 58.0 | 32.7 | 67.1 | 62.8 | 63.6 | 1.069× | 0.564× |
| Muse-Glimmer-30B | long-long | 132.9 | 115.7 | 68.2 | 159.6 | 133.8 | 155.9 | 1.149× | 0.589× |
| Muse-Glimmer-30B | prefill-short | 6767.0 | 6773.1 | 3282.0 | 6732.3 | 6695.1 | 6499.6 | 0.999× | 0.485× |
| Muse-Glimmer-30B | prefill-long | 6459.0 | 6435.4 | 3260.3 | 6457.2 | 6590.6 | 6002.2 | 1.004× | 0.507× |

## SDPA-cuDNN

端到端尝试目录：`/data/fa4-rsi-v3/e2e/six-qwen-20261111-sdpa-cudnn-001-sdpa_cudnn_rsi`、`/data/fa4-rsi-v3/e2e/six-muse-20261111-sdpa-cudnn-001-sdpa_cudnn_rsi`。Qwen 的 d256 与 Muse 的实际 GQA 路径均在 `torch_native_backend.py` 返回 `RuntimeError: Invalid backend`，因此没有吞吐数据，不能填 0。最小同卡预检显示普通 MHA d128 可用，但 Qwen d256 和模型 GQA 形状不可用；预检脚本为 `projects/FA4/comparisons/v3/e2e/sdpa_preflight.py`。

## 原始证据

- 原五后端：`/data/fa4-rsi-v3/e2e/six-qwen-20261111-006-*`、`six-muse-20261111-001-*`。
- FA2：`/data/fa4-rsi-v3/e2e/six-qwen-20261111-fa2-001-*`、`six-muse-20261111-fa2-001-*`。
