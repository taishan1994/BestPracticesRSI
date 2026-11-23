# Qwen/Muse 六场景 attention 吞吐对比（graph-enabled 修正版）

固定并发 8；每点 8 个正式请求。prefill 输出 1，仅比较输入吞吐；decode 比较输出吞吐。FA2 使用独立 flash-attn 2.8.4 CUDA runtime，并确认服务日志为 `cuda graph: True`。

| 模型 | 场景 | V405 FA4 | 官方 PR FA4 | FA2 | FlashInfer | Torch Flash | Triton | V405/PR | FA2/PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | short-short | 86.2 | 129.5 | 132.9 | 136.8 | 132.9 | 117.8 | 0.665× | 1.026× |
| Qwen3.8-27B | short-long | 174.6 | 174.5 | 182.6 | 190.7 | 182.7 | 190.4 | 1.001× | 1.046× |
| Qwen3.8-27B | long-short | 61.4 | 61.3 | 63.9 | 66.9 | 64.0 | 65.8 | 1.001× | 1.042× |
| Qwen3.8-27B | long-long | 126.1 | 126.1 | 139.2 | 155.3 | 139.4 | 153.6 | 1.000× | 1.104× |
| Qwen3.8-27B | prefill-short | 6654.4 | 6663.3 | 6658.1 | 6656.0 | 6672.7 | 6625.7 | 0.999× | 0.999× |
| Qwen3.8-27B | prefill-long | 6548.7 | 6555.3 | 6549.0 | 6522.7 | 6563.3 | 6376.6 | 0.999× | 0.999× |
| Muse-Glimmer-30B | short-short | 72.9 | 127.8 | 121.9 | 137.8 | 121.1 | 100.1 | 0.570× | 0.954× |
| Muse-Glimmer-30B | short-long | 159.1 | 167.0 | 159.5 | 188.8 | 158.5 | 188.4 | 0.952× | 0.955× |
| Muse-Glimmer-30B | long-short | 62.0 | 58.0 | 62.9 | 67.1 | 62.8 | 63.6 | 1.069× | 1.085× |
| Muse-Glimmer-30B | long-long | 132.9 | 115.7 | 134.6 | 159.6 | 133.8 | 155.9 | 1.149× | 1.164× |
| Muse-Glimmer-30B | prefill-short | 6767.0 | 6773.1 | 6690.4 | 6732.3 | 6695.1 | 6499.6 | 0.999× | 0.988× |
| Muse-Glimmer-30B | prefill-long | 6459.0 | 6435.4 | 6598.7 | 6457.2 | 6590.6 | 6002.2 | 1.004× | 1.025× |

## SDPA-cuDNN

Qwen d256 和 Muse 实际 GQA 在当前 PyTorch/SM120 组合均返回 `RuntimeError: Invalid backend`；普通 MHA d128 最小预检可用，但不能代表模型路径。

## 证据边界

- 旧的 `six-*-fa2-001` 和 `confirm-fa2-pr-001` FA2 结果因 `torch native attention backend` 关闭 CUDA Graph，均标记为诊断数据，不用于本表。
- 修正版原始数据：`/data/fa4-rsi-v3/e2e/six-qwen-20261111-fa2-graphfix-001-*`、`six-muse-20261111-fa2-graphfix-001-*`。
