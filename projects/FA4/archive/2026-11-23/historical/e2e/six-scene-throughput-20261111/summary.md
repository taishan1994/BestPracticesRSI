# Qwen/Muse 六场景注意力吞吐对比（探索版）

固定并发 8；每点 8 个正式请求（1×并发），无前缀缓存；prefill 输出 1，仅比较输入吞吐。decode 场景比较输出吞吐。

| 模型 | 场景 | V405 FA4 | 官方 PR FA4 | FlashInfer | Torch Flash | Triton | V405/PR | V405/FlashInfer | 最佳后端 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.8-27B | short-short | 86.2 | 129.5 | 136.8 | 132.9 | 117.8 | 0.665× | 0.630× | flashinfer |
| Qwen3.8-27B | short-long | 174.6 | 174.5 | 190.7 | 182.7 | 190.4 | 1.001× | 0.916× | flashinfer |
| Qwen3.8-27B | long-short | 61.4 | 61.3 | 66.9 | 64.0 | 65.8 | 1.001× | 0.918× | flashinfer |
| Qwen3.8-27B | long-long | 126.1 | 126.1 | 155.3 | 139.4 | 153.6 | 1.000× | 0.812× | flashinfer |
| Qwen3.8-27B | prefill-short | 6654.4 | 6663.3 | 6656.0 | 6672.7 | 6625.7 | 0.999× | 1.000× | torch_flash_rsi |
| Qwen3.8-27B | prefill-long | 6548.7 | 6555.3 | 6522.7 | 6563.3 | 6376.6 | 0.999× | 1.004× | torch_flash_rsi |
| Muse-Glimmer-30B | short-short | 72.9 | 127.8 | 137.8 | 121.1 | 100.1 | 0.570× | 0.529× | flashinfer |
| Muse-Glimmer-30B | short-long | 159.1 | 167.0 | 188.8 | 158.5 | 188.4 | 0.952× | 0.842× | flashinfer |
| Muse-Glimmer-30B | long-short | 62.0 | 58.0 | 67.1 | 62.8 | 63.6 | 1.069× | 0.925× | flashinfer |
| Muse-Glimmer-30B | long-long | 132.9 | 115.7 | 159.6 | 133.8 | 155.9 | 1.149× | 0.832× | flashinfer |
| Muse-Glimmer-30B | prefill-short | 6767.0 | 6773.1 | 6732.3 | 6695.1 | 6499.6 | 0.999× | 1.005× | fa4_reference_rsi |
| Muse-Glimmer-30B | prefill-long | 6459.0 | 6435.4 | 6457.2 | 6590.6 | 6002.2 | 1.004× | 1.000× | torch_flash_rsi |

## 口径

- `short` 输入=2048，`long` 输入=8192；`short` 输出=128，`long` 输出=1024。
- `prefill-short`=2048→1，`prefill-long`=8192→1。
- 结果是单批探索测量，不是多批置信区间；所有请求均完成并通过 token 数量校验。
- 原始 JSON、SSE 和服务日志位于 `/data/fa4-rsi-v3/e2e/six-<model>-20261111-*`。
