# Qwen/Muse 六场景端到端对照

BF16 权重/QKV/KV，普通自回归、无推测解码、关闭 radix cache；固定并发 8，每场景 1 轮 × 8 请求，另有 warmup。此轮是初测，不能代替重复验收。baseline 是原始 PR3109；KDA 是历史 serving-policy-dev；EAT 是 V412 源码 + 原 adapter 的 EAT overrides。

| 模型 | Arm | 场景 | 输入 token/s | 输出 token/s | TTFT ms | TPOT ms |
|---|---|---|---:|---:|---:|---:|
| Qwen3.8-27B | pr3109 | prefill-short | 6647.927 | 3.246 | 1878.134 | N/A |
| Qwen3.8-27B | pr3109 | prefill-long | 6585.343 | 0.804 | 5632.759 | N/A |
| Qwen3.8-27B | pr3109 | short-short | 2196.551 | 137.284 | 1863.874 | 43.954 |

状态：running
