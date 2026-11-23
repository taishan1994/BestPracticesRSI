# Qwen/Muse 六场景端到端对照

BF16 权重/QKV/KV，普通自回归、无推测解码、关闭 radix cache；固定并发 8，每场景 1 轮 × 8 请求，另有 warmup。此轮是初测，不能代替重复验收。baseline 是原始 PR3109；KDA 是历史 serving-policy-dev；EAT 是 V412 源码 + 原 adapter 的 EAT overrides。

| 模型 | Arm | 场景 | 输入 token/s | 输出 token/s | TTFT ms | TPOT ms |
|---|---|---|---:|---:|---:|---:|

状态：running
