# PR3132 FP8 vs PR3109 BF16：285 场景逐项误差与加速比

每个场景使用同一随机输入种子。输入先转为 FP8 E4M3，PR3132 走 FP8 kernel；同时把同一 FP8 输入反量化成 BF16，使用 PR3132 的 BF16 path 作为误差 reference。`speedup = PR3109 BF16 latency / PR3132 FP8 latency`。

| 阶段 | 场景数 | 几何平均加速比 | 最大绝对误差 | 平均最大绝对误差 | 有限输出 |
|---|---:|---:|---:|---:|---:|
| Forward | 165 | 1.30455× | 0.093750 | 0.045236 | 165/165 |
| Decode | 120 | 0.90448× | 见 CSV | 见 CSV | 120/120 |

误差是 FP8 输出相对“同一 FP8 输入反量化到 BF16 后的 BF16 reference”的 max absolute error，不是 BF16 原始输入 reference，也不是最终模型精度指标。逐场景的 PR3109 latency、PR3132 FP8 latency、reference latency、加速比和误差全部在 CSV 中。
