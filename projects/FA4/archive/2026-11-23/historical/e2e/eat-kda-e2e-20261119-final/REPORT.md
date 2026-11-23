# PR3109 / EAT / KDA 模型端到端实测

2026-11-19，RTX PRO 6000 Blackwell Server Edition（GPU0）。同一张卡串行运行 Qwen3.8-27B 和 Muse-Glimmer-30B。BF16 权重/QKV，KV dtype=auto（BF16）；本轮没有启用 FP8。普通自回归，关闭推测解码和 radix cache。固定并发 8，每场景 1 轮 × 8 请求，另有 1 个 warmup 请求（排除计时）。这是初测；未做独立重复轮次，接近 1% 的差异不能判定为稳定收益。

Baseline 是 PR3109 的固定源码提交 `064643897f27c3d7b2058743d9e4dbdbf0d66325`，不是 main。EAT 是之前 V412 candidate 源码加原 adapter 的 EAT 开关；Qwen prefill 使用 64×64，Muse 未附加手工 tile 覆盖。KDA 使用历史 b38 serving-policy-dev 路径，包含 paged KV→packed varlen 的 gather；Muse 长上下文按历史策略走 legacy CuTe 路径。此表比较整条已集成服务路径，不代表同版本单 kernel 的加速。

历史原版 KDA 在 Qwen decode CUDA Graph 捕获阶段触发 `CUDA_LAUNCH_INVALID_CONFIG`，没有有效 Qwen 测量。主表 KDA 为明确标识的适配版 `kda_deployable`：decode 统一 splits=1 并使用 policy 的固定 base tile（Qwen 64×64；Muse legacy 分支保留原行为），诊断日志按唯一记录去重。所有场景共用相同规则；没有逐场景调参。原始失败日志和 Muse 历史 KDA 数值保留在输入目录与 historical-kda.csv。

集成审计：kda_deployable 的非 legacy 路径只向 `_flash_attn_fwd` 传入 tile/packing/splits，没有传入策略中的 num_threads、Q_in_regs、num_stages。因此额外测试 Qwen `kda_profile`：所有 Qwen 场景使用同一显式 CuTe 编译选择器，prefill 使用 policy 的 128×32 / 256 threads / Q_in_regs=True / 1 stage，decode 使用 64×64 / 128 threads / Q_in_regs=False / 1 stage；pack_gqa=True。这是额外的集成对照，不覆盖前述失败和适配版结果。

六场景输入→输出 token 数：prefill-short 2048→1；prefill-long 8192→1；short-short 2048→128；short-long 2048→1024；long-short 8192→128；long-long 8192→1024。prefill 加速比用输入吞吐 candidate/baseline，其余用输出吞吐；大于 1 表示更快。TPOT 在输出只有 1 token 时无定义。

本轮观察：EAT 吞吐比范围 0.9921–1.0062×，96/96 请求完整 greedy 输出一致，尚未证明稳定端到端收益。Muse KDA 适配版吞吐比范围 0.8244–1.1099×，48/48 完整输出一致；短 prefill 有单轮收益，decode 均退化。Qwen KDA 的两条路径均有 greedy 输出分歧，不能当作已通过精度验收的优化。显式 CuTe 修复了参数未生效的问题，但 decode 仍退化；原始服务适配版的严重 prefill 退化不能代表完整参数路径。

Qwen 显式 CuTe 的 2K prefill 首轮效应在新服务进程中重现：第 1 个 cohort 3408.3 token/s，后两轮 6727.1、6696.5 token/s（相对 baseline 1.0119×、1.0073×）。因此原始首轮不能用来描述已运行同类 cohort 之后的吞吐；也不能拿两次暖态复测宣称统计稳定加速。

| 模型 | 场景 | PR3109 tok/s | EAT tok/s | EAT 比值 | KDA 适配版 tok/s | KDA 比值 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | prefill-short | 6647.927 | 6595.666 | 0.992 | 5250.714 | 0.790 |
| Qwen3.8-27B | prefill-long | 6585.343 | 6574.992 | 0.998 | 3216.668 | 0.488 |
| Qwen3.8-27B | short-short | 137.284 | 137.315 | 1.000 | 122.689 | 0.894 |
| Qwen3.8-27B | short-long | 191.107 | 191.210 | 1.001 | 178.801 | 0.936 |
| Qwen3.8-27B | long-short | 67.010 | 67.007 | 1.000 | 38.564 | 0.576 |
| Qwen3.8-27B | long-long | 155.892 | 155.959 | 1.000 | 116.333 | 0.746 |
| Muse-Glimmer-30B | prefill-short | 6147.618 | 6129.567 | 0.997 | 6823.214 | 1.110 |
| Muse-Glimmer-30B | prefill-long | 6674.105 | 6659.853 | 0.998 | 6516.625 | 0.976 |
| Muse-Glimmer-30B | short-short | 136.449 | 137.003 | 1.004 | 132.121 | 0.968 |
| Muse-Glimmer-30B | short-long | 190.369 | 191.549 | 1.006 | 177.528 | 0.933 |
| Muse-Glimmer-30B | long-short | 68.284 | 68.464 | 1.003 | 62.114 | 0.910 |
| Muse-Glimmer-30B | long-long | 161.484 | 162.311 | 1.005 | 133.125 | 0.824 |

## Qwen 完整参数集成对照

| 场景 | KDA 显式 CuTe 吞吐 token/s | 相对 PR3109 | TTFT ms | TPOT ms | 完整输出一致请求 |
|---|---:|---:|---:|---:|---:|
| prefill-short | 3402.769 | 0.512 | 4229.545 | — | 7/8 |
| prefill-long | 6612.514 | 1.004 | 5612.226 | — | 8/8 |
| short-short | 132.873 | 0.968 | 1862.502 | 45.917 | 0/8 |
| short-long | 181.338 | 0.949 | 1858.969 | 42.328 | 0/8 |
| long-short | 63.571 | 0.949 | 5629.033 | 82.239 | 0/8 |
| long-long | 136.617 | 0.876 | 5627.949 | 53.080 | 0/8 |

## Qwen 显式 CuTe 2K prefill 异常复测

首轮 2K prefill 明显慢于后续同长度 decode 场景的 prefill，补测一个独立服务进程内的 3 个完整并发 8 cohort，每个 cohort 仍排除 1 个 warmup 请求。保留原始首轮，不用复测替换原值。这里只能检查异常能否重复；没有 JIT 时间线证据，不能断言异常由编译导致。baseline 仍为原来的单轮，不能把差异视为统计确认。

| 复测 | 输入 token/s | 相对 PR3109 | TTFT ms | 首 token 一致请求 |
|---|---:|---:|---:|---:|
| prefill-short-recheck1 | 3408.347 | 0.513 | 4222.192 | 7/8 |
| prefill-short-recheck2 | 6727.149 | 1.012 | 1849.623 | 7/8 |
| prefill-short-recheck3 | 6696.531 | 1.007 | 1859.017 | 7/8 |

## 完整指标

输入/输出吞吐单位 token/s；TTFT、TPOT 单位 ms。请求有效表示长度、流式结束和缓存协议检查通过，不等于数值精度通过。Greedy 一致请求数表示与 baseline 对应输入的完整输出 token 序列一致；也不能代替 logits/attention 的精度误差测试。

| 模型 | Arm | 场景 | 输入吞吐 | 输出吞吐 | TTFT | TPOT | 吞吐比 | 有效请求 | Greedy 一致请求 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | pr3109 | prefill-short | 6647.927 | 3.246 | 1878.134 | — | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_deployable | prefill-short | 5250.714 | 2.564 | 2377.919 | — | 0.790 | 8/8 | 7/8 |
| Qwen3.8-27B | eat_v412 | prefill-short | 6595.666 | 3.221 | 1902.798 | — | 0.992 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_profile | prefill-short | 3402.769 | 1.662 | 4229.545 | — | 0.512 | 8/8 | 7/8 |
| Qwen3.8-27B | pr3109 | prefill-long | 6585.343 | 0.804 | 5632.759 | — | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_deployable | prefill-long | 3216.668 | 0.393 | 11503.521 | — | 0.488 | 8/8 | 8/8 |
| Qwen3.8-27B | eat_v412 | prefill-long | 6574.992 | 0.803 | 5642.212 | — | 0.998 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_profile | prefill-long | 6612.514 | 0.807 | 5612.226 | — | 1.004 | 8/8 | 8/8 |
| Qwen3.8-27B | pr3109 | short-short | 2196.551 | 137.284 | 1863.874 | 43.954 | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_deployable | short-short | 1963.032 | 122.689 | 2345.450 | 47.144 | 0.894 | 8/8 | 0/8 |
| Qwen3.8-27B | eat_v412 | short-short | 2197.039 | 137.315 | 1863.768 | 43.943 | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_profile | short-short | 2125.967 | 132.873 | 1862.502 | 45.917 | 0.968 | 8/8 | 0/8 |
| Qwen3.8-27B | pr3109 | short-long | 382.215 | 191.107 | 1862.777 | 40.069 | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_deployable | short-long | 357.602 | 178.801 | 2337.667 | 42.488 | 0.936 | 8/8 | 0/8 |
| Qwen3.8-27B | eat_v412 | short-long | 382.421 | 191.210 | 1864.065 | 40.045 | 1.001 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_profile | short-long | 362.676 | 181.338 | 1858.969 | 42.328 | 0.949 | 8/8 | 0/8 |
| Qwen3.8-27B | pr3109 | long-short | 4288.638 | 67.010 | 5659.352 | 75.518 | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_deployable | long-short | 2468.118 | 38.564 | 11510.188 | 118.171 | 0.576 | 8/8 | 0/8 |
| Qwen3.8-27B | eat_v412 | long-short | 4288.425 | 67.007 | 5664.622 | 75.434 | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_profile | long-short | 4068.568 | 63.571 | 5629.033 | 82.239 | 0.949 | 8/8 | 0/8 |
| Qwen3.8-27B | pr3109 | long-long | 1247.139 | 155.892 | 5657.710 | 45.803 | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_deployable | long-long | 930.667 | 116.333 | 11511.174 | 57.549 | 0.746 | 8/8 | 0/8 |
| Qwen3.8-27B | eat_v412 | long-long | 1247.675 | 155.959 | 5663.219 | 45.777 | 1.000 | 8/8 | 8/8 |
| Qwen3.8-27B | kda_profile | long-long | 1092.936 | 136.617 | 5627.949 | 53.080 | 0.876 | 8/8 | 0/8 |
| Muse-Glimmer-30B | pr3109 | prefill-short | 6147.618 | 3.002 | 2030.679 | — | 1.000 | 8/8 | 8/8 |
| Muse-Glimmer-30B | kda_deployable | prefill-short | 6823.214 | 3.332 | 1799.493 | — | 1.110 | 8/8 | 8/8 |
| Muse-Glimmer-30B | eat_v412 | prefill-short | 6129.567 | 2.993 | 2036.385 | — | 0.997 | 8/8 | 8/8 |
| Muse-Glimmer-30B | pr3109 | prefill-long | 6674.105 | 0.815 | 5516.511 | — | 1.000 | 8/8 | 8/8 |
| Muse-Glimmer-30B | kda_deployable | prefill-long | 6516.625 | 0.795 | 5651.507 | — | 0.976 | 8/8 | 8/8 |
| Muse-Glimmer-30B | eat_v412 | prefill-long | 6659.853 | 0.813 | 5527.345 | — | 0.998 | 8/8 | 8/8 |
| Muse-Glimmer-30B | pr3109 | short-short | 2183.186 | 136.449 | 1851.922 | 44.410 | 1.000 | 8/8 | 8/8 |
| Muse-Glimmer-30B | kda_deployable | short-short | 2113.929 | 132.121 | 1810.259 | 46.669 | 0.968 | 8/8 | 8/8 |
| Muse-Glimmer-30B | eat_v412 | short-short | 2192.049 | 137.003 | 1854.072 | 44.149 | 1.004 | 8/8 | 8/8 |
| Muse-Glimmer-30B | pr3109 | short-long | 380.738 | 190.369 | 1791.794 | 40.300 | 1.000 | 8/8 | 8/8 |
| Muse-Glimmer-30B | kda_deployable | short-long | 355.055 | 177.528 | 1784.054 | 43.351 | 0.933 | 8/8 | 8/8 |
| Muse-Glimmer-30B | eat_v412 | short-long | 383.098 | 191.549 | 1794.091 | 40.039 | 1.006 | 8/8 | 8/8 |
| Muse-Glimmer-30B | pr3109 | long-short | 4370.179 | 68.284 | 5551.263 | 73.763 | 1.000 | 8/8 | 8/8 |
| Muse-Glimmer-30B | kda_deployable | long-short | 3975.277 | 62.114 | 5687.710 | 84.322 | 0.910 | 8/8 | 8/8 |
| Muse-Glimmer-30B | eat_v412 | long-short | 4381.666 | 68.464 | 5546.583 | 73.548 | 1.003 | 8/8 | 8/8 |
| Muse-Glimmer-30B | pr3109 | long-long | 1291.871 | 161.484 | 5553.581 | 44.090 | 1.000 | 8/8 | 8/8 |
| Muse-Glimmer-30B | kda_deployable | long-long | 1064.997 | 133.125 | 5686.218 | 54.504 | 0.824 | 8/8 | 8/8 |
| Muse-Glimmer-30B | eat_v412 | long-long | 1298.485 | 162.311 | 5561.270 | 43.832 | 1.005 | 8/8 | 8/8 |

## KDA 输出一致性诊断

以下共同前缀是从第一个输出 token 开始连续一致的 token 数，不是 attention/logits 数值误差。出现 greedy 分歧的结果只能作为性能诊断，不能视为已通过精度验收；需要独立的 logits/数值验证解释分歧。

| 模型 | 场景 | 完整序列一致请求 | 共同前缀 min / mean / max | 输出长度 |
|---|---|---:|---:|---:|
| Qwen3.8-27B (kda_deployable) | prefill-short | 7/8 | 0 / 0.9 / 1 | 1 |
| Qwen3.8-27B (kda_profile) | prefill-short | 7/8 | 0 / 0.9 / 1 | 1 |
| Qwen3.8-27B (kda_deployable) | prefill-long | 8/8 | 1 / 1.0 / 1 | 1 |
| Qwen3.8-27B (kda_profile) | prefill-long | 8/8 | 1 / 1.0 / 1 | 1 |
| Qwen3.8-27B (kda_deployable) | short-short | 0/8 | 0 / 24.9 / 69 | 128 |
| Qwen3.8-27B (kda_profile) | short-short | 0/8 | 0 / 24.9 / 69 | 128 |
| Qwen3.8-27B (kda_deployable) | short-long | 0/8 | 0 / 24.9 / 69 | 1024 |
| Qwen3.8-27B (kda_profile) | short-long | 0/8 | 0 / 24.9 / 69 | 1024 |
| Qwen3.8-27B (kda_deployable) | long-short | 0/8 | 5 / 8.9 / 20 | 128 |
| Qwen3.8-27B (kda_profile) | long-short | 0/8 | 5 / 8.9 / 20 | 128 |
| Qwen3.8-27B (kda_deployable) | long-long | 0/8 | 5 / 8.9 / 20 | 1024 |
| Qwen3.8-27B (kda_profile) | long-long | 0/8 | 5 / 8.9 / 20 | 1024 |
| Muse-Glimmer-30B (kda_deployable) | prefill-short | 8/8 | 1 / 1.0 / 1 | 1 |
| Muse-Glimmer-30B (kda_deployable) | prefill-long | 8/8 | 1 / 1.0 / 1 | 1 |
| Muse-Glimmer-30B (kda_deployable) | short-short | 8/8 | 128 / 128.0 / 128 | 128 |
| Muse-Glimmer-30B (kda_deployable) | short-long | 8/8 | 1024 / 1024.0 / 1024 | 1024 |
| Muse-Glimmer-30B (kda_deployable) | long-short | 8/8 | 128 / 128.0 / 128 | 128 |
| Muse-Glimmer-30B (kda_deployable) | long-long | 8/8 | 1024 / 1024.0 / 1024 | 1024 |

原始逐请求结果和 SSE token 事件路径见 comparison.csv；artifact-sha256.json 保存哈希。dispatch-summary.json 保存真实路径与覆盖参数摘要（记录数量不是 CUDA Graph 的 GPU 执行次数）。全部模型使用 page size 256、prefill chunk 8192、max running 16、decode graph max batch 16、context length 10240、seed 0。Qwen 的 hybrid GDN 层仍使用原有 Triton backend，所以 FA4 的局部改善不会按相同比例作用到整个模型。

完整性：42/42 已完成（36 项主矩阵 + 6 项 Qwen 显式 CuTe 对照）。 额外 2K prefill 复测 3/3。
