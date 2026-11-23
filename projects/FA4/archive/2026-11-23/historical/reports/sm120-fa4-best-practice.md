# SM120 FA4 最佳实践（PR3109 forward + decode 口径）

## 结论

SM120 的主实现应以 Dao-AILab PR3109 的 SM120-owned forward/decode/runtime/scheduler 为基线。策略按形状族使用 SM120 host planner，不把 165 个 forward 和 120 个 decode 场景写成 285 个独立配置。PR3132 是基于 PR3109 的独立 FP8 forward 扩展，不能混入 BF16/FP16 的主验收结果。

## 上游证据清单

### Dao-AILab/flash-attention

| PR | 结论 | 对最佳实践的影响 |
|---|---|---|
| [#3108](https://github.com/Dao-AILab/flash-attention/pull/3108) | SM120 per-batch SplitKV building blocks、compact split grid、single-split bypass | decode/varlen 必须让 scheduler 根据每批 KV 长度决定 split |
| [#3109](https://github.com/Dao-AILab/flash-attention/pull/3109) | SM120-owned forward/backward；D256、paged KV、varlen、SplitKV、L2-aware scheduler | BF16/FP16 主基线；host planner 优先于手写 tile 表 |
| [#3132](https://github.com/Dao-AILab/flash-attention/pull/3132) | 基于 #3109 的 E4M3/E5M2 FP8 forward；无 backward | 作为独立 FP8 候选；需要 FP8 输入和 descale，不能替代 BF16 结果 |
| [#2634](https://github.com/Dao-AILab/flash-attention/pull/2634) | 另一条 SM120 集成路线，补公共 API、paged KV、D256、FP8 decode | 作为交叉候选，不与 #3109 的结果混报 |

### vLLM

| Issue/文档 | 发现 | 最佳实践动作 |
|---|---|---|
| [#45260](https://github.com/vllm-project/vllm/issues/45260) | FA4 capability selector 漏掉 SM120，导致已有 SM120 dispatch 被判不支持 | 先验证 architecture gate；SM120 必须命中 SM120-owned dispatch |
| [#51776](https://github.com/vllm-project/vllm/issues/51776) | 首调用 `mDynamicCausal` / `is_split_kv` Python 错误 | 必须做 eager、graph、varlen 首调用 smoke gate |
| [attention backends](https://github.com/vllm-project/vllm/blob/main/docs/design/attention_backends.md) | D256 FA4 需要 KV block size 128；不满足时回退 | paged-KV 的 page/block size 作为显式验收维度，不能固定成旧值 |

### SGLang

| Issue/PR | 发现 | 最佳实践动作 |
|---|---|---|
| [#32991](https://github.com/sgl-project/sglang/pull/32991) | 架构拥有的 SM12x FA4 kernel 集成 | serving 路由应使用 SM120-native wrapper |
| [#36531](https://github.com/sgl-project/sglang/issues/36531) | QSA fallback 在 SM120 错导到 pip 通用 FA4；应转 `flash_attention_v4` | 禁止 SM120 fallback 到通用 SM100/pip FA4 |
| [#36806](https://github.com/sgl-project/sglang/pull/36806) | QSA exact SM120 路由到 FlashInfer sparse decode | sparse/QSA 路径和 dense FA4 路径分开验收 |
| [#42012](https://github.com/sgl-project/sglang/issues/42012) | hybrid extend 在 CUDA graph capture 中出现零长度 reshape 崩溃 | graph capture 必须覆盖空/零 extend 和 hybrid shape |
| [#31640](https://github.com/sgl-project/sglang/issues/31640) | FP8 FA4 decode descale 参数不对称 | FP8 prefill/decode 参数和数值 gate 分开验证 |

## 配置原则

配置文件：[sm120-fa4-best-practice.json](../configs/sm120-fa4-best-practice.json)。只有 6 个形状族，运行时 tile、split 和 L2 分组由 SM120 host planner 决定；D256 受 99 KiB shared-memory gate 约束，失败时回退，不为单个 case 记特例。

| 形状族 | 路径 | 核心约束 |
|---|---|---|
| D64–128 forward | SM120-owned forward | scheduler metadata 随 batch/Q/K 形状重建 |
| D192–256 forward | SM120-owned forward | 先过 SMEM gate；否则回退 |
| local/score/mask/modifier | SM120-owned forward | 保留 modifier 语义并单独做 graph gate |
| D64–128 decode | SM120-owned decode | auto SplitKV、L2-aware order |
| D256 decode | SM120-owned decode | transposed decode + SMEM gate |
| paged decode | SM120-owned decode | page/block size 与 D256 约束一起验收 |

## PR3109 同口径验收

- GPU：RTX PRO 6000 Blackwell Server Edition，188 SM。
- Forward：165 行，CUDA Graph，30 replays × 10 calls，mean。
- Decode：120 行，CUDA Graph，30 replays × 1 call，每次 256 MiB L2 flush，mean。
- ABAB 独立进程，固定输入字节，编译/初始化不计入计时。
- 每个 family policy 必须覆盖对应全部行；不允许 per-case override。
- 数值：`abs_error <= .02 + .02*abs(reference)`，同时做 eager、graph、扰动输入和 poisoned output 检查。
- 报告同时保留 PR3109 latency、candidate latency、owner、family、resource fallback、最大误差和每一条样本。

## 当前状态与下一步

已有 PR3109 KDA 285 行验收只能证明“逐行 policy”的上限，不能证明本 family policy。下一步必须用本文件的 6 个 family policy 重跑完整 165+120，并将结果按 family 汇总；只有 family policy 在 PR3109 全域通过后，才进入 Qwen/Muse E2E。

## 真实 smoke / 反例证据

在 `local/fa4-sm120:1.0.0`、GPU1、PR3109 原始 runner 上，family smoke 的数值门全部通过，但统一 `tile_m=64,tile_n=64,num_threads=128,num_splits=0` 在同一形状族内出现回退（例如 FP16 D128 MQA S2048 global 约 0.977×），不能作为通用 family 配置。

随后启动的 165-row forward 尝试在大量行上出现低于 1× 的结果，并在部分 local/D256 配置上触发资源或编译失败，已中止且不计入任何收益结论。原始部分 receipt 保留在 `sm120-family-forward-full.json`，结论在 `sm120-family-forward-full-summary.json`。这条反例验证了：family policy 的“配置”必须是 SM120 host planner + resource gate + runtime metadata，而不是一个固定 tile 复制到所有形状。


### Decode family experiment

在同一容器和 GPU1 上，完整 120 行 decode 的静态 `64x64, num_splits=0` family 候选全部通过数值门，但几何平均只有 **1.02385x**，仍有 **14/120** 行低于 1x，最差 **0.47160x**，因此不晋升。原始明细见 `sm120-family-decode-full.json`，汇总见 `sm120-family-decode-full-summary.json`。

这两次反例实验的结论是一致的：最佳实践不能把 tile/split 写死到 family；family 只负责选择 SM120-owned route、页大小、SMEM gate 和 metadata 生命周期，具体 tile、wave、L2 order、SplitKV 由 PR3109 host planner 运行时决定。

## 最终采用的实践

最终采用 `sm120_host_planner`：调用 PR3109 的 SM120 public interface 时不传 per-case `tile_mn` 或 `num_splits` 覆盖；上层只做 SM120 capability gate、D256/99-KiB resource gate、paged-KV page-size gate、metadata rebuild 和明确的 fallback。这样同一份 policy 可以被 standalone operator runner 和后续 Qwen/Muse E2E backend 复用，不会把算子验收里的精确 case 配置泄漏到模型服务。

这不是“没有优化”，而是把优化放回 PR3109 已验证的 SM120 host planner：它按当前 stream 的 SM 数、Q/K 长度、head ratio、L2 分组和每批 KV 尾部选择 tile、wave、LPT 顺序和 SplitKV。静态 family tile 实验已经证明，把这些决策搬到外层会造成跨 dtype、GQA ratio 和长尾 batch 的回退。
