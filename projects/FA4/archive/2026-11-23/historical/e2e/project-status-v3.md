# 15-project current evidence overlay

Fixed-source snapshot; documentation, versioned API recipe measurement and E2E runtime are distinct. Fifteen investigated projects do not mean fifteen independent executed FA4 algorithms.

## Dao-AILab 官方 FA4

SM120专用类继承SM80；前向、反向和varlen已具备基本路径。

b38 versioned API: 23/24 comparable; default D256 resource failure retained. Our separate safe serving tile does not retroactively fix the stock operator baseline.

Pinned source/version: 4.0.0b38；源码主分支7c53542e

## qu0b/fa4-sm120-research

在SM120 Max-Q工作站上探索线程、tile和流水级。

Four published shapes, EXPERIMENT_B parameters adapted to b38 ABI. This is an adapted recipe, not unmodified old source reproduction.

Pinned source/version: b4133047880b2ffbb164f7deec58426bd033176d

## blake-snc 原始 TMA PR2349

TMA搬运与warp专门化的早期SM120实现。

Superseded original proposal investigated from fixed sources. No new standalone local performance run; later source bundles do not count as unmodified reproduction of this PR.

Pinned source/version: PR2349；2026-11-07网页快照

## SecondNatureComputing HF SM120 bundle

将多个未合并PR打包，提供TMA、paged、block-sparse和dropout等功能入口。

Prior dependency failure resolved in isolated DSL4.5.2/Quack0.4.0 using only an existing Arch enum re-export. 23/24 comparable; D256 rejected by source resource guard. Source API bundle measured; source dispatch has TMA and non-TMA paths.

Pinned source/version: 60117041e10fcc6f19882afd274318c755a5ef6e

## LingYeAI TMA 八 MMA warp PR2599

在原始TMA方案上增加MMA warp，试图缓解寄存器压力和spill。

Prior dependency failure resolved in isolated DSL4.5.2/Quack0.4.0 using only an existing Arch enum re-export. 18/24 comparable; five GQA cases fail compilation and D256 fails construction in this pinned recipe. No claim of permanent hardware non-support.

Pinned source/version: 44779ed18e6c92c79d8abbb04f27c450a66bf3cf

## sisgrad TMAOptimized PR2406

独立TMA优化路线，覆盖SplitKV、paged、varlen、pack-GQA及更多调度策略。

Distinct source proposal investigated; no standalone local runtime/performance claim. Documentation coverage does not imply measured comparison.

Pinned source/version: PR2406；分支dz/sm120_tma_optimized

## thad0ctor SM120 集成 PR2634

整合TMA、分派、paged、decode和较大head维度的完整SM120路径。

Pinned public default API measured for all24 shapes. Separate exact decode profile uses an explicitly disclosed return-ABI patch and direct GEMV/combine path. Default and modified product are not interchangeable.

Pinned source/version: baabf5a1b106c3fb95c365852e2e75c0078af837

## johnnynunez SM120 下游集成

汇集TMA八warp、score_mod反向及CUTLASS适配修复。

Downstream integration inspected; no standalone local runtime/performance claim. Overlap with related TMA lineage is not an independent speed result.

Pinned source/version: johnnynunez/flash-attention PR1网页快照

## Nebius Physical AI Workbench

在PRO6000 Server Edition上锁定FA4/CUTLASS/Quack并发布数值、算子和SDXL记录。

Exact source helper with pinned FA4/DSL/Quack, only12 qualified shapes; tile assertion forbids silent generic fallback. No confirmed graph-time wins over this recipe; three confirmed call-time wins.

Pinned source/version: FA4 eed1971f；helper源b002a53e；当前仓库72845438

## PyTorch SDPA 与 FlexAttention

SDPA提供Flash/cuDNN等后端；FlexAttention可用FA4或Triton生成注意力。

Explicit SDPA flash, SDPA cuDNN and Flex Triton only; no claim that Flex FA4 integration was measured.

Pinned source/version: 本机Torch2.11开发构建；文档2026-11-07快照

## NVIDIA Transformer Engine / cuDNN

将FlashAttention、cuDNN fused与原生unfused作为不同后端管理。

cuDNN comparator is PyTorch SDPA route. Transformer Engine module itself was not executed, and no TE FA4-dispatch claim is made.

Pinned source/version: TE2.19文档；本地cuDNN由Torch运行时读取

## FlashInfer

提供MHA/GQA prefill和多种decode后端；本地可强制FA2作为竞争方案。

Explicit FA2-family operator route, batch ragged execution for batched shapes. D96 rejected due prior illegal access, not counted as an infinite gain. Serving version is separate0.6.18 and requires independent E2E data.

Pinned source/version: 0.7.3；单请求与ragged接口分别对齐

## Meta xFormers

提供memory-efficient attention和后端封装，作为生态覆盖与兼容性线索。

Documentation-only coverage, no independent local SM120 FA4 performance assertion.

Pinned source/version: 2026-11-07来源快照

## SageAttention

采用INT8/INT4或FP8/FP4等量化注意力，属于相关替代路线。

Different quantized-attention precision protocol; excluded from exact BF16 ranking. No broad claim that quantized attention is unsupported.

Pinned source/version: 2026-11-07项目快照

## CloudRift Emmy

提供模型编译、内核调优与部署工具，相关RTX注意力研究是优化方法线索。

Compiler/tuning-system research coverage; no independently executable SM120 FA4 recipe established in this snapshot.

Pinned source/version: 2026-11-07README快照

