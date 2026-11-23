# FA4 PR3109 与 PR3132 测试归档

本归档保留 RTX PRO 6000 Blackwell Server Edition（SM120，188 SM）的重要结论、逐场景数据、测量协议、可复用知识和复现入口。归档日期为 2026-11-23。没有进行新的性能测量。

## 重要结论

| 对照 | Forward | Decode | 口径 |
|---|---:|---:|---|
| PR3132 对 PR3109，同精度历史形状对照 | 0.99620×，165 项 | 0.96805×，120 项 | 原 runner 为 eager CUDA Events，5 次均值；不是 CUDA Graph 结果 |
| PR3132 FP8 对 PR3109 BF16/FP16，带误差轮次 | 1.30455×，165 项 | **0.91307×，120 项** | E4M3 输入，BF16 输出；跨精度比较 |
| PR3109，80 SM 固定调度对 188 SM | 0.47817× | 0.83934× | CUDA Graph 热缓存，9 轮中位数；188 SM=1× |
| PR3109，80 SM 自动调度对 188 SM | 0.47965× | 0.84113× | 实际使用 CUDA Green Context 限制为 80 SM |

PR 分支比较的加速比为 `PR3109 耗时 / PR3132 耗时`；SM 对照为 `188 SM 耗时 / 80 SM 耗时`。均按场景等权计算几何平均，大于 1 才表示候选更快；不能将混合场景的比值当成生产流量的总吞吐。

- [PR 比较结论和精度口径](reports/pr-comparison/README.md)
- [188 与 80 SM 完整报告](reports/sm80/REPORT.html) · [性能变化分析](reports/sm80/EXPLANATION.html)
- [285 场景 SM 对照 CSV](reports/sm80/analysis/comparison.csv) · [CTA 对照](reports/sm80/analysis/cta-comparison.csv)
- [10 场景整体耗时](reports/sm80/analysis/representative10-timing.csv) · [prepare/combine CTA 与单 kernel 耗时](reports/sm80/analysis/representative10-helpers-comparison.csv)

固定调度时，285 项的每 CTA 寄存器分配、每 SM 并驻上限和 spill 计数全部不变；205 项包含寄存器并驻限制，17 项原本已有 spill。自动组有 106 项主 kernel 配置或 grid 改变，但没有新增 spill 或 spill 计数增加。减少 SM 的主要影响是整卡并行容量降低，而非缩小单 SM RF。

## 证据边界

现有清单为 165 forward＋120 decode，共 285 项，排除 backward。它是按 PR 公布范围重建的本地清单，不是作者原始 165＋89 项的逐行恢复，也不穷举所有 score_mod、mask_mod、softcap、block sparsity 等组合。

历史同精度 runner 中的 `correct=True` 主要来自分支内部重复调用、自比较，不能证明 PR3109 与 PR3132 数值等价。部分 V 输入使用未显式传入 generator 的 `randn_like`，不能仅凭相同 seed 宣称两个独立进程的全部输入字节一致。FP8 的 finite 和误差记录也不等于严格 upstream FP32/LSE 验收或模型质量验收。

旧说明中带误差轮次 decode 的 0.90448× 沿用了上一轮数值；对应 CSV 重算为 **0.91307×**。所有归档主结论以 [recomputed-summary.json](reports/pr-comparison/recomputed-summary.json) 为准。初始精度失败、旧论文和后续测量属于不同阶段，均有明确分类，没有将它们合并为单次验收。

二进制 NCU、重复源码快照、镜像离线包、缓存和运行日志不保留。完整提取指标、输入形状、源 SHA、采样信息和验证摘要仍在归档中；[NCU 证据边界](reports/sm80/RAW_EVIDENCE.md) 说明哪些内容可离线复算、哪些必须重新采集。

## 可复用知识

- [SM120 测量与优化知识](knowledge/measurement-and-sm120.md)
- [容器复现与离线验证](reproduce/README.md)
- [版本固定与源码校验](provenance/source-pins.json) · [原始输入映射](provenance/copied-files.json)
- [流水线和 ping-pong 参考](knowledge/pipeline-reference/fa4-sm120-pingpong-timeline.md)

每 SM 有 256 KiB RF 和最高 100 KiB SMEM；单 CTA 的 opt-in SMEM 上限为 99 KiB，另有 1 KiB driver 保留。实际 carveout 可能只有 64 KiB。必须综合 RF 分区、SMEM、warp 和 block 限制判断并驻数，不能仅用总 RF 除以每 CTA 占用。

用户提供的原 PDF 和截图保存在项目的 `references-local/`，不包含在本次 GitHub 发布文件中。历史代码与配置保留在 `historical/reference-code/`，供理解算法与失败案例；其中旧绝对路径不作为当前可执行入口。

## 历史论文与端到端结论

[论文索引](papers/README.md) 保留重要 PDF 和匹配 LaTeX。v140 明确属于开发阶段，不是最终全域优化发布。旧论文中的 24 场景或其他 PR 结果不能冒充这里的 285 项 PR3109/PR3132 结果。

[2026-11-19 模型端到端报告](historical/e2e/eat-kda-e2e-20261119-final/REPORT.md) 保留 EAT/KDA 的集成边界：EAT 尚未证明稳定端到端收益；部分 Qwen KDA 路径有 greedy 输出分歧，不属于已通过质量验收的优化。历史逐场景策略或单 kernel 数字不能直接代表模型服务收益。

## 验证

```bash
python3 tools/verify_archive.py
```

只使用 Python 标准库，检查保留文件 SHA256、285 项覆盖、几何平均、源脚本指纹、SM 硬件证明以及归档主文档链接。GPU 测量仅在显式运行复现命令时执行，环境隔离在容器内。
