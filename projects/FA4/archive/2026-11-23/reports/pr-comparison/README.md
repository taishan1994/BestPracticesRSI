# PR3109 与 PR3132 比较结论

测量硬件是 RTX PRO 6000 Blackwell Server Edition，SM120、188 SM。测试源码 pin 为 PR3109 `064643897f27c3d7b2058743d9e4dbdbf0d66325` 和 PR3132 `e1f207e9e8453a7343a1a72f741225de93673763`。

## 同精度历史形状对照

[285 项 CSV](pr3132-vs-pr3109-bf16-full-285.csv) 重算几何平均：forward **0.99620×**、decode **0.96805×**，比值定义为 PR3109 latency / PR3132 latency。Forward 包括 138 个 BF16 与 27 个 FP16 场景，decode 120 项为 BF16。原脚本使用 eager CUDA Events，warmup 2 次、5 次均值，不是后来 SM 分区实验的 CUDA Graph 协议。

原脚本中 `correct=True`、`max_abs_error=0` 不提供独立的跨 PR 数值等价验证：比较对象主要是同一分支重复调用，部分代码共享输出 buffer。V 的 `randn_like` 也没有显式使用局部 generator，因此不能仅凭 case seed 认定跨进程全部输入相同。保留这些字段作为历史原记录，而不将它们升级为严格精度结论。

## FP8 跨精度结果

[带误差轮次 CSV](pr3132-fp8-vs-pr3109-bf16-full-285-with-error.csv) 的结果：

| 阶段 | 项数 | 几何平均加速比 | 最大绝对误差 | 有限输出 |
|---|---:|---:|---:|---:|
| Forward | 165 | 1.30455× | 0.09375000 | 165/165 |
| Decode | 120 | **0.91307×** | 0.01953125 | 120/120 |

PR3132 使用 FP8 E4M3 输入、BF16 输出和 unit descale。误差 reference 是同一 FP8 输入反量化为 BF16 后的 PR3132 BF16 路径；不包含原浮点输入到 FP8 的量化误差，也不代表模型 C-Eval/MATH500 精度。原 CSV 的 `pr3109_bf16_ms` 对那 27 个 FP16 forward 行命名不准确，baseline dtype 应以 case 清单为准。

原说明把此前 decode 的 0.90448× 沿用到带误差轮次，但带误差 CSV 重算为 0.91307×。这里没有跨轮次拼接；重新计算结果与极值见 [recomputed-summary.json](recomputed-summary.json)。原始 JSON 保留每场景状态、误差和测量耗时。

原 FP8 误差轮次记录的 source_sha256 与当前留下的 runner 文件并不全部匹配；[runner 审计](../../provenance/runner-audit.json) 同时保留记录指纹与文件指纹。复现入口是经过路径适配的新快照，不能假装恢复了缺失的原始脚本字节。

## NCU

[指定 10 项 FP8 指标](pr3132-fp8-selected-10-ncu.md) 保留历史 NCU 摘要；MFU 必须采用对应精度、dense/sparse 峰值和采样时频率。Tensor Active、SM throughput、MFU 含义不同，不可互相替代。

早期 PR3132 pytest 出现 LSE 阈值失败；后续 285 项 finite/误差测量并没有自动补齐该 upstream 全量验收。初始状态留在历史区，当前归档不宣称 FP8 已通过严格 FP32/LSE 或模型质量门禁。
