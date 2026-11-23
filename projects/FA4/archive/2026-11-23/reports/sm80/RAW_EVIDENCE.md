# NCU 原始记录的归档口径

按仅保留重要报告和可复用知识的要求，二进制 `.ncu-rep` 和运行日志已清理。完整逐 kernel 指标保留于 [kernels.json](analysis/kernels.json) 和 [kernels.csv](analysis/kernels.csv)，原始 NCU 文件的 SHA256、1375 条记录的计数与提取验证见 [validation.json](analysis/validation.json)。补采指标见 [explanation-kernels.json](analysis/explanation-kernels.json) 与 [explanation-summary.json](analysis/explanation-summary.json)。

这些提取结果可以离线复算汇总，但不能重新打开已经清理的原二进制报告，也不能从提取结果重新证明原 NCU 文件内容。需要新的 GPU 证据时使用归档复现脚本重新采集。
