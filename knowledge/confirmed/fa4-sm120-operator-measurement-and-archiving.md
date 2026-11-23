# FA4 SM120 测量与归档知识

[PR3109 与 PR3132 归档](../../projects/FA4/archive/2026-11-23/README.md)保留结论、精简逐项数据、协议和容器复现入口。

[完整可复用知识](../../projects/FA4/archive/2026-11-23/knowledge/measurement-and-sm120.md)覆盖 RF/SMEM/CTA、Green Context、spill、跨精度 reference、eager/graph/NCU 计时、采样频率及源码指纹。

关键规则：按实际运行的源文件 hash 固定版本；finite 和自比较不能证明数值正确；相同 seed 不保证所有输入字节一致；不同精度、缓存策略和计时模式分开汇总。缺失或无效的性能计数不得当作零。

全量 188/80 SM 对照显示，285 项固定调度的寄存器分配、每 SM 并驻上限和 spill 计数不变；该观察不等于寄存器对性能没有影响。FP8 历史精度与跨 PR 对照的局限见归档主说明。
