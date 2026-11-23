# SM120 注意力测量与优化知识

以下规则来自 PR3109、PR3132 和 188/80 SM 对照，适用于继续研究 SM120 attention；性能数字仅适用于对应输入、版本和协议。

## 区分资源和工作

CTA 是线程块，一个 CTA 在一个 SM 上执行。总 CTA 数描述工作划分；最大 CTA/SM 描述 RF、SMEM 等资源允许的并驻容量。整卡并驻容量上限为 `SM 数 × 最大 CTA/SM`，还需以 grid CTA 总数封顶。不同 tile 的每 CTA 工作量不同，CTA 减半不等于 FLOPs 减半。

每 SM RF 为 65536 个 32 位寄存器，即 256 KiB。`launch__registers_per_thread` 是编译分配，`launch__registers_per_thread_allocated` 包含硬件粒度取整；RF/CTA 为后者乘线程数乘 4 字节。运行时 live registers、分配容量、整卡时间平均 RF 使用量是不同量。

SMEM 每 SM 最高 100 KiB；单 CTA opt-in 可用 99 KiB，加 driver 1 KiB 保留。还必须检查 `launch__shared_mem_config_size`：例如 d96 local 使用 64 KiB carveout 时，38 KiB/CTA 只能放一个。换为 100 KiB carveout、50 KiB/CTA 才可能放两个，但 RF 分区和其他限制也必须通过。

## 区分并驻限制与 spill

寄存器可能限制每 SM 可驻留 CTA 数，同时 SMEM 也限制它；这不单独证明 RF 主导耗时。Spill 是编译器将临时状态放到寄存器之外，可能增加访存。local 请求不全部等于 spill，L1 sector 请求也不是实际 DRAM 字节。

NCU 通用 `sass__inst_executed_register_spilling` 要通过 InstructionStats 采集。只请求部分派生 spilling 指标可能得到缺项；缺失不能当 0。若 local/shared 子计数与通用计数不一致，保留原始值并说明，不用子计数覆盖通用计数。

## 减少 SM 的受控实验

使用 CUDA Green Context 实际划分硬件，再查询 stream 资源和读取 `%smid` 验证。`FLASH_ATTENTION_NUM_SMS` 只是 host planner 输入覆盖，不会限制硬件。至少保留完整 GPU 自动组、限制 SM 但固定原 planner 的控制组、限制 SM 且自动规划的真实组。

固定调度的 285 项 RF、CTA/SM 和 spill 计数全部不变；原有 205 项寄存器并驻限制、17 项 spill 仍存在。自动组有 106 项主 kernel/grid 变化，仍没有增加 spill。减少 SM 主要减少整卡并行容量，而不缩小每 SM RF。

## 判断计算与访存

Forward d128 global 的补采 Tensor Active 约 87%、DRAM 吞吐仅约 5%/2%，支持对 SM 计算资源敏感。d64 大 batch 长 KV paged decode 两边 DRAM 吞吐约 96%，80 SM 仍能接近饱和，耗时几乎不变。小 grid 的并行任务数不足时，完整 GPU 本来就未用满。

频率不锁定时，用采样频率解释缩放，而不是固定标称频率。GPU2 同一次 NCU 中，d128 global 按 SM 数和频率预测耗时倍率 2.232×，实测 2.237×；不能把 GPU2 的频率代入 GPU1 的原始 CUDA Graph 数据做精确因果分解。

## 防止比较口径错误

- 同精度回归和 FP8 对 BF16/FP16 的跨精度收益分开。
- FP8 误差必须注明 reference 是否含量化误差，并同时说明 descale、输出精度和 LSE 门禁。
- finite、自比较、共享输出别名导致的零误差，不替代独立 reference。
- 相同 seed 不保证相同输入字节；每个随机输入都显式传 generator，并保留输入指纹。
- CUDA Graph 热缓存、eager CUDA Events、L2 flush decode、NCU 单 kernel 时间不能混报。
- prepare、attention、combine 依次统计 CTA 和资源；它们的 CTA 数不能相加当成同时驻留数。辅助 kernel 的 NCU 单次微秒值不能替代稳定的 CUDA Graph 分项计时。
- MFU 使用采样时频率及正确精度峰值；SM throughput、Tensor Active、MFU 不是同一个指标。
- NCU 无效比率（如短 kernel L2 hit 超过 100%）应标记为无效，不用于瓶颈结论。
- 源码目录名或 git SHA 标签不能证明文件未修改；固定源文件 SHA256，明确区分观察、推断和独立干预结论。

## 归档原则

保留版本 pin、场景清单、计时协议、逐项结果、必要计数、失败边界、报告和离线校验。重复源码通过固定版本获取；缓存、构建产物、大日志和 NCU 二进制按归档政策清理，并明确说明提取结果的复核边界。旧论文与开发阶段数字不能冒充后来的全域发布结论。
