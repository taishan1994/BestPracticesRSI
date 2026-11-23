# FA4-FP8 分析对 V412/SM120 优化的可迁移结论

这份交叉分析用于把 `FA4-FP8性能分析和优化.pdf` 中的性能方法映射到当前
PR FA4/V412 工作。FP8 的数值协议与当前 BF16 serving 协议分开统计。

## 可以直接迁移的部分

1. **先按瓶颈分类，而不是统一调大 tile。** 当前 paged decode 的 NCU 数据中，
   DRAM active 高而 Tensor Active 低，优化重点应是 page-table/KV gather、L2
   重用和 SplitKV 尾部；dense prefill 的 SM/Tensor Active 高，重点才是
   `tile_m/tile_n`、pipeline stage 和寄存器占用。
2. **scale/metadata 必须纳入端到端时间。** FP8 分析中量化、scale 读取和反量化
   不能从 kernel 时间中删除；同理，V412 的 prepare metadata、paged gather 和
   combine 必须包含在 TTFT/TPOT 中。
3. **按形状选择配置。** PDF 中的 FP8 tile/warp 结论可迁移为 SM120 的配置搜索：
   d128 prefill、d256 decode、paged 与 dense、global 与 local 分开搜索，不能
   使用一个全局 tile 配置。
4. **用采样时钟计算 MFU。** 频率变化会同时改变计算和内存模型；V412 的计划
   选择应继续使用实测 SM/memory clock，而不是规格峰值。

## 不能直接迁移的部分

- 当前官方 SM120 FA4 是 SM80 派生 warp-MMA 路径，不是 SM100 的 tcgen05/TMEM
  主路径。
- FP8 KV、FP8 MMA、scale layout 和 BF16 KV 不是同一个数值协议；FP8 的收益必须
  单独报告精度、量化/反量化开销和模型质量。
- PDF 中 FP8 的 TOPS 或 kernel-only 数字不能替代当前 Qwen/Muse BF16 的端到端
  吞吐，也不能证明 standard six scenes 有收益。

## 对当前 V412 的优先级

1. 保留已验证的 d256 global decode、batch≥32 dynamic SplitKV；它在 Qwen
   one-long/short 混合场景平均带来约 3.34% 吞吐提升和 5.37% TPOT 降低。
2. 对 prefill 单独搜索 `M128×N64`、stage 数和 L2 group，而不是把 decode 的
   dynamic split 逻辑带入 prefill；六场景并发 8 的 V412 已验证基本中性。
3. 对 paged KV gather 做 page-table coalescing/L2 分组实验；比较时必须保留
   gather、prepare 和 combine 的时间。
4. 最后再建立 FP8 KV 的独立实验协议：BF16、FP8-E4M3/E5M2、scale 粒度、精度
   门槛和端到端开销分别记录，不并入 BF16 PR/V412 主排名。
