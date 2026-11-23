# GEMM NCU 与采样频率归一化利用率

## 范围和来源

测量时间2026-11-20，独立复算2026-11-23。正式融合服务在Nsys前以8192→1、C6/N60得到24796.90 tok/s、mean TTFT1897.837ms；Nsys本身的画像吞吐21214.48和TTFT2219.945ms受工具影响，不作为SLA成绩。

完整TP4 Nsys使用6个warmup+60个正式冷请求，40层主MoE每阶段每GPU2640次调用。清点32类独立GEMM/GEMV符号，独立重放覆盖32/32；按代表性真实操作数/shape及原执行变体采集47组，输出52条kernel记录。50条计算利用率，另2条split-K归约辅助kernel不重复分配整个GEMM的FLOPs。覆盖依符号规范化和launch/ABI核对完成，不是所有层/所有调用实例逐一画像。

8类融合attention/indexer/MHC Gram-reduction另行清点，未拆分为独立GEMM，也不为其填造GEMM MFU。完整kernel符号、实际M/N/K/L、采样频率、时间、FLOPs和指标在[data/gemm-ncu.csv](data/gemm-ncu.csv)。排序按kernel+实际工作形状，重复符号的不同shape分别保留。

## 真实输入与画像边界

使用实际模型请求产生的激活、原始权重和scale、实际MoE路由及runtime参数，保留stride/offset/storage别名。请求prompt为合成token序列，真实指实际前向张量，不代表真实业务语料。

提取阶段用eager且关闭adaptive verification，独立加载原可执行/相同算子合同；这会改变正式CUDA Graph的padding/执行形状。原始请求8192不表示所有层都是M8192：CED后实际有128行，DSpark还有3行/1行。

最后的FP32置信度头M3/N1/K5376通过同一次请求的pre-norm hidden、3个真实Markov embedding和已加载FP32权重，按原forward的cat(...).float()组装并诊断执行。它是实际模型操作数按原语义构造的输入，不冒称已截获正式adaptive图内部输入。若干小M Dense补充案例使用未改值的真实张量，在ABI/布局兼容核对后加载原trace对应tile版本；两项cuBLAS补充调用也保留相同张量。

NCU 2026.3：`cache-control=all`、`clock-control=none`，单调用独立冷缓存重放。L2是该kernel内部访存命中率，不能等同服务prefix-cache命中率。输出保存抽样比对通过，不是全张量逐元素正确性或模型精度证明。并行采集使用独立network/tmp/HOME与不同NCU端口；早期共享端点导致串接的报告已隔离，未进入本表。

## 指标与计算

| 指标 | 原始NCU字段／计算 |
| --- | --- |
| L2 hit | `lts__t_sector_hit_rate.pct` |
| SM综合吞吐 | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM吞吐 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| kernel时间 | `gpu__time_duration.sum`，base单位ns |
| 当次平均SM频率 | `sm__cycles_elapsed.avg.per_second`，单位Hz |
| 非稀疏峰值 | 实测SM数188 × 当条kernel实测平均频率 × 对应精度FLOPs/SM/cycle |
| 算子MFU | 有效FLOPs ÷ kernel时间 ÷ 上述非稀疏峰值 |

FP8/BF16/FP16/TF32/FP32 SIMT分别使用2048/1024/1024/512/256 FLOPs/SM/cycle，FMA计2。重算峰值与NCU FP8显式`sparsity_off`峰值交叉核对；不混入稀疏翻倍或FP4宣传峰值。使用采集窗口平均频率，不使用标称boost或之后查询的频率。

Dense有效FLOPs=`2*M*N*K*L`；MoE FC1=`4*有效route*H*I`（gate+up），FC2=`2*有效route*H*I`，H5120、TP局部I576。MHC权重高低TF32分解有两次MMA，逻辑FLOPs只计一次，所以在该分母下算法利用率上限50%，不直接与单次FP8 MMA相同解释。

这里的MFU是**单kernel逻辑有效FLOPs利用率**，计时来自NCU画像，不是端到端整模型MFU或未画像CUDA-event测量。多kernel程序event时间不能冒充裸FC1/FC2时间；BF16输入走SIMT的记录仍按表内BF16硬件峰值定义，实际执行路径利用率需要另列分母。

例：8192×1792×5120，FLOPs150323855360，时间272.896µs，频率2.23901553331GHz，峰值862.074716697TFLOP/s，得到63.8977885%。该例实际发射FP8 FLOPs恰等有效FLOPs，Tensor管线为SM综合最大组成项，所以MFU与SM同为63.90%。这不是普遍等式：MoE padding或其他SM管线主导时二者不同。指标定义参考 [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

## MFU降序前20条

| 排名 | 类型／阶段 | 精度 | 实际形状 | L2 hit | SM | DRAM | 算子MFU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Dense GEMM | FP8 | 8192×25600×6144 | 97.02% | 79.15% | 9.92% | 79.15% |
| 2 | Dense GEMM | FP8 | 8192×4096×1280 | 89.14% | 71.90% | 14.81% | 71.90% |
| 3 | Dense GEMM | FP8 | 8192×8192×1280 | 89.81% | 69.38% | 23.39% | 69.38% |
| 4 | Dense GEMM | FP8 | 8192×1024×4096，L=2 | 93.49% | 67.50% | 20.78% | 67.50% |
| 5 | Dense GEMM | FP8 | 8192×5120×2048 | 94.87% | 66.87% | 13.67% | 66.87% |
| 6 | Dense GEMM | FP8 | 8192×5120×640 | 82.23% | 64.59% | 26.95% | 64.59% |
| 7 | Dense GEMM | FP8 | 8192×1792×5120 | 95.60% | 63.90% | 12.11% | 63.90% |
| 8 | Dense GEMM | FP8 | 8192×1152×5120 | 94.31% | 58.85% | 16.13% | 58.85% |
| 9 | BF16 projection | BF16 | 8192×512×5120 | 93.22% | 42.64% | 20.80% | 36.16% |
| 10 | Dense GEMM | FP8 | 128×5120×15360 | 79.63% | 35.77% | 80.57% | 35.77% |
| 11 | BF16 projection | BF16 | 8192×512×5120 | 93.37% | 42.12% | 20.57% | 35.71% |
| 12 | FC1 | FP8 dense tensor path | 8192 tokens，top-6 | 88.42% | 62.92% | 45.91% | 33.52% |
| 13 | FC2 | FP8 dense tensor path | 8192 tokens，top-6 | 88.22% | 51.59% | 57.65% | 26.62% |
| 14 | Dense GEMM | FP8 | 128×5120×2048 | 63.95% | 24.10% | 57.48% | 24.10% |
| 15 | BLAS matrix projection | BF16 | 8192×128×512 | 68.04% | 24.14% | 55.50% | 24.07% |
| 16 | Dense GEMM | FP8 | 128×8192×1280 | 77.26% | 22.64% | 50.16% | 22.64% |
| 17 | Dense GEMM | FP8 | 128×4096×1280 | 77.02% | 19.10% | 46.01% | 19.10% |
| 18 | Dense GEMM | FP8 | 128×1792×5120 | 78.28% | 16.78% | 42.34% | 16.78% |
| 19 | MHC projection | TF32 | 8192×24×20480 | 43.44% | 32.70% | 89.80% | 14.93% |
| 20 | Dense GEMM | FP8 | 128×5120×640 | 74.65% | 14.18% | 34.21% | 14.18% |

## FC1/FC2低于SM综合吞吐的分解

8192行真实top-6共49152条route，370个活跃expert；M16补齐52112行；I576按640执行，有效/发射运算比例 **84.8879%**。FC1的Tensor有效峰值利用率39.4848%×84.8879%=**33.5178%**；FC2为31.3559%×84.8879%=**26.6174%**。两者SM综合62.92%/51.59%由LSU指标主导，不是纯Tensor计算利用率。

额外WarpStateStats在另一轮独立画像获得：

| 8192 tokens | FC1 | FC2 |
| --- | ---: | ---: |
| 每warp每条发射指令间隔 | 7.91周期 | 10.02周期 |
| eligible warps/scheduler | 0.73 | 0.58 |
| issue active | 45.15% | 39.78% |
| barrier | 1.78周期，22.50% | 0.68周期，6.79% |
| MIO throttle | 1.36周期，17.19% | 0.78周期，7.78% |
| long scoreboard | 0.45周期，5.69% | 4.66周期，46.51% |

上述停顿百分比为平均warp指令间隔构成，**不是kernel墙钟时间占比**，不能相加当可兑现加速。两者理论occupancy33.33%，寄存器与shared memory分别约束至4 CTA/SM、16 active warps/SM（硬件48）。FC1有同步/MIO与固定依赖等待，FC2最突出为L1TEX访存结果长依赖；DRAM45.91%/57.65%并未跑满持续峰值，结论支持延迟隐藏不足而不是统一的DRAM带宽饱和说法。

128行MoE的有效比例只有37.8947%；FC1/FC2 MFU4.9976%/4.7560%，不能将它们当成8192行主prefill。详见[data/moe-stalls.csv](data/moe-stalls.csv)。缺少同轮baseline独立NCU对照，本报告不伪造优化前后硬件指标收益，也不将工具estimated speedup当实测收益。
