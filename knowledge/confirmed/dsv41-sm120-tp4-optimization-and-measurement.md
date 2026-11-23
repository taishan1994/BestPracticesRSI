# DSV4.1 / SM120 / TP4：可复用优化与测量知识

证据等级：本项目实测支持的规则及明确限定的推断；不是对全部模型/硬件的最优性证明。来源：[结项报告](../../projects/pro6000se/deepseek-v4.1-flash/reports/closeout-20261123/README.md)、[GEMM数据与口径](../../projects/pro6000se/deepseek-v4.1-flash/reports/closeout-20261123/gemm-ncu.md)、[来源manifest](../../projects/pro6000se/deepseek-v4.1-flash/reports/closeout-20261123/data/evidence-manifest.json)。

## 部署与通信

1. 先冻结模型revision、镜像digest、实际overlay hash、CLI/env与编译缓存身份。版本字符串相同不代表执行文件或kernel相同，缓存键必须包含影响launch的容量/布局/策略。
2. TP通信里的all-to-all可以只是AllReduce的传输组织，不能据此推断模型开启EP；应区分TP collective与MoE专家token dispatch。
3. INT8 block32 ring减少线载，但引入量化误差；模型权重未改不能证明整个推理无损。保留精度、数值误差门槛和退回原路径的能力。
4. PCIe分块更多不保证更快：复制重叠、同步、信号和量化开销需要一起测。单次链路GB/s只约束指定消息/算法，不是整模型物理上界。
5. NUMA改善可能与加载、页分布和缓存预热混合。稳态性能归因必须由相同协议服务对照支持。

## SLA与结果晋级

1. 选择规则在实验前固定。8192/1用mean TTFT≤2s，8192/1024用mean TPOT≤30ms，N=10C；p95、请求达标率及不合格原始TPS不能替代。
2. 客户端并发、实际admitted/running batch、内部序列容量、请求总数分开记录。capacity-limited/underfilled点保留为观察，不伪装为完整并发复现。
3. 单次筛选、同实例多seed、新部署复验、三对独立部署和全冷重建是不同门槛。局部kernel或输入收益不能绕过输出和质量回归；不同父配方的收益也不能任意相加。
4. 合成固定长度服务测试、冷长prefill、持续decode、真实数据集精度、动态agent轨迹各自单列。AgentX自动执行并不保证不同配方产生相同轨迹，并发1也不足以保证身份相同。

## 算子与架构

1. 外层8192输入不等于所有GEMM M8192；按CED后实际M、batch L、有效route和padding计算。MoE FC1覆盖gate+up，FC2为down；TP局部中间维必须与执行形状对应。
2. SMEM carveout、寄存器预算、grid和tile必须联合评估。更高occupancy不自动等于更快；FP8/MXFP4存储也不意味着使用硬件FP4峰值，应核对实际Tensor指令路径。
3. SM120的适配以实际编译代码、warp-level MMA及launch为准；不能把其他Blackwell架构的tcgen05路径按产品名直接套用。
4. TF32高低分解为保持FP32权重语义会执行额外矩阵乘法。逻辑FLOPs与发射FLOPs分开列，不能因算法展开而人为抬高有效利用率。

## NCU与MFU

1. 用每kernel当次测量窗口的平均SM频率和相应精度非稀疏峰值；记录时间/频率单位。该项目188SM下FP8/BF16/TF32分别2048/1024/512 FLOPs/SM/cycle，并与显式sparsity_off峰值核对。
2. 单kernel有效FLOPs利用率与整模型MFU分开；画像时间与未画像native时间分开；多kernel程序event不能冒称单kernel时间。
3. SM综合吞吐为组成项最大值，不能自动解释为Tensor计算利用率。整齐Dense可能MFU≈SM；MoE补齐、MHC双TF32或LSU主导时通常不同。
4. L2 hit、DRAM吞吐、warp依赖停顿一起看。long scoreboard高并不必然代表DRAM带宽饱和；单看L2 hit高也不能证明没有访存瓶颈。stall份额为warp指令间隔构成，不是墙钟可节约比例。
5. 真输入提取保留值、stride、offset、storage alias、精度和路由，不能仅用同shape随机张量替代。eager提取、图padding、独立冷缓存、输入构造或兼容执行变体必须披露。
6. 并行profiler需要独立通信端点、私有tmp/HOME和明确设备映射；串接会让报告属于错误目标。先审计符号/ABI/launch，再计算MFU。

## 精度与知识沉淀

原生评分与答案提取修正并列保存；按本轮wire response ID去重，区分API错误、空答、截断与非空未解析。修正解析不应按参考答案改写输出。单次GPQA多对1题不能证明统计显著提升。

保存轻量汇总、来源指纹、失败边界与验证条件足以传播知识；不把权重、真实激活、试题全文、缓存和海量日志塞进知识仓库。指纹不能代替原始证据可访问性，也不能把未完成的发布门禁写成已完成。
