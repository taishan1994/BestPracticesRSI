# DeepSeek-V4.1-Flash 四卡优化结项报告

结项日期：2026-11-23。硬件为四张 NVIDIA RTX PRO 6000 Blackwell **Server Edition 96GB（SE，非 MAX-Q）**，TP4，SM120；工作卡物理编号4–7。本项目采用本机实测证据，不将同事机器或网页公开成绩当作本机复现结果。

## 结项结论

用户选定融合配方作为默认交付，保留原非融合发布版作为已完整确认的回退参考。融合版继承网页 fork/B12X 与本项目部署、通信、MoE/MHC 策略，具体身份见[配方记录](deployment.md)。N64-only、FC1 StreamB 等候选不因单项吞吐更高而自动替代它。

结项关闭本轮优化与画像任务，保留后续机会和明确边界：现有归档中融合 manifest 仍有全冷重建及最终包门禁未闭合，不在本次文档整理中伪造通过。原非融合版三对独立部署、完整干净重建的证据不能转借给融合版。2026-11-20 的部署复测完成的是运行与 prefill SLA 验证，不是全可执行缓存清空的重建证明。

## 测试协议与环境

- 服务目标1：8192输入、1输出，搜索客户端并发C，正式请求N=10C，按 **mean TTFT ≤ 2000ms** 选最高输入吞吐。
- 服务目标2：8192输入、1024输出，N=10C，按 **mean TPOT ≤ 30ms** 选最高输出吞吐。
- 请求精确长度、错误数和prefix-cache增量分别检查；p95和请求级达标比例不参与筛选。
- 96GB×4、TP4、NUMA1；CPU32–63和96–127，模型 DeepSeek-V4.1-Flash-1017，revision `1bae0e1088993b9b82de08f59de5be5a34e42170`。
- 融合镜像 vLLM `0.1.dev21832+g54e054785`；镜像digest、七项覆盖源文件SHA256及有效参数见[deployment.md](deployment.md)。画像时驱动615.71.09，CUDA兼容版本13.4；这是11月20日环境快照，不把不同日期的驱动版本当作同一固定环境。

## 服务吞吐结果

| 版本／证据 | 8192→1 输入吞吐 tok/s | 8192→1024 输出吞吐 tok/s | 说明 |
| --- | --- | --- | --- |
| Baseline 回测 | 18,692.26，C4 | 689.49，C22 | 原始配方；正式比较分母 |
| 原非融合发布版 | 24,892.34，C6 | 901.94，C29 | 三对新部署确认后，另作完整干净重建 |
| 融合版前沿筛选 | 24,913.11，C6 | 920.15，C30 | 单部署筛选；C30 mean TPOT 29.9693ms，余量很小 |
| 融合版独立部署复验 | 24,879.13，C6 | 901.36，C29 | seed1002；mean TTFT 1891.955ms、TPOT 29.5512ms |
| 融合版 Nsys 前无画像复测 | 24,796.90，C6 | 未重测 | 2026-11-20；60请求有效，mean TTFT 1897.837ms，cache增量0 |

原非融合完整重建点相对 Baseline 回测点的算术增幅为输入 **+33.17%**、输出 **+30.81%**；这是两个具体观测值的比例，不是从原始配方到最终版本直接完成三对测量所得的累计置信区间。不同阶段的收益区间不能相乘后冒称一个直接配对区间。

融合版C30为筛选峰值；独立部署复验的是C29，不能将920.15写成多次独立部署稳定确认的输出成绩。11月20日没有补测decode，不能把最新prefill和旧decode拼成同一轮联合确认。

完整机器可读记录见[data/serving-results.csv](data/serving-results.csv)。这里的32/64序列容量、客户端并发和请求总量是三个不同量。

## Local-inference-lab 协议

采用本地保留的 `llm-inference-bench v0.7.9`（commit `dcdfe05b19e489c4ae27321131aafb6fc5028f50`）测试：冷prefill 32K/128K，decode上下文0/64K/128K、并发1/8/16。它的持续decode窗口与8192输入、固定输出长度、mean SLA协议不同。

冷prefill（Baseline／原非融合版／融合）：32K **15,506／20,780／21,864 tok/s**；128K **15,466／19,950／20,911 tok/s**。32K配置实际prompt约32256 tokens，配置标签不等于精确prompt计数。

完整decode九格及实际并发、capacity-limited/underfilled标记保存在[data/local-inference-lab-results.csv](data/local-inference-lab-results.csv)。融合记录中0上下文C16、64K C8/C16、128K C8/C16存在capacity限制，其中128K C8另有underfilled；这些是保留的观测，不能作为所有指定并发均已充分供给的证明。对照侧未记录相同字段时留空，不能把空字段解释为无此限制。

网页原始版本、网页增强G1C1与本项目融合是不同配置。原始网页结果不得被增强结果替换，也不把该协议吞吐直接与8192的SLA吞吐混排。[结果CSV](data/local-inference-lab-results.csv)只收录有明确本机原始来源的Baseline、原非融合版、选定融合版。

## 精度与答案提取

| 数据集／版本 | 原生正确数 | 原生精度 | 答案提取审计后 |
| --- | --- | --- | --- |
| C-Eval／融合 | 1273/1346 | 94.58% | 94.58% |
| MATH-500／融合 | 487/500 | 97.40% | 491/500，98.20% |
| GPQA Diamond／Baseline | 180/198 | 90.91% | 未更改原评分 |
| GPQA Diamond／非融合发布版 | 180/198 | 90.91% | 未更改原评分 |
| GPQA Diamond／融合 | 181/198 | 91.41% | 未更改原评分 |

C-Eval/MATH-500 使用 EvalScope 1.14.1，temperature1、top_p0.95、thinking/high、max_tokens32768；该轮seed947。融合共1846个唯一HTTP200响应，请求参数检查通过。C-Eval旧输出目录曾混有历史行，按本轮wire response ID去重后再审计；不能用文件行数充当本轮样本数。

MATH-500的四个格式恢复来自同一批已有生成，包括等式右侧、向量赋值、boxed标量/范围后的解释文本；不重新生成、不按参考答案改写输出。**原生97.40%与提取审计98.20%并列保留**。该审计未发现非空最终答案仍未解析的问题，不意味着模型所有推理或答案都正确。

GPQA使用198题、并发8、temperature1/top_p0.95/high、max_tokens32768；三版本HTTP错误均0，Baseline/非融合/融合仍分别有5/2/4个截断无最终答案的题，保留在分母。融合仅多对1题，不能声称已统计证明精度提升。精度表和原始评分边界见[data/quality-results.csv](data/quality-results.csv)。本发布包不含试题、答案全文或推理轨迹。

## 优化、画像与可复用知识

- [逐轮做法、状态、负结果与来源](optimization-history.md)，以及[迭代图](throughput-iterations.svg)。26K级输入候选保留为未整配方晋级记录。
- [GEMM NCU结论](gemm-ncu.md)：32/32类独立GEMM/GEMV符号覆盖、47组重放、52条kernel记录，其中50条给出算子MFU，2条split-K归约辅助kernel不重复计GEMM FLOPs。另8类融合attention/indexer/MHC Gram-reduction只有清点，不冒称已分拆GEMM MFU。
- [RSI知识沉淀](../../../../../knowledge/confirmed/dsv41-sm120-tp4-optimization-and-measurement.md)：mean协议、来源/缓存身份、真实工作量、通信与EP区分、数值门禁、频率归一化及画像边界。

## 实测极限与未决事项

已确认的是已探索配方在固定硬件/工作负载下的合格吞吐与资源约束，没有全局最优证明。相邻并发失败说明当前配方的实测边界，不说明所有实现不能突破。

FC1同步/MIO与FC2访存依赖等待仍有优化线索；N64、FC2 M32、FC1 compact-A/StreamB的原生或输入收益不保证联合SLA/精度收益。GC-MoE/MegaMoE适配探索未得到可替代融合版的已确认端到端成绩，按用户要求停止；不将依赖构建或单kernel可运行写成整模型成功。AgentX为动态agent轨迹，不将它的输入吞吐当成固定8192服务吞吐，也不以固定并发推断轨迹相同。

## 交付与证据保留

本目录只包含报告、汇总CSV、轻量SVG和[evidence-manifest.json](data/evidence-manifest.json)。manifest记录本地原始证据逻辑路径与SHA256；哈希提供身份指纹，不等于GitHub上能取回原始文件或独立重现结果。

不上传模型权重、真实激活/路由张量、试题全文、容器导出、运行日志、编译缓存、runtime、Nsys/NCU二进制或上游代码快照。真实输入与原始画像继续保留在本机归档；没有原始输入和冻结覆盖源文件时，公开文档不构成可直接重建整个服务的自包含软件发行包。

结项不改变已有运行服务的状态，本次只整理文档与证据，未启动新GPU实验。11月20日画像任务结束时已退出模型和测试进程。
