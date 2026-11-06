# OptimizationRSI 新项目入口

在本仓库打开一个 Codex 聊天，填写下面的任务卡并发送启动指令。当前入口是一套由 Codex 或操作者执行的工作流，部署和压测使用项目提供的脚本与目标机环境。

## 1. 建立任务卡

把模板保存为 `projects/<硬件标识>/<模型标识>/task.yaml`。同一硬件的 PCIe 拓扑、NUMA 和 P2P 等探测结果放在 `projects/<硬件标识>/hardware/`。模板中的 `null` 表示待确认。开跑前补齐关键条件并冻结任务卡，形成可对照的测量协议。

```yaml
schema_version: 1
model:
  repository: null
  revision: null
  weight_variant: null
  quantization: null
  tokenizer_revision: null
  chat_template: null
hardware:
  gpu_model: null
  gpu_count: null
  inventory_ref: null
  resource_limits: null
workload:
  primary_scenario: null
  matrix_ref: null       # 输入/输出长度、到达率或并发、缓存、预热与计时规则
  seeds: []
acceptance:
  sla_tier: null         # InferX Base、InferX Pro 或 InferX Max
  min_sla_pass_rate: 0.95
  max_error_rate: null
  quality_spec_ref: null # 质量下限、相对基线容差、长上下文和 Tool Calling 要求
baseline:
  local_baseline_id: null
  selection_rule: null
budget:
  max_gpu_hours: null
  max_wall_clock_hours: null
  max_candidates: null
  confirmation_gpu_hours: null
  max_rounds_without_gain: null
  minimum_gain_pct: null
```

`matrix_ref` 和 `quality_spec_ref` 指向项目内 `benchmarks/` 和 `quality/` 下的文件。已有可用部署时，可额外提供配方路径、镜像 digest 和压测命令。已有本地基线时，填入运行 ID 并提供可追溯的配方与测量记录。

## 2. 发送启动指令

替换下面的项目路径，再发送整段指令。

```text
为 projects/<硬件标识>/<模型标识> 执行 OptimizationRSI。
从该目录的 task.yaml 读取目标、验收条件和预算，按 framework.md 的优化循环执行。
先检查目标机、任务卡和已有基线的可比性。已有合格基线就复用，否则先建立本地基线。
需要外部方案时接入 DeepResearchRSI 查证配方，再用 ReproduceRSI 在目标机验证。
优先按实测瓶颈提出候选，逐轮筛选和配对确认。每轮记录变更、运行 ID、门禁结果和下一步。
关键输入或目标机缺失时，在 reports/ 记录缺口和恢复条件。
达到停止条件后生成最终报告、部署与回滚说明，再按框架规则归档和清理。
```

## 3. 检查每个阶段的产物

| 阶段 | 开始条件与动作 | 到哪里看结果 |
| --- | --- | --- |
| 建立对照 | 已有基线的实际环境、负载和指标口径须与任务卡一致。已有配方时直接本地实测。只有模型和硬件时，先调研并复现可执行配方。 | `configs/`、`runs/`、`reports/` |
| 筛选候选 | 有可部署、通过质量门禁的配方后，针对实测瓶颈，每轮优先改一个因素。先做冒烟和短压测，记录失败和淘汰原因。 | `configs/`、`runs/` |
| 确认收益 | 筛选通过后，用相同请求和资源与当前最优、锁定基线做配对测量。通过 SLA、质量、稳定性和收益门禁后更新对应场景的最优方案。没有合格基线时，首次达标结果只报告绝对指标。 | `runs/`、`reports/` |
| 收尾归档 | 达成目标、预算用尽或达到停止条件后，汇总适用范围、已验证结果、重建与回滚步骤。先核验归档及引用，再清理可重建的缓存、临时文件和已有校验副本的重复产物。 | `reports/` 中的最终报告、发布清单和清理记录 |

上游接入说明见 [DeepResearchRSI](../DeepResearchRSI/README.md) 和 [ReproduceRSI](../ReproduceRSI/README.md)。完整流程见 [架构设计](../framework.md)。
