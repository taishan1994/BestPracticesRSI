# 选定融合配方与来源身份

本文件记录用户选定的融合版本；仅有同一镜像版本号不足以复现实际服务。基础镜像之外还有七项只读文件覆盖，必须核对实际挂载文件与编译缓存身份。

## 基础与有效配置

- 镜像：`ghcr.io/local-inference-lab/vllm@sha256:a7dc6c5f23310c0949e72fc1977fdcdaa5d8a4e1a9e7ee8f2482e6163b296178`
- vLLM：`0.1.dev21832+g54e054785`；模型revision：`1bae0e1088993b9b82de08f59de5be5a34e42170`。
- TP4、PP1、EP未启用；B12X attention/linear/MoE，BF16 dtype，FP8 KV，GPU memory utilization0.95。
- `max-num-batched-tokens=8192`、`max-num-seqs=64`、`max-cudagraph-capture-size=256`、`max-parallel-prefills=1`。
- 图模式 `FULL_AND_PIECEWISE`；async scheduling、chunked prefill、prefix caching开启，测试使用冷请求检查缓存增量。
- DSpark深度3、draft TP4、probabilistic draft、standard rejection；adaptive verification开启、cost scale1.25。
- `prefill-compute-share=0.4`、round-robin、interval1；block-size256、SWA block128。
- generation：temperature1、top_p0.95；chat template thinking=true、reasoning_effort=high。
- Engram：cpu_offload=false、table_memory=ram、disk_resident_scales=false、projection_tp=false；lazy safetensors loading。
- 物理GPU4–7，CPU32–63及96–127，NUMA memory node1。移机按实际拓扑重新确定绑定，不能照抄本机GPU UUID。

## 通信与执行环境

```text
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=96KB
VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=96KB
VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=6MB
VLLM_PCIE_TWOSHOT_ROW_ELEMS=1024
VLLM_PCIE_DMA_MIN_BYTES=6MB
VLLM_PCIE_DMA_FP8=i8b32_ring
VLLM_USE_BREAKABLE_CUDAGRAPH=1
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_USE_AOT_COMPILE=1
VLLM_USE_STANDALONE_COMPILE=1
VLLM_USE_MEGA_AOT_ARTIFACT=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
B12X_AUTOTUNE=0
CUDA_DEVICE_MAX_CONNECTIONS=32
CUDA_MODULE_LOADING=LAZY
NCCL_IB_DISABLE=1
NCCL_P2P_LEVEL=SYS
NCCL_NET_PLUGIN=spcx
NCCL_BUFFSIZE=2097152
NCCL_MIN_NCHANNELS=16
NCCL_MAX_NCHANNELS=16
NCCL_PROTO=LL,LL128,Simple
```

原始配方还包括固定的模型parser/tokenizer与工具调用配置；完整命令在本地 `web54e-final-confirmation-20261111-v1/recipe.sh`，哈希见证据manifest。这里不提供缺失覆盖源文件时看似可直接运行的启动脚本。

## 七项覆盖源文件

下表是11月20日实际部署记录中的冻结身份，包含图分支、两项PCIe通信文件和四项MoE/MHC文件。不能将“在B12X框架中运行”自动解释为“该镜像原本已经包含所有优化”。

| 覆盖文件（相对B12X或模型目录） | SHA256 |
| --- | --- |
| model_state.py | 3b903f684d1efc82bfc4ce2ec09b7f834505277243c3c7e4ab30513c97a983b8 |
| comm/pcie/pcie_dma.py | 42099b001b91434721fec5fb1a98d580d54feae0bcbd2d2003139b8d784bf58e |
| comm/pcie/_dma_kernels.py | 4ecdcc9b2469e858e5aabc01be7d3eada098dbca281767ac7cc40262ee251027 |
| flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_phase1.py | 42a9b8b584f2921adb8b5a9bacb85a10fb46ffce723a18a289198bccef170278 |
| flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_phase2.py | 71cf57c9b821c33e3719499ee33064a1e7ab44a11b539cd8ebe86877429e1623 |
| flashinfer/experimental/b12x/moe/fused_moe/_impl.py | c9bd7863a059639ef0f31b14daeb58b4477c6fa058d628e258d01d7ecf1caf47 |
| flashinfer/experimental/b12x/norm/mhc/_tuning.py | 43ea8866359a110796f3b0465bdfdcf76223805bdfa9507c19ce227983f2dd52 |

## 实现来源与融合边界

网页镜像提供SM120模型适配、B12X attention/linear/MoE与其通信框架。本项目调整部署/并发、启用8192单请求prefill图路径，并使用审计过的PCIe覆盖文件与MoE/MHC调度覆盖文件。MoE策略按容量区分grid、寄存器和shared-memory carveout；MHC调优涉及投影tile/pipeline/K-split及post_pre56/64的P8策略。逐文件hash比“参考了某网页”更能界定实际执行身份。

历史INT8 all-to-all为TP AllReduce的一种传输组织；没有开启MoE EP，也不表示执行专家token dispatch。最终选定 `i8b32_ring`，每32个值共享BF16 scale，通信有量化误差；不能因模型权重不变就称整条推理路径无损。

PR #60762的SM120 64-token声明overlay做过本项目A/B，但未合入此选定配方：prefill C6约24960.68（对照24913.11），decode C29约905.37（对照910.77）；C30 mean TPOT30.62ms越线。局部prefill微增不足以抵消decode退化。FC1 compact-A+FC2 M32最小overlay也未获得联合收益。具体数值由本项目测试提供，不能当作该PR在所有机器上的结论。

## 重建与回退边界

恢复同一服务需要镜像digest、模型revision、七项覆盖源文件、有效CLI/env及可核对的缓存/编译产物来源。源码和overlay未按本次“只发布报告与知识”的范围上传；文档不保证从GitHub单独重建。应向本机归档取回冻结源，再在隔离缓存中加载、检查生效配置、执行精度与两个mean-SLA协议，最后验证重建与回退。

原非融合版已完成三对部署和完整干净重建；融合版现有档案确认全量精度、一次独立性能复验及11月20日prefill部署复测，但没有证据将其写成三对联合晋级/全冷重建完成。回退目标为原非融合已确认配方，而不是任一26K候选。
