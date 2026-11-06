# DeepSeek-V4.1-Flash 最初复现部署与启动指令

本文记录最开始复现外部效果时的正常服务部署，依据保存的 `run_external.sh`、原部署说明与 `dsv41-ext4` 容器配置整理。

## 1. 部署概况

| 项目 | 配置 |
| --- | --- |
| 模型 | `/data/DeepSeek-V4.1-Flash-1017` |
| 对外模型名 | `DeepSeek-V4.1-Flash` |
| GPU | 物理 GPU 4、5、6、7；4×RTX PRO 6000 Blackwell Server Edition 96GB |
| 并行 | TP=4、PP=1、decode context parallel=1 |
| 推理框架 | 外部 vLLM fork＋B12X |
| Python 入口 | `/opt/venv/bin/python -m vllm.entrypoints.cli.main` |
| 容器名称 | `dsv41-ext4` |
| 服务地址 | `http://127.0.0.1:8000` |
| 宿主 venv | `/home/lbj001947/lil-venv`，挂载为 `/opt/venv` |
| 宿主启动脚本 | `/home/lbj001947/run_external.sh` |

基础镜像为 TensorRT-LLM 镜像，实际推理入口为挂载 venv 中的 vLLM fork。

## 2. 镜像及运行环境

镜像标签：

```text
nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2.dev202610260007
```

本机核实的镜像 ID：

```text
sha256:35327adc947817092c96e1f03750c370c0ddf211420486f640cc8785e996a942
```

仓库 digest：

```text
nvcr.io/nvidia/tensorrt-llm/release@sha256:35327adc947817092c96e1f03750c370c0ddf211420486f640cc8785e996a942
```

| 组件 | 记录版本 |
| --- | --- |
| vLLM（当前挂载 venv METADATA） | `0.1.dev21986+g4a379ed42.cu134` |
| vLLM fork commit | `4a379ed42881ee022aaf5d9ada554f9096e5acf3` |
| Python（原部署记录） | 3.12.3 |
| PyTorch（部署日志） | `2.14.0a0+b2c75dd062.nv26.09` |
| CUDA（镜像运行环境） | 13.4 |
| FlashInfer（原部署记录） | 0.6.18 |
| B12X（当前 venv METADATA） | 1.5.0 |
| TileLang（当前 venv METADATA） | 0.1.15 |
| apache-tvm-ffi（当前 venv METADATA） | 0.1.12 |

宿主的 `/usr/local/cuda-13.1` 曾用于 nsys/ncu，不代表容器推理使用 CUDA 13.1。基础镜像本身不包含完整的外部 fork 环境，复现时必须同时准备 `/home/lbj001947/lil-venv`，容器内挂为 `/opt/venv`。

## 3. 显式设置的环境变量

启动容器时显式设置：

| 环境变量 | 值 | 作用 |
| --- | --- | --- |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600` | 允许模型加载和编译等待 3600 秒 |
| `B12X_AUTOTUNE` | `0` | 关闭这套 B12X 的运行时 autotune 设置 |
| `B12X_COMPILE_DISK_CACHE` | `1` | 启用 B12X 磁盘编译缓存 |
| `B12X_COMPILE_CACHE_DIR` | `/b12x-cache` | 指定容器内缓存路径 |

镜像继承的部分相关变量如下；复用相同 digest 时自动继承，不需要全部重复传入：

| 环境变量 | 镜像值 |
| --- | --- |
| `CUDA_HOME` | `/usr/local/cuda` |
| `CUDA_MODULE_LOADING` | `LAZY` |
| `CUDA_BINARY_LOADER_THREAD_COUNT` | `8` |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `1` |
| `TORCH_NCCL_USE_COMM_NONBLOCKING` | `0` |
| `PYTORCH_ALLOC_CONF` | `garbage_collection_threshold:0.99999` |
| `LD_LIBRARY_PATH` | `/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64` |
| `TRITON_PTXAS_PATH` | `/usr/local/cuda/bin/ptxas` |
| `TORCH_CUDA_ARCH_LIST` | `7.5 8.0 8.6 9.0 10.0 12.0+PTX` |

## 4. 原始正常服务完整脚本

保存为宿主 `/home/lbj001947/run_external.sh`。以下来自保存的原始脚本，未用省略号隐藏参数。

```bash
#!/bin/bash
set -e
mv /usr/local/lib/python3.12/dist-packages/tvm_ffi /usr/local/lib/python3.12/dist-packages/tvm_ffi.disabled 2>/dev/null || true
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /data/DeepSeek-V4.1-Flash-1017 \
  --async-scheduling \
  --attention-backend B12X \
  --block-size 256 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --decode-context-parallel-size 1 \
  --decode-refill-target auto \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-force-include-usage \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-request-id-headers \
  --engram-config '{"cpu_offload":false,"table_memory":"ram","disk_resident_scales":false,"projection_tp":false}' \
  --generation-config vllm \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 \
  --jit-monitor-mode warn \
  --kv-cache-dtype fp8 \
  --linear-backend b12x \
  --load-format instanttensor \
  --max-cudagraph-capture-size 128 \
  --max-model-len -1 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --max-parallel-prefills 1 \
  --moe-backend b12x \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --pipeline-parallel-size 1 \
  --port 8000 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/prof","torch_profiler_with_flops":true,"torch_profiler_with_stack":false}' \
  --prefill-compute-share 0.4 \
  --prefill-policy round-robin \
  --prefill-schedule-interval 1 \
  --reasoning-parser deepseek_v41 \
  --safetensors-load-strategy lazy \
  --no-scheduler-reserve-full-isl \
  --served-model-name DeepSeek-V4.1-Flash \
  --speculative-config '{"method":"dspark","attention_backend":"B12X","draft_sample_method":"probabilistic","rejection_sample_method":"standard","num_speculative_tokens":7,"draft_tensor_parallel_size":4,"enable_adaptive_verification":true,"adaptive_verification_cost_scale":1.0}' \
  --swa-block-size 128 \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41
```

`mv tvm_ffi` 用于屏蔽基础镜像的重复 FFI 注册，只发生在容器可写层；原 venv 仍由宿主挂载提供。

主要配置含义：

| 类别 | 配置 |
| --- | --- |
| attention／linear／MoE | `B12X`／`b12x`／`b12x` |
| 数据精度 | `dtype=bfloat16`，KV cache=`fp8`；模型量化由 checkpoint／fork 识别，不能将整套权重理解为 BF16 |
| CUDA Graph | `FULL_AND_PIECEWISE`，最大捕获 batch=128 |
| 调度 | async scheduling、chunked prefill、每轮最多 4096 tokens、最多 32 sequences、最多 1 个 parallel prefill |
| 缓存 | prefix caching 开启；block=256、SWA block=128 |
| Engram | `cpu_offload=false`、`table_memory=ram`、`disk_resident_scales=false`、`projection_tp=false`；其中 table 使用主机 RAM |
| 推测解码 | DSpark、7 个 speculative tokens、draft TP=4、adaptive verification 开启 |
| 上下文 | `--max-model-len -1`，由 fork 按模型配置及可用资源自动选择 |
| 采样默认值 | temperature=1.0、top_p=0.95；thinking=true、reasoning_effort=high |
| 权重加载 | `instanttensor`，safetensors strategy=`lazy` |

## 5. 最初正常服务如何启动容器

以下按 `dsv41-ext4` 的实际 mounts、环境变量与端口还原。先准备好模型、venv 和上一节脚本。

```bash
mkdir -p /home/lbj001947/b12x-cache

docker run -d --name dsv41-ext4 \
  --gpus '"device=4,5,6,7"' \
  --ipc=host --shm-size=64g \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e B12X_AUTOTUNE=0 \
  -e B12X_COMPILE_DISK_CACHE=1 \
  -e B12X_COMPILE_CACHE_DIR=/b12x-cache \
  -v /data:/data:ro \
  -v /home/lbj001947/lil-venv:/opt/venv \
  -v /home/lbj001947/run_external.sh:/run_external.sh:ro \
  -v /home/lbj001947/b12x-cache:/b12x-cache \
  -p 8000:8000 \
  --entrypoint bash \
  nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2.dev202610260007 \
  -c 'exec bash /run_external.sh'
```

这是历史启动命令。同名容器已存在时，若另建实例应更换名称，端口和 GPU 也应按目标环境调整；本文未执行启动。

GPU 已由 Docker 限定为物理 4–7，因此该正常服务没有再设置 `CUDA_VISIBLE_DEVICES=4,5,6,7`。使用 UUID 或限定设备时，不要把宿主编号和进程可见编号混用。

## 6. 新机器一次性准备外部 venv

以下摘自原始部署记录；目标目录应为新建环境，避免覆盖已有 venv。新机器的模型、venv 和缓存宿主路径可以调整，容器内路径需与启动脚本一致。


### 6.1 下载外部 vLLM fork 的 release bundle

```bash
B="https://github.com/local-inference-lab/vllm/releases/download/vllm-jovian-cu134-beta-4a379ed42881ee022aaf5d9ada554f9096e5acf3"
cd ~
curl -C - -L "$B/vllm-jovian-cu134-4a379ed42881ee022aaf5d9ada554f9096e5acf3.tar.zst" -o external-runtime.tar.zst
mkdir -p ext-runtime && tar --zstd -xf external-runtime.tar.zst -C ext-runtime
# wheels/vllm-0.1.dev21986+g4a379ed42.cu134-cp312-cp312-linux_x86_64.whl
```

### 6.2 生成依赖列表（去掉 torch/torchvision 精确 pin）

```bash
python3 - <<'PY'
import zipfile, glob
z = zipfile.ZipFile(glob.glob('/home/lbj001947/ext-runtime/wheels/*.whl')[0])
md = [n for n in z.namelist() if n.endswith('METADATA')][0]
txt = z.read(md).decode()
deps = [l.split(': ', 1)[1] for l in txt.splitlines() if l.startswith('Requires-Dist:')]
keep = [d for d in deps if not d.lower().startswith(('torch==', 'torchvision=='))]
open('/home/lbj001947/vllm_deps.txt', 'w').write('\n'.join(keep))
print('deps', len(keep))
PY
```

### 6.3 创建 venv 并安装

```bash
mkdir -p ~/lil-venv
docker run --rm \
  -v /home/lbj001947/lil-venv:/opt/venv \
  -v /home/lbj001947/ext-runtime:/wheels \
  -v /home/lbj001947/vllm_deps.txt:/deps.txt \
  --entrypoint bash \
  nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2.dev202610260007 -c '
    set -e
    python3 -m venv --system-site-packages /opt/venv
    /opt/venv/bin/pip install -q --upgrade pip
    /opt/venv/bin/pip install -q --no-deps /wheels/wheels/*.whl
    /opt/venv/bin/pip install -q --no-deps -r /deps.txt
    /opt/venv/bin/pip install -q b12x==1.5.0 tokenspeed-mla humming-kernels quack-kernels tilelang
    /opt/venv/bin/pip install -q loguru
    /opt/venv/bin/pip install -q --upgrade tilelang
  '
```

要点：
- wheel 精确 pin `torch==2.14.0a0+4fdf77b940.nv26.8`，PyPI 没有；用 `--no-deps` 复用镜像的 nv26.09。
- **tilelang 必须 >= 0.1.15**（0.1.11 自带 tvm 与 venv 冲突，import 直接崩）。
- **必须补 `loguru`**（`compressed_tensors` 依赖）。
- `b12x 1.5.0` 恰好依赖 `cutlass-dsl 4.7.1`，与外部 runtime.lock 一致。
- 验证：
```bash
docker run --rm --gpus '"device=4"' -v ~/lil-venv:/opt/venv \
  --entrypoint /opt/venv/bin/python \
  nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2.dev202610260007 -c \
  "import torch, vllm, vllm._C_stable_libtorch, b12x, flashinfer; print(torch.__version__, vllm.__version__, flashinfer.__version__)"
```

## 7. 检查服务与复现依赖

启动后检查：

```bash
docker logs -f dsv41-ext4
curl -s http://127.0.0.1:8000/v1/models
```

迁移需要同时准备基础镜像、模型目录、外部 venv 和启动脚本；宿主路径可按目标机器调整，容器内路径要与脚本一致。B12X 编译缓存可以重新生成。本文保留原部署命令，未执行启动。

参考材料：

- [最初部署说明](deepseek_v41_flash_deploy.md)
- [保存的原始 run_external.sh](benchmarks/deepseek-v41/profiles/tp4-prefill-20261102/original-run_external.sh)
