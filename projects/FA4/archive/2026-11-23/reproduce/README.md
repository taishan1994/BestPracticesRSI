# 容器复现与离线验证

本归档不保留重复上游源码、离线镜像包、NCU 二进制或编译缓存。当前数据可以离线复算；重新测量需要获取固定源码和可用容器镜像，结果写入新的目录。

## 离线复核

```bash
python3 tools/verify_archive.py
```

仅用 Python 标准库检查保留文件、几何平均、逐项数值关系、覆盖范围和链接，不占 GPU，不安装主机依赖。

## 软件与源版本

原镜像为 `local/fa4-sm120:1.0.0`，ID 为 `sha256:99adb5ffbb5bead2dd590e108bac7f3ffd727aa0f45b9a1375d587a56f37adbf`。这是本机镜像名，不假装它是可公共拉取的镜像。重新构建须在容器内使用 [记录的软件版本](../provenance/container-package-versions.json)，通过 `FA4_IMAGE` 指定镜像。主机驱动为 615.71.09；NCU 2025.4.1 原安装路径由 wrapper 只读挂载。

```bash
python3 reproduce/fetch_sources.py --revision pr3109
python3 reproduce/fetch_sources.py --revision pr3132
```

源码落在被 Git 忽略的 `_sources/`。PR3109 获取后自动校验 74 个原始文件 SHA256。不会修改原始源码或安装主机包。

## 两个 PR 的历史计时协议

下列脚本是保留 runner 的路径适配版本，不冒充缺失的原始 runner 字节。eager、warmup 2 次、5 次 CUDA Events 均值的协议保留；同精度精度字段仍存在自比较局限。它们不应当被用作严格质量验收。

从归档根目录运行，将 UUID 换为实际设备 UUID；不要用容器内的序号猜物理 GPU。

```bash
bash reproduce/run_container.sh GPU_UUID pr3109 ./_work/pr3109-forward -- python /archive/reproduce/compare_bf16_source_forward.py --output /out/result.json
bash reproduce/run_container.sh GPU_UUID pr3132 ./_work/pr3132-forward -- python /archive/reproduce/compare_bf16_source_forward.py --output /out/result.json
bash reproduce/run_container.sh GPU_UUID pr3109 ./_work/pr3109-decode -- python /archive/reproduce/compare_bf16_source_decode.py --output /out/result.json
bash reproduce/run_container.sh GPU_UUID pr3132 ./_work/pr3132-decode -- python /archive/reproduce/compare_bf16_source_decode.py --output /out/result.json
bash reproduce/run_container.sh GPU_UUID pr3132 ./_work/fp8-forward -- python /archive/reproduce/pr3132_fp8_full_forward.py --output /out/result.json
bash reproduce/run_container.sh GPU_UUID pr3132 ./_work/fp8-decode -- python /archive/reproduce/pr3132_fp8_full_decode.py --output /out/result.json
```

输出目录和依赖均隔离在容器挂载路径中；容器内网络关闭。wrapper 使用 privileged 以支持 NCU 计数器，且同时设置 `CUDA_VISIBLE_DEVICES=GPU_UUID`，防止 privileged 暴露全部 GPU 后误用第 0 卡。

## PR3109 的 188 与 80 SM 三组实验

需要原始 188 SM 设备，Green Context 及 SM ID 证明均会检查精确的 188/80 SM。9 轮 CUDA Graph、每 graph 10 调用、每样本 10 次重放；完整 GPU 自动、80 SM 固定 planner=188、80 SM 自动三组保留。计时与 NCU 不在同一 GPU 同时运行。

```bash
bash reproduce/run_sm80.sh timing GPU_UUID ./_work/sm80
bash reproduce/run_sm80.sh profile GPU_UUID ./_work/sm80
bash reproduce/run_sm80.sh analyze GPU_UUID ./_work/sm80
```

也可顺序在同一张卡运行计时和 NCU；或两张同型号卡分别运行，输出阶段文件可共用目录。新输出会包含 NCU 原文件，供进一步检查；需要另行归档时才按策略清理。本归档的旧 HTML 是原测量的报告，不会被新测量覆盖。
