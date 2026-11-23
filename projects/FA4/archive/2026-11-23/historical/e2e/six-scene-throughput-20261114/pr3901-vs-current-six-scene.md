# PR3901 对比：Qwen / Muse 六场景

同 GPU、并发 8、graph enabled、radix cache disabled、每场景 1 次正式批次，warmup 排除。仓库没有独立的 `PR3901` 目录，PR3901 对照映射到已冻结官方 PR arm（commit `064643897f27c3d7b2058743d9e4dbdbf0d66325`）。

| model | scene | PR3901 tok/s | current tok/s | delta |
|---|---|---:|---:|---:|
| Qwen3.8-27B | short-short | 134.670 | 134.028 | -0.48% |
| Qwen3.8-27B | short-long | 190.972 | 191.142 | +0.09% |
| Qwen3.8-27B | long-short | 67.888 | 67.879 | -0.01% |
| Qwen3.8-27B | long-long | 156.236 | 156.349 | +0.07% |
| Qwen3.8-27B | prefill-short | 3.314 | 3.310 | -0.12% |
| Qwen3.8-27B | prefill-long | 0.812 | 0.809 | -0.39% |
| Muse-Glimmer-30B | short-short | 140.008 | 140.322 | +0.22% |
| Muse-Glimmer-30B | short-long | 191.763 | 191.907 | +0.07% |
| Muse-Glimmer-30B | long-short | 69.128 | 69.186 | +0.08% |
| Muse-Glimmer-30B | long-long | 162.757 | 162.864 | +0.07% |
| Muse-Glimmer-30B | prefill-short | 3.348 | 3.345 | -0.08% |
| Muse-Glimmer-30B | prefill-long | 0.823 | 0.822 | -0.13% |

Qwen3.8-27B 六场景几何平均：-0.140%。

Muse-Glimmer-30B 六场景几何平均：+0.039%。

fresh runner 尝试在启动前因历史容器 `bprsi-fa4-qwen-N32-v351-gpu2` 已不存在而停止；Qwen 候选值沿用已保存六场景审计表，Muse 候选值来自已保存逐场景 summary，没有把失败启动当成新测量。
