# PR3901 vs V412 fresh E2E：Qwen / Muse 六场景

真实模型权重、GPU2、并发 8、graph enabled、radix cache disabled、每场景 1 次正式批次。PR3901 使用 pinned official PR arm；候选是当前可运行的 V412 serving arm。

| model | scene | PR3901 tok/s | V412 tok/s | delta | TTFT delta | TPOT delta |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | short-short | 134.500 | 134.815 | +0.23% | -0.97% | +0.02% |
| Qwen3.8-27B | short-long | 191.026 | 190.917 | -0.06% | -0.33% | +0.08% |
| Qwen3.8-27B | long-short | 67.663 | 67.790 | +0.19% | -0.40% | -0.09% |
| Qwen3.8-27B | long-long | 156.266 | 156.244 | -0.01% | -0.18% | +0.04% |
| Qwen3.8-27B | prefill-short | 3.312 | 3.307 | -0.16% | +0.10% | +0.00% |
| Qwen3.8-27B | prefill-long | 0.812 | 0.809 | -0.31% | +0.17% | +0.00% |
| Muse-Glimmer-30B | short-short | 140.180 | 140.256 | +0.05% | -0.18% | -0.01% |
| Muse-Glimmer-30B | short-long | 191.853 | 191.842 | -0.01% | +0.07% | +0.00% |
| Muse-Glimmer-30B | long-short | 69.254 | 69.367 | +0.16% | -0.20% | -0.11% |
| Muse-Glimmer-30B | long-long | 162.891 | 162.915 | +0.01% | -0.15% | -0.00% |
| Muse-Glimmer-30B | prefill-short | 3.346 | 3.343 | -0.07% | +0.07% | +0.00% |
| Muse-Glimmer-30B | prefill-long | 0.822 | 0.819 | -0.39% | +0.18% | +0.00% |

Qwen3.8-27B 六场景输出吞吐几何平均：-0.021%。

Muse-Glimmer-30B 六场景输出吞吐几何平均：-0.039%。
