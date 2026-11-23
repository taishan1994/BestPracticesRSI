# EAT vs PR3901 fresh E2E：Qwen / Muse 六场景

真实权重、GPU2、并发8、graph enabled、radix cache disabled；EAT 使用当前 runtime-safe policy 与 V412 source binding。

| model | scene | PR3901 tok/s | EAT tok/s | delta | TTFT delta | TPOT delta |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | short-short | 134.500 | 129.665 | -3.59% | -0.26% | +5.18% |
| Qwen3.8-27B | short-long | 191.026 | 179.647 | -5.96% | +0.32% | +6.61% |
| Qwen3.8-27B | long-short | 67.663 | 64.158 | -5.18% | -0.83% | +9.16% |
| Qwen3.8-27B | long-long | 156.266 | 136.431 | -12.69% | -0.78% | +16.38% |
| Qwen3.8-27B | prefill-short | 3.312 | 3.316 | +0.11% | -0.05% | +0.00% |
| Qwen3.8-27B | prefill-long | 0.812 | 0.816 | +0.53% | -0.42% | +0.00% |
| Muse-Glimmer-30B | short-short | 140.180 | 132.471 | -5.50% | +0.13% | +7.62% |
| Muse-Glimmer-30B | short-long | 191.853 | 176.685 | -7.91% | -0.52% | +8.98% |
| Muse-Glimmer-30B | long-short | 69.254 | 65.680 | -5.16% | -0.37% | +8.82% |
| Muse-Glimmer-30B | long-long | 162.891 | 143.183 | -12.10% | -0.62% | +15.51% |
| Muse-Glimmer-30B | prefill-short | 3.346 | 3.355 | +0.26% | -0.28% | +0.00% |
| Muse-Glimmer-30B | prefill-long | 0.822 | 0.827 | +0.66% | -0.55% | +0.00% |

Qwen3.8-27B EAT 六场景输出吞吐几何平均：-4.568%。

Muse-Glimmer-30B EAT 六场景输出吞吐几何平均：-5.061%。
