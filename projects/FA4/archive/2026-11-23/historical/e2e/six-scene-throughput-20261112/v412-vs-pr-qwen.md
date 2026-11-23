# Qwen V412 vs official PR, six standard scenes

Same GPU2, same model/server settings, concurrency 8, one formal request per
concurrent slot, graph-enabled, radix cache disabled. The V412 gate requires a
d256 global decode batch of at least 32, so these six standard scenes are a
regression check rather than the workload where V412 is expected to win.

| scene | PR output tok/s | V412 output tok/s | throughput delta | TTFT delta | TPOT delta |
|---|---:|---:|---:|---:|---:|
| short-short | 134.670 | 134.028 | -0.48% | +2.01% | -0.08% |
| short-long | 190.972 | 191.142 | +0.09% | +0.26% | -0.10% |
| long-short | 67.888 | 67.879 | -0.01% | +0.04% | -0.04% |
| long-long | 156.236 | 156.349 | +0.07% | -0.11% | -0.07% |
| prefill-short | 3.314 | 3.310 | -0.11% | -0.23% | n/a |
| prefill-long | 0.812 | 0.809 | -0.40% | +0.17% | n/a |

All requests were valid. The result confirms that V412 is neutral on the
standard concurrency-8 scenes; its measured gain belongs to the separate
batch-32 one-long/short workload.
