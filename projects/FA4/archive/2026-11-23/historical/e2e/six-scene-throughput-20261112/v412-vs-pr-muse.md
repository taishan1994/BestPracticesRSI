# Muse V412 vs official PR, six standard scenes

Same GPU2, same model/server settings, concurrency 8, graph-enabled, radix
cache disabled, and eight valid formal requests per scene.

| scene | PR output tok/s | V412 output tok/s | throughput delta | TTFT delta | TPOT delta |
|---|---:|---:|---:|---:|---:|
| short-short | 140.008 | 140.322 | +0.22% | -0.71% | -0.07% |
| short-long | 191.763 | 191.907 | +0.07% | -1.42% | -0.02% |
| long-short | 69.128 | 69.186 | +0.08% | -0.16% | -0.04% |
| long-long | 162.757 | 162.864 | +0.07% | -0.24% | -0.05% |
| prefill-short | 3.348 | 3.345 | -0.08% | +0.09% | n/a |
| prefill-long | 0.823 | 0.822 | -0.13% | +0.07% | n/a |

All requests were valid. Muse is effectively neutral on these standard
concurrency-8 scenes, as expected: V412's d256 global decode gate does not
activate here.
