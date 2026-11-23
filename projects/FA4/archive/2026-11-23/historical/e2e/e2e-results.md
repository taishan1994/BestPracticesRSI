# End-to-end audited

Each repeat is a complete10*C cohort. Pool tokens/sum(cohort duration) for two confirmations, request-weight latency, require both eligible; search peak separately retained. Maximum observed within searched range, not global optimum.

| Model | Backend | Workload | Searched C | Confirmed C | SLA latency ms | Admitted token/s |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | fa4_rsi | prefill | [1, 2, 4] | 1 | 1309.394 | 6239.749 |
| Qwen3.8-27B | fa4_rsi | decode | [1, 2, 4] | None | N/A | 0.000 |
| Qwen3.8-27B | fa4_reference_rsi | prefill | [1, 2, 4] | 1 | 1319.594 | 6195.276 |
| Qwen3.8-27B | fa4_reference_rsi | decode | [1, 2, 4] | None | N/A | 0.000 |
| Qwen3.8-27B | torch_flash_rsi | prefill | [1, 2, 4] | 1 | 1323.029 | 6177.964 |
| Qwen3.8-27B | torch_flash_rsi | decode | [1, 2, 4] | None | N/A | 0.000 |
| Qwen3.8-27B | flashinfer | prefill | [1, 2, 4] | 1 | 1267.511 | 6447.660 |
| Qwen3.8-27B | flashinfer | decode | [1, 2, 4] | None | N/A | 0.000 |
| Qwen3.8-27B | triton | prefill | [1, 2, 4] | 1 | 1344.371 | 6078.395 |
| Qwen3.8-27B | triton | decode | [1, 2, 4] | None | N/A | 0.000 |
| Muse-Glimmer-30B | triton | prefill | [1, 2, 4] | 1 | 1411.681 | 5789.013 |
| Muse-Glimmer-30B | triton | decode | [1, 2, 4] | None | N/A | 0.000 |
| Muse-Glimmer-30B | flashinfer | prefill | [1, 2, 4] | 1 | 1342.257 | 6087.316 |
| Muse-Glimmer-30B | flashinfer | decode | [1, 2, 4] | None | N/A | 0.000 |
| Muse-Glimmer-30B | torch_flash_rsi | prefill | [1, 2, 4] | 1 | 1317.957 | 6200.369 |
| Muse-Glimmer-30B | torch_flash_rsi | decode | [1, 2, 4] | None | N/A | 0.000 |
| Muse-Glimmer-30B | fa4_reference_rsi | prefill | [1, 2, 4] | 1 | 1360.104 | 6008.945 |
| Muse-Glimmer-30B | fa4_reference_rsi | decode | [1, 2, 4] | None | N/A | 0.000 |
| Muse-Glimmer-30B | fa4_rsi | prefill | [1, 2, 4] | 1 | 1345.023 | 6075.102 |
| Muse-Glimmer-30B | fa4_rsi | decode | [1, 2, 4] | None | N/A | 0.000 |
