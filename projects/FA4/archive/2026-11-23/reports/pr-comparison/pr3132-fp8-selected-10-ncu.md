# PR3132 FP8 NCU：指定 10 场景

输入为 FP8 E4M3，输出 BF16，unit descale。MFU 使用 FP8 dense peak：188 SM × 2048 FP8 FLOPs/SM/cycle × NCU 采样 SM 频率。Tensor Active 单独列出。

| 场景 | kernel µs | L2 hit % | SM % | DRAM % | FP8 MFU % | Tensor Active % |
|---|---:|---:|---:|---:|---:|---:|
| bf16-d128-h16-hk4-b8-dense-uniform4096 | 63.776 | 41.70 | 7.20 | 23.12 | 0.450 | 43.97 |
| bf16-d256-h8-hk2-b128-paged256-one_long_short128 | 645.280 | 2.48 | 0.56 | 6.64 | 0.069 | 15.60 |
| bf16-d256-h8-hk2-b32-paged256-loguniform64to32768 | 539.136 | 0.69 | 1.63 | 19.82 | 0.202 | 31.91 |
| bf16-d64-h32-hk8-b1-dense-uniform512 | 7.232 | 99.82 | 1.03 | 0.02 | 0.064 | 29.99 |
| bf16-d64-h32-hk8-b32-paged256-uniform32768 | 749.472 | 0.08 | 19.65 | 94.08 | 1.228 | 20.12 |
| bfloat16-d128-gqa-varlen-causal | 661.216 | 93.36 | 66.09 | 9.34 | 65.340 | 67.01 |
| bfloat16-d128-mha-s512-global | 161.696 | 54.56 | 48.94 | 63.87 | 48.937 | 51.25 |
| bfloat16-d256-mqa-s8192-causal | 920.896 | 95.52 | 66.27 | 6.56 | 66.024 | 67.60 |
| bfloat16-d64-mqa-s8192-causal | 947.456 | 95.57 | 68.70 | 6.39 | 68.173 | 69.13 |
| bfloat16-d96-mha-s8192-local1024 | 213.248 | 81.39 | 62.54 | 37.07 | 55.645 | 64.98 |

MFU 公式：有效 FP8 attention FLOPs /（188 × 2048 × 实际 SM 频率 × NCU kernel 时间）。
