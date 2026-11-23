# PR3109 vs PR3132：完整 NCU 指标

| 指标 | PR3109 | PR3132 |
|---|---:|---:|
| Kernel duration (µs) | 399.616 | 201.376 |
| SM clock (MHz) | 2290.702 | 2238.112 |
| SM throughput % | 58.264 | 62.649 |
| Tensor Active % | 58.264 | 62.649 |
| DRAM throughput % | 33.964 | 39.302 |
| L2 hit % | 89.997 | 81.396 |
| L2 throughput % | 52.524 | 56.661 |
| L1TEX throughput % | 35.300 | 33.411 |
| SMEM wavefront count | 59413654 | 27163093 |
| XU/ALU/FMA proxy: tensor op count | 102676561920 | 108716359680 |
| Eligible warps / SMSP | N/A | N/A |
| Issue Active | N/A | N/A |
| Barrier stall % | 1.037 | 2.053 |
| Long Scoreboard stall % | 8.568 | 7.589 |
| Short Scoreboard stall % | 6.880 | 7.134 |
| Mio Throttle stall % | 0.710 | 2.899 |
| Wait stall % | 38.134 | 27.670 |
| Math Pipe Throttle stall % | 15.735 | 20.180 |
| Not Selected stall % | 0.315 | 4.696 |
| Lg Throttle stall % | 0.024 | 0.250 |

Stall 数据来自 NCU SourceCounters PC sampling；完整原始 CSV 为 `pr3109-d96-localmetrics2.csv` 和 `pr3132-d96-localmetrics.csv`。
