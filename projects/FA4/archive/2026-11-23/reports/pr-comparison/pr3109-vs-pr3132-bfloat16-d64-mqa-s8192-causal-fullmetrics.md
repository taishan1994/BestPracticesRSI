# PR3109 vs PR3132：完整 NCU 指标

| 指标 | PR3109 | PR3132 |
|---|---:|---:|
| Kernel duration (µs) | 1520.032 | 950.304 |
| SM clock (MHz) | 2243.077 | 2205.318 |
| SM throughput % | 84.410 | 68.664 |
| Tensor Active % | 84.410 | 69.117 |
| DRAM throughput % | 5.418 | 6.351 |
| L2 hit % | 96.999 | 95.578 |
| L2 throughput % | 36.484 | 30.155 |
| L1TEX throughput % | 29.432 | 36.826 |
| SMEM wavefront count | 185457400 | 141863024 |
| XU/ALU/FMA proxy: tensor op count | 554050781184 | 554050781184 |
| Eligible warps / SMSP | N/A | N/A |
| Issue Active | N/A | N/A |
| Barrier stall % | 0.342 | 0.511 |
| Long Scoreboard stall % | 3.301 | 6.290 |
| Short Scoreboard stall % | 1.500 | 5.716 |
| Mio Throttle stall % | 1.972 | 5.293 |
| Wait stall % | 27.272 | 25.015 |
| Math Pipe Throttle stall % | 40.485 | 22.013 |
| Not Selected stall % | 5.187 | 6.982 |
| Lg Throttle stall % | 0.001 | 0.006 |

Stall 数据来自 NCU SourceCounters PC sampling；完整原始 CSV 为 `pr3109-fullmetrics.csv` 和 `pr3132-fullmetrics.csv`。
