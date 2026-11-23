# Qwen/Muse 六场景端到端对照

BF16 权重/QKV/KV，普通自回归、无推测解码、关闭 radix cache；固定并发 8，每场景 1 轮 × 8 请求，另有 warmup。此轮是初测，不能代替重复验收。baseline 是原始 PR3109；KDA 是历史 serving-policy-dev；EAT 是 V412 源码 + 原 adapter 的 EAT overrides。

| 模型 | Arm | 场景 | 输入 token/s | 输出 token/s | TTFT ms | TPOT ms |
|---|---|---|---:|---:|---:|---:|
| Muse-Glimmer-30B | pr3109 | prefill-short | 6147.618 | 3.002 | 2030.679 | N/A |
| Muse-Glimmer-30B | pr3109 | prefill-long | 6674.105 | 0.815 | 5516.511 | N/A |
| Muse-Glimmer-30B | pr3109 | short-short | 2183.186 | 136.449 | 1851.922 | 44.410 |
| Muse-Glimmer-30B | pr3109 | short-long | 380.738 | 190.369 | 1791.794 | 40.300 |
| Muse-Glimmer-30B | pr3109 | long-short | 4370.179 | 68.284 | 5551.263 | 73.763 |
| Muse-Glimmer-30B | pr3109 | long-long | 1291.871 | 161.484 | 5553.581 | 44.090 |
| Muse-Glimmer-30B | kda_serving | prefill-short | 6128.442 | 2.992 | 2037.336 | N/A |
| Muse-Glimmer-30B | kda_serving | prefill-long | 6513.943 | 0.795 | 5654.289 | N/A |
| Muse-Glimmer-30B | kda_serving | short-short | 2093.204 | 130.825 | 1852.696 | 46.944 |
| Muse-Glimmer-30B | kda_serving | short-long | 352.430 | 176.215 | 2298.855 | 43.184 |
| Muse-Glimmer-30B | kda_serving | long-short | 3975.103 | 62.111 | 5688.252 | 84.307 |
| Muse-Glimmer-30B | kda_serving | long-long | 1065.008 | 133.126 | 5687.905 | 54.500 |
| Muse-Glimmer-30B | eat_v412 | prefill-short | 6129.567 | 2.993 | 2036.385 | N/A |
| Muse-Glimmer-30B | eat_v412 | prefill-long | 6659.853 | 0.813 | 5527.345 | N/A |
| Muse-Glimmer-30B | eat_v412 | short-short | 2192.049 | 137.003 | 1854.072 | 44.149 |
| Muse-Glimmer-30B | eat_v412 | short-long | 383.098 | 191.549 | 1794.091 | 40.039 |
| Muse-Glimmer-30B | eat_v412 | long-short | 4381.666 | 68.464 | 5546.583 | 73.548 |
| Muse-Glimmer-30B | eat_v412 | long-long | 1298.485 | 162.311 | 5561.270 | 43.832 |

状态：incomplete

失败：Qwen3.8-27B / pr3109
```
Traceback (most recent call last):
  File "/home/gob001915/compare/github/BestPracticesRSI/projects/FA4/comparisons/v3/e2e/campaign-20261119/run.py", line 37, in <module>
    if proc.poll() is not None:raise RuntimeError('server exited; inspect '+str(root/'server.log'))
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: server exited; inspect /home/gob001915/compare/github/BestPracticesRSI/projects/FA4/comparisons/v3/reports/eat-kda-e2e-20261119-r5/Qwen3.8-27B-pr3109/server.log

```

失败：Qwen3.8-27B / kda_serving
```
Traceback (most recent call last):
  File "/home/gob001915/compare/github/BestPracticesRSI/projects/FA4/comparisons/v3/e2e/campaign-20261119/run.py", line 37, in <module>
    if proc.poll() is not None:raise RuntimeError('server exited; inspect '+str(root/'server.log'))
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: server exited; inspect /home/gob001915/compare/github/BestPracticesRSI/projects/FA4/comparisons/v3/reports/eat-kda-e2e-20261119-r5/Qwen3.8-27B-kda_serving/server.log

```

失败：Qwen3.8-27B / eat_v412
```
Traceback (most recent call last):
  File "/home/gob001915/compare/github/BestPracticesRSI/projects/FA4/comparisons/v3/e2e/campaign-20261119/run.py", line 37, in <module>
    if proc.poll() is not None:raise RuntimeError('server exited; inspect '+str(root/'server.log'))
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: server exited; inspect /home/gob001915/compare/github/BestPracticesRSI/projects/FA4/comparisons/v3/reports/eat-kda-e2e-20261119-r5/Qwen3.8-27B-eat_v412/server.log

```
