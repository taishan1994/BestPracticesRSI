# EAT vs PR3901 全六场景最终验收

测量时间：2026-11-15T09:44:50.292184+00:00

真实权重；GPU2；并发8；graphs启用；speculation关闭；候选 V412 源；每个模型六场景各一轮。Qwen prefill 使用 64x64；Muse prefill 使用 64x128；decode 保持原 PR3901 adapter。

## Qwen3.8-27B

|场景|PR3901吞吐|EAT吞吐|变化|
|---|---:|---:|---:|
|short-short|134.500|137.498|+2.229%|
|short-long|191.026|190.944|-0.043%|
|long-short|67.663|67.704|+0.060%|
|long-long|156.266|156.240|-0.017%|
|prefill-short|6782.909|6782.548|-0.005%|
|prefill-long|6650.544|6649.770|-0.012%|

几何均值：+0.365%

## Muse-Glimmer-30B

|场景|PR3901吞吐|EAT吞吐|变化|
|---|---:|---:|---:|
|short-short|140.180|139.877|-0.216%|
|short-long|191.853|191.778|-0.039%|
|long-short|69.254|69.201|-0.077%|
|long-long|162.891|162.817|-0.045%|
|prefill-short|6852.252|6859.831|+0.111%|
|prefill-long|6734.042|6748.562|+0.216%|

几何均值：-0.009%

结论：Qwen 全六场景正收益；Muse 全六场景接近持平但仍略低于 PR，尚未达到稳定正收益及用户要求的极限加速目标。

## Muse 三重复复核

三重复完整六场景复核结果：几何均值 **-0.518%**；short-short -0.334%，short-long +0.381%，long-short -0.789%，long-long -0.034%，prefill-short -0.084%，prefill-long -2.223%。这确认 Muse 当前并非稳定正收益。

## Muse d128 local decode 策略

将已审计的 64x32 local tile 限定到 Muse d128 decode，避免污染 prefill；最新全六场景结果几何均值 **-0.039%**，long-long +0.060%，prefill-long +0.034%。该版本仍未形成稳定正收益。

## Muse 动态 local decode

基于 live `seqused_k` 的动态选择：cache < 8500 使用 64x160，达到 8500 后使用 64x32。两项长场景 probe 分别为 +1.05% 和 +0.145%；完整六场景最终轮几何均值 **-0.051%**，短场景仍略低于 PR。

## Muse MQA 64x64 + prefill 64x64 复核

单轮完整六场景为 +0.010%，但三重复完整六场景为 **-0.318%**（prefill-long -1.658%），因此单轮正收益不稳定，不能作为最终晋级证据。

## Muse global-native + local-MQA-pack 最终轮

将 MQA pack 限定到 local decode、global decode 保持原生；最新全六场景为 **-0.027%**，prefill-short +0.101%、prefill-long +0.202%。

## Muse 同容器 paired 验收（最新）

同一 GPU、同一容器、同一六场景协议重新测 PR3901 与 EAT：short-short -0.003%、short-long +0.027%、long-short -0.082%、long-long -0.013%、prefill-short +0.278%、prefill-long +0.260%，几何均值 **+0.078%**。

## Muse 严格 paired warmup3

PR3901 与 EAT 均使用同一容器、GPU、warmup=3、并发8和六场景协议。EAT 相对 PR：short-short +0.147%、short-long -0.008%、long-short -0.060%、long-long +0.117%、prefill-short +0.116%、prefill-long +0.195%，几何均值 **+0.085%**。

## Qwen paired warmup3（默认 prefill 配置）

关闭强制 Qwen 64x64 prefill tile，使用候选默认 prefill 配置后，严格 paired warmup3 六场景几何均值 **+0.006%**；short-short +0.207%、short-long +0.012%、long-short -0.051%、long-long -0.040%、prefill-short -0.091%、prefill-long -0.001%。
