# NCU evidence and useful-attention MFU

Warm cache is application replay after 50 warmups with cache-control none. Cold is kernel replay with cache-control all. MFU uses five-round unprofiled graph timing, independent of cache-control experiments; never interpret its repeated value as a measured cold-cache MFU. DRAM bytes unsupported, not zero. Multi-pass L2 >100% stays raw and is unavailable as a probability.

| Case | Method | Cache | Kernels | L2 hit % | SM % | DRAM % | Unprofiled us | Useful MFU % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mha-s2k-d128-causal | official | warm | 1 | 100.00 | 35.89 | 0.54 | 114.339 | 32.13 |
| mha-s2k-d128-causal | ours | warm | 1 | 100.00 | 48.83 | 0.86 | 87.252 | 42.11 |
| mha-s2k-d128-causal | sdpa-cudnn | warm | 1 | 99.98 | 40.31 | 0.90 | 100.557 | 36.54 |
| gqa-s2k-d128-causal | ours | warm | 1 | 99.99 | 49.09 | 0.89 | 86.765 | 42.35 |
| gqa-s2k-d128-causal | sdpa-cudnn | warm | 1 | N/A | 40.38 | 0.81 | 100.579 | 36.53 |
| sdxl-q1024-k77 | official | warm | 1 | N/A | 31.06 | 7.43 | 8.029 | 21.50 |
| sdxl-q1024-k77 | ours | warm | 1 | 99.98 | 37.05 | 7.32 | 7.505 | 23.00 |
| sdxl-q1024-k77 | flashinfer | warm | 1 | 100.00 | 32.54 | 6.73 | 8.568 | 20.14 |
| qu0b-b1-s8192-d128 | ours | warm | 1 | 98.36 | 78.34 | 3.25 | 1543.859 | 76.12 |
| qu0b-b1-s8192-d128 | sdpa-cudnn | warm | 1 | 98.39 | 80.81 | 3.14 | 1540.169 | 76.30 |
| qu0b-b2-s4096-d96 | ours | warm | 1 | 99.97 | 79.00 | 0.20 | 572.007 | 77.04 |
| qu0b-b2-s4096-d96 | fa2 | warm | 1 | 99.95 | 78.24 | 0.18 | 586.923 | 75.09 |
| gqa-decode-k8k-d128 | official | warm | 1 | N/A | 1.72 | 0.18 | 273.791 | 0.05 |
| gqa-decode-k8k-d128 | ours | warm | 1 | 99.77 | 1.66 | 0.38 | 140.862 | 0.10 |
| gqa-decode-k8k-d128 | flashinfer | warm | 2 | 99.98 | 7.57 | 5.70 | 8.531 | 1.68 |
| gqa-decode-k8k-d128 | thad-default | warm | 2 | N/A | 0.37 | 0.38 | 163.600 | 0.09 |
| gqa-decode-k8k-d128 | ours-decode | warm | 2 | N/A | 12.48 | 2.44 | 19.600 | 0.73 |
| mha-s2k-d128-causal | official | cold | 1 | 80.60 | 34.35 | 12.83 | 114.339 | 32.13 |
| mha-s2k-d128-causal | ours | cold | 1 | 88.90 | 46.91 | 16.96 | 87.252 | 42.11 |
| sdxl-q1024-k77 | official | cold | 1 | 32.70 | 26.07 | 32.71 | 8.029 | 21.50 |
| sdxl-q1024-k77 | ours | cold | 1 | 52.03 | 28.04 | 35.11 | 7.505 | 23.00 |
| qu0b-b1-s8192-d128 | official | cold | 1 | 97.18 | 72.72 | 3.83 | 1638.685 | 71.72 |
| qu0b-b1-s8192-d128 | ours | cold | 1 | 96.89 | 78.37 | 4.06 | 1543.859 | 76.12 |
