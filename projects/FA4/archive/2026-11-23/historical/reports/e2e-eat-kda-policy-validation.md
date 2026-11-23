# EAT / KDA policy E2E validation

| Policy | Runtime result | Details |
|---|---|---|
| EAT 20261115 | blocked | policy source hashes are FA4 b19/CUTLASS 4.6.2; current validation container is b38/CUTLASS 4.8.1, so source contract rejects before execution |
| KDA nopack | partial | Muse cases passed; Qwen short cases passed; Qwen long case failed because qwen-full 128×32/256-thread config requests 131072 B SMEM, above SM120 101376 B limit |

KDA readbacks show the E2E path did consume the policy config (Qwen prefill `tile_m=128,tile_n=32,num_threads=256`; Muse uses its policy profile).
