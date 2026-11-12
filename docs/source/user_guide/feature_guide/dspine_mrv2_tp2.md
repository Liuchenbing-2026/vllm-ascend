# DSpine MRV2 on Qwen3.8-27B with TP=2

The 27B draft is the same DSpine architecture as the 8B one, but a single
64 GB 910B4 cannot serve it: the bf16 target alone loads 54.79 GiB, and the
Qwen3.5 hybrid GDN cache requires at least 2.84 GiB of KV (the mamba page
floor), while only 2.1-2.8 GiB remained. Tensor parallel 2 splits the target to
~30.6 GiB per chip and leaves plenty of room.

## Verified run

- Core: vLLM v0.33.0 + `Liuchenbing-2026/vllm`
  `adapt/dspine-v033-main-20261111` (`fccf27a0`) with the device-validation
  patches from `wangzhao-11a/vllm` `adapt/dspine-dev-validation-20261111`.
- Ascend: vLLM-Ascend main `1544fe25` + this branch.
- Target `Qwen/Qwen3.8-27B` (bf16, 64 layers, Qwen3.5 hybrid GDN) and draft
  `tencent/Qwen3.8-27B-DSpine-b16`; greedy draft, eager, target TP=2,
  draft TP=1, `max_model_len=2048`, `max_num_seqs=2`, one request at a time.
- Two 910B4-1 chips, `VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS=1`,
  `VLLM_DSPINE_ALLOW_SLIDING=1`, `VLLM_DSPINE_FREE_DRAFT_IO=1`.

| run | wall | throughput | acceptance length |
|---|---:|---:|---:|
| no speculation | 139.13 s | 3.7 tok/s | — |
| DSpine b16 | 125.83 s | 4.1 tok/s | 4.26 |

8 real prompts x 64 tokens, batch 1. Per-position acceptance starts
0.71 / 0.51 / 0.43 / 0.31 / 0.26 ...; DSpine is 1.11x over no speculation.

## Why TP=2 needs two patches

- Core (`vllm/v1/worker/gpu/spec_decode/dspine/speculator.py`): the speculator
  rejected any sharded target head. It now accepts clean TP shards and
  all-gathers the full head only for the serial-table build; candidate top-k
  keeps using the sharded head and Core's `get_top_k_tokens` reduces across TP.
- Ascend (`worker/v2/spec_decode/dspine/speculator.py`): the TP==1 guard is
  removed; PP/PCP/DCP must still be 1.

`draft_tensor_parallel_size` stays 1 (Core requires it); only the target is
sharded. The draft bridge and serial tables are replicated on both ranks.

## Launch

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS=1
export VLLM_DSPINE_ALLOW_SLIDING=1
export VLLM_DSPINE_FREE_DRAFT_IO=1
vllm serve "$TARGET_MODEL" --tensor-parallel-size 2 --enforce-eager \
  --speculative-config '{"method":"dspine","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":15,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","enforce_eager":true}'
```

See `tools/dspine/serve_dspine_mrv2_tp2.sh` for the full flag set.