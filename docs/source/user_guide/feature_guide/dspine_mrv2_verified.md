# DSpine MRV2 on Ascend: verified recipe

This note records the first successful NPU run of the DSpine MRV2 candidate
(paired Core + Ascend speculator) and the environment gap that had to be worked
around. It complements `dspine.md`, which described the candidate before any
device run.

## Verified pair

- Core: vLLM v0.33.0 baseline plus `Liuchenbing-2026/vllm`
  `adapt/dspine-v033-main-20261111` (`fccf27a0`, adds the Qwen3 DSpine draft
  model and the `dspine` method wiring).
- Ascend: vLLM-Ascend main `1544fe25` plus this branch (the
  `AscendDSpineSpeculator` adapter and the `DSpineDraftModel` routing).
- Models: `Qwen/Qwen3-8B` + `tencent/Qwen3-8B-DSpine-b16`
  (`num_speculative_tokens=15`, greedy draft, eager draft and target).
- Device: one Atlas 910B4-1 (64 GB HBM), single card, TP=PP=PCP=DCP=1.

## Environment gap: missing fused add-rmsnorm-bias

Every available Ascend image ships CANN 9.2.1, whose `libopapi.so` does not
provide `aclnnAddRmsNormBias` / `aclnnAddRmsNormBiasGetWorkspaceSize`. The
Ascend `AscendRMSNorm.forward_oot` fused path calls it for every layer, so the
engine dies at the first forward with:

```
RuntimeError: aclnnAddRmsNormBias or aclnnAddRmsNormBiasGetWorkspaceSize not in libopapi.so
```

This branch adds an env-gated fallback to the plain math path:

```bash
export VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS=1
```

The fallback keeps the same semantics (`x, residual = normed, x + residual`) and
is only a stop-gap: it applies to the target as well, so absolute throughput is
lower than a fused-path build. The proper fix is an image whose CANN provides
the op; nothing else changed.

## Measured result

Same prompt set and client as the v1 lane: 16 real technical prompts x 128
tokens, greedy, batch 1, `max_model_len=1024`, `max_num_seqs=2`, eager, one
910B4-1, with the fallback enabled for both runs.

| MRV2 run | wall time | throughput | acceptance length |
|---|---:|---:|---:|
| no speculation | 195.63 s | 10.5 tok/s | — |
| DSpine b16 | 138.77 s | 14.8 tok/s | 3.37 |

- DSpine vs no-spec on MRV2: **1.41x**.
- Acceptance length 3.37 vs 3.05 for the v1-lane DSpine port on the same
  prompts (the Core tail is slightly different from the v1 implementation).
- Because the fallback slows the target, these absolute tok/s numbers are not
  comparable with the fused-path v1 measurements; compare ratios within a run.

## How it was run

The paired trees are mounted over the image's editable install paths (the
container's `/vllm-workspace/vllm-main` and `/vllm-workspace/vllm-ascend-main`),
because the stock image carries a different Core without the DSpine model:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS=1
vllm serve "$TARGET_MODEL" \
  --served-model-name dspine-mrv2 \
  --dtype bfloat16 --tensor-parallel-size 1 --enforce-eager \
  --max-model-len 1024 --max-num-seqs 2 \
  --speculative-config '{"method":"dspine","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":15,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","enforce_eager":true}'
```

A convenience launcher is `tools/dspine/serve_dspine_mrv2.sh`.

## Remaining work

- Rebuild the image against a CANN that ships `aclnnAddRmsNormBias` so the fused
  norm path (and the normal performance profile) is restored.
- Re-run the benchmark without the fallback and with batch > 1.
- The Core-side DSpine model/tail and the paired `dspine` method still need to
  land in the vLLM release the image is built from.