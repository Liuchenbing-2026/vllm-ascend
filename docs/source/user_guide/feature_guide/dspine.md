# DSpine on Model Runner V2 (development candidate)

This candidate connects the Core DSpine speculator to the Ascend MRV2 DFlash
attention, input preparation, and candidate-walk implementation. It does not
fall back to Model Runner V1. The DSpine model definition and bridge scoring
remain in the paired vLLM checkout.

## Required versions and scope

- vLLM v0.32.0 (`5df63a7203cdb9328d824a27b682e77c7b9928be`) plus the
  `adapt/dspine-v032-main-20261111` branch in `Liuchenbing-2026/vllm`.
- vLLM-Ascend main (`e6e554efcc3eda48364ad5bbd0e6a055befcbcb1`) plus the
  same task branch in `Liuchenbing-2026/vllm-ascend`.
- Draft eager execution and greedy draft sampling. TP, PP, PCP, and DCP must
  each equal one. A full unquantized target embedding and BF16/FP32 LM head
  are required. The draft block size must match its checkpoint.

The Ascend main base normally pairs with a newer Core commit. This branch
includes narrow imports/planner dispatch compatibility for v0.32; it does not
backport newer Step attention to v0.32 or establish compatibility for all models.

## Planned device smoke command

The following is a validation recipe, **not a recorded successful NPU run**.
Obtain a device allocation, pin both model revisions, install/build both task
branches in an approved Ascend image, and set the two local model paths first.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve "$TARGET_MODEL" \
  --served-model-name dspine-mrv2 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --port 8352 \
  --speculative-config '{"method":"dspine","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":15,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","enforce_eager":true}'
```

The intended first pair is `Qwen/Qwen3-8B` and
`tencent/Qwen3-8B-DSpine-b16`. Verify their exact revisions/configurations
before running; the command assumes one anchor plus 15 proposal rows.
Global `--enforce-eager` is only for the initial smoke. Removing it later
allows target graph testing while keeping the draft eager.

```bash
vllm bench serve --backend vllm --model dspine-mrv2 \
  --base-url http://127.0.0.1:8352 --endpoint /v1/completions \
  --tokenizer "$TARGET_MODEL" --dataset-name random \
  --random-input-len 512 --random-output-len 256 \
  --num-prompts 100 --max-concurrency 1
```

Random-input benchmarking is not a quality evaluation. Compare real fixed
prompts with the same target/precision/greedy parameters, then run the target's
full quality suite and long-running mixed-batch/preemption tests. Record
accepted draft tokens per proposed token separately from accepted length
including the bonus token and end-to-end speed.

## Validation boundary

CPU tests check the shared bridge math against float64 references, checkpoint
validation, public config normalization, MRV2 dispatch, loader ordering, and
paired worker imports. Ascend CPU tests use the repository's device mocks.
They do not execute NPU attention, Triton candidate walk, cache updates, graph
replay, real weights, quality evaluation, or performance measurement.

```bash
# In the paired vLLM source, using an existing remote CPU environment:
python -m pytest tests/v1/spec_decode/test_dspine.py \
  tests/transformers_utils/test_speculators_dspine_config.py
python -m pytest tests/config/test_speculative_draft_hf_overrides.py -k dspine
# In vLLM-Ascend, with the repository CPU mock setup:
python -m pytest tests/ut/spec_decode/test_dspine_v2.py
```
