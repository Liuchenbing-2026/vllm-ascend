# DSpine on Model Runner V2

This integration connects the Core DSpine speculator to the Ascend MRV2 DFlash
attention, input preparation, candidate walk, and ACL Graph hooks. It preserves
the target model's compilation configuration instead of forcing Model Runner V1
or eager execution. The DSpine model definition and bridge scoring remain in the
paired vLLM checkout.

## Required versions and scope

- vLLM v0.33.0 base commit
  `4ba787ec9959e566d3b587a796dff07673324da5`, plus the DSpine Core change
  delivered in the paired vLLM PR. Pass that PR's head commit as
  `VLLM_COMMIT` when building the image.
- vLLM main2main verified commit
  `24420b7a341af37a9bc3186e6fc6062512e0aa58`.
- vLLM-Ascend main
  `c251a6a5ed5cc10f7ec6daa5b87d410d731bf6fd`, plus this DSpine change.
- `VLLM_USE_V2_MODEL_RUNNER=1` is mandatory; this integration targets MRV2 and
  does not enable or validate a Model Runner V1 path.
- Target TP and PP are not artificially restricted to one. The target-parallel
  configuration in the tests is TP8/PP4.
- Draft `draft_tensor_parallel_size` must be `1`.
- Greedy and probabilistic draft sampling are accepted. Ascend currently fails
  closed when `use_fp64_gumbel=True`.
- A full unquantized target embedding and BF16/FP32 row-contiguous LM head are
  required. By default, `num_speculative_tokens` is the checkpoint block size
  minus one. Non-default runtime windows are accepted for memory or latency
  experiments, but log a warning and require acceptance and output-accuracy
  validation.

The v0.33 main2main change
`760d7ecbb962135fff738726825cffacc74f5d2f` updates every Dockerfile to
`VLLM_TAG=v0.33.0` while leaving `VLLM_COMMIT=""`. The annotated tag
`v0.33.0` is `3a4d82f569a9f57054ba81633e9b6c73b34b51fe` and peels to
`4ba787ec9959e566d3b587a796dff07673324da5`, so the default Docker build checks
out that release commit. `.github/vllm-main-verified.commit`
(`24420b7a341af37a9bc3186e6fc6062512e0aa58`) is the separately verified main
revision and is not the Dockerfile default.

The A5 nightly image
`swr.cn-north-12.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend:nightly-ci-main-a5`
was resolved to registry index
`sha256:3146eb9ae8b32b00c307c0c56548d9bb8ab057bae986743b1060c945e7aad7ae`
(amd64) and `sha256:ef89d685807ec1e2da1f8d566eeb5a4c565056d34abb381a9afd6d25bd49a089`
(arm64). The image manifest exists, but its OBS-backed layers were not available
to this environment. Therefore the Dockerfile default and expected vLLM commit
are verified; the actual `/vllm-workspace/vllm` HEAD inside the nightly image
remains unverified.

## Reproducible image build

The Dockerfile gives a nonempty `VLLM_COMMIT` precedence over `VLLM_TAG`.
Select the paired Core delivery commit explicitly so the image contains the
DSpine model, bridge configuration, and MRV2 speculator:

```bash
VLLM_COMMIT=<head-commit-of-paired-vllm-pr>
docker build -f Dockerfile.a5 \
  --build-arg VLLM_REPO=https://github.com/Liuchenbing-2026/vllm.git \
  --build-arg VLLM_TAG=v0.33.0 \
  --build-arg VLLM_COMMIT="$VLLM_COMMIT" \
  -t dspine-mrv2:v033 .
```

This image build has not been executed as part of the code validation described
below.

## Target-parallel graph recipe

The following is a validation recipe, **not a recorded successful NPU run**.
Obtain a device allocation, pin both model revisions, and set the two local
model paths first. The draft remains TP1 while the target uses TP8/PP4 and
full-decode ACL Graph.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve "$TARGET_MODEL" \
  --served-model-name dspine-mrv2 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 4 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 8352 \
  --speculative-config '{"method":"dspine","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":15,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","enforce_eager":false}'
```

Use `"method":"dspine"` as the public explicit entry. It validates the DSpine
checkpoint and dispatches the DSpine speculator, which reuses the shared DFlash
candidate walk, draft-logit cache, and Ascend attention/graph hooks. The
`"method":"dflash"` form remains accepted for backward compatibility when the
draft checkpoint carries `selected_prefix_code_bridge`.

The intended first pair is `Qwen/Qwen3-8B` and
`tencent/Qwen3-8B-DSpine-b16`. Verify their exact revisions/configurations
before running; the command assumes one anchor plus 15 proposal rows.
`num_speculative_tokens` defaults to the checkpoint block size minus one. An
explicit runtime width is also accepted for memory or latency tuning, but a
non-default width logs a warning because the checkpoint contains the full
proposal window.

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

The CPU tests validate the shared bridge math against float64 references,
checkpoint validation, public config normalization, MRV2 dispatch, loader
ordering, target TP8/PP4 plus draft TP1 configuration, ACL Graph configuration
preservation, and greedy/probabilistic sampling acceptance. The Ascend CPU
tests also verify that DSpine reuses Core proposal/scoring and the Ascend DFlash
attention/graph hooks.

```bash
# In the paired vLLM source, using an existing remote CPU environment:
python -m pytest tests/v1/spec_decode/test_dspine.py \
  tests/transformers_utils/test_speculators_dspine_config.py
python -m pytest tests/config/test_speculative_draft_hf_overrides.py -k dspine
# In vLLM-Ascend, with the repository CPU mock setup:
python -m pytest tests/ut/spec_decode/test_dspine_v2.py \
  tests/ut/worker/test_model_runner_v2.py::test_init_spec_pp_full_graph_and_speculator
```

| Validation item | Status |
| --- | --- |
| Core DSpine CPU unit tests | 52 passed, 2 skipped |
| Core DSpine config selection CPU tests | 7 passed |
| DSpine Speculators config CPU tests | 4 passed |
| vLLM-Ascend MRV2 CPU unit tests | 15 passed |
| Remote CI | Not triggered |
| NPU single-card execution | Not run |
| TP8/PP4 with Model Runner V2 | Not run |
| ACL Graph capture/replay | Not run |
| Real target and draft weights | Not run |
| Output accuracy | Not evaluated |
| TTFT, TPOT/ITL, throughput, and acceptance rate | Not measured |

These results were collected in a CPU/mock environment. The CPU validation
used `torch_npu` 2.10.0.post4, while this branch declares
`torch_npu==2.10.0.post7`; the target NPU dependency combination was not
executed. CPU tests do not execute NPU attention, the Triton candidate walk,
cache updates, ACL Graph capture/replay, real weights, output quality,
stability, or performance measurement.
