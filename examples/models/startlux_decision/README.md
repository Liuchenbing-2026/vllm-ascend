# StartLux native text decisions on Ascend

This recipe uses the native vLLM decision model and `/v1/decisions` strategy.
It requires paired vLLM and vLLM-Ascend checkouts containing the StartLux model,
its typed decision strategy, and the appropriate Ascend encoder attention path.
The legacy standalone `serve_startlux_decision.py` example is not required.

## Paired code and validation scope

The model and typed request strategy belong to vLLM. This recipe adds no model
registration or attention implementation to the Ascend plugin. The current
standard attention backend already routes encoder-only pooling groups through
its encoder path even when shared metadata is causal.

The fixed experimental pair used for the v1 dense 4B HTTP smoke test was:

- vLLM [2f76c9845517a247a6741c68dadb13246d2472f1](https://github.com/Liuchenbing-2026/vllm/commit/2f76c9845517a247a6741c68dadb13246d2472f1).
- vLLM-Ascend [97d3b33d7694362914e4f508e8472ac8b694502c](https://github.com/Liuchenbing-2026/vllm-ascend/commit/97d3b33d7694362914e4f508e8472ac8b694502c).

These include compatibility changes outside this recipe. They are not the
unmodified official verified pair. The smoke test used installed candidate
wheels and real checkpoint weights on an Ascend 910B4-1, with Torch 2.10.0+cpu,
torch_npu 2.10.0.post7 and CANN 9.2.1. It exercised text choice, predicate and
score requests through the native endpoint. It did not run these launchers on
this latest main snapshot or validate v1.1 Instant, MoE, images or AutoThink.

Before deployment, inspect the selected Ascend revision's Dockerfile, image
build workflow and `.github/vllm-main-verified.commit`; record the installed
commits and image digest. Do not combine independent latest main revisions
or treat the historical v0.30.0rc2 recipe as validation of this pair.

## Start the service

Keep `decision_config.json`, the tokenizer, and the official chat template with
the local checkpoint. Activate the environment containing the paired packages.

```bash
export MODEL_PATH='<WEIGHTS_DIR>'
export ASCEND_RT_VISIBLE_DEVICES='<ASSIGNED_NPU_IDS>'
export PYTHON_BIN='<ENVIRONMENT_DIR>/bin/python'
export TENSOR_PARALLEL_SIZE=1
export MAX_MODEL_LEN=8192
bash examples/models/startlux_decision/serve.sh --enforce-eager
```

Use the same launcher for v1 dense checkpoints. For the v1 MoE checkpoint,
select the corresponding architecture explicitly:

```bash
bash examples/models/startlux_decision/serve.sh --enforce-eager \
  --hf-overrides '{"architectures":["StartLuxDecisionMoeForSequenceClassification"],"num_labels":26}'
```

The launcher binds localhost and preserves the assigned device selection.
`PORT`, `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and `PYTHON_BIN` configure the
service. Both launchers disable prefix caching and chunked prefill and set the
batch token budget to the full prompt budget. This conservative recipe also
works for causal v1 prompts; it does not claim a v1 prefix-cache optimization.

Call the native `/v1/decisions` endpoint using its choice, predicate and score
schema. `/classify` returns raw candidate logits, not calibrated probabilities.
The old independent `/v1/systemone` gateway has a different deployment contract.

## Validation boundaries

This recipe targets text-only, single-pass reads. It does not implement
AutoThink, images, or long-running service acceptance. Launcher tests verify
argument forwarding and reject partial-prompt scheduling overrides; they do
not prove that a real checkpoint started or passed a model benchmark.
Pin and record the actual paired commits and installed artifacts before model
validation; a package version string or historical rc2 result is insufficient.
