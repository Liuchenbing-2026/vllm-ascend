# StartLux native text decisions on Ascend

This recipe uses the native vLLM decision model and `/v1/decisions` strategy.
It requires paired vLLM and vLLM-Ascend checkouts containing the StartLux model,
its typed decision strategy, and the appropriate Ascend encoder attention path.
The legacy standalone `serve_startlux_decision.py` example is not required.

## Start the service

Keep `decision_config.json`, the tokenizer, and the official chat template with
the local checkpoint. Activate the environment containing the paired packages.

```bash
export MODEL_PATH=/models/StartLux-Decision-v1.1-4B-Instant
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHON_BIN=/path/to/paired/environment/bin/python
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
