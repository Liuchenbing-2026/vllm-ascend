# DeepSeek V4.1 LoRA

This implementation enables language-model LoRA projections for the existing
DeepSeek V4.1 model on Ascend. It refreshes cached attention projections,
uses the upstream replicated-linear wrapper, and adds W4A8 per-channel expert
updates at floating-point activation boundaries.

## Configuration and scope

Use matching adapter module names, dimensions and rank. HF `.attn.` names are
mapped to `self_attn`; text-backbone names also receive the language-model
prefix. V4-specific target mapping is not included in this branch. Vision and connector adapters are excluded.

- LoRA with expert parallelism uses AllGather when graph mode is enabled and
  AlltoAll with eager execution. MC2 and fused MC2 are not supported.
- `wo_a` adapters require one local output-projection group per rank and no
  fine-grained output-projection TP.
- W4A8 requires per-channel weight scales. Dynamic EPLB and antiquant offsets
  are unsupported by this path.
- Loaded adapter slots and active request slots are distinct. With DP=1,
  inactive expert adapters do not select the LoRA expert path. With EP and
  DP>1, expert execution is conservative because remote ranks can send adapter
  tokens; this combination still requires device validation.
- This change does not add an MXFP4 backend or modify global NZ format policy.

## Example launch

Set `WEIGHTS_DIR` and `ADAPTER_ROOT` to compatible local artifacts. This command
is a reproduction template, not a claim of device validation of this branch.

```bash
vllm serve "${WEIGHTS_DIR}" \
  --served-model-name deepseek-lora \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --enable-lora --max-loras 1 --max-lora-rank 8 \
  --lora-modules adapter="${ADAPTER_ROOT}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_fused_mc2":0}'
```

For eager validation, replace the compilation configuration with
`--enforce-eager`. Keep model, prompts, adapter and sampling configuration fixed.
Compare base requests, a zero adapter and a non-zero adapter within each mode.
Verify loaded targets and logprob/output deltas; startup or graph capture alone
is not proof of an effective adapter. Numerical agreement with a reference
implementation requires a separate accuracy check.

## Regression coverage

CPU tests cover projection rebinding, target mapping, W4A8 boundary injection,
dispatch selection, upstream request-ID conversion, active-slot selection and
expert-index boundaries. These tests do not establish NPU correctness or
performance; real-weight A3 and distributed validation remain required.
