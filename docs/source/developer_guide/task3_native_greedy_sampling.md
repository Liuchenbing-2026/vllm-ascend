# Task 3 native greedy sampling experiment

This experiment extends the PA delivery at `68b3478` with an opt-in
`additional_config.enable_native_greedy_sampling` switch, defaulting to false.
It avoids materializing full-vocabulary FP32 logits for plain greedy requests.
BF16 and FP16 values are exactly representable in FP32, so converting them does
not change their ordering. The NPU tests also check tied maxima and NaNs.

The fast path requires no penalties, allowed-token mask, bad words, requested
logprobs, thinking-budget holder, or bonus-token prediction. Active builtin
logit-bias/min-token processors and all unknown processors use the inherited
implementation. Exact builtin types with empty CPU state are recognized because
the engine retains these processors even when no requests use them. Subclasses
always fall back. Outputs retain the parent's int32 `[batch, 1]` contract.

On the pinned Ascend 910B4-1 stack, 171 sample/config unit tests passed. The actual
candidate forward matched its parent in 32 synthetic full-vocabulary NPU cases:
BF16/FP16, batches 1/8/256/352, changed values, ties, NaNs and allowed-token
fallback. These checks are not an OCR ground-truth evaluation. At BF16 B256,
the standalone complete reduction measured 378.52 to 148.79 microseconds;
this is not end-to-end throughput. Service acceptance is tracked separately in
the task archive, and the switch must not be enabled on that basis alone.

Run the CPU suite with matching CANN driver libraries mounted read-only and
without NPU devices:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q \
  tests/ut/sample/ tests/ut/test_ascend_config.py
```
