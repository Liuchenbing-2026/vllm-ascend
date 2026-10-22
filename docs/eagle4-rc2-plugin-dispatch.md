# EAGLE-4 external plugin dispatch on rc2

This development checkpoint keeps models and the EAGLE algorithm in the external
plugin. It lets the Ascend factory instantiate that plugin's registered NPU class
and refreshes runner hooks after replacing the parent-created speculator.

| Component | Fixed identity |
| --- | --- |
| Image | `quay.io/ascend/vllm-ascend:v0.30.0rc2` |
| ARM64 digest | `sha256:7859b75177884252d2b703eff727831ccab4e33a8bde0a9b4ac7563facc98b33` |
| Ascend base | `8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| Task branch | `feat/eagle4-rc2-plugin-dispatch-20261022` |
| vLLM base | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Required vLLM interface checkpoint | `9ecbe2903bd567d0eca7eee0669f0294196ef685` |
| vLLM task branch | `feat/eagle4-rc2-engine-hooks-20261022` |
| External plugin branch | `feat/eagle4-ascend-portability-20261022` |
| Runtime in the image | CANN 9.2.0, torch 2.10.0+cpu, torch-npu 2.10.0.post7 |

Record `git rev-parse HEAD` for this repository and the external plugin before
packaging. A branch name is not a fixed runtime identity.

## What changed

- `worker/v2/spec_decode/__init__.py`: registered methods precede built-in
  EAGLE detection. Otherwise a plugin declaring target-hidden-state use would
  silently become the stock EAGLE speculator.
- `worker/v2/model_runner.py`: refresh the two per-forward hooks from the actual
  replacement speculator. The parent runner owns their invocation order.
- Unregistered methods preserve the old factory branches. A vLLM without the
  registry accessor keeps the existing behavior.

The registered class must provide the correct hardware implementation; this
factory does not turn an arbitrary GPU class into an NPU class. The external
plugin selects its Ascend implementation. At this checkpoint the plugin rejects
NPU tree mode explicitly. Tree attention, state compaction and numerical
validation remain work in progress.

## Executed verification and limits

```bash
PYTHONPATH="$PWD:../vllm" ../vllm/.venv/bin/python -m pytest \
  --noconftest tests/ut/worker/v2/test_spec_decode.py -q
```

12 CPU cases passed through the real imported factory. Concrete hardware
speculator constructors are test doubles. This does not validate the real NPU
class, runner construction, model inference or hardware arithmetic.

Changed-file pre-commit manual checks passed. `bash format.sh ci` was also run:
it reported formatting changes in 34 unrelated baseline files and a missing
local `shellcheck`. Those unrelated automatic edits were preserved as private
failure evidence and reverted. Full repository lint/CI is not claimed green.

No model-service startup, full-quality, load-test or stability command has been
validated for this checkpoint. The earlier 14 passing NPU candidate primitives
belong to the separate plugin checkpoint and do not prove this dispatch works.

Next: import the actual Ascend/plugin modules in an isolated CPU-only rc2 image,
assert the real inheritance/call order without device allocation, then implement
the remaining NPU algorithm paths and request a separate device window. Keep
model accuracy, sampling behavior, graph combinations, HCCL and performance as
separate acceptance gates. Do not deploy this as a completed EAGLE-4 port.
