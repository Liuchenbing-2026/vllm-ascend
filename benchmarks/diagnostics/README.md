# Experimental DSpark TND device-tiling validation

This branch adds an optional device-length attention path to the fixed
`f06916fc3f9aa8cabe9b0edc5e74a40170a85b72` baseline. It is an experiment,
not a claim that the original review comments or merge gates are resolved.

The normal opt-in remains `enable_dspark_draft_kv_optimistic_bound`.
When a matching FIA sink extension is registered, eligible eager draft calls
use TND query tensors and device INT64 sequence-length snapshots. A per-build
cache shares the device tiling across layers. Native torch-npu FIA v2's
host-list interface is not used for these lengths. Missing operators and
unsupported layouts retain the original optimized/exact fallback. Kernel
errors are surfaced rather than hidden by a fallback. Graph capture and
noncausal sliding-window handling are unchanged.

The original BSND implementation is deliberately retained for comparison until
the replacement passes precision and serving validation. This intermediate
branch does not yet reduce the production implementation's code size.

**Observed limitation:** the pinned dependency's `CheckGqaDSupport` excludes
equal Q/K/V head size 256. The initial A2 probe reproduced a tiling failure at
D=256; the integration now returns to the existing path before launching that
unsupported call. The deployed Qwen3 DSpark draft has head dimension 256.
Consequently this branch is not a TND replacement for that model and cannot
claim new Qwen serving gains. Equal head sizes 64/128/192 are admitted by the
dependency, but only D=128 was device-validated in this experiment.

## Separate compiled dependency

The experiment uses the unmodified C++/AICPU source tree from
[`vllm-project/vllm-ascend@bec19b8dea1e538e64838f3064f7e91a3fa9ee72`](https://github.com/vllm-project/vllm-ascend/commit/bec19b8dea1e538e64838f3064f7e91a3fa9ee72).
Its two required operators are `fused_infer_attention_score_v2_sink` and
`fused_infer_attention_score_v2_sink_metadata`. These are **not** supplied by
the baseline image's native torch-npu API. The source archive, its licenses,
and compiled package remain separate from this Python integration patch.

Build the dependency in an isolated source directory with CANN 9.1.0,
torch 2.10.0+cpu and torch-npu 2.10.0.post4, on an A2 build host:

```bash
git clone https://github.com/vllm-project/vllm-ascend.git sink-source
git -C sink-source fetch origin bec19b8dea1e538e64838f3064f7e91a3fa9ee72
git -C sink-source checkout --detach bec19b8dea1e538e64838f3064f7e91a3fa9ee72
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd sink-source/csrc
MAKEFLAGS=-j8 CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=8 bash build.sh \
  --pkg --soc=ascend910b --vendor_name=dspark_tnd \
  --ops=fused_infer_attention_score_v2_sink,fused_infer_attention_score_v2_sink_metadata -j8
```

The actual experiment used `git archive` of the same pinned revision instead
of installing/replacing any existing vLLM checkout. For a standalone Torch
binding, run from this candidate checkout with absolute dependency/build paths:

```bash
MAX_JOBS=2 python benchmarks/diagnostics/build_fia_sink_probe.py \
  --source /absolute/path/to/sink-source \
  --build /absolute/path/to/probe-binding \
  --cann /usr/local/Ascend/ascend-toolkit/latest
```

The resulting shared library registers only the dependency's two operators.
It is for the standalone probe; it does not replace the serving extension.
Export the built package's vendor directory through `ASCEND_CUSTOM_OPP_PATH`
and its library directories through `LD_LIBRARY_PATH` before loading it.

## Validation commands

```bash
# CPU routing/cache checks; Python >= 3.11 for unittest.enterContext.
python tests/ut/ops/test_draft_tnd.py
# Check physical-card ownership and container mapping before allocating.
npu-smi info
npu-smi info -m
# Device 0 below is a container-local ID, not necessarily physical card 0.
python benchmarks/diagnostics/probe_draft_tnd.py \
  --library /absolute/path/to/probe-binding/dspark_fia_sink_probe.so \
  --device 0 --head-dims 128 --output /absolute/path/to/result128.json
python benchmarks/diagnostics/profile_draft_tnd.py \
  --library /absolute/path/to/probe-binding/dspark_fia_sink_probe.so \
  --device 0 --output /absolute/path/to/new-profile-directory
```

The device probe requires the new path to run: a fallback is a failure. It
uses random BF16 tensors, FP32 attention references, paged/interleaved KV,
ragged queries, changed live lengths, causal/window modes, and NaN/Inf tails.
The inherited relative-L2 threshold remains 0.02; cache storage must be
unchanged with zero numeric tolerance (NaNs compare equal). These checks do not establish model
quality, graph correctness, TP4 throughput, or the absence of synchronization
inside the C++/CANN implementation. Those need separate validation.

## A2 results, 2026-10-08

- Matching CANN package and standalone Torch binding built successfully.
- Seven CPU routing/cache/fallback tests passed, including the D=256 guard.
- D=128: 36 parameter combinations, each with two live-length steps, passed
  the unchanged 0.02 relative-L2 threshold and cache-preservation checks.
- D=256: the initial unguarded candidate failed at CANN tiling; the final
  guard preserves the original implementation. It does not add D=256 support.
- A separate 3-second D=128 trace contained 4,041 candidate calls, no scalar
  extraction, no per-call stream synchronization, and two device synchronizes
  at capture boundaries. Parsed kernel details contain 47 columns.
- The standalone one-layer-plus-metadata timing was 0.5383 ms per call. It is
  not a serving metric or proof of a speedup; the service can amortize metadata
  across layers, which this timing intentionally does not do.
- Model quality, TP4 serving performance and graph replay were not tested.
  The original PR was not updated. Full lint was attempted; local gitleaks
  and shellcheck dependencies were unavailable. Other executed lint checks
  passed; this does not establish upstream CI or merge approval.

The default device probe still includes D=256 and intentionally reports the
unsupported candidate as a failure. Selecting `--head-dims 128` explicitly
narrows the diagnostic scope and must not be reported as full-model coverage.
