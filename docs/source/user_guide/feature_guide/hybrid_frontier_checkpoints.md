# Hybrid frontier checkpoints (experimental)

Retain the recurrent state at a finished request's exact computed end so a
continuation can reuse it instead of replaying from a block boundary.

This integration requires the matching vLLM Core branch
`feat/flashstates-native-20261117-c` at
[`ca09945c1bb89659c845ebeaca97409d306cccf1`](https://github.com/Liuchenbing-2026/vllm/commit/ca09945c1bb89659c845ebeaca97409d306cccf1), based on
`ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`. The unmodified nightly Core does
not contain the checkpoint manager. Install both components together.

Core owns indexing, allocation, scratch lifetime and eviction. Ascend keeps
its native page layout and uses the existing segmented copy implementation.
There is no additional process-wide Scheduler or BlockPool monkey patch.

Enable with prefix caching, `--mamba-cache-mode align`,
`--prefix-cache-retention-interval 0` and `--mamba-frontier-checkpoint`.
The feature defaults to off. Use Model Runner V2 and the default scheduler.
Custom recompute, DynTra and balance schedulers are rejected
because their schedule overrides bypass the Core lifecycle.

Speculative decoding and context parallelism are unsupported. External KV
store recovery and GDS performance are outside the initial Ascend validation
scope. CPU tests establish logical contracts, not device copy completion,
model quality or performance. Validate the final paired revisions on NPU
before deployment; do not attribute measurements from older branches to this
implementation.
