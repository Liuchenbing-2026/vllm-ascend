# Hybrid Frontier Checkpoints (Experimental)

Retain the recurrent state at a finished request's exact computed end so a
continuation can reuse it instead of replaying from a block boundary.

This integration requires the matching vLLM Core hybrid checkpoint support
(`HybridCheckpointManager` and `CacheConfig.mamba_frontier_checkpoint`). Unmodified
upstream vLLM builds that do not yet include the checkpoint manager cannot enable
this feature; install both components together.

vLLM Core owns indexing, allocation, scratch lifetime, and eviction. vLLM Ascend
keeps its native page layout, uses Core's `AscendCircularBufferManager` for
resumable K-pool tails, and reuses the segmented inplace copy implementation.

Enable with prefix caching (`--enable-prefix-caching`, which selects Mamba
`align` mode), `--prefix-cache-retention-interval 0`, and
`--mamba-frontier-checkpoint`.
The feature defaults to off. Use the default scheduler (`Scheduler` /
`AsyncScheduler`); custom recompute, DynTra, balance scheduling, and custom
`scheduler_cls` overrides are rejected because their scheduling paths bypass
Core's checkpoint admission and snapshot hooks.

Speculative decoding, KV cache events, and context parallelism are unsupported.
External KV store recovery and GreedyDual-Size eviction performance on NPU are
outside the initial validation scope. CPU unit tests verify scheduler contracts
and host page views only; verify device copy completion, model accuracy, and
throughput on NPU hardware before deployment.
