# FlashStates on Ascend — branch status

Branch: `flashstate-ascend` (based on vllm-ascend `main`, vLLM v0.30.0 build).

This branch contains **two tracks**. Only the first reproduced the measured
gains; the second is a work-in-progress runtime-patch port.

## Track A — validated: vLLM v0.32.0 core (what produced the numbers)

`patches/flashstate_v032/` is a 3-commit patch series against vLLM **v0.32.0**
(`5df63a7`): the `hybrid_checkpoint` package, scheduler/KV-cache/block-pool
wiring, MRV2 input-batch field, and Ascend manager classification
(`KpoolTailManager` skip).

It was validated on 8×910B4 with
`quay.io/ascend/vllm-ascend:nightly-releases-v0.32.0rc` and
`/data02/GLM-5.3-Flash-w8a8-b0829` (W8A8, TP8):

- multi-turn 4 sessions × 3 turns × 128 output: later-turn latency
  **33.53s → 31.72s (−5.4%)**, frontier hits **2,284 tokens**;
- `vllm bench serve` prefix_repetition 16K input / 1K output / 8 concurrency:
  TTFT **8.05s → 3.10s (−62%)**, E2EL **262.02s → 255.96s (−2.4%)**,
  output throughput **31.22 → 31.97 tok/s**.

Reproduction commands, raw outputs and the 30K-context OOM note are in
[`benchmarks/flashstate/README.md`](benchmarks/flashstate/README.md) and
`benchmarks/flashstate/results/`.

**30K context is not feasible on this hardware**: 32K/8 exits 137 during
startup (38.31 GB weights per card + 32K KV in 64 GB HBM). 20K is the largest
`max-model-len` that starts; the benchmark uses 20,480.

## Track B — WIP: vllm-ascend runtime patch (main / vLLM 0.30)

`vllm_ascend/hybrid_checkpoint/` plus
`vllm_ascend/patch/platform/patch_hybrid_checkpoint.py` apply the same logic as
a runtime monkey-patch for vllm-ascend `main` (vLLM v0.30.0), gated by:

```bash
--additional-config '{"flashstate": {"enable_frontier": true}}'
```

Status: it imports on vLLM 0.30 and installs the BlockPool / KVCacheManager /
Scheduler lifecycle hooks, but it has **not** been runtime-validated on NPU and
does **not** produce the numbers above. The 0.30 `get_computed_blocks` /
coordinator API differs from 0.32, so the frontier hit extension is
best-effort. Do not use this track to reproduce the gains yet.

## Next steps

1. Finish Track B: port the full `KVCacheManager.get_computed_blocks` and
   `Scheduler.schedule` frontier hooks to vLLM 0.30, run the CPU tests and an
   NPU smoke test on `nightly-main`.
2. Or push Track A to the paired vLLM fork (`Liuchenbing-2026/vllm`), which
   currently has no write access from `wangzhao-11a`.
3. Add a fixed-length (`ignore_eos`) and longer-output (256–512) benchmark to
   show the frontier tail beyond one block.