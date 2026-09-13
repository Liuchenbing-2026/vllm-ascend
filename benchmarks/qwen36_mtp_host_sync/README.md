# Qwen3.6-35B-A3B + MTP: the two host syncs

Reproduction material for the two commits on this branch. It answers one
question: why does turning on MTP k=1 make Qwen3.6-35B-A3B *slower* than not
speculating at all, when its acceptance length is 1.8?

## The short version

It is not the drafter, and it is not the acceptance rate. Two `synchronize()`
calls in `_prepare_inputs` both wait on the same device milestone -- step N's
sampling -- and both run only on the speculative path. Under async scheduling
the host is a full step ahead of the device, so that milestone has not happened
yet and the host blocks at the *top* of every step. Nothing overlaps.

Without speculation the host also blocks, but inside `forward` -- *after* it has
issued its prep work. Same wall-clock wait, completely different consequence.

## Measured

Qwen3.6-35B-A3B, 910B4 x4, TP4 + EP, async scheduling,
`FULL_DECODE_ONLY`, MTP k=1, C64, 512-in / 2048-out, 128 prompts.
All four cells in one run of `run_ab.sh`, ms per decode step:

| cell | tok/s | vs nospec | TPOT | step | prolog | acc_sync | seq_sync | fwd (host) |
|---|---|---|---|---|---|---|---|---|
| nospec | 1881.79 | — | 33.50 | 33.4 | 6.87 | — | — | 23.7 |
| MTP, unpatched | 1835.57 | −2.5% | 33.64 | 55.03 | 41.50 | 28.92 | ~0 | 3.35 |
| MTP + device gather | 1848.57 | −1.8% | 32.72 | 54.01 | 40.49 | 0.34 | 28.66 | 3.35 |
| MTP + both | **2027.31** | **+7.7%** | 30.18 | 48.89 | 12.35 | 0.36 | off | 25.58 |

The third row is the point of the experiment: removing the first sync moves
28.58 ms into the second one and buys almost nothing. Only removing both wins.
In the last row `fwd` host time goes back up to 25.58 ms, i.e. the host is
waiting on the dispatch queue again -- the device-bound state nospec is in.

Acceptance length with both patches is 1.806 (145084 drafted / 116988
accepted), inside the 1.79-1.83 band from eight earlier runs. That is the
evidence the device gather did not corrupt GDN state selection.

Baseline drift across rounds was ±1% (1880.38 / 1900.91 / 1881.79 tok/s), and a
fixed-length 2-generation benchmark systematically penalises speculative
decoding through wave-tail dispersion. Read +7.7% as a direction and a
magnitude, not a constant.

## What was ruled out first

Each of these was a plausible story that measurement killed:

| hypothesis | verdict |
|---|---|
| drafter costs ~20 ms/step | refuted: `draft_dev` is 1.3-3.5 ms. The 20 ms came from a `nospec@2n` proxy that used 2n *requests* to stand in for n, over-counting KV reads and charging the excess to the drafter. |
| prefill got much more expensive | refuted: bucketed timing puts prefill steps at 307.40 -> 324.26 ms, +5.5%. The +28%/+99% figures came from a residual method (`total − steps × step`). |
| `prepare_inputs_event.synchronize()` | refuted: 0.03 ms, identical to nospec. |
| per-step pinned allocations (4x `torch.tensor(pin_memory=True)`) | refuted: 0.64 ms. |
| "it is one sync" | refuted by the third cell above. |

## Files

- `segment_timing_patch.py` — applies the instrumentation (and the two fixes) to
  a pristine checkout. Modes: `probe` (timers only), `A` (device gather),
  `AB` (both). Every edit is anchored and asserted; a miss aborts.
  Timers: `prolog / sync_ip / upd_st / prep_in / acc_sync / seq_sync / fwd /
  sample / bookkeep / draft`, plus `fwd_dev` / `draft_dev` from NPU events read
  through `query()` rather than `synchronize()`, so reading them never stalls
  the host and never perturbs what is being measured.
- `run_ab.sh` — the four-cell harness. Expects the model at
  `/models/Qwen3.6-35B-A3B` and `segment_timing_patch.py` at
  `/work/scripts/patch.py`. Takes a `mkdir` lock, restores from
  `model_runner_v1.py.ntorig` between cells, warms up before every measured run
  (a cold server runs at ~68% of warm), and reports `vllm:spec_decode` counters
  around each benchmark.
- `greedy_digest.py` — temperature-0 digest over 16 fixed prompts sent
  concurrently. **Its verdict is void on this stack**: HCCL all-reduce under
  TP4 + EP is non-deterministic across calls, and three consecutive runs of the
  same unpatched server produced three different digests even with
  `HCCL_DETERMINISTIC=true`. Kept because the harness calls it and because it is
  still a usable smoke test on a deterministic configuration. To check
  equivalence here, use `prompt_logprobs` against a matched control, or a task
  benchmark.

## Still open

The second fix is lossy by construction and ships off by default
(`VLLM_ASCEND_MTP_SKIP_SEQ_LENS_CORRECTION`). Its accuracy cost has not been
measured; a task benchmark is the only instrument left that can settle it.
