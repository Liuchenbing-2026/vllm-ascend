# DFlash2 vs MTP benchmark harness (Ascend 910B4, vllm-ascend 0.26)

Harness and evidence for a head-to-head of **DFlash2** against Qwen3.8-27B's **built-in MTP head**,
run so that the numbers can be compared against the published SGLang / LMSYS results rather than
only against each other.

Findings: [RESULTS.md](RESULTS.md). Exact bytes that produced each number: [PROVENANCE.md](PROVENANCE.md).

## Why it is shaped this way

Three measurement traps cost real time here; the harness encodes the fixes so they are not
re-discovered.

**1. The first bench against a fresh service is a warm-up, not a measurement.**
It lands at ~68% of warm throughput (graph capture + Triton JIT). Every sweep therefore opens with a
discarded warm-up round. A single-shot A/B across two freshly started services measures start-up
cost, not the arms.

**2. Run order contaminates a sweep.** Each sweep measures its highest-concurrency cell at *both*
ends and reports the ratio (`order_control_ok`). Numbers are quoted only when the pair agrees within
5%. Two sweeps here failed that check and are labelled as trend-only rather than dropped.

**3. `acceptance_rate` is not comparable across draft depths** — it averages over a different number
of positions at each K. The comparable quantity is
`advance = 1 + accepted_tokens / drafts` (expected tokens produced per iteration), and the win line
is `advance > T_iter_ratio` where `T_iter = TPOT x advance`.

A corollary that cost a wrong conclusion: **advance compares throughput across depths but NOT drafter
quality.** A wider draft trivially emits more tokens. Isolating quality needs matched verify width —
hence the K3-vs-MTP3 (width 4) and K7-vs-MTP7 (width 8) pairs.

## Layout

```
templates/    two-phase sweep templates; __ARM__ is substituted per arm
              *_sgl_*      short GSM8K, 256-token cap, C1/C8/C16/C32
              *_sglthink_* EOS respected, up to 2048 tokens (thinking not truncated)
audits/       card-free read-only probes (drafter cost, weight identity, runtime
              provenance, prompt fidelity, MTP depth feasibility)
drivers/      PowerShell chain runners: publish task, await terminal marker, publish next
```

**Two-phase split is mandatory, not stylistic.** Cold start is ~7.5 min and a C32 bench is ~283 s,
against a ~10 min remote wall clock. Phase A starts a detached service and *leaves it running* (so
being killed at the wall clock is harmless); phase B attaches, benches, and tears down. A phase B
that finds the service not yet ready exits **without** tearing it down, so it can simply be re-run.

## Reproducing

```bash
for arm in nospec mtp3 mtp7 k3 k7; do
  sed "s/__ARM__/$arm/" templates/benchserve_start_sgl_template_m18_20260911.sh > start_$arm.sh
  sed "s/__ARM__/$arm/" templates/benchserve_bench_sgl_template_m18_20260911.sh > bench_$arm.sh
done
# then, per arm: run start_$arm.sh, wait for BENCHSERVE_START_READY, run bench_$arm.sh
```

Paths are absolute and specific to the host they ran on (weights under `/data1`, runtime and evidence
under `/data2/dflash2-v026-official`). Adjust `root`, `model`, `draft`, `image` at the top of each
template.

Each start script gates on the image id and the runtime manifest hash before touching a card, and
refuses to start if the NPUs are busy — it polls and waits rather than taking cards from whoever
holds them. Each bench script tears its own service down and prints the released HBM usage.

## What the arms are

| arm | config | verify width |
|---|---|---|
| `nospec` | no speculative config | 1 |
| `mtp3` | `qwen3_5_mtp`, 3 tokens | 4 |
| `mtp7` | `qwen3_5_mtp`, 7 tokens | 8 |
| `k3` | `dflash`, 3 tokens | 4 |
| `k7` | `dflash`, 7 tokens | 8 |

MTP depth is a **loop count, not a checkpoint property**: Qwen3.8-27B ships exactly one MTP module
(`mtp.layers.0.*`), and `llm_base_proposer.py` loops it `num_speculative_tokens - 1` extra times.
DFlash takes the `parallel_drafting` branch instead. The `mtp7` arm asserts the engine actually came
up at depth 7 and fails loudly (exit 47) rather than silently measuring a clamped depth.

The `mtp3`/`mtp7` arms serve through a `/tmp` symlink: vLLM's Speculators updater rewrites
`Qwen3_5MTP` -> `DFlashQwen3_5MTP` (unregistered) when the served path contains "dflash".

## Environment the numbers were taken on

Ascend 910B4 x2 (TP2), `vllm-ascend:pr14171-v026-runtime-20260821`
(`sha256:e6519803e088d655...`), vLLM engine v0.26.0, Qwen3.8-27B target,
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) draft — verified
byte-identical to the published checkpoint (sha256 `67fc76d68dc5a941...`), config matching on all
13 checked fields.

**The runtime tree is not an upstream branch.** It is an unversioned local patch series
(`dflash2-mrv1-full-kv-device-queryloc-v2`) whose manifest self-describes as
`classification=speculative local DFlash-only Python optimization over immutable c1b3 runtime`. Its
delta against its declared base is **9 lines in one file** (`vllm_ascend/attention/attention_v1.py`):
reuse of an existing device `query_start_loc` instead of a pinned H2D copy, gated on
`speculative_config.parallel_drafting` — so it applies to **DFlash only**, and it makes DFlash
*faster*. DFlash loses anyway, so the asymmetry does not threaten the direction of the result; it is
recorded because it means the DFlash arm ran on slightly-patched code while MTP ran stock.

No source file in either tree was modified during the measurement window (0 `.py` files changed since
2026-09-10 across 1201 + 3841 files), so every run shares one frozen tree.
