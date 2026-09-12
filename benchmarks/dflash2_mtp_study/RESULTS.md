# DFlash2 vs MTP on vllm-ascend 0.26 / Ascend 910B4 — aligned to the published SGLang setup

2026-09-12 · kylin10-6018 · NPU 0+2 · TP2 · Qwen3.8-27B · vllm-ascend 0.26 · image `vllm-ascend:pr14171-v026-runtime-20260821`

## 0. One line

**DFlash2 never beats the built-in MTP head on this stack, at any concurrency, in any setup tested —
including the MTP-7 baseline the published "DFlash 1.5x MTP" claim uses.** At the user's own
benchmark shape (C32, 4096-token inputs) speculative decoding as a whole is worth ~4%.

The published results were checked and are not disputed. They measure a different workload.

## 1. The user's benchmark shape

`--dataset-name random --random-input-len 4096 --random-output-len 256 --max-concurrency 32`

| arm | tok/s | vs nospec |
|---|---|---|
| nospec | 184.25 | — |
| **MTP3** | **191.66** | **1.04x** |
| DFlash K3 (best DFlash depth) | 162.76 | 0.88x |
| DFlash K7 | 149.87 | 0.81x |
| DFlash K1 | 96.40 | 0.52x |

The point of this cell is not that DFlash loses; it is that **speculation is worth almost nothing
here** — the best arm beats no-speculation by 4%.

DFlash's depth curve is **non-monotone with an interior optimum at K=3** (96.40 -> 162.76 -> 149.87).
This curve can only be measured, not modelled: fitting `T = c + alpha*(K+1)` to three observations
yields a *negative* drafter cost. Two predictions made from that fit were both wrong.

## 2. Aligned to the published setup

Real GSM8K questions (16, mean 59.8 tokens), greedy, prefix caching off (= their RadixCache off),
TP2, `/v1/chat/completions`. Each curve is swept **inside one service instance**, with C32 measured
at both ends as an order control.

### 2.1 Output capped at 256 (`--ignore-eos`)

| conc | nospec | MTP3 | MTP7 | K3 | K7 |
|---|---|---|---|---|---|
| 1 | 28.83 | 64.18 | 66.94 | 45.23 | 59.14 |
| 8 | 187.42 | 265.71 | 247.86 | 203.23 | 210.00 |
| 16 | 333.03 | 403.82 | 367.10 | 301.14 | 283.26 |
| 32 | 540.53 | 587.96 | 394.34 | 424.95 | 281.93 |

All five passed their order control (drift 1.0012 / 0.9602 / 1.0248 / 1.0004 / 0.9783).

### 2.2 EOS respected — the published setup

| | C1 | vs nospec | C16 | vs nospec |
|---|---|---|---|---|
| nospec | 28.53 | — | 224.86 | — |
| MTP3 | 63.37 | 2.22x | 339.87 | 1.51x |
| **MTP7** | **71.35** | **2.50x** | **363.94** | **1.62x** |
| K3 | 49.17 | 1.72x | 281.66 | 1.25x |
| K7 | 69.41 | 2.43x | 313.12 | 1.39x |

**MTP-7 is the best arm at both points.** K7 beats MTP-3 at C1 (1.10x) but loses to MTP-7 (0.97x),
and loses to both at C16.

> `--ignore-eos` was **our** choice, not theirs, and it is not a neutral one: forcing 256 tokens
> drops K7's advance from 5.18 to 3.90 (-25%) while K3 loses only 13%. **The bias is not uniform
> across arms**, so orderings taken from the 256-capped table cannot be trusted. Under §2.1 K7
> appeared to fall to 0.85x of no-spec at C16; with EOS respected all three speculative arms are
> above no-spec there.

### 2.3 Long output (EOS, up to 2048 — thinking not truncated)

| conc | MTP3 | K7 | K7/MTP3 |
|---|---|---|---|
| 1 | 69.63 | 73.62 | 1.06 |
| 8 | 213.35 | 200.67 | 0.94 |
| 16 | 364.50 | 289.06 | 0.79 |

K7 advance 4.86–5.14, mean output 382–427 tokens (the 256 cap had been truncating: mean 191–200
against a 256 limit). MTP3 passed its order control (0.9594); **K7's did not (0.9233) — trend only.**

## 3. Why: three-factor decomposition

`throughput = N_decoding x advance / T_iter`, balanced at all four points
(C32: 0.624 x 3.9252 / 4.693 = 0.522, measured 0.522).

| conc | K7/nospec | N_dec ratio | advance | T_iter ratio |
|---|---|---|---|---|
| 1 | 2.05x | 0.93 | 4.21 | 1.92 |
| 8 | 1.12x | 0.88 | 3.75 | 2.95 |
| 16 | 0.85x | 0.84 | 3.81 | 3.81 |
| 32 | 0.52x | 0.62 | 3.93 | 4.69 |

**The dominant term is T_iter ratio climbing 1.92 -> 4.69.** A no-spec iteration barely gets more
expensive with batch (32x the tokens for +58%: 33.38 -> 52.90 ms) because at low concurrency the
target forward is memory-bound. DFlash pushes 8x the width every iteration: free when latency-bound,
paid in full once compute-bound. Advance is constant, so **the problem is entirely on the cost side.**

Win line is `advance > T_iter ratio`; advance ~3.9 is overtaken at C16, and with slot losses the
break-even lands at **C ~ 11**.

Secondary: at C32 DFlash keeps only 17.8 of 32 slots decoding versus no-spec's 28.5. KV is not the
constraint (32 x ~316 tokens vs a 135,399-token pool) — this is a scheduler-side loss, not localised.

## 4. Matched verify width — the drafter is not stronger

**Advance compares throughput across depths, but NOT drafter quality**: a wider draft trivially emits
more tokens. Isolating quality requires matching verify width.

| width | pair | advance ratio | T_iter ratio | throughput |
|---|---|---|---|---|
| 4 | K3 vs MTP3 | **0.95** | 1.21–1.38 | 0.70–0.76 |
| 8 | K7 vs MTP7 | 1.07 | **0.81–0.95** | MTP7 wins 1.13–1.40x |

Per-drafted-position hit rate, both widths favour MTP:

```
width 4:  MTP3 (3.08-1)/3 = 69.3%     K3 (2.93-1)/3 = 64.3%
width 8:  MTP7 (4.21-1)/7 = 45.9%     K7 (3.93-1)/7 = 41.9%
```

MTP-7 is **seven sequential single-layer head calls**; DFlash is **one block-parallel five-layer
forward**. "Block-parallel is more hardware friendly" does not hold here.

## 5. The drafter's cost is not an implementation defect

Two earlier hypotheses, both **refuted**:

- **"The drafter runs eager, outside the graph."** All three arms, both TP ranks, log
  `Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL`.
- **"The drafter costs 15-30x more than it should."** Back-solving the published numbers:

  | | drafter overhead per iteration | in units of one target forward |
  |---|---|---|
  | H200 (68.9 -> 236.1, accept 5.46) | 8.62 ms | **0.59** |
  | 910C, SGLang (33.2 -> 101.2, accept 5.8) | 27.2 ms | **0.90** |
  | 910B4, here (28.5 -> 69.4, accept 5.31) | 31.2 ms | **0.94** |

  The drafter is ~1.5x more expensive relative to the target on Ascend than on GPU, and **this
  implementation is within 5% of the other Ascend implementation.** The original estimate used the
  wrong denominator ("5 of 64 layers = 8%"): with `block_size=8` the drafter pushes `num_reqs x 8`
  rows through a 248320-wide lm_head every iteration (651 GFLOP at C32 against MTP3's 96 rows /
  244 GFLOP), then a full-vocabulary top-k, then candidate scoring. The code refuses to shrink it:
  `"DFlash2 does not support a reduced draft vocabulary; the selector top-k needs the unquantized target LM head."`

Draft weights are 1.924B on disk (not the 4.14B a naive config read gives): embeddings **and**
lm_head are shared with the target. Sharing saves memory, not arithmetic.

KV per token is **6.03x** no-spec (816,578 vs 135,399 tokens; max concurrency 24.92x vs 4.13x), from
`retaining full KV history for 5 sliding-window draft layers`. Turning off
`dflash_full_kv_allocation` is worth +9.4% but is **not** the cause (single-variable: -0.23% on pool size).

## 6. Setup reconciliation

| axis | LMSYS blog | SGLang PR#35629 | MindStudio H200 | here |
|---|---|---|---|---|
| target | Qwen3.5-397B-A17B (MoE) | Qwen3.8-27B | Qwen3.8-27B | Qwen3.8-27B |
| hardware | 8xB200 | 910C TP2 | 1xH200 | 910B4 TP2 |
| concurrency | 1–8 | 1 / 8 / 16 | 1 / 8 / 32 | 1 / 8 / 16 / 32 |
| MTP baseline | **MTP-7** | **none** | 7 draft tokens | MTP-3 and MTP-7 |
| draft block size | 16 | 8 | 8 | 8 |

Three things worth carrying forward:

1. **The widely quoted "DFlash 1.5x MTP" is measured on a 397B-A17B MoE, not on the 27B dense model.**
   A ~1.9B drafter amortises far better against a 397B target forward. The sign can flip between the
   two models for structural reasons.
2. **The Ascend adaptation PR contains no MTP comparison at all** — only versus no-speculation.
3. **Their no-spec baseline scales worse than ours**: C16/C1 = 7.4x against our 11.6x. Had their
   baseline scaled like ours, their C16 figure would be 431.2/333 ~ 1.29x rather than 1.75x.
   *When reading an "Nx speedup", ask how the denominator scales.*

Acceptance itself **is** aligned: 5.31 here versus 5.46 published for the same model on H200, and one
of our 16 reconstructed prompts is malformed (a `q[:400]` fallback truncated the actual question),
which accounts for roughly the remaining 2%. That defect is identical across all arms, so it depresses
absolute acceptance without affecting any ratio.

## 7. Recommendations

- **C32 + 4096-token inputs: do not deploy DFlash2.** Do not expect much from MTP either — 4%. The
  bottleneck is that concurrency has already pushed the target forward into the compute-bound regime,
  which removes the premise speculative decoding runs on.
- **C <= 8 with short inputs**: speculation pays well (2.2–2.5x). Use **MTP-7**; it is the best arm at
  every point measured.
- **DFlash2 has no regime here where it is the right choice.** Its single win is against MTP-**3** at
  C1 with EOS respected (1.10x), and MTP-7 beats it there too.
- The only remaining order-of-magnitude lever is the **selector path** — SGLang demonstrated 6 ms of
  TPOT recoverable on NPU by replacing full-vocabulary selector sampling with argmax.

## 8. Measurement protocol

Every number above was produced under, and every claim is conditional on:

- **Warm protocol.** The first `vllm bench serve` against a fresh service runs at ~68% of warm
  throughput (graph capture + Triton JIT). A discarded warm-up round precedes every sweep.
- **Order control.** C32 (or C8 in the long-output sweep) is measured at both ends of each sweep;
  results are quoted only when the pair agrees within 5%. Failures are labelled in place.
- **One instance per curve**, so a curve's shape carries no restart variance.
- **advance = 1 + accepted/drafts**, not `acceptance_rate`, which averages over a different number of
  positions at each K and is not comparable across depths.

Known defects, left in place deliberately so that all arms share them: one malformed prompt in 16
(~2% absolute acceptance drag); `--ignore-eos` in §2.1 (non-uniform across arms — see the note there).
