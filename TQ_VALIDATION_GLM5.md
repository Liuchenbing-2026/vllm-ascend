# TurboQuant (glm5_tq_best) Validation on GLM-5-w4a8 / Ascend 910B4

Validation of the `glm5_tq_best` branch (TurboQuant **Plan A — MLA latent target**)
on a real GLM-5 model, Ascend NPU, fake-quant round-trip path.

## Setup
- **Model**: `GLM-5-w4a8` (`glm_moe_dsa`, `kv_lora_rank=512`, 78 layers, `index_topk=2048`)
- **Hardware / stack**: 8× Ascend 910B4, TP8, `vllm-ascend v0.20.2rc1` + this branch (patch applies clean, `git apply`)
- **Path**: `--kv-cache-dtype turboquant_4bit_nc`, `VLLM_ASCEND_TURBOQUANT_TARGET=latent`,
  `VLLM_ASCEND_TURBOQUANT_BACKEND=reference` (torch reference quantizer, fake-quant round-trip in `exec_kv`; KV kept at model dtype)
- **Metric**: PPL via `prompt_logprobs` over a fixed prompt; A/B = TQ-off (BF16 KV) vs TQ-on (`turboquant_4bit_nc`)

## Unit test
`tests/ut/quantization/test_turboquant_reference.py` → **21 passed**.

## Compression ratios (reference quantizer, 512-dim latent) — matches the preset table
| preset | key/val bits | NC | compression (ours) | paper |
|---|---|---|---|---|
| `turboquant_k8v4`    | 8 / 4 | no  | 2.56× | 2.6× |
| `turboquant_4bit_nc` | 4 / 4 | yes | 3.77× | 3.8× |
| `turboquant_k3v4_nc` | 3 / 4 | yes | 4.27× | ~3.5× |
| `turboquant_3bit_nc` | 3 / 3 | yes | 4.92× | 4.9× |

## On-card PPL A/B (GLM-5-w4a8, `turboquant_4bit_nc`, latent target)
| corpus | tokens | TQ-off PPL (BF16) | TQ-on PPL (4bit_nc) | ΔPPL | note |
|---|---|---|---|---|---|
| short, distinct           |  663 | 2.0065 | 2.0073 | **+0.04%** | context too short — KV barely exercised |
| long, repeated            | 4496 | 1.1145 | 1.1126 | −0.17% | degenerate (repeated text → PPL→1.1), within noise |
| **long, distinct (GSM8K)** | 4584 | **1.4299** | **1.4401** | **+0.71%** | clean, meaningful long-context measurement |

## Conclusion
- The **Plan A latent `turboquant_4bit_nc`** scheme is **near-lossless on GLM-5**:
  **+0.71% PPL** on distinct long context (NLL +1.99%), squarely in the paper's claimed
  regime (4bit_nc +2.71%; single-digit %).
- This is the latent target (`kv_lora_rank=512`), confirming the key levers do the work:
  **random sign-flip + Hadamard rotation → per-vector unit-norm → per-coordinate Lloyd-Max
  (N(0,1/d) centroids) → norm-correction → boundary-skip first/last 2 layers**.
- A naive single-scale joint 4-bit latent quant (no rotation, no per-coord codebook,
  no boundary skip) on the same latent measured **+18% PPL** — i.e. the levers in this
  branch take 4-bit MLA-latent quantization from +18% down to <1%.

## Notes / limitations
- `reference` backend only (torch fake-quant round-trip); NPU `aclnn`/`triton` kernels not exercised here.
- GSM8K (math) corpus differs from the paper's likely WikiText benchmark; exact +2.71%
  reproduction would need the same corpus. The near-lossless conclusion holds across corpora.
- Eval harness: single long sequence, chunked prefill (`max_num_batched_tokens=512`) to bound
  the `prompt_logprobs` memory peak at TP8.

---

## Appendix: service-level benefit picture (a *different*, production joint-4bit MLA path)

> The numbers below are **not** from the `glm5_tq_best` Plan A reference path validated above.
> They come from a **separate, simpler** TQ scheme actually deployed in serving:
> a single per-token-scale joint 4-bit Lloyd-Max over the 512-dim MLA latent
> (`turboquant_mla_4bit`, real compression + in-kernel dequant). Same hardware
> (GLM-5-w4a8, 8× Ascend 910B4, TP8). Included for context on real-compression trade-offs.

### Capacity — the one positive
- 2.19× KV compression → same-HBM KV pool **67,840 → 148,736 tokens**.
- → serve **2.19× longer context / more concurrent sessions** (contexts that OOM/reject under
  BF16 run under TQ).

### Throughput — honest: negative in this regime
- TQ decode is **~2.5× slower** per token (in-kernel dequant of 4-bit KV in the sparse-FA path).
- W4A8 / 910B4 MoE decode is **compute-bound** (expert GEMM + TP8 comms dominate; KV read is a
  small fraction), so the 2.2× capacity advantage is outweighed by the per-token dequant cost.
- Net: **TQ throughput < BF16** at every batch size tested, no crossover. The "2× throughput"
  result is a **different regime** (910B3 / W8A8 / KV-bandwidth-bound decode), not reproducible here.

### Task accuracy — held
- GSM8K (n=200, greedy): **TQ 80.0% vs BF16 81.5%** — within noise; task quality essentially
  held despite 2.19× compression.

### Summary
| dimension | result |
|---|---|
| KV compression | 2.19× (joint-4bit) / up to 3.77× (4bit_nc) |
| max context @ same HBM | 67,840 → ~149k tokens |
| decode throughput | ~0.4× (slower; structural on compute-bound MoE) |
| task accuracy (GSM8K) | 80.0% vs 81.5% (held) |
| token-level PPL | joint-4bit +18% · `glm5_tq_best` 4bit_nc reference +0.71% |

→ Delivered value of TQ here = **compression × quality (capacity), not throughput**.
