# DeepSeek V4 DSpark W8A8/W4A8 integration report

## Provenance

- Public baseline: vllm-ascend PR #11196, commit `8bca4f2061baefec54b18005839eb17ed3654d02`.
- Implementation method: clean-room changes derived from the public loader/runtime interfaces, the ModelSlim checkpoint schema, and independently written coordinate-system tests.
- No code was copied, ported, or paraphrased from another implementation or from the abandoned worktree.
- Submitted functional commit: `b1909f574e0ce6642379e8ef3fb63d26d802570e`.
  Its 2,436 in-scope regular source/test Git blobs match the benchmark runtime
  worktree exactly; see
  [`runtime/source-provenance.json`](runtime/source-provenance.json).

## Supported checkpoint contract

| Component | W8 checkpoint | Hybrid W4 checkpoint |
| --- | --- | --- |
| Attention/dense projections | `W8A8_DYNAMIC` | `W8A8_DYNAMIC` |
| Shared experts | `W8A8_DYNAMIC` | `W8A8_DYNAMIC` |
| Routed experts | `W8A8_DYNAMIC` | `W4A8_DYNAMIC` |
| `main_proj` | `FLOAT` or `W8A8_DYNAMIC` | `FLOAT` or `W8A8_DYNAMIC` |
| `wo_a` | `FLOAT` | `FLOAT` |
| Norm/router/Markov/HC parameters | `FLOAT` | `FLOAT` |

Dense `W4A8_DYNAMIC` is intentionally rejected. The current dense W4 linear path needs a positive group size, while the validated DSpark W4 profile is per-channel W4 for routed experts only.

## Changes

1. Raw ModelSlim names under `mtp.<stage>.*` receive canonical `model.mtp.<stage>.*` aliases using path-segment mappings. Direct expanded `model.layers.<draft>.*` descriptions are also recognized. Raw names remain present for safetensors loading; conflicting entries fail immediately.
2. `main_proj` selects the existing W8 linear method only when its per-weight metadata is `W8A8_DYNAMIC`. Explicit `FLOAT` preserves the floating implementation; a dynamic draft profile with a missing entry, dense W4, or an unknown type fails closed.
3. A pre-construction validator checks stage coverage, all routed experts, projection precision consistency, W4 version/explicit group size, and the `wo_a=FLOAT` invariant. Unknown or non-string draft weight types are rejected before allocation. It accepts both weight-only and full physical-manifest ModelSlim descriptions; when companions are listed, partial manifests are rejected.
4. The loader derives required W8/W4 physical companions even from a weight-only description, compares them with tensors actually supplied by the checkpoint, and requires instantiated draft quantization parameters to load. Base weights declared W8/W4 must be physically I8; FLOAT weights must be floating. `main_proj` and `wo_a` additionally receive global checkpoint-shape validation in both safetensors-header and iterator paths.
5. Local expert loads that return `False` fail closed, while proven non-local EP experts remain legal and redundant mappings may continue until a later replica succeeds. EP weight-filter exemption applies only to non-local base weights, never scales/offsets.
6. QuaRot uses an explicit row-vector contract, `h_rotated = h_unrotated @ Q`. Whenever a rotation exists, the draft basis must be explicit regardless of precision. The rotation must be finite, square, correctly sized, and pass full row/column norm plus fixed-seed full-dimensional `Q.T @ Q` probes.

## QuaRot basis modes

Set `dspark_quarot_draft_basis` in `config.json`, or set `optional.quarot.dspark_draft_basis` in `quant_model_description.json`.

| Value | Checkpoint meaning | Runtime transitions |
| --- | --- | --- |
| `legacy` | PR #11196 bridge-format `main_proj`; draft decoder is canonical | embedding `@ Q.T`, head `@ Q`, context unchanged |
| `unrotated` | Draft and `main_proj` are canonical/unrotated | embedding `@ Q.T`, each context block `@ Q.T`, head `@ Q` |
| `rotated` | Complete draft, including `main_proj`, is in target QuaRot coordinates | embedding/context/head basis unchanged |
| `rotated_decoder` | Shared embedding/head and decoder residuals are rotated; `main_proj`/`main_norm` stay canonical | context blocks `@ Q.T` before `main_proj`, normalized projection output `@ Q`; HC-head basis is declared independently |

For canonical `main_proj` weight `W_C` and block-diagonal `Q_k` over captured target layers, the exact weight contracts are:

- `legacy`: `W_L = W_C @ Q_k` (canonical output, input rotation folded offline).
- `unrotated`: `W_U = W_C` (runtime converts context with `Q_k.T`).
- `rotated`: `W_R = Q.T @ W_C @ Q_k` (rotated output and input).
- `rotated_decoder`: `W_D = W_C`; runtime converts context blocks with `Q_k.T` and converts the normalized canonical output with `Q`.

The final RMSNorm gamma removal remains active whenever QuaRot is present; it is scale fusion, not a basis transition.
`unrotated` performs an FP32 rotation for every context block and is a diagnostic/compatibility path; production weights should prefer an offline-folded `legacy` or fully `rotated` layout.

## Local verification

- `python -m compileall`: passed for all changed production and test files.
- Ruff lint and format checks: passed.
- Clean-room pure-contract harness: passed alias/conflict, W8, hybrid W4, missing companion, unsafe `wo_a`, dense W4 rejection, main projection selection, physical dtype/shape gates, EP locality/replica behavior, QuaRot directions, blockwise context conversion, explicit-basis enforcement, orthogonality probes, and raw/canonical/direct physical-versus-logical ModelSlim key filtering.
- The targeted suite was rerun in `ds-v4-w8a8`: 277 tests passed with 16 deprecation warnings and no failures.
- Independent source safety review: passed with no blocker.
- Full clean package audit: passed for all 38 requested DS-V4 operators, 216 parseable JSON files, 176 kernel JSON/object pairs, and 272 binary/JSON references. The audited v14 package was atomically activated for the device tests; the previous package remains available in the rollback slot.

## Real checkpoint evidence

The W8 checkpoint `DeepSeek-V4-Flash-DSpark-w8a8-cleanroom-rotated-decoder-headfix` matches the strict contract: explicit `rotated_decoder/canonical` basis fields, I8 `main_proj [4096,12288]`, BF16 `wo_a [8192,4096]`, and I8 routed-expert base weights. Its F32 `[4096,4096]` rotation has row/column norm-squared maximum error `2.13742256e-4`; the runtime-equivalent 16-probe maximum error is `3.34531069e-6` versus the FP32 limit `5e-3`.

The available `DeepSeek-V4-Pro-w4a8-mtp` artifact confirms I8-packed W4 routed experts and FLOAT `wo_a`, but it is standard one-layer MTP rather than DSpark: it has no DSpark block/layer metadata, `main_proj`, or draft basis. It is not sufficient for final DSpark-W4 acceptance/performance validation.

## Container validation order

1. Check the weight index and `quant_model_description.json` have identical raw `mtp.*` tensor sets, excluding the intentionally unused confidence head.
2. Run the targeted unit tests:

   ```bash
   pytest -q tests/ut/quantization/test_modelslim_config.py -k dspark
   pytest -q tests/ut/spec_decode/test_dspark_config.py -k 'quant or quarot or load_weights'
   ```

3. Run BF16 DSpark as the deterministic reference.
4. Run W8A8 first with eager execution and communication/graph optimizations disabled. Confirm the first request accepts at least one draft token; stop immediately if accepted tokens remain zero.
5. Compare first-step draft logits/top-k against BF16 before enabling ACLGraph or MC2. A basis mismatch should be diagnosed here, not masked by performance paths.
6. Enable ACLGraph, then the default communication path, one at a time.
7. Repeat for hybrid W4A8. Keep dense/shared layers W8 and routed experts W4.
8. Final gate: no unmatched or missing tensors; no NaN/Inf; acceptance rate is non-zero on the smoke set and within 1–2 absolute percentage points of the BF16 reference on the agreed benchmark.

DeepSeek V4 DSpark is selected through vLLM's integrated MTP interface. The serving config must use `"method":"mtp"`; `dspark_block_size` in the checkpoint routes it to `AscendDSparkProposer`. There is no separate draft-model path in the speculative config, and `num_speculative_tokens` must equal `dspark_block_size`.

## Measured-gain protocol

For each target artifact, compare the same service twice: speculative decoding disabled and DSpark enabled. A BF16-versus-W8-versus-W4 draft comparison is attributable to draft quantization only after hashes confirm that every non-`mtp.*` target tensor, tokenizer/config field, and unquantized draft tensor is identical. Otherwise report each artifact's DSpark-on versus DSpark-off gain separately.

Keep the container, CANN/torch_npu/vLLM versions, NPU allocation, TP/DP/EP, graph/communication settings, seed, dataset, request order, lengths, concurrency, KV settings, and `num_speculative_tokens` fixed. Disable block/entropy verification for correctness and attribution.

Before performance measurement:

- Greedy token IDs from no-spec, BF16, W8, and W4 runs must match exactly on the frozen correctness set.
- Acceptance must be non-zero during the first smoke requests; zero at every position stops the run immediately.
- Record Prometheus deltas for `vllm:spec_decode_num_drafts_total`, `vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total`, and the available per-position accepted-token counters after warmup.
- With deltas `D`, `T_draft`, and per-position accepted counts `A_i`, report `AR=sum(A_i)/T_draft`, `AL_draft=sum(A_i)/D`, and `AL_step=1+AL_draft`.
- Relative to the BF16 draft on the same target and workload, W8 AR may lose at most 1 absolute percentage point and W4 at most 2.

Measure at least single-stream decode latency and a fixed-concurrency saturated workload. Warm up graph capture, then run at least three repetitions and report every raw result plus the median. Claim a real gain only when greedy correctness and acceptance gates pass and median TPOT improves by at least 5% or output-token throughput improves by at least 5%, with run-to-run spread no greater than about 3%.

## On-machine W8A8 result

The audited W8A8 package passed the real SparseAttnSharedkv NPU precision probe with maximum absolute error `3.052e-5`, then passed both the eager-draft and full draft-graph acceptance gates. The final AISBench comparison used the requested 8k input, 1k output, concurrency 8, and 32 formal requests per run. AISBench reported 8196 actual input tokens and performs one precheck outside the formal 32, so acceptance counters cover 33 requests.

| Mode | Throughput runs (output tok/s) | Median | Spread | Median acceptance | Median TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| DSpark5 full graph | 217.7340 / 217.6976 / 210.5191 | 217.6976 | 3.3142% | 77.3109% | 31.9 ms |
| DSpark checkpoint, no spec | 136.1390 / 136.9068 / 135.8115 | 136.1390 | 0.8045% | N/A | 55.4 ms |
| MTP1 full graph | 191.8100 / 190.6709 / 193.5156 | 191.8100 | 1.4831% | 94.6289% | 37.9 ms |
| MTP checkpoint, no spec | 137.4186 / 136.8683 / 137.5040 | 137.4186 | 0.4626% | N/A | 54.8 ms |

In the direct same-machine speculative-service sample, DSpark5 has 13.4965% higher median output throughput and 15.8311% lower median TPOT than the fresh MTP1 comparator. Its slowest run is still 8.7866% faster than MTP1's fastest run. DSpark throughput spread is 3.3142%, slightly above the approximate 3% stability target, so all raw runs are reported.

The same-checkpoint target-only baselines remove the checkpoint-path performance confound. DSpark5 improves its own checkpoint's median throughput by 59.9083%, versus 39.5808% for MTP1 on its checkpoint: a 20.3275 percentage-point advantage. Dividing the two on/off factors leaves a 14.5633% normalized DSpark advantage. Median TPOT falls 42.4188% for DSpark versus 30.8394% for MTP. The two no-spec baselines differ by only 0.9312% and are substantially more stable than the speculative DSpark sample.

The checkpoint audit found identical target tensor key/dtype/shape and logical metadata sets, but not strict payload identity: one of 4,194,304 I8 values in `layers.0.attn.wq_a.weight` differs by one, with identical scale/offset. Same-checkpoint on/off is therefore the attribution source of truth. All 12 formal runs completed 32/32 requests (384/384 total), both no-spec smoke requests had zero speculative-counter deltas, and all service logs were clean during benchmark execution. Shutdown-only tracebacks appended after SIGTERM in two logs are separated in the lifecycle audit.

The frozen BF16/no-spec greedy token-ID gate remains pending. A post-run audit confirmed identical performance inputs, but exact predictions varied even between repeated `temperature=0` runs of the same service; this dynamic-batching workload is therefore not used as a token-exact correctness test.

The normalized comparison is stored at [`comparisons/normalized/comparison.json`](comparisons/normalized/comparison.json) (SHA256 `160bf16d1a9c65ddf80549aa66b01295d1135a77b764eab5e48a629a1ec0eb66`). The benchmark services are stopped, port 8900 is free, and the previous custom-op package remains in the rollback slot. W4A8 remains locally/schema validated only because the available W4 checkpoint is standard MTP rather than a genuine DSpark artifact.
