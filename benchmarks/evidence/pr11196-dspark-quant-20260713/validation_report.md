# DSpark W8A8/W4A8 validation report

Date: 2026-07-13

## Scope

The original sections cover schema and loader tests, DSpark metadata semantics, build-cache safety, staged-package validation, host compilation, and a clean custom-op package build. The dated NPU addendum records the completed W8A8 precision, graph, acceptance, and performance evidence. A genuine DSpark hybrid-W4 artifact is still unavailable, so W4A8 remains contract-tested but not device-certified.

## NPU addendum (2026-07-13)

- Functional commit `b1909f574e0ce6642379e8ef3fb63d26d802570e`
  is bound to the benchmark runtime worktree by an exact Git-blob manifest:
  2,436 of 2,436 regular files under `csrc`, `tests`, and `vllm_ascend` match,
  with zero mismatch. See
  [`runtime/source-provenance.json`](runtime/source-provenance.json). Raw
  `code.commit` and `code.diff.sha256` files retain the deployment repository's
  original lineage for auditability.
- The audited full v14 package was atomically exchanged into the active repository package. The previous active package remains intact at `.cann_ops_custom.rollback_v14_0713` pending the user's permanent-deployment decision.
- SparseAttnSharedkv passed the real NPU PTA comparison for 64 query heads, one KV head, head dimension 512, sequence length 133, K=256, window 128, and a non-contiguous block table `[[2, 0]]`.
- Result shape: `(5, 64, 512)`; maximum absolute error: `3.052e-5`; mean absolute error: `1.38e-6`; output finite: true.
- The first target-graph/eager-draft startup exposed a startup-only device inheritance bug: a CPU `torch.Generator` was paired with an implicitly NPU probe tensor during QuaRot validation. No AISBench request was sent.
- The fix pins both safetensors rotation loading and random probe allocation to CPU, then transfers the validated rotation to the requested NPU. A non-CPU-default-device regression passed remotely, followed by the existing QuaRot suite (`8 passed`, `107 deselected`).
- The v14b target-graph/eager-draft 8192/128 gate passed at 58.7879% acceptance, with 194 accepted of 330 draft tokens. This cleared the safety gate before draft graph capture.
- The v14c full target+draft graph emitted the DSpark drafter ACLGraph marker and passed its 8192/128 gate at 59.6875% acceptance, with 191 accepted of 320 draft tokens.
- The exact requested 8k/1k/concurrency-8 comparison then completed for both DSpark5 and a fresh MTP1 service. Every formal run completed 32/32 requests; server-log error scans were zero. Both services were stopped afterward and port 8900 was confirmed free.

## Confirmed runtime configurations

- Draft checkpoint: `DeepSeek-V4-Flash-DSpark-w8a8-cleanroom-rotated-decoder-headfix`.
- Historical MTP1 bring-up baseline: 191.1903 output token/s, 94.0088% acceptance, average accepted length 1.94009.
- Historical DSpark5 target-graph/eager-draft result: 126.0376 output token/s, 77.2332% acceptance. It proved non-zero acceptance but was not used for the final performance claim.
- Final DSpark5 and MTP1 both used the current code and v14 custom-op package, TP8/EP, target `FULL_DECODE_ONLY` graph mode, identical serving/benchmark knobs and seed, and port 8900. They use different complete checkpoint paths and speculative architectures, so this is an end-to-end service comparison rather than a draft-quantization-only ablation.
- The requested workload was configured as 8192 input tokens, 1024 output tokens, concurrency 8, and 32 formal requests. AISBench reported an actual prompt length of 8196 tokens and sends one same-length precheck; therefore CSV/detail metrics cover 32 requests while Prometheus acceptance deltas cover 33.

## Final AISBench performance evidence

| Drafter | Run | Output tok/s | Acceptance | Avg accepted length | TTFT avg/P90 (ms) | TPOT avg/P90 (ms) | AISBench duration (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DSpark5 | 1 | 217.7340 | 81.4241% | 5.07120 | 3469.5 / 7959.2 | 31.9 / 42.5 | 150.496 | 216.031 |
| DSpark5 | 2 | 217.6976 | 77.3109% | 4.86555 | 3299.1 / 7747.7 | 32.1 / 39.4 | 150.521 | 214.579 |
| DSpark5 | 3 | 210.5191 | 77.2772% | 4.86386 | 3439.6 / 7709.8 | 31.9 / 40.0 | 155.653 | 221.308 |
| MTP1 | 1 | 191.8100 | 94.6289% | 1.94629 | 3531.5 / 7154.6 | 37.9 / 40.8 | 170.836 | 246.717 |
| MTP1 | 2 | 190.6709 | 94.1427% | 1.94143 | 3467.8 / 6990.7 | 38.0 / 41.8 | 171.856 | 244.518 |
| MTP1 | 3 | 193.5156 | 95.6046% | 1.95605 | 3292.4 / 6985.7 | 37.9 / 40.4 | 169.330 | 243.062 |

DSpark5 median throughput was 217.6976 output token/s versus 191.8100 for MTP1, a 13.4965% gain in this end-to-end sample. Median TPOT decreased 15.8311%, median AISBench measured duration decreased 11.8915%, and median wall time decreased 11.6502%. The slowest DSpark run remained 8.7866% faster than the fastest MTP run.

DSpark run-to-run throughput spread was 3.3142%, slightly above the approximate 3% target by 0.3142 percentage points; MTP spread was 1.4831%. All three DSpark results exceed all three MTP results in this 3+3 sample, but the predeclared stability target was narrowly missed and the variance is disclosed rather than rounded away.

## Same-checkpoint no-spec attribution

Because the DSpark and MTP services use different complete checkpoint paths, each checkpoint was also served with speculative decoding fully disabled. The launch settings remained TP8/EP, target `FULL_DECODE_ONLY`, the same active v14 package, and the same AISBench protocol. A preceding 8192/128/concurrency-1 smoke request succeeded for each service and produced zero deltas for every exposed speculative counter.

| Mode | Throughput runs (output tok/s) | Median | Spread | Median TPOT | Median AISBench duration | Median wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DSpark checkpoint, DSpark5 on | 217.7340 / 217.6976 / 210.5191 | 217.6976 | 3.3142% | 31.9 ms | 150.521 s | 216.031 s |
| DSpark checkpoint, no spec | 136.1390 / 136.9068 / 135.8115 | 136.1390 | 0.8045% | 55.4 ms | 240.695 s | 331.076 s |
| MTP checkpoint, MTP1 on | 191.8100 / 190.6709 / 193.5156 | 191.8100 | 1.4831% | 37.9 ms | 170.836 s | 244.518 s |
| MTP checkpoint, no spec | 137.4186 / 136.8683 / 137.5040 | 137.4186 | 0.4626% | 54.8 ms | 238.454 s | 330.517 s |

Against its own checkpoint's target-only baseline, DSpark5 increases median output throughput by 59.9083%, reduces median TPOT by 42.4188%, reduces AISBench batch duration by 37.4642%, and reduces wall time by 34.7486%. MTP1 increases median throughput by 39.5808%, with corresponding reductions of 30.8394%, 28.3569%, and 26.0194%. DSpark's same-checkpoint throughput gain is therefore 20.3275 percentage points larger; dividing the two on/off factors leaves DSpark 14.5633% ahead after baseline normalization. The two no-spec medians differ by only 0.9312%.

All six no-spec runs completed 32/32 formal requests (192/192 total), and speculative counters were absent rather than non-zero. Together with the six speculative runs, all 384 formal requests succeeded. Every service log had zero error-pattern matches before its audited SIGTERM shutdown. Two services appended an eight-line `AsyncLLM output_handler`/`EngineDeadError` traceback only after shutdown began; the separate lifecycle audit records these as teardown artifacts, not benchmark-phase failures.

The checkpoint identity audit found exactly matching 100,829 non-draft physical tensor names, dtypes, and shapes, plus 100,834 identical non-draft logical-description entries. The payloads are not strictly byte-identical: `layers.0.attn.wq_a.weight` differs at one I8 element out of 4,194,304, by a value of one, while its scale and offset are identical. This is why the same-checkpoint on/off comparison, rather than absolute checkpoint identity, is used for attribution.

A post-run detail audit found 32 successful, error-free, 1024-token responses in every run and zero input mismatches across all six runs. The AISBench model config used `temperature=0`, but exact predictions also varied between repeated runs of the same service (dynamic batching/numerical replay is therefore not token-deterministic on this workload). These performance runs are not presented as the still-pending frozen no-spec/BF16 greedy token-ID gate.

Committed artifacts:

- DSpark speculative and no-spec runs: [`modes/dspark5_spec`](modes/dspark5_spec) and [`modes/dspark_target_nospec`](modes/dspark_target_nospec)
- MTP speculative and no-spec runs: [`modes/mtp1_spec`](modes/mtp1_spec) and [`modes/mtp_target_nospec`](modes/mtp_target_nospec)
- Direct speculative comparison: [`comparisons/spec/comparison.json`](comparisons/spec/comparison.json), SHA256 `b840c4bc5da3ae061822efe999d8f6c5261ec336fae44976adeb14c758b48a4b`
- Normalized comparison: [`comparisons/normalized/comparison.json`](comparisons/normalized/comparison.json), SHA256 `160bf16d1a9c65ddf80549aa66b01295d1135a77b764eab5e48a629a1ec0eb66`
- Service lifecycle audit: [`comparisons/normalized/service_log_audit.json`](comparisons/normalized/service_log_audit.json), SHA256 `0ba55b612bdfba247783035c51066d28d5a5d8b3b49e3db8953ed7979a7fd275`

## Root cause already isolated

The double-graph crash came from a mixed custom-op ABI: the host tiling library was rebuilt after adding `hasOriSparseIndices` and `oriSparseIndexWidth`, while packaged SparseAttnSharedkv kernel objects were stale. The old kernel therefore decoded later tiling fields at the wrong offsets and produced an AIV MTE DDR out-of-range failure.

A forced clean targeted build changed both SparseAttnSharedkv kernel hashes:

- old: `67e4d3f5...`, `1616731e...`
- clean: `0a3682df...`, `79d48a9e...`
- clean host tiling library: `701a4724...`

The earlier targeted diagnostic package remains isolated in its staging slot. The subsequently audited complete v14 package, containing the same corrected SparseAttnSharedkv objects, is the package used for the final device runs.

## Safety changes

- `build_aclnn.sh` performs a clean build by default; build reuse is an explicit opt-in.
- ccache is disabled by default both through `CCACHE_DISABLE=1` and `--ccache false`.
- A canonical-path lock covers cleanup, compilation, packaging, validation, and replacement.
- Installation occurs in a staging directory. The existing package is retained until the staged package passes validation.
- Validation requires the current-host opmaster library, non-empty SparseAttnSharedkv `.o` and kernel JSON files, parseable operator/binary configuration, and valid JSON-to-object references.
- Failed directory replacement restores the previous package; tests inject failure on the second `mv` to exercise the real rollback branch.
- Host attributes declared as `Int` are read as `int64_t`; missing attributes, workspace pointers, tiling buffers, and insufficient tiling capacity fail closed.

## CPU test evidence

Container: `ds-v4-w8a8`

- Isolated build-policy tests: 10 passed.
- Targeted integration suite: 277 passed, 16 deprecation warnings, 0 failed.
- Local Ruff lint and format: passed.
- Local Python bytecode compilation: passed.
- Remote `bash -n csrc/build_aclnn.sh`: passed.
- Remote `git diff --check`: passed.

The targeted suite includes W8/hybrid-W4 ModelSlim contracts, alias/conflict handling, required quantization companions, physical dtype/global-shape gates, local/non-local/redundant EP expert loading, QuaRot basis and orthogonality guards, proposer/loader behavior, graph scheduling guards, and draft5/window128 physical-slot metadata with multiple requests, non-contiguous physical blocks, and padded rows.

## Real checkpoint header evidence

The W8 artifact `DeepSeek-V4-Flash-DSpark-w8a8-cleanroom-rotated-decoder-headfix` matches the new fail-closed contract:

- It explicitly declares `dspark_quarot_draft_basis=rotated_decoder` and `dspark_quarot_hc_head_basis=canonical`.
- `mtp.0.main_proj.weight` is `I8 [4096, 12288]`; its scale and offset are `F32 [4096, 1]`.
- All three `wo_a` tensors are `BF16 [8192, 4096]` and remain `FLOAT` in the manifest.
- All 2,304 routed-expert base weights are physically `I8`; their scale/offset tensors are `F32`.
- `global_rotation` is `F32 [4096, 4096]`.
- Its row/column norm-squared maximum error is `2.13742256e-4`; the exact runtime 16-probe check reports `Q.T @ Q` max-absolute error `3.34531069e-6`, well below the FP32 gate `5e-3`.

The available `DeepSeek-V4-Pro-w4a8-mtp` artifact is a standard one-stage MTP checkpoint, not a DSpark checkpoint: it has no `dspark_block_size`, DSpark layer count, `main_proj`, or explicit DSpark basis. Its routed-expert W4 tensors are physically `I8`, but it cannot serve as the final on-machine DSpark-W4 acceptance artifact. A separate DSpark W4 checkpoint is still required for device validation.

## Full clean package evidence

The complete offline build passed in an isolated clean staging directory:

- 38 requested DS-V4 operators are present: 36 kernel operators plus two host-only metadata APIs.
- All 216 package JSON files parse; 176 kernel metadata JSON files have matching non-empty `.o` files, and all 272 binary/JSON references resolve.
- Host opmaster hash: `2994008a...`.
- SparseAttnSharedkv kernel hashes: `0a3682df...` and `79d48a9e...`, matching the independent forced-clean targeted build.
- The package was subsequently activated by an atomic directory exchange after review. The old package with SparseAttnSharedkv hashes `67e4d3f5...` and `1616731e...` remains in the rollback slot pending the user's permanent-deployment decision.
- Port 8000 remains closed; the unrelated service on port 8100 was not modified.

## Remaining scope

1. Produce or identify a genuine DSpark W4 checkpoint before claiming W4 device acceptance or performance; the available W4 artifact is standard MTP.
2. Run the frozen BF16/no-spec correctness reference if the final merge gate requires token-for-token attribution beyond the successful target-verified speculative runs.
3. Keep the old package rollback slot until the user decides the v14 deployment should be made permanent.
