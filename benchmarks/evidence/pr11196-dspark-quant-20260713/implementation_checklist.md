# PR #11196 draft quantization clean-room adaptation

## Boundary

- Baseline: public PR #11196 commit `8bca4f2061baefec54b18005839eb17ed3654d02`.
- Inputs: public repository interfaces, ModelSlim checkpoint schema, and independently derived linear algebra/tests.
- Do not inspect, transplant, paraphrase, or copy any private/third-party implementation.
- Preserve backward compatibility unless an explicit checkpoint/config contract opts in.

## Work

- [x] Create a fresh worktree and branch from the public PR head.
- [x] Write failing tests for raw ModelSlim MTP aliases and conflicts.
- [x] Write failing tests for quantized `main_proj` selection and fail-closed dynamic-profile handling.
- [x] Define and test an explicit QuaRot draft-basis contract.
- [x] Implement the minimum production changes.
- [x] Add W8A8/W4A8 schema guards that prevent silent bad loads.
- [x] Run isolated contract tests, lint, formatting, compilation, and a provenance-oriented diff review.
- [x] Prepare the container/on-device acceptance checklist.
- [x] Exclude stage-local logical metadata from physical checkpoint completeness checks.
- [x] Re-run loader, schema, QuaRot, lint, and static regression checks.
- [x] Inspect the ModelSlim artifacts in validation container `ds-v4-w8a8` and validate their tensor/schema contract.
- [x] Run the W8A8 draft precision, graph, acceptance, and end-to-end performance gates on NPU.
- [ ] Run the BF16 reference and genuine hybrid W4A8 DSpark correctness/acceptance comparisons on NPU.
- [x] Benchmark W8A8 against a freshly measured MTP1 comparator and report all raw runs plus the median.

## On-machine status (2026-07-13)

- [x] Validate the ModelSlim W8A8 draft checkpoint schema and run the clean-room loader/contract tests in the container.
- [x] Establish the valid MTP1 AISBench baseline (8192/1024, concurrency 8, 32 requests): 191.1903 output token/s.
- [x] Prove DSpark5 acceptance is non-zero with target graph + eager draft: 77.2332% main-run acceptance, but 126.0376 output token/s (negative performance result).
- [x] Diagnose double-graph failure as a stale custom-op binary, not zero acceptance or checkpoint incompatibility.
- [x] Force-clean rebuild the targeted SparseAttnSharedkv host and kernel and confirm that both kernel hashes changed.
- [x] Atomically activate the audited v14 package while retaining the old package in a rollback slot.
- [x] Pass the NPU precision probe with the same clean host/kernel package.
- [x] Launch target+draft full graph, pass graph markers and an acceptance warmup gate, then run the requested AISBench comparison against MTP1.

### Live device gate

- [x] SparseAttnSharedkv PTA comparison on NPU: shape `(5, 64, 512)`, max absolute error `3.052e-5`, mean absolute error `1.38e-6`, all finite.
- [x] Fix the startup-only QuaRot probe so a process-wide NPU default device cannot mismatch its CPU random generator.
- [x] Pass the non-CPU-default-device regression and the remote QuaRot suite (`8 passed`).
- [x] Target graph + eager/PTA draft v14b passed the 8192/128 gate at 58.7879% acceptance.
- [x] DSpark drafter ACLGraph v14c passed its 8192/128 gate at 59.6875% acceptance.
- [x] Complete the requested 8192/1024/concurrency-8 AISBench workload: all three 32-request runs succeeded, with 217.6976 output token/s median throughput and 77.3109% median acceptance.
- [x] Repeat the identical workload with fresh MTP1 full-graph v14d: 191.8100 output token/s median throughput and 94.6289% median acceptance.
- [x] Stop both services after testing and confirm port 8900 is free.

## CPU-only preparation gate

- [x] Make `build_aclnn.sh` clean `build/output/build_out` by default and disable ccache by default.
- [x] Lock the complete clean-build/install interval and validate staged host, kernel, and JSON artifacts before replacement.
- [x] Add failure tests for invalid environment values, missing kernel artifacts, invalid JSON, wrong host architecture, installer failure, lock contention, and swap rollback.
- [x] Correct host Int attribute reads to `int64_t` and fail closed on missing attributes or undersized tiling buffers.
- [x] Validate the draft5/window128 physical-slot prefix contract for multiple requests, non-contiguous physical blocks, and padded rows.
- [x] Fail closed on manifest/physical dtype mismatch, invalid critical weight shapes, ambiguous QuaRot basis, non-orthogonal rotation, and local expert load failure.
- [x] Pass the targeted CPU unit suite in `ds-v4-w8a8`: 277 tests passed.
- [x] Finish and audit the complete clean DS-V4 custom-op build in an isolated staging directory without activating it.

## Current verification

- Python bytecode compilation: passed.
- Ruff lint and format checks: passed after formatting.
- Clean-room contract harness: passed for raw/direct aliases, W8, hybrid W4, weight-only/full manifests, physical companions, and all QuaRot directions/modes.
- Native targeted pytest in `ds-v4-w8a8`: 277 passed, 16 deprecation warnings, no failures.
- Independent source safety re-review: passed with no blocker.
- Full clean staging package audit: passed; 38 requested ops present, 216 JSON files parse, 176 kernel JSON/object pairs and 272 binary references validate.
- Real W8 checkpoint headers match the strict I8/global-shape contract, and its QuaRot 16-probe max error is `3.34531069e-6`.
- The available W4 artifact is standard MTP rather than DSpark; a genuine DSpark W4 checkpoint is still required for device acceptance/performance validation.
- Latest schema review: the P2 metadata false positive is fixed with exact, fail-closed filtering for `indexer.quant_type` and `indexer.wq_b_weight` across raw/canonical/direct names.
- The benchmark services are stopped and port 8900 is free.
- The exact formal workload used 32/32 successful requests per run, concurrency 8, 1024 output tokens, and observed input length 8196 tokens for the requested 8k input. AISBench sends one same-length precheck, so Prometheus acceptance deltas cover 33 requests while the CSV/detail results cover the formal 32.
- DSpark5 full-graph raw throughput was 217.7340, 217.6976, and 210.5191 output token/s; acceptance was 81.4241%, 77.3109%, and 77.2772%.
- MTP1 full-graph raw throughput was 191.8100, 190.6709, and 193.5156 output token/s; acceptance was 94.6289%, 94.1427%, and 95.6046%.
- Median DSpark throughput gain over MTP1 is 13.4965%; the slowest DSpark run still exceeds the fastest MTP1 run by 8.7866%. DSpark spread is 3.3142%, 0.3142 percentage points above the approximate 3% target, so the raw-run separation is retained in the report instead of hiding the variance.
- Same-checkpoint no-spec baselines completed with zero speculative activity: DSpark checkpoint 136.1390/136.9068/135.8115 output token/s (median 136.1390, spread 0.8045%); MTP checkpoint 137.4186/136.8683/137.5040 (median 137.4186, spread 0.4626%).
- DSpark5 improves same-checkpoint median throughput by 59.9083%, versus 39.5808% for MTP1. The gain delta is 20.3275 percentage points; normalized DSpark factor exceeds the MTP factor by 14.5633%.
- Both no-spec suites completed 96/96 formal requests, their one-request 8192/128 smoke gates had zero speculative-counter deltas, and every service log was clean throughout the benchmark phase.
- Checkpoint identity audit found identical 100,829 target tensor keys/dtype/shapes and identical 100,834 logical entries. Strict payload identity is false: `layers.0.attn.wq_a.weight` differs by one I8 element out of 4,194,304, with identical scale/offset; same-checkpoint on/off baselines therefore remain the attribution source of truth.
- Final normalized artifact: [`comparisons/normalized/comparison.json`](comparisons/normalized/comparison.json), SHA256 `160bf16d1a9c65ddf80549aa66b01295d1135a77b764eab5e48a629a1ec0eb66`.

## Acceptance-risk invariants

- `wo_a` remains floating point because its specialized path reads the raw weight.
- Every quantized base weight has all companion tensors required by its scheme.
- Routed-expert precision is complete and consistent within each draft stage.
- QuaRot basis transitions are explicit whenever a rotation exists and direction-tested; never inferred silently.
- Checkpoints without QuaRot retain the legacy PR behavior by default.
