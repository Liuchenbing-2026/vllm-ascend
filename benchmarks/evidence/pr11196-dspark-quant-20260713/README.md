# PR #11196 DSpark quantization evidence

This package records the implementation and validation evidence for adapting
the PR #11196 DSpark drafter to ModelSlim W8A8 and hybrid W4A8 checkpoint
contracts. The implementation was developed from the public PR interfaces and
the observed ModelSlim schema; no code was copied or transplanted from another
implementation.

## Status

- W8A8 is validated on an Ascend eight-device system for loading, custom-op
  precision, graph execution, non-zero acceptance, and end-to-end performance.
- Hybrid W4A8 is covered by fail-closed schema, loader, dtype, shape, expert,
  and QuaRot contract tests. It is not claimed as device-certified because the
  available W4A8 checkpoint is a standard one-stage MTP artifact rather than a
  genuine DSpark checkpoint.
- All benchmark services were stopped after collection; the test port was
  verified free.

## Workload and headline results

AISBench 3.1 requested 8192 input tokens and observed 8196, with 1024 output
tokens, concurrency 8, 32 formal requests per run, seed 1, TP8/DP1/EP, and
target `FULL_DECODE_ONLY` graph mode. Each mode was repeated three times.

| Mode | Throughput runs (output tok/s) | Median | Spread | Median acceptance | Median TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| DSpark5 | 217.7340 / 217.6976 / 210.5191 | 217.6976 | 3.3142% | 77.3109% | 31.9 ms |
| DSpark checkpoint, no spec | 136.1390 / 136.9068 / 135.8115 | 136.1390 | 0.8045% | n/a | 55.4 ms |
| MTP1 | 191.8100 / 190.6709 / 193.5156 | 191.8100 | 1.4831% | 94.6289% | 37.9 ms |
| MTP checkpoint, no spec | 137.4186 / 136.8683 / 137.5040 | 137.4186 | 0.4626% | n/a | 54.8 ms |

DSpark5 improves median output throughput by 59.9083% against its own
checkpoint with speculative decoding disabled. MTP1 improves its corresponding
baseline by 39.5808%. The same-checkpoint gain delta is 20.3275 percentage
points, and the normalized DSpark factor is 14.5633% ahead. The direct
speculative-service comparison is 13.4965% in favor of DSpark5.

All 384 formal requests succeeded. Service logs had zero benchmark-phase error
matches. The lifecycle audit separately records shutdown-only tracebacks from
two services after SIGTERM; they are not counted as benchmark failures.

## Evidence map

- [`integration_report.md`](integration_report.md) explains the checkpoint
  contract, implementation, safety decisions, and performance interpretation.
- [`validation_report.md`](validation_report.md) records CPU, package, device,
  precision, graph, acceptance, and benchmark validation.
- [`implementation_checklist.md`](implementation_checklist.md) preserves the
  implementation boundary and completion checklist.
- [`comparisons/normalized/comparison.json`](comparisons/normalized/comparison.json)
  is the normalized on/off attribution source of truth.
- [`comparisons/normalized/service_log_audit.json`](comparisons/normalized/service_log_audit.json)
  separates benchmark-phase health from service shutdown artifacts.
- `modes/` contains per-mode and per-run AISBench aggregates.
- `gates/` contains smoke and acceptance-gate summaries.
- `runtime/` contains code, model-identifier, protocol, package, and artifact
  fingerprints.
- [`runtime/source-provenance.json`](runtime/source-provenance.json) binds the
  submitted functional commit to the machine-tested worktree: all 2,436 regular
  source and test files in scope have identical Git blobs, with zero mismatch.
- [`MANIFEST.json`](MANIFEST.json) inventories the 98 collected payload
  files, including their original source paths, byte sizes, and SHA256 digests.
  Portable path operands and checkpoint identifiers are repository-normalized
  and marked as such. Documentation files added in Git are intentionally
  outside that collection manifest.

Integrity anchors:

- Normalized comparison SHA256:
  `160bf16d1a9c65ddf80549aa66b01295d1135a77b764eab5e48a629a1ec0eb66`
- Service lifecycle audit SHA256:
  `0ba55b612bdfba247783035c51066d28d5a5d8b3b49e3db8953ed7979a7fd275`

## Intentional exclusions

The package excludes model checkpoints and tensor payloads, per-request prompts
and generated predictions, full Prometheus dumps, and full server logs. These
are unnecessary for reviewing the aggregate claim and would add large,
potentially sensitive artifacts. The compact lifecycle audit and all formal
per-run result summaries are included instead.
