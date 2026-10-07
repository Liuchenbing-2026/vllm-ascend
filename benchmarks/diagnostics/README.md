# FIA TND length-interface diagnostics

This experimental branch is based on Ascend
`f06916fc3f9aa8cabe9b0edc5e74a40170a85b72`. It adds diagnostics only;
it does not replace the serving attention implementation or claim to resolve
upstream review concerns. The paired serving reference is vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`, but the operator probes do not load
vLLM or run a model.

## Question under test

Does keeping query layout TND and passing an NPU length Tensor to the installed
FIA v2 interface eliminate host length reads? A Tensor-looking Python call is
not sufficient evidence: the registered length argument may still be a host
`SymInt[]` and extract each scalar during argument conversion.

## Components and isolation

- `fia_length_contract.py`: CPU-only schema and scalar-extraction diagnostic.
  CPU dispatch failure for a host-list call is expected; this is not an NPU
  support or precision failure.
- `fia_tnd_probe.py`: paged BF16 TND output comparisons and synchronized
  operator wall times; batch 1/8, query length 1/9, two live-length values.
  Tolerance is `rtol=0.002, atol=0.002` against the existing FIA v1 operator.
  The scalar-read observer is absent from timed iterations.
- `profile_fia_lengths.py`: separately warmed three-second traces of FIA v2
  with host-list versus NPU Tensor lengths, Level1 + PipeUtilization.
  Profiler warmup and transition to NONE bracket each timed capture window.
- `guard_idle_npu.py`: fail-closed process-table check. Empty process names
  still count as occupied; missing information is not treated as idle.
  This is a snapshot, not a lock or a reservation.

## Reproduce

Base image used for the recorded A2 operator experiment:
`vllm-ascend:dspark-a2-028`, image ID
`sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14`.
The image tag refers to its original framework snapshot. The actual operator
probe uses CANN 9.1.0, torch 2.10.0+cpu, torch-npu 2.10.0.post4 and driver 25.5.1.
Do not describe this standalone probe as a vLLM 0.30 serving benchmark.

Check physical-card ownership on the host and inspect container mapping first.
In the recorded run, the new container exposed only physical cards 4–7;
`npu-smi info -m` mapped them to container logical devices 0–3. Existing work
on physical card 0 was not changed. Set visibility using verified logical IDs.

```bash
python3 benchmarks/diagnostics/guard_idle_npu.py --cards 4 5 6 7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python3 benchmarks/diagnostics/fia_length_contract.py > contract.json
python3 benchmarks/diagnostics/fia_tnd_probe.py \
  --device 0 --output /work/fia_tnd_probe_result.json
python3 benchmarks/diagnostics/profile_fia_lengths.py \
  --device 0 --seconds 3 --output /work/profile_lengths_clean
```

The profiler output directory must not already exist. Run the ownership guard
again before each device job. The standalone probe uses one card; reserving a
four-card container does not mean TP4 inference was tested.

## Interpretation and next gate

In the recorded A2 experiment, all 24 comparisons passed. For batch 8, the
native v2 Tensor-length call requested eight scalar reads per call. The same
shape with a host length list requested none. Direct Tensor passing therefore
did not establish a device-only length path on this installed stack.

The list case already has lengths on the host and excludes the work needed to
obtain them in a real model. Operator wall-time ratios are not serving speedups.
The cases are noncausal without a sliding window and do not validate the full
DSpark path, graph replay, model quality, acceptance or production throughput.

A genuine device metadata/tiling implementation must be compiled and verified
before deleting the existing upper-bound, gather, mask or fallback machinery.
Then compare old optimization off/on and the new implementation under identical
TP4 serving inputs, with warmup plus two formal rounds and separate profiling.
No production implementation is changed by these diagnostics.

## Validation

```bash
python3 tests/ut/benchmarks/test_guard_idle_npu.py
ruff check benchmarks/diagnostics tests/ut/benchmarks/test_guard_idle_npu.py
ruff format --check benchmarks/diagnostics tests/ut/benchmarks/test_guard_idle_npu.py
```

The target repository pins Ruff 0.14.0. Four ownership-guard regression cases
were run locally. NPU tests and recorded results belong to the documented
installed stack; upstream CI and TP4 serving validation remain pending.

## Recorded operator result

Batch 8, query length 9, BF16, paged TND; means across the two live-length
values (20 timed calls per value, three warmup calls). These are synchronized
operator wall times, not formal repeated serving benchmark rounds.

| Call | Operator wall time (ms) | Scalar reads per call |
| --- | ---: | ---: |
| FIA v1, host length list | 0.1303 | 0 |
| FIA v2, host length list | 0.1314 | 0 |
| FIA v2, NPU length Tensor | 0.4215 | 8 |

The separate three-second traces recorded 14,274 host-list calls and 4,450
Tensor-length calls. The latter contained 35,600 scalar reads, 35,600 stream
synchronizations and 35,600 memcpy API calls: eight of each per marked call.
The host-list trace had none of these per-call events. Both traces also contain
two device synchronizations used by capture boundaries; those are not attributed
to length conversion. Kernel detail CSVs have 47 columns and marker counts match
submitted calls. The second capture completed with explicit schedule transitions.

Raw summaries are in [results/20261007-a2](results/20261007-a2).
This disproves the direct-Tensor shortcut on this installed stack; it does not
measure or disprove a genuinely device-side tiling implementation.

The full repository `bash format.sh ci` was attempted. Relevant Ruff, format,
import-policy and ownership-guard checks passed after fixing the new diagnostic's
import policy. Full CI is not claimed: the local machine lacks gitleaks/wget and
shellcheck, so those repository-wide checks could not complete. No runtime
feature or merge-ready fix is included in this branch.
