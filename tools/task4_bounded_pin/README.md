# Task 4: bounded host-pin diagnostic on rc2

This is a diagnostic branch, not a production fix or an upstream PR candidate.
Its V2 runner requires an approved lease and never returns from model loading
to profiling, KV allocation, KV transfer, warmup or inference serving.
Do not deploy this branch as a service.

## Question and limits

After complete real TP8 model loading, can rank zero allocate increasingly large
pinned host blocks? The four retained allocations are 32, 128, 256 and 1024 MiB.
The new probe request is at most **1440 MiB = 1.40625 GiB**, both per rank and
across the host. Ranks 1–7 allocate no probe buffers. Model-loader allocations
and other runtime memory are additional and are captured, not covered by this
probe quota.

This isolates the single-region allocation step. It intentionally stops before
KV cache initialization and does not reconstruct the original 46-region pool.
Passing does not close the original 224 GiB failure, prove transfer correctness,
measure performance, or justify expanding to the original pool. There is no
command-line option to expand the fixed stage A budget.

All eight cards are needed to hold the complete TP8 model; a single-card process
would not reproduce the same full-weight host/device context. No device window
is reserved by preparing or packaging these files.

## Fixed versions

| Component | Identity |
| --- | --- |
| Ascend base | `v0.30.0rc2`, `8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| vLLM, unchanged | `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, the image's verified pairing |
| Diagnostic branch | `task4-207001-rc2-bounded-pin-20261018`; exact clean HEAD is recorded by `prepare.py` |
| Image ID | `sha256:0faf08fb90dd21807266cb719b4947b0111b3bd5f58c46424810a8ecabe82b98` |
| Image digest | `quay.m.daocloud.io/ascend/vllm-ascend@sha256:8e375bf1cb4a98ecf7d487c7f98a28d1c3a7e24f7f8cfbadbb37914ed68b2a30` |
| Installed torch_npu | `2.10.0.post7` |

The original failure used a different fixed nightly/0.31/post4 combination.
Results on this rc2/post7 pair must be reported separately. Git HEAD is not
treated as imported-source proof: container preflight checks the actual runner
and helper import paths and SHA256 against the packaged manifest. A changed
API must be reported before choosing a separately fixed nightly pair; no
automatic fallback or package installation is implemented.

## CPU checks and packaging

Run from this branch's worktree; these tests use fake allocators and do not
import torch or allocate large buffers:

```bash
PYTHONPATH=vllm_ascend:tools/task4_bounded_pin \
  python3 -m unittest discover -s tools/task4_bounded_pin -p 'test_*.py' -v
bash -n tools/task4_bounded_pin/entrypoint.sh tools/task4_bounded_pin/launch.sh
```

Tests cover eight-rank synchronization, rank-zero-only allocation, retained
buffers, exact four-step and total quotas, duplicate workers, repeated steps,
the first failure preventing subsequent allocations, old/different scope,
expired/changed leases, original 224 GiB configuration, dummy weights, wrong
parallelism, host/cgroup headroom, no Docker call without approval, container
ownership, idle-card evidence, and direct-child reaping with no future action.
CPU checks cannot prove NPU/backend behavior; no hardware test is claimed here.

After committing and verifying the independent branch has no linked PR:

```bash
python3 tools/task4_bounded_pin/prepare.py \
  --output /absolute/new/private/task4-stage-a-attempt \
  --model-path /data01/models/DeepSeek-V4-Flash-0731-w8a8
python3 /absolute/new/private/task4-stage-a-attempt/controller.py \
  --lease /absolute/new/private/task4-stage-a-attempt/lease.json
```

`prepare.py` requires a clean committed worktree. It creates an incremental Git
bundle, payload/source hash manifest, and a **PENDING** lease template. The
default controller invocation only previews the command; it never calls Docker.
The package contains no credentials, metrics or original failure logs.

## Only after a new supervisor reservation

Copy the prepared package to a fresh directory on the assigned Linux host.
The supervisor must supply an explicitly approved `lease_id`, integer
`start_epoch`/`end_epoch` for exactly 25 minutes, and `status: APPROVED`.
The reviewed scope, physical cards 0–7, exact SHAs, single allocating rank and
fixed byte quotas must remain unchanged. Never reuse an old attempt/output
directory. Host model path must point to the reviewed complete 74-shard model.

Start within the first minute of the newly assigned window:

```bash
python3 /absolute/new/private/task4-stage-a-attempt/controller.py \
  --lease /absolute/new/private/task4-stage-a-attempt/lease.json --execute
```

This is the only device-execution command; it has **not been executed** as part
of CPU preparation. No benchmark command is supplied because benchmarking is
outside this stage. The container uses the existing fixed image ID, no network,
no published port and no automatic restart. The model and driver are mounted
read-only. Source checkout happens only inside the new owned container, with
no changes to other users' containers or host repositories.

The 25-minute budget is 3 minutes preflight, 10 loading, 5 probing, 3 cleanup
and 4 audit. The controller conservatively counts its 10-minute load timer from
container start, including in-container preflight. Both worker leases and a
separate direct-process watchdog stop new work at minute 18, reserving cleanup
and audit time. Late preflight, missing samples, missing peers, identity or
quota mismatch, low headroom, allocation error or any timeout stops the attempt;
there is no retry, restart, pool fallback, cache flush or NUMA/memlock change.

## Evidence and cleanup

Every worker records before-model, model-loaded, four before/after-pin pairs,
completion and exceptions, including MemAvailable, per-node NUMA, numa_maps,
buddy info, process status/limits, memlock, cgroups and raw host allocator stats.
No arithmetic interprets inconsistent or negative allocator counters as cache
capacity. Every allocation follows all eight workers' pre-pin samples; rank
zero retains all successful buffers until controlled container stop.

The controller preserves command argv, immutable container ID, image identity,
actual source import hashes, full server log, first error, state/exit and NPU
snapshots. Docker exit 137 with `OOMKilled=false` after controlled stop is not
inference success. The controller, logger and watchdog have separate identity
records; direct children are terminated and reaped, without shell process
substitution or `tee`. Containers are retained stopped for evidence.

After the controller returns, the supervisor must independently inspect its
recorded PID/start identity, logger/watchdog absence, container state and cards
before releasing the reservation. A completion file alone is not proof of
resource release. All runtime logs and private metrics remain in the private
attempt directory; do not push them to the code repository.
