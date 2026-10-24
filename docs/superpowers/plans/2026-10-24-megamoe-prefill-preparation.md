# MegaMoe Prefill Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the measured MegaMoe routed-core advantage through preparation and into repeatable Qwen3.6-35B-A3B whole-model gains on m18 TP4/EP4.

**Architecture:** Keep the V30 CANN kernel, V31 numerical contract, final BF16 TP reduction, shared-expert overlap, and production slow-shape guard unchanged. Test one preparation-only variant in an isolated V31 source copy, then promote it to the existing `71fd094f` integration branch only if real-route correctness and performance gates pass.

**Tech Stack:** Python, Triton Ascend, PyTorch/torch_npu, HCCL, CANN MegaMoe, vllm-ascend, Bash, m18 910B4-1 devices 2/3/4/5.

**Spec:** `megamoe_optimization_20261023/progress.json`, `megamoe_optimization_20261023/replay068/prepare_core_summary126.json` on m18, and the user's request to continue until stable whole-model gain while retaining the baseline and slow-shape protection.

## Global Constraints

- Preserve baseline commits `38036f44fc24536d50e396f8c4d6319066fe0aa0` and `71fd094f4a85ff9c625f778ef2fa117007ffa964`, V30 vendor, model, container image `sha256:8dd949bff550c8e33211bbf6f0daa899c13eab3a1a140105bb2001509a6b568b`, and all known-good receipts.
- Do not relax EP4/EP8 production eligibility before whole-model proof. Any later eligibility change must be a separate test-driven change limited to the accepted configuration and shapes, with all remaining slow shapes protected.
- Use only merged upstream code or original local work; do not import an unmerged community PR.
- Change only `_COPY_BLOCK` in the first candidate. Keep `_ROUTING_BLOCK=256`, sentinel values, BF16/FP32 conversions, routing order, CANN binaries, and reduction order unchanged.
- Use fixed physical devices 2/3/4/5 and ABBA ordering. Do not run concurrent NPU benchmarks or treat profiler timing as performance evidence.
- Do not stop or modify the unrelated `dspark-028rc1` service. Start NPU work only after the selected devices, ports, and worker checks pass.
- Every candidate must pass real-route bitwise output and queued-repeat checks before timing. Whole-model promotion requires both ON arms to beat both OFF arms on output throughput without correctness regressions.
- Do not commit or push unless the user explicitly asks.

## Review Focus

- Sentinel layout: prepared tensors must retain 32 rows with hidden=1, cyclic expert IDs, probability=1/8, and active sentinel routes; Task 1 checks prepared tensors and final output bitwise.
- Mutable queued inputs: repeated calls must not overwrite earlier outputs; Task 1 retains the existing 20-call queued stability check. CANN may modify routing IDs when an active mask is present, so a production no-sentinel helper must clone `topk_ids` and preapply masks only to that clone; hidden states alone may be zero-copy.
- Rank asymmetry: per-rank medians alone can hide a critical-path regression; Tasks 1 and 3 aggregate each iteration by the slowest rank before comparing arms.
- Shape protection: the production selector must still reject EP4/EP8; Task 3 reruns the 15 configuration tests and real fallback startup checks.
- Resource ownership: an unrelated service may occupy the cards; every launch task must preflight devices, ports 8077/29541, process ownership, and source hashes, and must clean only its own workers.

---

### Task 1: Real-Route Preparation Regression Test

**Files:**
- Create remotely: `/data1/megamoe_optimization_20261023/prepare_block128/bench_prepare_candidate.py`
- Create remotely: `/data1/megamoe_optimization_20261023/prepare_block128/manifest.json`
- Reference: `vllm_ascend/ops/triton/megamoe_prepare.py`
- Reference remotely: `/data1/megamoe_optimization_20261023/replay068/bench_v31_operator.py`

**Interfaces:**
- Consumes: sealed real 4K/8K hidden states, top-k IDs/probabilities, packed 3D weights, captured MegaMoe outputs, and the existing four-rank HCCL setup.
- Produces: `baseline_results.json` containing per-rank prepared-output equality, final-output equality, queued stability, and ABBA preparation/full-chain timings.

- [x] **Step 1: Write the failing real-route test**

Implement `bench_prepare_candidate.py BASELINE_SOURCE CANDIDATE_SOURCE OUTPUT` so it calls the production `prepare_cann_megamoe_local_partial` from each source in separate controlled runs, checks all prepared tensors and final MegaMoe outputs against the sealed references, then measures slowest-rank ABBA medians. Its promotion assertion requires at least a 20% preparation-time reduction for both 4K and 8K with no final-output difference. Run it first with the unchanged V31 source as both inputs.

- [x] **Step 2: Run the test against unchanged V31 and verify RED**

Run: four-rank `torchrun` through the existing `replay068/run_env.sh`, after preflight and with no other NPU work.

Expected: all correctness checks PASS; promotion assertion FAILS because preparation reduction is 0%, not because of imports, missing files, ABI resolution, or a test harness exception.

- [x] **Step 3: Preserve the baseline receipt**

Save the script SHA256, source manifests, loaded libraries, device mapping, raw samples, failure assertion, and postflight device/port cleanup in `manifest.json`. Do not alter `replay068` or its receipts.

### Task 2: Single-Factor Copy-Block Candidate

**Files:**
- Modify in an isolated V31 copy only: `vllm_ascend/ops/triton/megamoe_prepare.py:13`
- Modify later only after GREEN: the same path in the integration worktree
- Test: Task 1's `bench_prepare_candidate.py`
- Existing regression: `tests/e2e/pull_request/one_card/test_megamoe_local_partial_prepare.py`

**Interfaces:**
- Consumes: Task 1's baseline harness and sealed inputs.
- Produces: `candidate_results.json` with the same schema as `baseline_results.json`; the public preparation function signature remains unchanged.

- [ ] **Step 1: Apply the minimal candidate**

Change `_COPY_BLOCK` from `4096` to `16384`. Do not change any other source line, Triton launch dimension, dtype, allocation, or CANN file.

- [ ] **Step 2: Run the real-route test and verify GREEN**

Run the same Task 1 command against the candidate source.

Expected: prepared tensors and final outputs remain bitwise equal, 20 queued repeats pass, and both shapes exceed the 20% preparation-time reduction gate. If the gate fails, reject this candidate and return to root-cause analysis rather than tuning several constants together.

- [ ] **Step 3: Run existing correctness and static regressions**

Run the existing local-partial preparation tests, the 15 configuration/startup tests, Ruff on the changed Python file, and `git diff --check`.

Expected: all previously passing tests remain green. Record any broader suite collection failure separately; never convert an unexecuted suite into a pass claim.

- [ ] **Step 4: Profile only if the timing gate passes**

Collect native profiler data for the candidate to confirm that `_prepare_local_partial_kernel` itself accounts for the measured reduction. Do not use the profiled run as the performance result.

### Task 2A: Protected No-Sentinel Candidate

**Files:**
- Modify: `vllm_ascend/ops/triton/megamoe_prepare.py`
- Test: `tests/e2e/pull_request/one_card/test_megamoe_local_partial_prepare.py`
- Diagnostic: `megamoe_optimization_20261023/bench_zero_sentinel.py`

**Interfaces:**
- Consumes: the rejected COPY_BLOCK result and V30's zero-row expert support.
- Produces: hidden zero-copy preparation with independent routing IDs and CANN-compatible active-mask handling.

- [x] **Step 1: Establish the no-copy RED and characterize V30**

The padded implementation failed the no-copy contract. An unmodified V30 diagnostic without sentinel rows then passed real 4K/8K final-output and queued bitwise checks, proving that the large hidden copy was unnecessary for those cases.

- [x] **Step 2: Write protected preparation tests and verify RED**

Cover no-sentinel shapes, hidden aliasing, independent routing IDs, BF16/FP32 probabilities, BOOL/INT8 masks, preapply behavior, and queued storage. After fixing a pytest collection-signature error, unchanged `71fd094f` produced 43 failures.

- [x] **Step 3: Implement and verify GREEN**

Return hidden unchanged, clone IDs, preapply inactive IDs=256 only on that clone, return None for preapply, otherwise return an INT8 mask. Result: 43 NPU tests passed, 52 config/local-partial CPU tests passed, Ruff/format/diff-check passed.

- [x] **Step 4: Revalidate real routes and active masks**

Real 4K/8K plus periodic BOOL/INT8 mask replay passed final-output and queued bitwise checks. Preparation reductions were 37.03%/49.55%; routed-with-prepare reductions were 2.18%/0.73%. Only this protected candidate proceeds to whole-model ABBA.

### Task 3: Whole-Model ABBA Acceptance

**Files:**
- Create remotely: `/data1/megamoe_optimization_20261023/whole_candidate/{serve.sh,configs.json,benchmark.py,acceptance.py}`
- Modify only after the real-route no-sentinel candidate passes: `vllm_ascend/ops/triton/megamoe_prepare.py`
- Evidence: `megamoe_optimization_20261023/progress.json` and `STATUS.txt`

**Interfaces:**
- Consumes: the single-file preparation candidate and unchanged V30 vendor/runtime ABI.
- Produces: matched OFF A1 → ON B1 → ON B2 → OFF A2 whole-model results and a promote/reject decision.

- [x] **Step 1: Validate startup and output before benchmarking**

Start TP4/EP4 with 4K/6K inputs, 256-token outputs, C32, shared overlap enabled, compiled model enabled, and the candidate source. Run the existing 511/513/2K/4K/6K fixed-output, logprob, repeated-output, and loaded-library checks. Verify the production guarded branch still resolves EP4/EP8 to ordinary MoE separately from the diagnostic V31 enablement.

- [x] **Step 2: Run matched whole-model ABBA**

Use 160 requests per shape per arm, no profiler, fixed seed, identical input/output lengths, and the same four physical cards. Stop and verify cleanup between arms.

Expected: all requests complete; both candidate ON arms have higher output throughput than both OFF arms for the accepted shape set. Report TTFT separately and do not hide a TTFT regression.

- [x] **Step 3: Decide promotion from evidence**

Promote the preparation file only if both ON arms beat both OFF arms and correctness passes. Otherwise keep the candidate out of the integration branch and document the measured limitation. A local preparation improvement alone is not a whole-model gain claim. Keep production eligibility unchanged in this task; any proof-backed narrowing of the guard requires a new failing test and a separate minimal change.

- [x] **Step 4: Update durable status and clean resources**

Record exact source diffs, hashes, raw metrics, aggregate formulas, unexecuted checks, and remaining limitations. Verify devices 2/3/4/5, ports 8077/29541, and owned workers are clear. Leave `dspark-028rc1` and all other unrelated services untouched.

### Task 4: Production Enablement and Edge Acceptance

**Files:**
- Modify after a successful Task 3: `vllm_ascend/megamoe_policy.py`
- Modify after a successful Task 3: `vllm_ascend/ascend_config.py`
- Modify after a successful Task 3: `vllm_ascend/ascend_forward_context.py`
- Test: `tests/ut/ops/test_megamoe_config_resolution.py`
- Test: `tests/e2e/pull_request/one_card/test_megamoe_local_partial_prepare.py`

**Interfaces:**
- Consumes: the frozen preparation implementation and Task 3's whole-model evidence.
- Produces: a narrow EP4/E256 local-partial eligibility exception; ordinary EP4 and all EP8 BF16 requests retain the existing fallback.

- [x] **Step 1: Write and run the failing selector test**

Require a validated TP4/EP4 local-partial configuration to retain `enable_fused_mc2=1` and select FUSED_MC2 at 512/4096 tokens, while 4097 uses ALLGATHER and 32 retains the existing EP4 ALLGATHER decode fallback. Keep ordinary EP4 and EP8 fallback assertions unchanged. Run the config suite before the policy edit and record the expected failures.

- [x] **Step 2: Implement the minimal policy exception**

Pass the already validated `mega_moe_local_partial` flag into the shared shape policy from config resolution and runtime selection. Exempt only EP4/E256 local-partial; do not alter other shape thresholds or configuration contracts.

- [x] **Step 3: Rerun regressions and static checks**

Expected: all config/local-partial tests, all 43 preparation NPU tests, Ruff, and `git diff --check` pass. The frozen preparation file's SHA256 must remain unchanged so Task 3's performance evidence still applies.

- [x] **Step 4: Run all-inactive correctness-only replay**

Use the sealed 4K/8K inputs with an all-inactive INT8 mask. Require bitwise equality with the padded V31 reference and 20 queued repeats on all four ranks, followed by device cleanup.

- [x] **Step 5: Run actual production startup smoke**

Start the integration source with TP4/EP4 local-partial enabled. Verify resolved config, actual MegaMoe vendor/operator path, shared-expert overlap, 511/513/2K/4K/6K outputs and logprobs, repeated-output stability, and owned cleanup. Do not repeat ABBA merely because eligibility changed; Task 3 already measures the same frozen runtime path.
