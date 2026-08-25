# GLM-5.2 W4A8 CANN MegaMoe A2 integration report

## Scope

- Repository: `Liuchenbing-2026/vllm-ascend`
- Migration branch: `pr11200_glm52_w4a8_megamoe_v027`
- Official base: `vllm-project/vllm-ascend@6fd6ea161b904ee4b379f6a18bc29d3e076cf25a`
- Paired vLLM line: `v0.27.1` (`vllm-project/vllm@ba07e4a` in the base commit message)
- Source adaptation baseline: `pr11200_glm52_w4a8_megamoe_ready@c3c06f1f5c7f35af976dc3be88ce8b75e4d7255b`

## Integration strategy

The 0.27 main line already contains the generic CANN MegaMoe W4A8 path and
several merged follow-up fixes. The migration therefore keeps those upstream
implementations and adds only the A2-specific compatibility layer:

- Select CANN MegaMoe through the upstream `enable_fused_mc2=1` API for validated
  V1, DP1, W8A8/W4A8 shapes; unsupported batches retain the standard MoE fallback.
- Preserve packed INT32 W4 parameters for fallback while exposing per-expert
  INT8 views over the same storage to CANN MegaMoe.
- Add deterministic dummy routes covering every expert and account for their
  capacity in symmetric/HCCL buffer sizing.
- Retain graph execution for supported uniform decode batches and skip the
  compiled path for unsupported dynamic batches.
- Disable shared-expert multistream overlap for this path, matching the upstream
  MegaMoe restriction.

No unmerged third-party PR was imported. The old private
`VLLM_ASCEND_ENABLE_FUSED_MC2=2` mode is intentionally not restored.

## Preserved known-good baseline

- Runtime image: `codex/vllm-ascend-megamoe:fix-dde3e817-v2`
- Image ID: `sha256:f0ed636b85bd999c47028226cea51b43063dbfb8e2ff1a4918fd8bfed8f1fe25`
- Old vLLM Ascend runtime: `dde3e817`
- Old vLLM runtime: `568afb3a`
- CANN: `9.1`
- Hardware: 8 x Ascend 910B4-1
- Model: `/data01/models/GLM-5.2-w4a8c8`
- Draft model: `/data01/models/GLM-5.2-DSpark-NPU-0805`
- Port/devices: `8077`, devices `0-7`

These runtime results are historical regression evidence, not evidence that the
new 0.27.1 migration has completed NPU service validation.

## Validation

- Exact-base blob identity: PASS (eight tracked source/test files)
- Ruff 0.14.0 format/check: PASS (nine changed Python files)
- Python AST/bytecode parse: PASS (local AST and remote `py_compile`)
- LF/no-CR byte check: PASS before SFTP, with SHA256 verified after upload
- Remote `git diff --check`: PASS
- Targeted unit tests: PASS (`8 passed`) in image
  `sha256:617cabd987784c883553a9f0f0f68a479d7aa1545fbbbca478d08d534f3b9edb`
  with vLLM `0.27.1+empty` and vLLM Ascend `g6fd6ea161`
- NPU service/correctness/performance: NOT RUN (not part of the requested rebase)

## Result

- Adaptation commit: `280620ff94ab09166bb9d552693a6db6b40bbe97`
- Deterministic test fix: `2c11b74abb95b182d2d8eb0ff556ec26e0fc4ed5`
- Documentation commit: this commit
- Remote branch: `Liuchenbing-2026/vllm-ascend:pr11200_glm52_w4a8_megamoe_v027`
