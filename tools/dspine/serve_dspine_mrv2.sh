#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve Qwen3-8B with a DSpine draft on the Ascend MRV2 lane (paired Core).
#
# Requires the paired Core (vLLM v0.33.0 + the DSpine core patch) and this
# Ascend branch installed; the image's stock vllm must be replaced/mounted,
# because the released Core does not carry the DSpine draft model yet.
#
# num_speculative_tokens must be block_size - 1 (16 - 1 = 15 for the b16 drafts).
set -e
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export VLLM_USE_V2_MODEL_RUNNER=1
# CANN builds without aclnnAddRmsNormBias (e.g. 9.2.1) need the math fallback.
export VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS="${VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

TARGET="${TARGET:-Qwen/Qwen3-8B}"
DRAFT="${DRAFT:-tencent/Qwen3-8B-DSpine-b16}"
PORT="${PORT:-8352}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_SPEC="${NUM_SPEC:-$((BLOCK_SIZE - 1))}"

SPEC="{\"method\":\"dspine\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":${NUM_SPEC},\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"greedy\",\"enforce_eager\":true}"

exec vllm serve "${TARGET}" \
  --served-model-name dspine-mrv2 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --enforce-eager \
  --port "${PORT}" \
  --speculative-config "${SPEC}"