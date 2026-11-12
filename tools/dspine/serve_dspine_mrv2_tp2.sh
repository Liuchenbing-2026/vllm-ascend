#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve Qwen3.8-27B with a DSpine draft on MRV2 with tensor parallel 2.
#
# The 27B target needs TP=2 on 64 GB chips (bf16 weights + the Qwen3.5 GDN
# mamba page do not fit on one card). draft_tensor_parallel_size stays 1; the
# draft bridge and serial tables are replicated on both ranks.
set -e
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS="${VLLM_ASCEND_DISABLE_FUSED_ADD_RMSNORM_BIAS:-1}"
export VLLM_DSPINE_ALLOW_SLIDING="${VLLM_DSPINE_ALLOW_SLIDING:-1}"
export VLLM_DSPINE_FREE_DRAFT_IO="${VLLM_DSPINE_FREE_DRAFT_IO:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

TARGET="${TARGET:-Qwen/Qwen3.8-27B}"
DRAFT="${DRAFT:-tencent/Qwen3.8-27B-DSpine-b16}"
PORT="${PORT:-8354}"
TP="${TP:-2}"
NUM_SPEC="${NUM_SPEC:-15}"

SPEC="{\"method\":\"dspine\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":${NUM_SPEC},\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"greedy\",\"enforce_eager\":true}"

exec vllm serve "${TARGET}" \
  --served-model-name dspine-mrv2-27b \
  --dtype bfloat16 \
  --tensor-parallel-size "${TP}" \
  --max-model-len 2048 \
  --max-num-seqs 2 \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --port "${PORT}" \
  --speculative-config "${SPEC}"