#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve DeepSeek-V4-Flash DSpark with the rbf16 (dequantized, rotated) draft.
# Topology: single node, 8x W8A8. Target enters the ACL graph; the DSpark draft
# runs eager (the reference design). Steady-state AL ~3.3 on math/coding traffic.
#
# The DSpark checkpoint is detected by the plugin (DeepSeekV4DSpark architecture)
# and routed to the semi-autoregressive DSpark proposer. method is "mtp" because
# vLLM v0.23.0 core has no dspark method literal; the plugin's config patch
# rewrites the arch and enables parallel drafting.
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh

export VLLM_VERSION="${VLLM_VERSION:-0.23.0}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True HCCL_BUFFSIZE=1024
export VLLM_ASCEND_APPLY_DSV4_PATCH=1 VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_ENABLE_FUSED_MC2=0

TARGET="${TARGET:-/data1/DeepSeek-V4-Flash-DSpark-w8a8}"
DRAFT="${DRAFT:-/data1/DeepSeek-V4-Flash-DSpark-rbf16-draft}"
PORT="${PORT:-8902}"
NUM_SPEC="${NUM_SPEC:-5}"

SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC},\"model\":\"${DRAFT}\"}"

cd /data1
exec vllm serve "${TARGET}" \
  --max_model_len 8192 --max-num-batched-tokens 4096 --served-model-name dsv4-dspark-w8a8 \
  --gpu-memory-utilization 0.9 --max-num-seqs 8 --tensor-parallel-size 8 --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 --safetensors-load-strategy prefetch --quantization ascend \
  --port "${PORT}" --speculative-config "${SPEC}" --api-server-count 1 --block-size 128 --enforce-eager
