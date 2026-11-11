#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve Qwen3-8B with a DSpine draft on the Ascend MRV1 speculative-decoding path.
#
# The DSpine checkpoint is a DFlash draft plus a per-layer code bridge, so it is
# served through the "dflash" method; the plugin detects the DSpine marker
# (dflash_config.selected_prefix_code_bridge), rewrites the architecture to
# DFlashDSpineDraftModel and routes the proposer to AscendDspineProposer.
#
# num_speculative_tokens must be block_size - 1 (16 - 1 = 15 for the b16 drafts).
set -e
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# vllm-ascend main paired with vllm main reports 0.33.0 as its version string; the
# MRV1 proposal path needs VLLM_VERSION to select the main-lane compatibility
# branches of the vllm-ascend patches.
export VLLM_VERSION="${VLLM_VERSION:-0.33.1.dev1}"
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

TARGET="${TARGET:-Qwen/Qwen3-8B}"
DRAFT="${DRAFT:-tencent/Qwen3-8B-DSpine-b16}"
PORT="${PORT:-8903}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
NUM_SPEC="${NUM_SPEC:-$((BLOCK_SIZE - 1))}"

SPEC="{\"method\":\"dflash\",\"num_speculative_tokens\":${NUM_SPEC},\"model\":\"${DRAFT}\"}"

exec vllm serve "${TARGET}" \
  --served-model-name qwen3-8b-dspine \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port "${PORT}" \
  --speculative-config "${SPEC}"