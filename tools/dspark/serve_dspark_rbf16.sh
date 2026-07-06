#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve DeepSeek-V4-Flash DSpark with the rbf16 (dequantized, rotated) draft.
# Topology: single node, 8x W8A8. Target enters the ACL graph; the DSpark draft
# runs eager (the reference design). Steady-state AL ~3.3 on math/coding traffic.
set -euo pipefail

TARGET="${TARGET:-/data1/DeepSeek-V4-Flash-DSpark-w8a8}"
DRAFT="${DRAFT:-/data1/DeepSeek-V4-Flash-DSpark-rbf16-draft}"
PORT="${PORT:-8902}"
NUM_SPEC="${NUM_SPEC:-5}"        # DSV4 DSpark checkpoints are trained block-5
TP="${TP:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"

# bug#1: append :$PYTHONPATH so the CANN acl module path added by set_env is
# preserved instead of being clobbered by our export.
if [[ -f "${ASCEND_HOME:-/usr/local/Ascend}/ascend-toolkit/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME:-/usr/local/Ascend}/ascend-toolkit/set_env.sh"
fi
export PYTHONPATH="${EXTRA_PYTHONPATH:-}:${PYTHONPATH:-}"

# Route to the DSpark proposer: the draft is the target checkpoint (self-draft),
# so no separate `model` is needed. method=mtp because v0.23.0 has no dspark
# method literal; the plugin detects the DeepSeekV4DSpark architecture and
# routes to the semi-autoregressive DSpark proposer.
SPEC=$(cat <<JSON
{"method": "mtp", "num_speculative_tokens": ${NUM_SPEC}, "model": "${DRAFT}"}
JSON
)

export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"

exec vllm serve "${TARGET}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --block-size "${BLOCK_SIZE}" \
    --trust-remote-code \
    --speculative-config "${SPEC}"
