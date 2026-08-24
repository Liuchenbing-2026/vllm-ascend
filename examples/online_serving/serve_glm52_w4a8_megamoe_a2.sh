#!/usr/bin/env bash
set -euo pipefail

# Validated with:
#   vLLM:        568afb3a13806beb53bb2e6bd518269357b237c0
#   vLLM Ascend: dde3e817c069ba22425c590f4ca79f85976a983e
#   CANN:        9.1.0
#   Device:      8 x Ascend 910B4-1

model_path="${MODEL_PATH:-/data01/models/GLM-5.2-w4a8c8}"
draft_model_path="${DRAFT_MODEL_PATH:-/data01/models/GLM-5.2-DSpark-NPU-0805}"
host="${HOST:-0.0.0.0}"
port="${PORT:-8077}"
served_model_name="${SERVED_MODEL_NAME:-glm-5.2-w4a8c8-dspark}"

[[ -d "$model_path" ]] || {
    printf 'Target model directory does not exist: %s\n' "$model_path" >&2
    exit 1
}
[[ -d "$draft_model_path" ]] || {
    printf 'Draft model directory does not exist: %s\n' "$draft_model_path" >&2
    exit 1
}
[[ "$port" =~ ^[0-9]+$ ]] || {
    printf 'PORT must be numeric: %s\n' "$port" >&2
    exit 1
}

export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=2
export VLLM_USE_V2_MODEL_RUNNER=0

speculative_config="$(printf \
    '{"method":"dspark","model":"%s","num_speculative_tokens":7,"enforce_eager":true}' \
    "$draft_model_path")"

exec vllm serve "$model_path" \
    --host "$host" \
    --port "$port" \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name "$served_model_name" \
    --reasoning-parser glm45 \
    --max-num-seqs 2 \
    --max-model-len 36864 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.97 \
    --quantization ascend \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --additional-config '{"multistream_overlap_shared_expert":false}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config "$speculative_config"
