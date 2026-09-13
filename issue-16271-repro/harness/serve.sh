#!/bin/bash
# /nt/serve.sh <tag>
# Qwen3.6-35B-A3B (arch Qwen3_5MoeForConditionalGeneration) x DSpark, MRV2, cards 4-7.
# env knobs (exported by the caller):
#   MRV2=1|0                 VLLM_USE_V2_MODEL_RUNNER
#   NT_SEQ_LENS_PROBE=1      count D2H .tolist() in the attention metadata builder
#   NT_SEQ_LENS_CHECK=1      cross-check host list against the device tensor
#   NT_SEQ_LENS_EVERY=N      probe log period
TAG=${1:-run}
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=${MRV2:-1}
PROF_DIR=/nt/prof/${TAG}
export NT_SEQ_LENS_PROBE=${NT_SEQ_LENS_PROBE:-0}
export NT_SEQ_LENS_CHECK=${NT_SEQ_LENS_CHECK:-0}
export NT_SEQ_LENS_EVERY=${NT_SEQ_LENS_EVERY:-2000}

mkdir -p /nt/logs "$PROF_DIR"
LOG=/nt/logs/serve_${TAG}.log
rm -f "$LOG"
rm -rf "${PROF_DIR:?}"/*

nohup vllm serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen3 \
    --trust-remote-code \
    --max-num-seqs 64 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.90 \
    --port 8100 \
    --speculative-config '{"method": "dspark", "model": "/home/models/Qwen3.6-35B-A3B-speculator.dspark", "num_speculative_tokens": 8, "enforce_eager": true}' \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir="$PROF_DIR" \
    --profiler-config.torch_profiler_with_stack=false \
    --profiler-config.torch_profiler_use_gzip=false \
    > "$LOG" 2>&1 &
echo "pid=$! tag=$TAG log=$LOG mrv2=$VLLM_USE_V2_MODEL_RUNNER probe=$NT_SEQ_LENS_PROBE check=$NT_SEQ_LENS_CHECK prof=$PROF_DIR"
