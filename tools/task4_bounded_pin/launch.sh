#!/usr/bin/env bash
set -Eeuo pipefail
# Existing runtime knobs, fixed to the historical full-model load recipe.
export OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=1024 HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
unset VLLM_USE_SIMPLE_KV_OFFLOAD
cd /vllm-workspace/vllm-ascend
# No offload/connector configuration. Diagnostic load_model never returns.
# No ports are published; the container has no external network.
exec vllm serve /models/dsv4 \
  --host 127.0.0.1 --port 18031 --served-model-name task4-pin-diagnostic \
  --max-model-len 69632 --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 --num-gpu-blocks-override 4096 \
  --data-parallel-size 1 --tensor-parallel-size 8 --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 --default-chat-template-kwargs '{"thinking":false}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
  --quantization ascend --block-size 128 --attention_config.indexer_kv_dtype int8 \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --async-scheduling --trust-remote-code \
  --speculative-config '{"num_speculative_tokens":7,"method":"dspark","enforce_eager":true}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":false,"multistream_overlap_shared_expert":true}'
