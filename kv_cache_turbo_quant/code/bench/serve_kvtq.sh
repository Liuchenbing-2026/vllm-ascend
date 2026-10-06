#!/bin/bash
# usage: serve_kvtq.sh <0|1>  (1 = shadow turboquant on)
mkdir -p /root/kvtq_integration/bench
cd /root/kvtq_integration
export VLLM_ASCEND_KVTQ=$1
export ASCEND_RT_VISIBLE_DEVICES=2,3
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/models/Qwen3-30B-A3B \
  --served-model-name qwen3-30b \
  --tensor-parallel-size 2 \
  --max-model-len 40960 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --no-enable-prefix-caching \
  --port 8377 > /root/kvtq_integration/bench/serve_kvtq$1.log 2>&1
