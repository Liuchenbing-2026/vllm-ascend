#!/bin/bash
# usage: serve_base_eager.sh <model_dir> <served_name> <tp> <devices> <port> <max_len>
mkdir -p /root/kvtq_integration/bench
cd /root/kvtq_integration
export ASCEND_RT_VISIBLE_DEVICES=$4
python3 -m vllm.entrypoints.openai.api_server \
  --model $1 --served-model-name $2 \
  --tensor-parallel-size $3 \
  --max-model-len $6 \
  --max-num-batched-tokens $6 \
  --gpu-memory-utilization 0.92 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --port $5 > /root/kvtq_integration/bench/serve_base_eager_$2.log 2>&1
