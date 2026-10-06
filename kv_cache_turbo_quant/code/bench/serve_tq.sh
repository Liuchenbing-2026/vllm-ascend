#!/bin/bash
# usage: serve_tq.sh <model_dir> <served_name> <tp> <devices> <port> <max_len> <eager 0|1>
mkdir -p /root/kvtq_integration/bench
cd /root/kvtq_integration
export VLLM_ASCEND_KVTQ_STORE=1
export ASCEND_RT_VISIBLE_DEVICES=$4
python3 -m vllm.entrypoints.openai.api_server \
  --model $1 --served-model-name $2 \
  --tensor-parallel-size $3 \
  --max-model-len $6 \
  --max-num-batched-tokens $6 \
  --gpu-memory-utilization 0.92 \
  --no-enable-prefix-caching \
  $([ "$7" = "1" ] && echo --enforce-eager) \
  --port $5 > /root/kvtq_integration/bench/serve_tq_$2.log 2>&1
