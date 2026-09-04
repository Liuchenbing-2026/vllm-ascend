#!/bin/bash
# GLM-5.3-Flash w8a8 on Atlas A2 (8x 910B4-1), TP8 + EP.
# Prefix caching + ACL graph + MTP speculative decoding, all three on at once.
# Measured working: see docs/02-serving.md.
#
# Run INSIDE the container started by docker-run.sh:
#   docker exec glm53s bash -lc "bash /path/to/serve.sh"
#
# NOTE: do not `cd /vllm-workspace` -- the repo checkout there shadows the installed
# vllm package and you get
#   ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)
MODEL=${MODEL:-/data02/GLM-5.3-Flash-w8a8-b0829}
PORT=${PORT:-8000}
LOG=${LOG:-/tmp/glm53_serve.log}

pkill -f "vllm serve $MODEL" 2>/dev/null
pkill -f "VLLM::" 2>/dev/null          # reparented workers survive the first pkill and hold HBM
sleep 8

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_LAUNCH_BLOCKING            # incompatible with ACL graph -- vllm-ascend raises
cd /root

nohup setsid vllm serve "$MODEL" \
  --served-model-name glm53 \
  --trust-remote-code \
  --quantization ascend \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
  --additional-config '{"enable_cpu_binding": false}' \
  --port "$PORT" > "$LOG" 2>&1 < /dev/null &

echo "launched, log=$LOG"
