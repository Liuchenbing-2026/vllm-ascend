#!/bin/bash
# /nt/bench.sh <tag> <phase> [num_prompts] [concurrency]
# Fixed random-dataset benchmark against the running server on :8100.
TAG=${1:-run}
PHASE=${2:-measure}
NP=${3:-96}
CC=${4:-8}
OUT=/nt/logs/bench_${TAG}_${PHASE}.log
mkdir -p /nt/logs
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8100 \
  --model /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen3 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --num-prompts "$NP" \
  --max-concurrency "$CC" \
  --seed 1234 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  > "$OUT" 2>&1
rc=$?
echo "--- $OUT (rc=$rc)"
grep -E "Successful requests|Benchmark duration|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|P99 ITL|Mean E2EL|Median E2EL" "$OUT"
