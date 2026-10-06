#!/bin/bash
# usage: bench_cells.sh <label> <port>
cd /tmp
run() {
  echo "=== label=$1 input=$2 conc=$3 num_prompts=$4 ==="
  vllm bench serve --backend openai --model /home/models/Qwen3-30B-A3B \
    --served-model-name qwen3-30b --host 127.0.0.1 --port $5 \
    --dataset-name random --random-input-len $2 --random-output-len 1024 \
    --num-prompts $4 --max-concurrency $3 --ignore-eos \
    --save-result --result-dir /root/kvtq_integration/bench \
    --result-filename ${1}_in${2}_c${3}.json 2>&1 | grep -aE "Successful|Throughput|TTFT|TPOT|Error|error" | head -12
}
run $1 16384 32 48 $2
run $1 32768 32 48 $2
run $1 32768 48 64 $2
echo "CELLS $1 DONE"
