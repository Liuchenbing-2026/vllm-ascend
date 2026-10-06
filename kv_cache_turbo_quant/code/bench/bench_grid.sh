#!/bin/bash
# usage: bench_grid.sh <label>
cd /tmp
mkdir -p /root/kvtq_integration/bench
for INPUT in 16384 32768; do
  for CONC in 1 4 16 32; do
    case $CONC in 1) NP=4;; 4) NP=8;; 16) NP=32;; 32) NP=48;; esac
    echo "=== label=$1 input=$INPUT conc=$CONC num_prompts=$NP ==="
    vllm bench serve --backend openai --model /home/models/Qwen3-30B-A3B \
      --served-model-name qwen3-30b --host 127.0.0.1 --port 8377 \
      --dataset-name random --random-input-len $INPUT --random-output-len 1024 \
      --num-prompts $NP --max-concurrency $CONC --ignore-eos \
      --save-result --result-dir /root/kvtq_integration/bench \
      --result-filename ${1}_in${INPUT}_c${CONC}.json 2>&1 | grep -aE "Successful|Throughput|TTFT|TPOT|ITL|Mean|Median|Error|error" | head -20
  done
done
echo "GRID $1 DONE"
