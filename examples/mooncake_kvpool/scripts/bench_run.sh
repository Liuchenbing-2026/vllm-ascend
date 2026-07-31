#!/bin/bash
# 前缀复用压测。用法: bench_run.sh <tag> [num_prefixes]
#   num_prefixes=64 且 num_prompts=64  -> 冷路径（每个前缀只出现一次）
#   num_prefixes=8  且 num_prompts=64  -> 热路径（每个前缀复用 8 次）
set -e
TAG=${1:?usage: bench_run.sh <tag> [num_prefixes]}
NPREF=${2:-8}
MCSSD_ROOT=${MCSSD_ROOT:-/data1/mcssd}
MODEL=${MCSSD_MODEL:-/data1/Qwen3-14B}
PORT=${MCSSD_PORT:-8100}
mkdir -p "$MCSSD_ROOT/bench"

# seed 固定，保证跨轮次前缀内容一致（否则热路径测不出复用）
vllm bench serve --backend vllm --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" \
  --dataset-name prefix_repetition \
  --num-prompts ${MCSSD_NUM_PROMPTS:-64} \
  --prefix-repetition-prefix-len ${MCSSD_PREFIX_LEN:-8192} \
  --prefix-repetition-suffix-len 256 \
  --prefix-repetition-output-len 128 \
  --prefix-repetition-num-prefixes "$NPREF" \
  --max-concurrency ${MCSSD_CONCURRENCY:-8} \
  --seed 42 \
  --percentile-metrics ttft,tpot,e2el \
  2>&1 | tee "$MCSSD_ROOT/bench/$TAG.txt"

# 池模式下必查：写成功但读失败时，命中率指标依然漂亮，只有这里能看出问题
SERVE_LOG=$(ls -t "$MCSSD_ROOT"/logs/serve_*.log 2>/dev/null | head -1)
if [ -n "$SERVE_LOG" ]; then
  nfail=$(grep -c 'TRANSFER_FAIL\|error: -800' "$SERVE_LOG" 2>/dev/null || echo 0)
  echo "---- transfer failures in $(basename "$SERVE_LOG"): $nfail ----"
  [ "$nfail" -gt 0 ] && echo "!! KV 池读路径异常，本轮 TTFT 数据反映的是 recompute 回退，不是池命中"
  grep -oE 'External prefix cache hit rate: [0-9.]+%' "$SERVE_LOG" | tail -1
fi
