#!/bin/bash
# Round 12 -- do the two syncs actually cost what the segment timing says, and does
# removing them survive a correctness check?
#
# Four cells, all MTP k=1, identical shape (C64 / 512-in / 2048-out / 128 prompts):
#   nospec      the bar to beat (1880.38 / 1900.91 tok/s in rounds 5 and 7)
#   mtp_probe   unpatched MTP with instrumentation -- the paired baseline
#   mtp_A       device-side gather replaces sync A
#   mtp_AB      A plus skipping the seq_lens correction (sync B)
#
# The whole point of mtp_A as a separate cell is the prediction that it is worth only
# ~2 ms because the wall moves from A to B. The seq_sync segment makes that visible
# instead of inferable: if A's 28.92 ms reappears as seq_sync in the mtp_A cell, the
# two-sync model is confirmed and mtp_AB is the only configuration that can win.
#
# Every cell also runs a greedy byte-exact dump before its benchmark. A patched cell
# whose SHA differs from mtp_probe's has changed the model's output and its throughput
# number is void regardless of how good it looks.
set -u
LOCKDIR=/tmp/nt_mtp_harness.lock
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "LOCK_HELD"; exit 3; fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -f "$LOCKDIR/pid" 2>/dev/null; rmdir "$LOCKDIR" 2>/dev/null' EXIT

PORT=8011
MODEL=/models/Qwen3.6-35B-A3B
RES=/work/results
OUT=$RES/round12.txt
A=/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py

metrics() {
  curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
    | grep -E '^vllm:spec_decode' | awk '{printf "%s=%s ", $1, $2}'
}

apply_patch() {   # $1 = probe|A|AB
  cp "$A.ntorig" "$A"
  python3 /work/scripts/patch.py "$1" || return 1
  python3 -m py_compile "$A" || return 1
  find /vllm-workspace -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  return 0
}

start() {   # $1=arm(nospec|mtp1) $2=tag
  local arm=$1 tag=$2 log=/work/logs/serve_$2.log
  : > "$log"
  local SPEC=""
  [ "$arm" = "mtp1" ] && SPEC='--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":1}'
  rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null
  (
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export HCCL_IF_IP=192.168.99.42
    export GLOO_SOCKET_IFNAME=enp67s0f0 TP_SOCKET_IFNAME=enp67s0f0 HCCL_SOCKET_IFNAME=enp67s0f0
    export HCCL_OP_EXPANSION_MODE=AIV PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024 OMP_NUM_THREADS=100 TASK_QUEUE_ENABLE=1
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
    # shellcheck disable=SC2086
    exec vllm serve "$MODEL" --served-model-name Qwen36 \
      --host 0.0.0.0 --port $PORT --tensor-parallel-size 4 \
      --max-num-seqs 64 --max-model-len 131072 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.90 --async-scheduling --trust-remote-code \
      --enable-expert-parallel --no-enable-prefix-caching --enable-chunked-prefill \
      --mamba-ssm-cache-dtype bfloat16 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
      $SPEC
  ) >> "$log" 2>&1 &
  echo $! > /tmp/s_$tag.pid
  for _ in $(seq 1 360); do
    sleep 5
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200 && { echo "### $tag healthy"; return 0; }
    kill -0 "$(cat /tmp/s_$tag.pid)" 2>/dev/null || { echo "### $tag DIED"; return 1; }
  done
  echo "### $tag NO_HEALTH"; return 1
}

stop() {
  local tag=$1
  kill "$(cat /tmp/s_$tag.pid 2>/dev/null)" 2>/dev/null
  for _ in 1 2 3; do
    for q in $(pgrep -f 'VLLM::' 2>/dev/null); do kill -9 "$q" 2>/dev/null; done
    for q in $(pgrep -f 'vllm serve' 2>/dev/null); do kill -9 "$q" 2>/dev/null; done
    sleep 10
  done
  echo "### $tag stopped"
}

cell() {   # $1=arm $2=mode(probe|A|AB|none) $3=tag
  local arm=$1 mode=$2 tag=$3
  if [ "$mode" = none ]; then
    cp "$A.ntorig" "$A"
    find /vllm-workspace -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  else
    apply_patch "$mode" || { echo "### $tag PATCH_FAILED"; return 1; }
  fi
  start "$arm" "$tag" || { stop "$tag"; return 1; }

  # warmup first: a cold server runs at ~68% of warm
  vllm bench serve --backend openai-chat --model "$MODEL" \
    --base-url "http://localhost:$PORT" --endpoint /v1/chat/completions \
    --num-prompts 64 --trust-remote-code --ignore-eos --seed 12345 \
    --served-model-name Qwen36 --max-concurrency 64 \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --random-range-ratio 0 > /work/logs/b12_${tag}_warm.log 2>&1

  local sha
  sha=$(python3 /work/scripts/greedy.py "$tag" 2>&1 | tail -1)

  local pre post out=/work/logs/b12_${tag}_dec.log
  pre=$(metrics)
  vllm bench serve --backend openai-chat --model "$MODEL" \
    --base-url "http://localhost:$PORT" --endpoint /v1/chat/completions \
    --num-prompts 128 --trust-remote-code --ignore-eos --seed 12345 \
    --served-model-name Qwen36 --max-concurrency 64 \
    --dataset-name random --random-input-len 512 --random-output-len 2048 \
    --random-range-ratio 0 > "$out" 2>&1
  post=$(metrics)

  {
    echo "### $tag (arm=$arm mode=$mode) C64 in=512 out=2048 128 prompts"
    grep -E 'Successful requests|Output token throughput|Mean TTFT|Mean TPOT|Mean ITL' "$out" | sed 's/  */ /g'
    echo "$sha"
    echo "spec_pre : $pre"
    echo "spec_post: $post"
    echo "--- NTSEG4 steady (rank0) ---"
    grep -a NTSEG4 "/work/logs/serve_${tag}.log" | grep -a TP0_EP0 | grep -a "nreq=64" | tail -3
    echo
  } >> "$OUT"
  echo "### cell $tag done :: $sha"
  stop "$tag"
}

echo "##### ROUND12 START $(date -u +%FT%TZ)" >> "$OUT"
cell nospec none  r12_nospec    || exit 1
cell mtp1   probe r12_mtp_probe || exit 1
cell mtp1   A     r12_mtp_A     || exit 1
cell mtp1   AB    r12_mtp_AB    || exit 1
cp "$A.ntorig" "$A"
find /vllm-workspace -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "ROUND12_DONE"
