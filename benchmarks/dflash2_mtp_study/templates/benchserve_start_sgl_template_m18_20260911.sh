#!/usr/bin/env bash
set -euo pipefail

arm_id=__ARM__

# PHASE A of a two-task split. The single-task version cannot work: cold service startup on this
# box measured 7.5 min (09:39:43 -> 09:47:07 in 20260910aa) and the random-data bench itself runs
# 283 s (20260910z), so start+bench is ~12 min against a ~10 min remote wall clock. Arm x died
# exactly there (QUEUE_REMOTE_EXIT_STATUS=-1 at readiness attempt 8). Arm z only survived because
# x and y had already pulled the weights into page cache.
#
# So: this task starts the service and LEAVES IT RUNNING. Being truncated at the wall clock is
# harmless here -- the container is detached, and a SIGKILL of the driving shell does not touch it.
# Phase B (benchserve_bench_template) attaches to the ready service, benches, and tears down.
#
# THIS VARIANT reproduces the published SGLang kou-jing instead of the user's.
# sgl-project/sglang#35629 reports, on 910C TP2 / GSM8K / greedy / RadixCache off:
#     C1  33.2 -> 101.2 tok/s (3.05x, accept 5.8)
#     C8  176.2 -> 329.4     (1.87x)
#     C16 247.1 -> 431.2     (1.75x)
# and the LMSYS blog's "DFlash beats MTP" figure is C1-C8, short prompts, MTP at SEVEN steps.
# My own C32 / gsm8k-exact4096 run puts DFlash at 149.87 against nospec 184.25 -- 0.81x, i.e.
# slower than not speculating at all. Four axes separate the two claims (input length 4096 vs
# ~100 tokens, C32 vs C16, non-greedy vs greedy, MTP3 vs MTP7). This run pins three of them to
# SGLang's side so input length and concurrency can be read off a single service instance.
#
# The capture ladder is widened per arm because the sweep now visits C1/C8/C16/C32, and a decode
# batch whose descriptor misses every captured size silently falls back to eager with no warning
# ([[vllm-cudagraph-silent-downgrade]]). Sizes are batch x verify width: nospec 1, mtp3 4, k7 8.
# The C32 entries (256 for k7, 128 for mtp3, 32 for nospec) are unchanged from the 20260911 runs,
# so the long-input-vs-short-input comparison at C32 stays single-variable.

devices=0,2
tp=2
concurrency=32
root=/data2/dflash2-v026-official
vllm_src="$root/vllm"
runtime="$root/runtime-source/dflash2-mrv1-full-kv-device-queryloc-v2-runtime-e6519803"
runtime_gate="$root/upstream-audit/dflash2-device-queryloc-v2-20260827"
evidence="$root/upstream-audit/benchserve-sgl-${arm_id}-20260911b"
model=/data1/dflash2-models/Qwen3.8-27B
draft=/data1/dflash2-models/Qwen3.8-27B-DFlash2
image=vllm-ascend:pr14171-v026-runtime-20260821
expected_image=sha256:e6519803e088d655590cfe3b2ef9b429c1c4509e790f85b1fae4a37acd94ba75
expected_manifest=929fb5220edfa11e7804872a14d610b07504ede7b32869a192120bbbc3f71622
container="benchserve-${arm_id}-sgl-20260911b"

# Default: serve the model path directly. The mtp3 arm overrides both, because vLLM's Speculators
# updater rewrites Qwen3_5MTP -> DFlashQwen3_5MTP (unregistered) when the served path contains
# "dflash", which is what killed arm y with exit 43. Serving through a /tmp symlink avoids it.
pre=''
serve_path="$model"

case "$arm_id" in
  nospec)
    spec=''
    addl=''
    needle='vllm serve'
    caps='[1,8,16,32,64,128,256]'
    ;;
  mtp3)
    spec='--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}"'
    addl=''
    needle='qwen3_5_mtp'
    caps='[4,8,32,64,128,256]'
    pre="mtp_path=/tmp/qwen38_sgl_20260911b; ln -sfn $model \$mtp_path; "
    serve_path='$mtp_path'
    ;;
  # The baseline the published comparison actually used. LMSYS's "DFlash 1.5x MTP" is measured
  # against MTP at SEVEN steps; every MTP number I have is at three. That is not a fair swap in
  # either direction, because MTP-7 costs seven SEQUENTIAL head calls per iteration against
  # MTP-3's three, and block-parallel drafting is precisely what cashes that difference in.
  #
  # Reachable? The checkpoint ships exactly ONE MTP module (20260911y: mtp.layers.0.*, 15 tensors,
  # step index 0 only), and llm_base_proposer.py:1481 loops it `num_speculative_tokens - 1` extra
  # times -- the sequential path, taken because `parallel_drafting` is false for mtp (line 1448).
  # So depth is a loop count, not a checkpoint property. Whether vLLM clamps it to the checkpoint's
  # num_nextn_predict_layers is NOT settled by reading (speculative.py:566 reads that value but
  # the mtp branch does not obviously gate on it), so the readiness log is the judge: this arm
  # asserts the served config really came up at 7, and fails loudly if it was silently clamped.
  mtp7)
    spec='--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":7}"'
    addl=''
    needle='"num_speculative_tokens":7'
    caps='[8,16,64,128,256]'
    pre="mtp_path=/tmp/qwen38_sgl_20260911b; ln -sfn $model \$mtp_path; "
    serve_path='$mtp_path'
    ;;
  k7)
    spec="--speculative-config \"{\\\"method\\\":\\\"dflash\\\",\\\"model\\\":\\\"$draft\\\",\\\"num_speculative_tokens\\\":7}\""
    addl='\"dflash_full_kv_allocation\":true,\"draft_window_size\":2048,'
    needle='dflash_full_kv_allocation'
    caps='[8,16,64,128,256,512]'
    ;;
  # K is the only variable against k7. The win line says DFlash beats MTP3 whenever
  # T_d/T_m <= advance_d/2.183, and at K=3 its measured Sigma is already 0.896 against a
  # required 0.856 at cost ratio 0.85 -- so the whole question is what K=3 costs per iteration.
  # Plausibly less than MTP3: one 5-layer block-parallel forward pays one collective latency,
  # MTP3's three sequential single-layer head calls pay three.
  k3)
    spec="--speculative-config \"{\\\"method\\\":\\\"dflash\\\",\\\"model\\\":\\\"$draft\\\",\\\"num_speculative_tokens\\\":3}\""
    addl='\"dflash_full_kv_allocation\":true,\"draft_window_size\":2048,'
    needle='"num_speculative_tokens":3'
    caps='[4,8,32,64,128]'
    ;;
  # K=1 and K=3 bracket the depth curve at C32. A model cannot pick K here: fitting
  # T = c + alpha*(K+1) to the three measured points (nospec width 1, MTP3 width 4, K7 width 8)
  # gives a NEGATIVE drafter cost, i.e. cost is sublinear in verify width -- a width-8 forward
  # runs 32x8 tokens at better arithmetic intensity than 32x4. So advance/(K+1) understates deep
  # K, and the only honest way to choose is to measure both ends.
  k1)
    spec="--speculative-config \"{\\\"method\\\":\\\"dflash\\\",\\\"model\\\":\\\"$draft\\\",\\\"num_speculative_tokens\\\":1}\""
    addl='\"dflash_full_kv_allocation\":true,\"draft_window_size\":2048,'
    needle='"num_speculative_tokens":1'
    caps='[32,64,128,256,512]'
    ;;
  # Two single-variable probes of the 5.41x KV inflation. Everything else is byte-identical to
  # the k7 arm, so whichever recovers KV capacity identifies the cause.
  k7nofullkv)
    spec="--speculative-config \"{\\\"method\\\":\\\"dflash\\\",\\\"model\\\":\\\"$draft\\\",\\\"num_speculative_tokens\\\":7}\""
    addl='\"dflash_full_kv_allocation\":false,\"draft_window_size\":2048,'
    needle='"dflash_full_kv_allocation":false'
    caps='[32,64,128,256,512]'
    ;;
  k7win256)
    spec="--speculative-config \"{\\\"method\\\":\\\"dflash\\\",\\\"model\\\":\\\"$draft\\\",\\\"num_speculative_tokens\\\":7}\""
    addl='\"dflash_full_kv_allocation\":true,\"draft_window_size\":256,'
    needle='"draft_window_size":256'
    caps='[32,64,128,256,512]'
    ;;
  *) echo "unknown arm_id=$arm_id" >&2; exit 2 ;;
esac

IFS=',' read -r -a device_list <<< "$devices"

kill_own_service() {
  if docker ps -aq --filter name="^/${container}$" | grep -q .; then
    docker logs "$container" > "$evidence/service.$1.log" 2>&1 || true
    timeout -k 10s 60s docker kill "$container" >/dev/null 2>&1 || true
    timeout -k 10s 60s docker rm -f "$container" >/dev/null 2>&1 || true
  fi
}
# Only tears down on a *fatal* failure. A plain timeout must leave the service up for phase B.
on_error() {
  status=$?
  printf 'BENCHSERVE_START_ERROR arm=%s status=%s command=%q\n' "$arm_id" "$status" "$BASH_COMMAND" >&2
  kill_own_service failed
  exit "$status"
}
trap on_error ERR

test "$(hostname)" = kylin10-6018
test "$(docker image inspect "$image" --format '{{.Id}}')" = "$expected_image"
test "$(sha256sum "$runtime/RUNTIME_MANIFEST.txt" | awk '{print $1}')" = "$expected_manifest"
grep -Fxq 'runtime_import=pass' "$runtime_gate/runtime-import.known-good-order.log"
test ! -e "$evidence"
mkdir -p "$evidence"

# Reclaim only containers we named ourselves. Cards 1/3/4/6/7 are held by someone else's
# VLLMEngineCore processes (56 GB each, confirmed in 20260910ae); those are never touched.
docker ps -a --format '{{.Names}}' | grep -E '^(fourarm|onearm|csweep|mbtc8|c8delta|benchserve)-[a-z0-9_-]+-2026091[01]' > "$evidence/own-leftovers.txt" || true
while read -r name; do
  [[ -n "$name" ]] || continue
  echo "reclaiming own leftover $name"
  timeout -k 10s 60s docker kill "$name" >/dev/null 2>&1 || true
  timeout -k 10s 60s docker rm -f "$name" >/dev/null 2>&1 || true
done < "$evidence/own-leftovers.txt"

free=0
for attempt in $(seq 1 10); do
  busy=0
  for device in "${device_list[@]}"; do
    test "$(npu-smi info -t health -i "$device" -c 0 | awk '/Health/ {print $NF; exit}')" = OK
    hbm="$(npu-smi info -t usages -i "$device" -c 0 | awk -F: '/HBM Usage Rate/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
    test -n "$hbm"
    if (( hbm > 10 )); then busy=1; fi
  done
  if [[ -n "$(ss -ltnH 'sport = :8000' || true)" ]]; then busy=1; fi
  echo "preflight attempt=$attempt busy=$busy"
  if (( busy == 0 )); then free=1; break; fi
  sleep 10
done
test "$free" = 1

if python3 -c "
import json,sys
from pathlib import Path
p = Path('$model/tokenizer_config.json')
c = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
sys.exit(0 if c.get('chat_template') else 1)
" 2>/dev/null; then
  backend=openai-chat
  endpoint=/v1/chat/completions
else
  backend=openai
  endpoint=/v1/completions
fi
echo "backend=$backend endpoint=$endpoint" | tee "$evidence/backend.txt"

device_flags=()
for device in "${device_list[@]}"; do device_flags+=(--device "/dev/davinci${device}"); done
prelude='set -euo pipefail; export LD_LIBRARY_PATH=/dflash-src/vllm-ascend/vllm_ascend/lib:${LD_LIBRARY_PATH:-}; unset ASCEND_LAUNCH_BLOCKING VLLM_ASCEND_DEBUG_GDN_MIXED_SYNC HCCL_OP_EXPANSION_MODE HCCL_BUFFSIZE OMP_PROC_BIND OMP_NUM_THREADS VLLM_ASCEND_BALANCE_SCHEDULING TORCH_COMPILE_DEBUG TORCH_COMPILE_DEBUG_PATH TORCH_DEVICE_BACKEND_AUTOLOAD; '
tail_serve="--limit-mm-per-prompt \"{\\\"image\\\":0,\\\"video\\\":0}\" --tensor-parallel-size $tp --port 8000 --max-model-len 32768 --max-num-seqs $concurrency --max-num-batched-tokens 16384 --no-enable-prefix-caching --trust-remote-code"
graph_cfg="--compilation-config \"{\\\"cudagraph_mode\\\":\\\"FULL_DECODE_ONLY\\\",\\\"cudagraph_capture_sizes\\\":${caps},\\\"pass_config\\\":{\\\"fuse_norm_quant\\\":false}}\""

docker run -d --name "$container" \
  --privileged --network host --ipc=host \
  "${device_flags[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --mount "type=bind,src=$vllm_src,dst=/dflash-src/vllm,readonly" \
  --mount "type=bind,src=$runtime,dst=/dflash-src/vllm-ascend,readonly" \
  --mount "type=bind,src=$root,dst=/dflash-out" \
  --mount type=bind,src=/data1,dst=/data1,readonly \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --mount type=bind,src=/usr/local/sbin/npu-smi,dst=/usr/local/bin/npu-smi,readonly \
  --env PYTHONPATH=/dflash-src/vllm:/dflash-src/vllm-ascend:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe \
  --env "ASCEND_RT_VISIBLE_DEVICES=$devices" \
  --env VLLM_USE_V2_MODEL_RUNNER=0 \
  --env VLLM_DISABLE_COMPILE_CACHE=1 \
  --env VLLM_LOGGING_LEVEL=INFO \
  --entrypoint /bin/bash "$image" -lc \
  "${prelude}${pre}exec vllm serve ${serve_path} --served-model-name $model $spec --additional-config \"{\\\"enable_reduce_sample\\\":true,${addl}\\\"ascend_compilation_config\\\":{\\\"enable_npugraph_ex\\\":false,\\\"fuse_norm_quant\\\":false}}\" $tail_serve $graph_cfg" \
  > "$evidence/service.cid"

echo "container=$container started at $(date +%H:%M:%S)"

fatal='Traceback \(most recent call last\)|Engine core initialization failed|WorkerProc hit an exception|OutOfMemoryError|AIVEC|507011|507033|507034|107025|561002'
ready=0
for attempt in $(seq 1 42); do
  state="$(docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null || true)"
  docker logs "$container" > "$evidence/readiness.log" 2>&1 || true
  health="$(curl -sS --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)"
  echo "attempt=$attempt state=$state health=$health time=$(date +%H:%M:%S)"
  if grep -Eq "$fatal" "$evidence/readiness.log"; then tail -n 200 "$evidence/readiness.log" > "$evidence/readiness.tail.txt"; exit 43; fi
  if [[ "$state" != running ]]; then tail -n 200 "$evidence/readiness.log" > "$evidence/readiness.tail.txt"; exit 44; fi
  if [[ "$health" = 200 ]]; then ready=1; break; fi
  sleep 10
done

grep -iE 'GPU KV cache size|Available KV cache memory|Maximum concurrency' "$evidence/readiness.log" | tail -n 5 || true

# What depth did the engine ACTUALLY come up at? A request for 7 that the config layer silently
# clamps to the checkpoint's single MTP module would otherwise be indistinguishable from MTP-7,
# and the whole point of the mtp7 arm is the depth. Record it for every arm; assert it for mtp7.
grep -oE "SpeculativeConfig\(method='[^']*'[^)]{0,160}" "$evidence/readiness.log" | tail -n 1 \
  > "$evidence/spec-config.txt" || true
cat "$evidence/spec-config.txt"
case "$arm_id" in
  mtp7) grep -q 'num_spec_tokens=7' "$evidence/spec-config.txt" \
          || { echo "MTP7_CLAMPED: engine did not come up at depth 7 -- see spec-config.txt"; exit 47; } ;;
  mtp3) grep -q 'num_spec_tokens=3' "$evidence/spec-config.txt" || echo "WARNING: mtp3 depth not confirmed" ;;
esac
echo "EVIDENCE=$evidence"
if (( ready == 1 )); then
  echo "BENCHSERVE_START_READY arm=$arm_id"
else
  # Not an error. The service is still loading and stays up; phase B will keep polling.
  echo "BENCHSERVE_START_STILL_LOADING arm=$arm_id"
fi
