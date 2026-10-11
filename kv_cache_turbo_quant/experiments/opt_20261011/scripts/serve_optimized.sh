#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv
mode=$1
integration=/root/kvtq_integration
store=1
case "$mode" in
  bf16) store=0 ;;
  original) ;;
  optimized) integration=/ws/opt_20261011/candidate/integration ;;
  *) exit 2 ;;
esac
export PYTHONPATH="$integration:/ws/source/vllm:/ws/source/vllm-ascend:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${TASK_NPU_CARDS:-2,5}" VLLM_ASCEND_KVTQ=0 VLLM_ASCEND_KVTQ_BITS=4
export VLLM_ASCEND_KVTQ_STORE="$store" TORCH_EXTENSIONS_DIR=/ws/torch-extensions OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=/ws/opt_20261011/model-triton-cache
unset ASCEND_LAUNCH_BLOCKING
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model,ascend_model_loader,kv_cache_turbo_quant_shadow
profile_args=()
if [ -n "${2:-}" ]; then
  profile_args=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$2\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"delay_iterations\":0,\"max_iterations\":0}")
fi
exec /ws/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-30B-A3B --served-model-name qwen3-30b-tq-ab \
  --tensor-parallel-size 2 --dtype bfloat16 --enforce-eager \
  --no-enable-prefix-caching --no-async-scheduling \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --kv-cache-memory-bytes 25769803776 --gpu-memory-utilization 0.92 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}}' \
  --host 127.0.0.1 --port 18377 "${profile_args[@]}"
