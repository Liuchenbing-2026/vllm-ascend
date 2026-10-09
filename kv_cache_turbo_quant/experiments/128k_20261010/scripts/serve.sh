#!/usr/bin/env bash
# Same serving configuration for both arms; only the real store switch changes.
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv
export PYTHONPATH=/ws/source/vllm:/ws/source/vllm-ascend:/root/kvtq_integration:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_ASCEND_KVTQ=0 VLLM_ASCEND_KVTQ_BITS=4
export VLLM_ASCEND_KVTQ_STORE="$1" TORCH_EXTENSIONS_DIR=/ws/torch-extensions
export OMP_NUM_THREADS=1
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model,ascend_model_loader,kv_cache_turbo_quant_shadow
exec /ws/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-30B-A3B --served-model-name qwen3-30b-tq-ab \
  --tensor-parallel-size 2 --dtype bfloat16 --enforce-eager \
  --no-enable-prefix-caching --no-async-scheduling \
  --max-model-len 132096 --max-num-batched-tokens 4096 --max-num-seqs 32 \
  --kv-cache-memory-bytes 25769803776 --gpu-memory-utilization 0.92 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960}}' \
  --host 127.0.0.1 --port 18377
