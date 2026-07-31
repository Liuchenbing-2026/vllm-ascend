#!/bin/bash
# 启动 vLLM。用法: serve_launch.sh a|b|c
#   a = 基线（引擎内 prefix caching，无 KV 池）
#   b = DRAM KV 池（AscendStoreConnector + mooncake）
#   c = SSD 分层（小 DRAM 池强制下沉到 SSD）
set -e
MODE=${1:?usage: serve_launch.sh a|b|c}
MCSSD_ROOT=${MCSSD_ROOT:-/data1/mcssd}
CFG_DIR=${MCSSD_CFG_DIR:-$MCSSD_ROOT/cfg}
source "$CFG_DIR/env_common.sh"

mkdir -p "$MCSSD_ROOT/logs"
LOG=$MCSSD_ROOT/logs/serve_${MODE}.log

MODEL=${MCSSD_MODEL:-/data1/Qwen3-14B}
TP=${MCSSD_TP:-2}
PORT=${MCSSD_PORT:-8100}

# --host 127.0.0.1: vLLM 默认绑 0.0.0.0，有公网 IP 的机器会被外部扫描
# --block-size 128: 大 block 减少 SSD 小随机 IO（page 式布局是 SSD 路径的主要瓶颈）
COMMON="--model $MODEL --host 127.0.0.1 --port $PORT \
  --trust-remote-code --enforce-eager \
  --tensor-parallel-size $TP \
  --max-model-len ${MCSSD_MAXLEN:-16384} \
  --block-size 128 \
  --max-num-batched-tokens 16384"

KVCFG='{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lookup_rpc_port":"1","backend":"mooncake"}}'

case $MODE in
  a)
    exec python3 -m vllm.entrypoints.openai.api_server $COMMON >>"$LOG" 2>&1
    ;;
  b)
    export MOONCAKE_CONFIG_PATH=$CFG_DIR/mooncake_dram.json
    exec python3 -m vllm.entrypoints.openai.api_server $COMMON \
      --no-enable-prefix-caching --kv-transfer-config "$KVCFG" >>"$LOG" 2>&1
    ;;
  c)
    export MOONCAKE_CONFIG_PATH=$CFG_DIR/mooncake_ssd.json
    exec python3 -m vllm.entrypoints.openai.api_server $COMMON \
      --no-enable-prefix-caching --kv-transfer-config "$KVCFG" >>"$LOG" 2>&1
    ;;
  *)
    echo "usage: serve_launch.sh a|b|c" >&2; exit 1 ;;
esac
