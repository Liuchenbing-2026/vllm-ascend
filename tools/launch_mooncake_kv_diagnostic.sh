#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Container entrypoint only. Host guard must already be armed before docker start.
set -euo pipefail
umask 077
ulimit -c 0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
: "${HCCL_HOST_SOCKET_PORT_RANGE:?Requires a supervisor-reviewed host range}"
: "${HCCL_NPU_SOCKET_PORT_RANGE:?Requires a supervisor-reviewed NPU range}"
export PATH="/work/.venv-core-stream/bin:$PATH"
export PYTHONPATH="/diag:/vllm-workspace/vllm-ascend:/vllm-workspace/vllm"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=240
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME=/output/cache
mkdir /output/servers
read -r p_device d_device < <(python -c 'import json; s=json.load(open("/output/allocation.json")); print(*s["devices"])')
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    # Independent host guard provides the absolute bound if a child hangs.
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' INT TERM
for role in prefill decode; do
    if [[ "$role" == prefill ]]; then
        device=$p_device; port=29510; kv_port=29520; kv_role=kv_producer
    else
        device=$d_device; port=29511; kv_port=29530; kv_role=kv_consumer
    fi
    ASCEND_RT_VISIBLE_DEVICES="$device" \
        python -m vllm.entrypoints.openai.api_server \
        --model /models/Qwen3-8B --served-model-name bidir --dtype bfloat16 \
        --host 127.0.0.1 --port "$port" --enforce-eager --no-enable-prefix-caching \
        --worker-extension-cls tools.mooncake_kv_snapshot.KVSnapshotWorkerExtension \
        --tensor-parallel-size 1 --max-model-len 24576 --max-num-seqs 4 \
        --kv-cache-memory-bytes 17179869184 \
        --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV2\",\"kv_role\":\"$kv_role\",\"engine_id\":\"bidir-$role\",\"kv_port\":$kv_port,\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"bidirectional_kv_xfer\":true,\"decoder_kv_blocks_ttl\":480,\"kv_recompute_threshold\":64}}" \
        > "/output/servers/$role.log" 2>&1 &
    pids+=("$!")
done
# Do not leave the sibling serving after the first process fails.
wait -n "${pids[@]}"
exit 1
