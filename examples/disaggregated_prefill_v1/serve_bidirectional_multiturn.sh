#!/usr/bin/env bash
# Run inside the task-owned rc2 NPU container with two free mapped devices.
set -euo pipefail
: "${MODEL:?Set MODEL to the fixed local model snapshot}"
: "${ROUTER_BIN:?Set ROUTER_BIN to the tested native Rust router}"
: "${RUN_DIR:?Set RUN_DIR to a new directory}"
: "${P_DEVICE:?Set P_DEVICE to a free physical NPU id}"
: "${D_DEVICE:?Set D_DEVICE to a different free physical NPU id}"
LEG=${LEG:-off}
P_PORT=${P_PORT:-29510}
D_PORT=${D_PORT:-29511}
ROUTER_PORT=${ROUTER_PORT:-29512}
P_KV_PORT=${P_KV_PORT:-29520}
D_KV_PORT=${D_KV_PORT:-29530}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
KV_MEMORY_BYTES=${KV_MEMORY_BYTES:-8589934592}
if [[ "$P_DEVICE" == "$D_DEVICE" || ( "$LEG" != on && "$LEG" != off ) ]]; then
  echo 'Choose distinct free NPUs and LEG=off or LEG=on.' >&2
  exit 2
fi
mkdir "$RUN_DIR"
python3 - "$RUN_DIR" "$MODEL" "$P_DEVICE" "$D_DEVICE" "$P_PORT" "$D_PORT" "$ROUTER_PORT" "$P_KV_PORT" "$D_KV_PORT" <<'PY'
import errno, hashlib, importlib.metadata, json, pathlib, re, socket, subprocess, sys
devices = subprocess.check_output(['npu-smi', 'info'], text=True)
if any(not pathlib.Path('/dev/davinci'+device).exists() for device in sys.argv[3:5]):
    raise SystemExit('Selected NPU device nodes are not mapped into this container')
if 'Process id' not in devices:
    raise SystemExit('Cannot read the NPU process table; nothing was started')
busy = {m.group(1) for m in re.finditer(r'^\|\s+(\d+)\s+\d+\s*\|\s*\d+\s*\|', devices, re.M)}
if busy.intersection(sys.argv[3:5]):
    raise SystemExit('Selected NPUs have existing processes; nothing was started')
for port in [*map(int, sys.argv[5:8]), int(sys.argv[8]), int(sys.argv[8])+1, int(sys.argv[9]), int(sys.argv[9])+1]:
    with socket.socket() as connection:
        connection.settimeout(0.2)
        if connection.connect_ex(('127.0.0.1', port)) == 0:
            raise OSError(errno.EADDRINUSE, f'Port {port} already has a listener')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('0.0.0.0', port))
        probe.listen(1)
model = pathlib.Path(sys.argv[2])
if not model.is_dir():
    raise SystemExit('Use a fixed local model snapshot')
files = [p for p in model.iterdir() if p.suffix in ('.json', '.safetensors')]
hashes = {}
for path in files:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    hashes[path.name] = digest.hexdigest()
receipt = {'devices': devices, 'model': str(model), 'model_sha256': hashes,
           'versions': {n: importlib.metadata.version(n) for n in
                        ['vllm', 'vllm-ascend', 'torch', 'torch-npu', 'mooncake-transfer-engine-npu']}}
pathlib.Path(sys.argv[1], 'environment.json').write_text(json.dumps(receipt, indent=2))
PY
BIDIR=false
ROUTER_EXTRA=()
if [[ "$LEG" == on ]]; then
  BIDIR=true
  ROUTER_EXTRA=(--bidirectional-kv-config '{"ttl_secs":450}')
fi
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
for role in prefill decode; do
  if [[ "$role" == prefill ]]; then
    device=$P_DEVICE; port=$P_PORT; kv_port=$P_KV_PORT; kv_role=kv_producer
    EXTRA=(--no-enable-prefix-caching)
  else
    device=$D_DEVICE; port=$D_PORT; kv_port=$D_KV_PORT; kv_role=kv_consumer
    EXTRA=()
  fi
  ASCEND_RT_VISIBLE_DEVICES="$device" \
    HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}" \
    vllm serve "$MODEL" --served-model-name bidir --dtype bfloat16 --enforce-eager \
    --tensor-parallel-size 1 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 \
    --kv-cache-memory-bytes "$KV_MEMORY_BYTES" --port "$port" "${EXTRA[@]}" \
    --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV2\",\"kv_role\":\"$kv_role\",\"engine_id\":\"bidir-$role\",\"kv_port\":$kv_port,\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"bidirectional_kv_xfer\":$BIDIR,\"decoder_kv_blocks_ttl\":480,\"kv_recompute_threshold\":64}}" \
    > "$RUN_DIR/$role.log" 2>&1 &
  pids+=("$!")
done
python3 - "$P_PORT" "$D_PORT" <<'PY'
import sys, time, urllib.request
for port in sys.argv[1:]:
    deadline=time.monotonic()+1200
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:'+port+'/health', timeout=2):
                break
        except Exception:
            if time.monotonic()>deadline:
                raise
            time.sleep(2)
PY
"$ROUTER_BIN" --host 127.0.0.1 --port "$ROUTER_PORT" --policy round_robin \
  --vllm-pd-disaggregation --kv-connector mooncake-pull \
  --prefill "http://127.0.0.1:$P_PORT" --decode "http://127.0.0.1:$D_PORT" \
  "${ROUTER_EXTRA[@]}" > "$RUN_DIR/router.log" 2>&1 &
pids+=("$!")
printf '%s\n' "P=$P_PORT D=$D_PORT Router=$ROUTER_PORT; use benchmark.py --transport mooncake."
wait
