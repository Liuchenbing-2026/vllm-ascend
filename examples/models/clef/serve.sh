#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to a local Clef checkpoint with its joint head}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select the assigned NPU devices}"
# Without the secondary head, upstream can interpret this as plain Qwen.
# File presence is only a launch guard, not a checkpoint integrity check.
for file in config.json joint_head_config.json joint_head.safetensors; do
    if [[ ! -s "$MODEL_PATH/$file" ]]; then
        printf 'Missing or empty Clef checkpoint file: %s\n' "$MODEL_PATH/$file" >&2
        exit 2
    fi
done
python_bin=${PYTHON_BIN:-python}
exec "$python_bin" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name clef \
    --runner pooling \
    --dtype bfloat16 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --enforce-eager \
    --host 127.0.0.1 --port "${PORT:-8000}" "$@"
