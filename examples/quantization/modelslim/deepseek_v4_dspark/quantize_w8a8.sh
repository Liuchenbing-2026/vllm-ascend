#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  quantize_w8a8.sh MODEL_PATH RAW_OUTPUT CANONICAL_HEAD_CHECKPOINT FINAL_OUTPUT [LINK_MODE]

Arguments:
  MODEL_PATH                  Full-precision DeepSeek V4 DSpark checkpoint.
  RAW_OUTPUT                  New directory for the raw ModelSlim W8A8 output.
  CANONICAL_HEAD_CHECKPOINT   Compatible target/MTP checkpoint providing head.weight.
  FINAL_OUTPUT                New directory for the serving-ready checkpoint.
  LINK_MODE                   symlink (default), hardlink, or copy for unchanged files.

Activate the ModelSlim environment and source the Ascend toolkit environment first.
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage >&2
    exit 64
fi

model_path=$1
raw_output=$2
canonical_head_checkpoint=$3
final_output=$4
link_mode=${5:-symlink}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
recipe="$script_dir/deepseek_v4_flash_dspark_w8a8.yaml"
finalizer="$script_dir/finalize_w8a8_checkpoint.py"

if [[ ! -d "$model_path" ]]; then
    echo "MODEL_PATH is not a directory: $model_path" >&2
    exit 2
fi
if [[ ! -d "$canonical_head_checkpoint" ]]; then
    echo "CANONICAL_HEAD_CHECKPOINT is not a directory: $canonical_head_checkpoint" >&2
    exit 2
fi
if [[ -e "$raw_output" ]]; then
    echo "RAW_OUTPUT already exists; refusing to overwrite it: $raw_output" >&2
    exit 2
fi
if [[ -e "$final_output" ]]; then
    echo "FINAL_OUTPUT already exists; refusing to overwrite it: $final_output" >&2
    exit 2
fi
if [[ "$link_mode" != symlink && "$link_mode" != hardlink && "$link_mode" != copy ]]; then
    echo "LINK_MODE must be symlink, hardlink, or copy; got: $link_mode" >&2
    exit 2
fi
if ! command -v msmodelslim >/dev/null 2>&1; then
    echo "msmodelslim is not on PATH; activate the ModelSlim environment first" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is not on PATH" >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

echo "Quantizing the full target + DSpark draft checkpoint to W8A8_DYNAMIC"
echo "ModelSlim executable: $(command -v msmodelslim)"
echo "NPU devices: $ASCEND_RT_VISIBLE_DEVICES"
msmodelslim quant \
    --model_path "$model_path" \
    --save_path "$raw_output" \
    --model_type DeepSeek-V4-Flash-DSpark \
    --config_path "$recipe" \
    --trust_remote_code True

echo "Finalizing the QuaRot basis and canonical shared head"
python3 "$finalizer" \
    --modelslim-output "$raw_output" \
    --canonical-head-checkpoint "$canonical_head_checkpoint" \
    --output "$final_output" \
    --link-mode "$link_mode"

echo "Serving-ready W8A8 DSpark checkpoint: $final_output"
