#!/bin/bash

ROOT_DIR=$(readlink -f -- "${1:?ROOT_DIR is not set}")
SOC_VERSION=$2
: "${ROOT_DIR:?ROOT_DIR is not set}"

log() {
    echo "[build_aclnn] $*"
}

setup_catlass_dependency() {
    local catlass_path="${ROOT_DIR}/csrc/third_party/catlass/include"
    local catlass_commit
    local absolute_catlass_path

    git config --global --add safe.directory "$ROOT_DIR"
    catlass_commit=$(git config -f "${ROOT_DIR}/.gitmodules" --get submodule.csrc/third_party/catlass.commit)
    if [[ ! -d "${catlass_path}" ]]; then
        echo "dependency catlass is missing, try to fetch it..."
        git submodule sync
        if ! git submodule update --init --recursive; then
            log "fetch failed"
            exit 1
        fi
        cd "${ROOT_DIR}/csrc/third_party/catlass" || exit 1
        git fetch origin
        git checkout "${catlass_commit}" || exit 1
        cd - || exit 1
    fi
    absolute_catlass_path=$(cd "${catlass_path}" && pwd)
    export CPATH="${absolute_catlass_path}${CPATH:+:${CPATH}}"
    log "catlass include=${absolute_catlass_path}"
}

resolve_op_dir() {
    local op_name=$1
    local candidate_dir
    for candidate_dir in \
        "${ROOT_DIR}/csrc/moe/${op_name}" \
        "${ROOT_DIR}/csrc/gmm/${op_name}" \
        "${ROOT_DIR}/csrc/attention/${op_name}" \
        "${ROOT_DIR}/csrc/mc2/${op_name}" \
        "${ROOT_DIR}/csrc/ffn/${op_name}" \
        "${ROOT_DIR}/csrc/posembedding/${op_name}"; do
        if [[ -d "${candidate_dir}" ]]; then
            echo "${candidate_dir}"
            return 0
        fi
    done
    find "${ROOT_DIR}/csrc" -maxdepth 3 -type d -name "${op_name}" -print -quit 2>/dev/null
}

log_selected_ops() {
    local op_name
    local op_path
    local kernel_cpp_file_count

    log "resolved SOC_ARG=${SOC_ARG}"
    log "resolved CUSTOM_OPS=${CUSTOM_OPS}"
    log "custom op count=${#CUSTOM_OPS_ARRAY[@]}"
    for op_name in "${CUSTOM_OPS_ARRAY[@]}"; do
        op_path=$(resolve_op_dir "${op_name}")
        if [[ -z "${op_path}" ]]; then
            log "op ${op_name}: dir=<missing>"
            continue
        fi
        kernel_cpp_file_count=0
        if [[ -d "${op_path}/op_kernel" ]]; then
            kernel_cpp_file_count=$(find "${op_path}/op_kernel" -maxdepth 1 -name '*.cpp' | wc -l | tr -d ' ')
        fi
        log "op ${op_name}: dir=${op_path} cmake=$([[ -f "${op_path}/CMakeLists.txt" ]] && echo yes || echo no) op_host_cmake=$([[ -f "${op_path}/op_host/CMakeLists.txt" ]] && echo yes || echo no) op_kernel_cpp_count=${kernel_cpp_file_count}"
    done
}

log "start: ROOT_DIR=${ROOT_DIR:-<unset>} SOC_VERSION=${SOC_VERSION:-<unset>} cwd=$(pwd)"
log "env: ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>} ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"

if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    log "matched SOC branch: ascend310"
    # ASCEND310P series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "causal_conv1d_v310"
        "recurrent_gated_delta_rule_v310"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend310p"
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    log "matched SOC branch: ascend910b"
    # ASCEND910B (A2) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "scatter_nd_update_v2"
        "moe_grouped_matmul"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer"
        "sparse_flash_attention"
        "kv_quant_sparse_flash_attention"
        "matmul_allreduce_add_rmsnorm"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "lightning_indexer_quant"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "hamming_dist_top_k"
        "reshape_and_cache_bnsd"
        "recurrent_gated_delta_rule"
        "fused_gdn_gating"
        "ngram_spec_decode"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    log "matched SOC branch: ascend910_93"
    # ASCEND910C (A3) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "scatter_nd_update_v2"
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list"
        "lightning_indexer"
        "sparse_flash_attention"
        "kv_quant_sparse_flash_attention"
        "dispatch_ffn_combine"
        "dispatch_ffn_combine_w4_a8"
        "dispatch_ffn_combine_bf16"
        "dispatch_gmm_combine_decode"
        "moe_init_routing_custom"
        "moe_gating_top_k"
        "moe_gating_top_k_hash"
        "add_rms_norm_bias"
        "apply_top_k_top_p_custom"
        "transpose_kv_cache_by_block"
        "copy_and_expand_eagle_inputs"
        "causal_conv1d"
        "moe_grouped_matmul"
        "lightning_indexer_quant"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "sparse_attn_sharedkv"
        "sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_pre"
        "hc_post"
        "inplace_partial_rotary_mul"
        "rms_norm_dynamic_quant"
        "dequant_swiglu_quant"
        "grouped_matmul_swiglu_quant"
        "grouped_matmul_swiglu_quant_v2"
        "hamming_dist_top_k"
        "reshape_and_cache_bnsd"
        "recurrent_gated_delta_rule"
        "fused_gdn_gating"
        "ngram_spec_decode"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
elif [[ "$SOC_VERSION" =~ ^ascend950 ]]; then
    log "matched SOC branch: ascend950"
    # ASCEND950 (A5) series
    # dependency: catlass
    setup_catlass_dependency

    CUSTOM_OPS_ARRAY=(
        "moe_gating_top_k_hash"
        "indexer_compress_epilog"
        "inplace_partial_rotary_mul"
        "kv_compress_epilog"
        "compressor"
        "compressor_metadata"
        "vllm_quant_lightning_indexer"
        "vllm_quant_lightning_indexer_metadata"
        "kv_quant_sparse_attn_sharedkv"
        "kv_quant_sparse_attn_sharedkv_metadata"
        "hc_pre_sinkhorn"
        "hc_pre_inv_rms"
        "hc_post"
        "hc_pre"
        "swiglu_group_quant"
        "load_index_kv_cache"
        "indexer_compress_epilog_v2"
        "causal_conv1d"
        "recurrent_gated_delta_rule"
        "chunk_fwd_o"
        "chunk_gated_delta_rule_fwd_h"
        "store_kv_block"
    )

    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend950"
else
    # others
    # currently, no custom aclnn ops for other series
    log "no custom ACLNN ops configured for SOC_VERSION=${SOC_VERSION}; skip build_aclnn"
    exit 0
fi

log_selected_ops


# # build custom ops
# cd csrc
# rm -rf build output build_out
# echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
# bash build.sh --pkg --ops="$CUSTOM_OPS" --soc="$SOC_ARG"

# # install custom ops to vllm_ascend/_cann_ops_custom
# ./build/cann-ops-transformer*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom


(
  set -euo pipefail

  : "${ROOT_DIR:?ROOT_DIR is not set}"

  log "subshell cwd before cd=$(pwd)"
  cd "${ROOT_DIR}/csrc"
  log "subshell cwd after cd=$(pwd)"

  reuse_build="${VLLM_ASCEND_ACLNN_REUSE_BUILD:-0}"
  case "${reuse_build,,}" in
    0|false|off)
      clean_build=1
      ;;
    1|true|on)
      clean_build=0
      ;;
    *)
      echo "ERROR: invalid VLLM_ASCEND_ACLNN_REUSE_BUILD=${reuse_build}; expected 0/1, false/true, or off/on" >&2
      exit 2
      ;;
  esac

  ccache_mode="${VLLM_ASCEND_ACLNN_CCACHE:-off}"
  build_args=(--pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}")
  case "${ccache_mode,,}" in
    0|false|off)
      # CCACHE_DISABLE also bypasses a compiler launcher retained in an
      # explicitly reused CMake cache.
      export CCACHE_DISABLE=1
      build_args+=(--ccache false)
      ;;
    auto)
      unset CCACHE_DISABLE || true
      ;;
    1|true|on)
      ccache_program=$(command -v ccache || true)
      [[ -n "${ccache_program}" ]] || {
        echo "ERROR: VLLM_ASCEND_ACLNN_CCACHE=${ccache_mode}, but ccache was not found" >&2
        exit 2
      }
      unset CCACHE_DISABLE || true
      build_args+=(--ccache "${ccache_program}")
      ;;
    /*)
      [[ -f "${ccache_mode}" && -x "${ccache_mode}" ]] || {
        echo "ERROR: ccache path is not an executable file: ${ccache_mode}" >&2
        exit 2
      }
      unset CCACHE_DISABLE || true
      build_args+=(--ccache "${ccache_mode}")
      ;;
    *)
      echo "ERROR: invalid VLLM_ASCEND_ACLNN_CCACHE=${ccache_mode}; expected off, auto, on, or an absolute path" >&2
      exit 2
      ;;
  esac

  custom_ops_install_dir="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"
  custom_ops_stage_dir="${custom_ops_install_dir}.staging.$$"
  custom_ops_backup_dir="${custom_ops_install_dir}.backup.$$"
  custom_ops_lock_key=$(printf '%s' "${custom_ops_install_dir}" | cksum | awk '{print $1}')
  custom_ops_lock_file="${TMPDIR:-/tmp}/vllm_ascend_build_aclnn.${custom_ops_lock_key}.lock"
  log "custom_ops_install_dir=${custom_ops_install_dir}"

  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for safe custom-op build and package replacement" >&2
    exit 1
  }
  exec {custom_ops_lock_fd}>"${custom_ops_lock_file}"
  flock -n "${custom_ops_lock_fd}" || {
    echo "ERROR: another custom-op build or package installation is in progress" >&2
    exit 1
  }

  [[ ! -e "${custom_ops_stage_dir}" && ! -e "${custom_ops_backup_dir}" ]] || {
    echo "ERROR: custom-op staging or backup path already exists" >&2
    exit 1
  }

  install_swapped=0
  cleanup_custom_op_install() {
    cleanup_status=$?
    set +e
    if [[ -e "${custom_ops_backup_dir}" ]]; then
      if (( install_swapped )); then
        rm -rf -- "${custom_ops_backup_dir}"
      elif [[ ! -e "${custom_ops_install_dir}" ]]; then
        mv -T -- "${custom_ops_backup_dir}" "${custom_ops_install_dir}"
      else
        echo "ERROR: package replacement failed and target path reappeared; old package kept at ${custom_ops_backup_dir}" >&2
      fi
    fi
    rm -rf -- "${custom_ops_stage_dir}"
    flock -u "${custom_ops_lock_fd}" || true
    return "${cleanup_status}"
  }
  trap cleanup_custom_op_install EXIT

  host_arch=$(uname -m)
  case "${host_arch}" in
    aarch64|arm64)
      package_arch="aarch64"
      ;;
    x86_64|amd64)
      package_arch="x86_64"
      ;;
    *)
      echo "ERROR: unsupported host architecture for custom-op package validation: ${host_arch}" >&2
      exit 1
      ;;
  esac

  if (( clean_build )); then
      # Host tiling declarations are compiled into both the host library and
      # AscendC kernels. Reusing csrc/build after that ABI changes can combine
      # a new host library with stale kernel objects, so correctness requires a
      # clean build by default.
      log "cleaning csrc/build and output dirs (set VLLM_ASCEND_ACLNN_REUSE_BUILD=1 to opt in to reuse)"
      rm -rf -- build output build_out
  else
    log "explicitly reusing csrc/build and cleaning output dirs"
    rm -rf -- output build_out
  fi

  : "${CUSTOM_OPS:?CUSTOM_OPS is not set}"
  : "${SOC_VERSION:?SOC_VERSION is not set}"
  : "${SOC_ARG:?SOC_ARG is not set}"

  log "build command: bash build.sh ${build_args[*]}"
  log "building custom ops ${CUSTOM_OPS} for ${SOC_VERSION}"
  bash build.sh "${build_args[@]}"
  log "build.sh finished"

  shopt -s nullglob
  installer_candidates=(./build/cann-ops-transformer*.run)
  shopt -u nullglob

  log "installer candidate count=${#installer_candidates[@]}"
  for installer_file in "${installer_candidates[@]}"; do
    log "installer candidate: $(ls -lh "${installer_file}")"
  done

  (( ${#installer_candidates[@]} == 1 )) || { echo "ERROR: expected 1 installer, got ${#installer_candidates[@]}" >&2; exit 1; }

  chmod +x -- "${installer_candidates[0]}" || true
  log "running installer into staging: ${installer_candidates[0]}"
  "${installer_candidates[0]}" --install-path="${custom_ops_stage_dir}"
  custom_vendor_dir="${custom_ops_stage_dir}/vendors/custom_transformer"
  [[ -s "${custom_vendor_dir}/op_api/lib/libcust_opapi.so" ]] || {
    echo "ERROR: staged custom-op package is missing libcust_opapi.so" >&2
    exit 1
  }
  opmaster_file="${custom_vendor_dir}/op_impl/ai_core/tbe/op_tiling/lib/linux/${package_arch}/libcust_opmaster_rt2.0.so"
  [[ -s "${opmaster_file}" ]] || {
    echo "ERROR: staged custom-op package is missing libcust_opmaster_rt2.0.so" >&2
    exit 1
  }

  # 默认 clean build 保证 host tiling 与 kernel 来自同一轮编译；这里再校验 SAS 内核及其配置完整性。
  if [[ ";${CUSTOM_OPS};" == *";sparse_attn_sharedkv;"* ]]; then
    sparse_kernel_base="${custom_vendor_dir}/op_impl/ai_core/tbe/kernel"
    sparse_kernel_dir="${sparse_kernel_base}/${SOC_ARG}/sparse_attn_sharedkv"
    sparse_config_dir="${sparse_kernel_base}/config/${SOC_ARG}"
    sparse_object=$(find "${sparse_kernel_dir}" -maxdepth 1 -type f -name '*.o' -size +0c -print -quit 2>/dev/null || true)
    sparse_kernel_json=$(find "${sparse_kernel_dir}" -maxdepth 1 -type f -name '*.json' -size +0c -print -quit 2>/dev/null || true)
    [[ -n "${sparse_object}" && -n "${sparse_kernel_json}" ]] || {
      echo "ERROR: staged sparse_attn_sharedkv package is missing non-empty kernel objects or metadata" >&2
      exit 1
    }
    [[ -s "${sparse_config_dir}/sparse_attn_sharedkv.json" &&
       -s "${sparse_config_dir}/binary_info_config.json" ]] || {
      echo "ERROR: staged sparse_attn_sharedkv package is missing kernel config metadata" >&2
      exit 1
    }
    python3 - "${sparse_kernel_base}" "${SOC_ARG}" <<'PY'
import json
import sys
from pathlib import Path

kernel_base = Path(sys.argv[1])
soc = sys.argv[2]
config_dir = kernel_base / "config" / soc
op_config = json.loads((config_dir / "sparse_attn_sharedkv.json").read_text(encoding="utf-8"))
binary_config = json.loads((config_dir / "binary_info_config.json").read_text(encoding="utf-8"))
bin_list = op_config.get("binList")
binary_entry = binary_config.get("SparseAttnSharedkv")
if not isinstance(bin_list, list) or not bin_list:
    raise SystemExit("sparse_attn_sharedkv.json has no binList entries")
if not isinstance(binary_entry, dict) or not binary_entry.get("binaryList"):
    raise SystemExit("binary_info_config.json has no SparseAttnSharedkv binaryList")
expected_prefix = f"{soc}/sparse_attn_sharedkv/"
for item in bin_list:
    relative_json = item.get("binInfo", {}).get("jsonFilePath")
    if not isinstance(relative_json, str) or not relative_json.startswith(expected_prefix):
        raise SystemExit(f"invalid sparse_attn_sharedkv jsonFilePath: {relative_json!r}")
    json_path = kernel_base / relative_json
    object_path = json_path.with_suffix(".o")
    if not json_path.is_file() or json_path.stat().st_size == 0:
        raise SystemExit(f"missing sparse_attn_sharedkv kernel metadata: {json_path}")
    if not object_path.is_file() or object_path.stat().st_size == 0:
        raise SystemExit(f"missing sparse_attn_sharedkv kernel object: {object_path}")
    json.loads(json_path.read_text(encoding="utf-8"))
PY
  fi
  touch "${custom_ops_stage_dir}/.gitkeep"

  if [[ -e "${custom_ops_install_dir}" ]]; then
    mv -T -- "${custom_ops_install_dir}" "${custom_ops_backup_dir}"
  fi
  mv -T -- "${custom_ops_stage_dir}" "${custom_ops_install_dir}"
  install_swapped=1

  # CANN leaves generated vendor script dirs owner-read-only; keep repo-local
  # editable-build artifacts removable by the non-root user who built them.
  if [[ -d "${custom_ops_install_dir}/vendors/custom_transformer/scripts" ]]; then
    chmod u+w "${custom_ops_install_dir}/vendors/custom_transformer/scripts"
  fi
  log "installer finished and validated staged package replacement completed"
  log "installed files under ${custom_ops_install_dir} (maxdepth=4, first 120 entries):"
  { find "${custom_ops_install_dir}" -mindepth 1 -maxdepth 4 -print | sort | head -n 120 | sed 's#^#[build_aclnn] install: #'; } || true

  # install batch_invariant run package and whl package
  if [[ "${VLLM_BATCH_INVARIANT:-0}" == "1" ]]; then
    log "VLLM_BATCH_INVARIANT=1, installing batch_invariant run package and whl package..."

    # call separate installation script
    batch_invariant_script="${ROOT_DIR}/csrc/build_batch_invariant_ops.sh"
    if [[ -f "${batch_invariant_script}" ]]; then
      log "Calling batch_invariant_ops build script: ${batch_invariant_script}"
      bash "${batch_invariant_script}" "${SOC_ARG}"
    else
      log "Warning: batch_invariant_ops build script not found at ${batch_invariant_script}"
    fi
  else
    log "VLLM_BATCH_INVARIANT is not set to 1, skipping batch_invariant ops build"
  fi

  cleanup_custom_op_install
  trap - EXIT
  set -e
)
