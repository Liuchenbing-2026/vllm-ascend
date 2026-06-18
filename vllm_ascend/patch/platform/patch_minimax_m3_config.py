#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch target: vllm/config/model.py
# - MiniMax-M3 on NPU: for ACL graph capture set HCCL_OP_EXPANSION_MODE=AIV
#   if the user didn't set it (mirrors the MiniMax-M2 config patch).
#
# Model-type note: the M3 VL checkpoint reports model_type "minimax_m3_vl" at
# the top level (text_config.model_type may be None). We therefore match the
# substring "minimax_m3", which covers both "minimax_m3" and "minimax_m3_vl".

import os

from vllm.config.model import ModelConfig
from vllm.logger import logger

try:
    from vllm.platforms import current_platform
except Exception:  # pragma: no cover - defensive
    current_platform = None  # type: ignore[assignment]

_original_verify_cuda_graph = getattr(ModelConfig, "_verify_cuda_graph", None)


def _get_model_type(cfg: ModelConfig) -> str | None:
    """Best-effort model_type lookup across vLLM versions / config nesting."""
    # Try the VL wrapper's top model_type first, then text sub-config.
    for src_attr in ("model_arch_config", "hf_config", "hf_text_config"):
        src = getattr(cfg, src_attr, None)
        if src is not None:
            mt = getattr(src, "model_type", None)
            if mt:
                return mt
            # Look one level deeper into a nested text_config.
            text_cfg = getattr(src, "text_config", None)
            if text_cfg is not None:
                mt = getattr(text_cfg, "model_type", None)
                if mt:
                    return mt
    return getattr(cfg, "model_type", None)


def _is_minimax_m3(cfg: ModelConfig) -> bool:
    mt = _get_model_type(cfg)
    return bool(mt) and "minimax_m3" in str(mt).lower()


def _is_npu() -> bool:
    return current_platform is not None and getattr(
        current_platform, "device_name", None
    ) == "npu"


def _patched_verify_cuda_graph(self: ModelConfig) -> None:
    assert _original_verify_cuda_graph is not None

    if (
        _is_npu()
        and _is_minimax_m3(self)
        and not getattr(self, "enforce_eager", True)
    ):
        expansion_mode = os.environ.get("HCCL_OP_EXPANSION_MODE")
        if expansion_mode is None:
            os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
            logger.info(
                "Set HCCL_OP_EXPANSION_MODE=AIV for MiniMax-M3 ACL graph "
                "capture on NPU."
            )
        elif expansion_mode != "AIV":
            logger.warning(
                "HCCL_OP_EXPANSION_MODE=%s may reduce ACL graph shape coverage "
                "for MiniMax-M3 on NPU. Recommended value: AIV.",
                expansion_mode,
            )

    return _original_verify_cuda_graph(self)


if _original_verify_cuda_graph is not None:
    ModelConfig._verify_cuda_graph = _patched_verify_cuda_graph
else:  # pragma: no cover - defensive
    logger.debug(
        "ModelConfig._verify_cuda_graph not found; skipping MiniMax-M3 "
        "HCCL_OP_EXPANSION_MODE patch."
    )

# ---------------------------------------------------------------------------
# Make transformers `_LazyConfigMapping` picklable for vLLM v1 EngineCore spawn.
# ---------------------------------------------------------------------------
# With trust_remote_code, the MiniMax-M3 VL config drags transformers'
# CONFIG_MAPPING (a `_LazyConfigMapping`) into the VllmConfig payload that vLLM
# pickles to spawn the EngineCore worker. transformers 5.5.4's
# `_LazyConfigMapping` defines no `__reduce__`, so the worker's
# `pickle.load(from_parent)` reconstructs it by calling `_LazyConfigMapping()`
# with no args -> `TypeError: __init__() missing 1 required positional
# argument: 'mapping'`, which aborts EngineCore init.
#
# `__reduce__` runs at PICKLE time in the PARENT, and vllm_ascend is imported at
# platform-plugin activation (before the config is pickled), so patching here
# fixes the parent's serialization. We reconstruct from `self._mapping` (the
# static name->class-name dict) and intentionally DROP `_extra_content`
# (runtime/trust_remote_code registrations) — those may hold unpicklable
# dynamic classes, and the worker re-registers them itself via trust_remote_code.
def _patch_lazy_config_mapping_picklable() -> None:
    try:
        from transformers.models.auto import configuration_auto as _ca

        _L = _ca._LazyConfigMapping
    except Exception:
        return
    if getattr(_L, "_vllm_ascend_picklable", False):
        return

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (self.__class__, (self._mapping,))

    _L.__reduce__ = __reduce__
    _L._vllm_ascend_picklable = True
    logger.info("[vllm-ascend] Patched transformers _LazyConfigMapping to be picklable (MiniMax-M3 trust_remote_code spawn fix).")


_patch_lazy_config_mapping_picklable()


# TODO(verify): Unlike MiniMax-M2 (which disabled an fp8 checkpoint on NPU), the
# M3 w8a8 checkpoint is int8 (ModelSlim). No fp8-disable hook is needed here.
# However, the ModelSlim quant layer must recognize model_type "minimax_m3" /
# "minimax_m3_vl" so the FusedMoE/linear quant-method lookup remaps `mlp` ->
# `block_sparse_moe` and strips expert indices. That recognition lives in
# vllm_ascend/quantization/modelslim_config.py (get_quant_method + the
# packed_modules_model_mapping table), which is OUTSIDE the 4-file scope of this
# change. See the delivery report: a `minimax_m3` entry mirroring `minimax_m2`
# is REQUIRED for the int8 weights to find their quant scheme at load time.
