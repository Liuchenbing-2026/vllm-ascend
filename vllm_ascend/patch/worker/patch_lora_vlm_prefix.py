#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# ---------------------------------------------------------------------------
# FIX (v2, self-verifying): VLM-wrapped hybrid dense LoRA produces delta=0.
#
# Applies to Qwen3.5-27B AND 35B (and any model whose text tower is wrapped
# under a submodule such as `language_model`).
#
# Root cause (bug.md ⑤q/⑤t/⑤u, confirmed by two independent probes):
#   * ⑤t: from_lora_tensors loads the adapter weights NON-zero, but stores them
#     under the adapter's bare key, e.g.  model.layers.N.mlp.down_proj
#   * ⑤u: the model registers its LoRA modules under the wrapped name, e.g.
#     language_model.model.layers.N.mlp.down_proj
#   The lookup at activation (`_get_lora_layer_weights`) uses the wrapped model
#   name -> misses the bare-keyed adapter weights -> set_lora receives a zero
#   buffer -> LoRA delta == 0 (LORA output bit-identical to BASE).
#
# Fix: after WorkerLoRAManager._load_adapter loads the LoRAModel (but BEFORE it
# is registered / packed-merged), rename the loras dict keys so they line up
# with the model's actual module names. Instead of hard-coding
# `language_model.`, the prefix is AUTO-DETECTED from the manager's own module
# table (LoRAModelManager.modules): find a model module whose name ends with a
# (non-packed) adapter key and take the leading difference as the prefix. This
# makes the fix work unchanged for 27B / 35B / any wrapper depth.
#
# Once the keys line up:
#   * plain modules (down_proj/o_proj) match directly, and
#   * _create_merged_loras_inplace finds q/k/v & gate/up sub-loras and packs the
#     non-zero qkv_proj / gate_up_proj automatically.
#
# This build is intentionally verbose: it logs the detected prefix, how many
# keys were remapped, how many now match real model modules, and the norm of a
# sample adapter tensor (to distinguish a real non-zero adapter from a zero
# dummy). One clean run therefore proves whether the remap actually lands.
#
# Safety: only touches the real-adapter load path (dummy LoRAs already use model
# names); leaves keys untouched if no prefix is detected (already-aligned plain
# text models); never raises into the serving path. Disable with
# VLLM_ASCEND_LORA_VLM_PREFIX=0.
# ---------------------------------------------------------------------------
import os

from vllm.logger import init_logger
from vllm.lora.worker_manager import WorkerLoRAManager

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_ASCEND_LORA_VLM_PREFIX", "1") not in (
    "0", "", "false", "False")

_orig_load_adapter = WorkerLoRAManager._load_adapter


def _tensor_norm(t):
    try:
        return round(float(t.norm().item()), 4)
    except Exception:  # noqa: BLE001
        return None


def _sample_adapter_norm(loras):
    """Norm of the first available lora_a, to tell a real adapter from a zero
    dummy."""
    for w in loras.values():
        a = getattr(w, "lora_a", None)
        if a is not None:
            return _tensor_norm(a)
    return None


def _detect_prefix(lora_keys, model_module_names):
    """Return the prefix P such that some model module == P + <adapter_key>.

    Uses only keys that are likely NON-packed (down_proj / o_proj / *_proj that
    already appear verbatim as a model module suffix), so packed q/k/v vs
    qkv_proj naming does not confuse detection. Returns "" if the adapter keys
    already match model modules (nothing to do).
    """
    model_set = set(model_module_names)
    for lk in lora_keys:
        # Already aligned? then no prefix needed.
        if lk in model_set:
            return ""
        for mm in model_module_names:
            if mm.endswith("." + lk):
                return mm[: -len(lk)]
    return None


def _load_adapter(self, lora_request):
    lora = _orig_load_adapter(self, lora_request)
    try:
        mgr = self._adapter_manager
        model_module_names = list(getattr(mgr, "modules", {}).keys())
        lora_keys = list(lora.loras.keys())
        src_norm = _sample_adapter_norm(lora.loras)
        if not model_module_names or not lora_keys:
            logger.warning(
                "[lora-vlm-prefix] skip: model_modules=%d lora_keys=%d "
                "(LoRA manager modules not populated yet?)",
                len(model_module_names), len(lora_keys))
            return lora

        prefix = _detect_prefix(lora_keys, model_module_names)

        if prefix is None:
            logger.warning(
                "[lora-vlm-prefix] could NOT detect a prefix aligning adapter "
                "keys to model modules; leaving keys unchanged (LoRA may be "
                "no-op). sample lora_a norm=%s, adapter keys=%s, "
                "model modules e.g. %s",
                src_norm, lora_keys[:3], model_module_names[:2])
            return lora
        if prefix == "":
            logger.info(
                "[lora-vlm-prefix] adapter keys already align with model "
                "modules (no remap). sample lora_a norm=%s, keys=%s",
                src_norm, lora_keys[:3])
            return lora

        remapped = {}
        changed = 0
        for key, weights in lora.loras.items():
            new_key = key
            if not key.startswith(prefix):
                new_key = prefix + key
                changed += 1
            remapped[new_key] = weights

        # How many remapped keys now correspond to a real model module, either
        # directly (down_proj/o_proj) or as a packed sub-module (q_proj under
        # qkv_proj, gate_proj under gate_up_proj)?
        model_set = set(model_module_names)
        direct = sum(1 for k in remapped if k in model_set)
        parents = {k.rsplit(".", 1)[0] for k in remapped}
        packed_hits = sum(
            1 for m in model_module_names if m.rsplit(".", 1)[0] in parents)

        lora.loras = remapped
        logger.info(
            "[lora-vlm-prefix] detected prefix=%r, remapped %d/%d adapter keys "
            "(%s). after remap: %d keys directly match model modules, "
            "%d model modules share a parent (packed q/k/v, gate/up). "
            "sample lora_a norm=%s (non-zero => real adapter). "
            "model modules e.g. %s",
            prefix, changed, len(lora_keys), type(mgr).__name__, direct,
            packed_hits, src_norm, model_module_names[:2])
    except Exception as e:  # noqa: BLE001 - never break serving on a diag/fix
        logger.warning(
            "[lora-vlm-prefix] remap skipped due to %s: %s",
            type(e).__name__, e)
    return lora


if _ENABLED:
    WorkerLoRAManager._load_adapter = _load_adapter
    logger.info("[lora-vlm-prefix] WorkerLoRAManager._load_adapter patched "
                "(auto-detect prefix; VLLM_ASCEND_LORA_VLM_PREFIX on).")
