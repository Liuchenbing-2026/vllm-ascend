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
# FIX: Qwen3.5 (VLM-wrapped hybrid) dense LoRA silently produces delta=0.
#
# Root cause (bug.md ⑤q/⑤t/⑤u, confirmed on v0.22.1rc1 AND v0.23.0):
#   Qwen3_5ForConditionalGeneration wraps the text tower under `language_model`,
#   so its LoRA-eligible modules are registered as
#       language_model.model.layers.N.self_attn.qkv_proj  (etc.)
#   The agent-lora adapter, trained on the standalone causal LM, stores its
#   tensors under bare
#       model.layers.N.self_attn.q_proj / k_proj / v_proj  (etc.)
#   The shared `hf_to_vllm_mapper` only has the rule
#       "model.language_model."  ->  "language_model.model."
#   which is for the *base* checkpoint (`model.language_model.*`), and does NOT
#   match the adapter's bare `model.*`. So `from_lora_tensors` stores the
#   (non-zero) weights under `model.layers.N.*`, while the manager looks them up
#   under `language_model.model.layers.N.*` -> miss -> the packed-merge finds no
#   sub-loras and `set_lora` ends up with a zero buffer -> LoRA delta == 0
#   (LORA output bit-identical to BASE).
#
# Why not just add "model." -> "language_model.model." to hf_to_vllm_mapper:
#   that mapper is SHARED with base-weight loading, where it would wrongly
#   rewrite `model.visual.*` and `model.language_model.*`. So the remap must be
#   applied to LoRA names ONLY.
#
# Fix: after WorkerLoRAManager._load_adapter loads the LoRAModel (but BEFORE it
# is registered / packed-merged), rename the loras dict keys to add the
# `language_model.` prefix, gated on the model actually having a
# `language_model` submodule. Once the keys line up:
#   * plain modules (down_proj/o_proj) match directly, and
#   * `_create_merged_loras_inplace` finds q/k/v & gate/up sub-loras and packs
#     the non-zero qkv_proj / gate_up_proj automatically.
#
# Scope / safety:
#   * gated by hasattr(model, "language_model") -> no effect on plain text
#     models (Qwen3ForCausalLM etc. have no such attr).
#   * only rewrites keys starting with bare `model.` (never `model.language_model.`
#     or `language_model.*`), so full-model adapters and base loading are
#     untouched.
#   * only the real-adapter load path; dummy LoRAs already use model names.
#   * disable with VLLM_ASCEND_LORA_VLM_PREFIX=0.
# ---------------------------------------------------------------------------
import os

from vllm.logger import init_logger
from vllm.lora.worker_manager import WorkerLoRAManager

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_ASCEND_LORA_VLM_PREFIX", "1") not in (
    "0", "", "false", "False")

# The submodule the text tower lives under, and therefore the prefix the
# adapter's bare `model.` keys must be rewritten to (`model.` ->
# `language_model.model.`).
_LM_ATTR = "language_model"

_orig_load_adapter = WorkerLoRAManager._load_adapter


def _load_adapter(self, lora_request):
    lora = _orig_load_adapter(self, lora_request)

    model = self._adapter_manager.model
    if not hasattr(model, _LM_ATTR):
        return lora

    remapped = {}
    changed = 0
    for key, weights in lora.loras.items():
        new_key = key
        # Adapter keys are already stripped of the peft `base_model.model.`
        # prefix here, so a bare `model.` corresponds to the wrapped text
        # tower's `language_model.model.`. Skip anything that is already
        # correctly scoped.
        if key.startswith("model.") and not key.startswith(
                f"model.{_LM_ATTR}."):
            new_key = f"{_LM_ATTR}.{key}"
            changed += 1
        remapped[new_key] = weights

    if changed:
        lora.loras = remapped
        logger.info(
            "[lora-vlm-prefix] remapped %d LoRA keys with '%s.' prefix so they "
            "match %s's wrapped text-tower modules (Qwen3.5 dense LoRA fix).",
            changed, _LM_ATTR, type(model).__name__)
    return lora


if _ENABLED:
    WorkerLoRAManager._load_adapter = _load_adapter
    logger.info("[lora-vlm-prefix] WorkerLoRAManager._load_adapter patched "
                "(VLLM_ASCEND_LORA_VLM_PREFIX on).")
