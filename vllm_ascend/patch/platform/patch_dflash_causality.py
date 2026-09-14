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

from typing import Any

import vllm.model_executor.models.qwen3_dflash as qwen3_dflash


def _dflash_layer_causal(config: Any, layer_idx: int) -> bool:
    """Resolve explicit causality before falling back to legacy layer defaults.

    Backport of vLLM #52816. DFlash2 checkpoints declare their attention
    semantics with a top-level ``is_causal``; vLLM 0.26 only reads
    ``dflash_config.causal`` and otherwise treats every sliding layer as causal.
    """
    is_causal = getattr(config, "is_causal", None)
    if is_causal is not None:
        return bool(is_causal)
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return bool(override)
    layer_types = getattr(config, "layer_types", None)
    return bool(layer_types) and layer_types[layer_idx] == qwen3_dflash._SLIDING_ATTENTION


# dflash_has_any_non_causal and _resolve_layer_attention look this helper up in
# the module namespace at call time, so rebinding it covers both.
qwen3_dflash._dflash_layer_causal = _dflash_layer_causal
