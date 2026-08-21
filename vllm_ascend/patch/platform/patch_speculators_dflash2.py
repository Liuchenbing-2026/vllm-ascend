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
"""Preserve DFlash2 draft architectures through the v0.23.0 dflash updater.

vLLM v0.23.0's ``update_dflash`` unconditionally rewrites the draft
checkpoint architecture to ``DFlashDraftModel``, which would silently
degrade a DFlash2 checkpoint (e.g. z-lab/Qwen3.8-27B-DFlash2) to the plain
DFlash model and skip the candidate selector. This patch re-registers the
``dflash`` speculator updater with a wrapper that restores the checkpoint's
declared architecture whenever it starts with ``DFlash`` (mirrors upstream
vllm#52816's preservation rule); DFlash1 checkpoints resolve exactly as
before.
"""

from __future__ import annotations

from typing import Any

from vllm.transformers_utils.configs.speculators.algos import (
    register_speculator,
    update_dflash,
)

_ORIGINAL_UPDATE_DFLASH = update_dflash


@register_speculator("dflash")
def update_dflash_preserve_dflash2(
    config_dict: dict[str, Any],
    pre_trained_config: dict[str, Any],
) -> None:
    declared_architectures = list(pre_trained_config.get("architectures") or [])
    _ORIGINAL_UPDATE_DFLASH(config_dict, pre_trained_config)
    if declared_architectures and all(str(name).startswith("DFlash") for name in declared_architectures):
        pre_trained_config["architectures"] = declared_architectures
