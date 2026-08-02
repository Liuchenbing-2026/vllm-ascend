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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import vllm_ascend.patch.worker.patch_lora_vlm_prefix as vlm_prefix_patch


def test_detect_prefix_for_wrapped_language_model() -> None:
    lora_keys = ["model.layers.0.mlp.down_proj"]
    model_modules = ["language_model.model.layers.0.mlp.down_proj"]

    assert vlm_prefix_patch._detect_prefix(lora_keys, model_modules) == "language_model."


def test_detect_prefix_leaves_aligned_modules_unchanged() -> None:
    lora_keys = ["model.layers.0.mlp.down_proj"]

    assert vlm_prefix_patch._detect_prefix(lora_keys, lora_keys) == ""


def test_load_adapter_remaps_keys_and_preserves_weights() -> None:
    weight = object()
    lora = SimpleNamespace(loras={"model.layers.0.mlp.down_proj": weight})
    worker = SimpleNamespace(
        _adapter_manager=SimpleNamespace(
            modules={"language_model.model.layers.0.mlp.down_proj": object()},
        ),
    )
    load_adapter = Mock(return_value=lora)

    with patch.object(vlm_prefix_patch, "_orig_load_adapter", load_adapter):
        loaded = vlm_prefix_patch._load_adapter(worker, "request")

    assert loaded is lora
    assert loaded.loras == {"language_model.model.layers.0.mlp.down_proj": weight}
    load_adapter.assert_called_once_with(worker, "request")
