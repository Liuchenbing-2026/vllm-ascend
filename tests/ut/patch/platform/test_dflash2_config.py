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

from types import SimpleNamespace

from vllm_ascend.patch.platform.patch_eagle_vocab_fallback import (
    _materialize_vocab_size,
)
from vllm_ascend.patch.platform.patch_speculators_dflash2 import (
    update_dflash_preserve_dflash2,
)


def _dflash_config() -> dict:
    return {
        "aux_hidden_state_layer_ids": [1],
        "draft_vocab_size": 32,
        "mask_token_id": 31,
        "target_hidden_size": 16,
    }


def test_update_dflash_preserves_dflash2_architecture() -> None:
    pretrained_config = {"architectures": ["DFlash2DraftModel"]}

    update_dflash_preserve_dflash2(_dflash_config(), pretrained_config)

    assert pretrained_config["architectures"] == ["DFlash2DraftModel"]
    assert pretrained_config["dflash_config"]["mask_token_id"] == 31


def test_update_dflash_keeps_dflash1_default() -> None:
    pretrained_config = {"architectures": ["Qwen3ForCausalLM"]}

    update_dflash_preserve_dflash2(_dflash_config(), pretrained_config)

    assert pretrained_config["architectures"] == ["DFlashDraftModel"]


def test_materialize_vocab_size_from_nested_config() -> None:
    model = SimpleNamespace(text_config=SimpleNamespace(vocab_size=151936))

    _materialize_vocab_size(model)

    assert model.vocab_size == 151936


def test_materialize_vocab_size_does_not_override_existing_value() -> None:
    model = SimpleNamespace(
        vocab_size=32000,
        text_config={"vocab_size": 151936},
    )

    _materialize_vocab_size(model)

    assert model.vocab_size == 32000
