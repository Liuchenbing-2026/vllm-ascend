# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from transformers import AutoConfig, PretrainedConfig

from vllm_ascend.transformers_utils import register_gemma4_assistant_config


def test_register_gemma4_assistant_config(tmp_path):
    config = {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 5376,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 4,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 0,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    register_gemma4_assistant_config()
    parsed = AutoConfig.from_pretrained(tmp_path)

    assert parsed.model_type == "gemma4_assistant"
    assert parsed.backbone_hidden_size == 5376
    assert isinstance(parsed.text_config, PretrainedConfig)
    assert parsed.text_config.hidden_size == 1024
    assert parsed.text_config.num_kv_shared_layers == 4


def test_gemma4_assistant_is_converted_to_vllm_mtp(tmp_path):
    config = {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 4,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 0,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    register_gemma4_assistant_config()

    parsed = AutoConfig.from_pretrained(tmp_path)
    from vllm_ascend.patch.platform.patch_speculative_config import hf_config_override

    converted = hf_config_override(parsed)

    assert converted.model_type == "gemma4_mtp"
    assert converted.architectures == ["Gemma4MTPModel"]
    assert converted.n_predict == 1
    assert converted.text_config.num_kv_shared_layers == 0
