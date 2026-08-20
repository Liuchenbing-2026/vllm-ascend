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

from typing import Any

from huggingface_hub.dataclasses import strict
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


@strict
class Gemma4AssistantConfig(PretrainedConfig):
    """Compatibility config for Transformers releases before Gemma4 MTP."""

    model_type = "gemma4_assistant"
    sub_configs = {"text_config": Gemma4TextConfig}

    text_config: Gemma4TextConfig | dict[str, Any] | None = None
    backbone_hidden_size: int = 1536
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = Gemma4TextConfig(**self.text_config)

        if self.text_config is not None and not self.text_config.num_kv_shared_layers:
            self.text_config.num_kv_shared_layers = self.text_config.num_hidden_layers

        super().__post_init__(**kwargs)


def register_gemma4_assistant_config() -> None:
    """Backport Gemma4 assistant config parsing when Transformers lacks it."""

    if "gemma4_assistant" not in CONFIG_MAPPING:
        AutoConfig.register("gemma4_assistant", Gemma4AssistantConfig)
