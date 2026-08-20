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

from types import SimpleNamespace

from vllm_ascend.spec_decode.llm_base_proposer import _get_image_token_index


def test_gemma4_multimodal_token_uses_image_token_id():
    config = SimpleNamespace(image_token_id=258880)

    assert _get_image_token_index("Gemma4ForConditionalGeneration", config) == 258880


def test_multimodal_token_fallback_uses_image_token_index():
    config = SimpleNamespace(image_token_index=32000)

    assert _get_image_token_index("OtherMultimodalModel", config) == 32000
