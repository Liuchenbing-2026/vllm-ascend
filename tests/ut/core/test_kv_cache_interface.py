#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _maybe_promote_dflash_swa_for_full_allocation,
)


def test_ascend_mla_page_size_honors_hybrid_cache_padding():
    spec = AscendMLAAttentionSpec(
        block_size=768,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=912384,
    )

    assert spec.real_page_size_bytes == 884736
    assert spec.page_size_bytes == 912384

    merged = AscendMLAAttentionSpec.merge([spec, spec])
    assert merged.real_page_size_bytes == 884736
    assert merged.page_size_padded == 912384
    assert merged.page_size_bytes == 912384


def test_dflash_full_kv_diagnostic_promotes_only_sliding_specs():
    sliding = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=2048,
        page_size_padded=16384,
    )
    hidden_state_sentinel = object()
    specs = {
        "draft.layers.0.attn": sliding,
        "target.hidden_states": hidden_state_sentinel,
    }
    config = SimpleNamespace(
        additional_config={"dflash_full_kv_allocation": True},
        speculative_config=SimpleNamespace(method="dflash"),
    )

    _maybe_promote_dflash_swa_for_full_allocation(config, specs)

    converted = specs["draft.layers.0.attn"]
    assert isinstance(converted, FullAttentionSpec)
    assert converted.block_size == sliding.block_size
    assert converted.sliding_window == sliding.sliding_window
    assert converted.page_size_padded == sliding.page_size_padded
    assert specs["target.hidden_states"] is hidden_state_sentinel


def test_dflash_full_kv_diagnostic_is_opt_in():
    sliding = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=2048,
    )
    specs = {"draft.layers.0.attn": sliding}
    config = SimpleNamespace(
        additional_config={},
        speculative_config=SimpleNamespace(method="dflash"),
    )

    _maybe_promote_dflash_swa_for_full_allocation(config, specs)

    assert specs["draft.layers.0.attn"] is sliding
