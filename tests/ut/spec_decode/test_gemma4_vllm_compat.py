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

import pytest
from torch import nn
from vllm.model_executor.models import gemma4, gemma4_mtp


class _DummyModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class _CaptureAttention(_DummyModule):
    def __init__(self, *args, num_kv_heads: int, **kwargs) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads


def _config(layer_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        attention_k_eq_v=True,
        attn_logit_softcapping=None,
        global_head_dim=512,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=5376,
        hidden_size_per_layer_input=0,
        intermediate_size=21504,
        layer_types=[layer_type],
        max_position_embeddings=5500,
        num_attention_heads=32,
        num_global_key_value_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=16,
        num_kv_shared_layers=0,
        rms_norm_eps=1e-6,
        use_double_wide_mlp=False,
    )


@pytest.mark.parametrize(
    ("layer_type", "expected_num_kv_heads"),
    [("full_attention", 4), ("sliding_attention", 16)],
)
def test_target_decoder_uses_attention_specific_kv_heads(
    monkeypatch, layer_type: str, expected_num_kv_heads: int
) -> None:
    monkeypatch.setattr(gemma4, "Gemma4Attention", _CaptureAttention)
    monkeypatch.setattr(gemma4, "Gemma4MLP", _DummyModule)
    monkeypatch.setattr(gemma4, "RMSNorm", _DummyModule)

    layer = gemma4.Gemma4DecoderLayer(_config(layer_type), prefix="model.layers.0")

    assert layer.self_attn.num_kv_heads == expected_num_kv_heads


@pytest.mark.parametrize(
    ("layer_type", "expected_num_kv_heads"),
    [("full_attention", 4), ("sliding_attention", 16)],
)
def test_mtp_decoder_matches_target_kv_heads(monkeypatch, layer_type: str, expected_num_kv_heads: int) -> None:
    monkeypatch.setattr(gemma4_mtp, "Gemma4MTPAttention", _CaptureAttention)
    monkeypatch.setattr(gemma4_mtp, "Gemma4MLP", _DummyModule)
    monkeypatch.setattr(gemma4_mtp, "RMSNorm", _DummyModule)

    layer = gemma4_mtp.Gemma4MTPDecoderLayer(_config(layer_type), prefix="model.layers.0")

    assert layer.self_attn.num_kv_heads == expected_num_kv_heads
