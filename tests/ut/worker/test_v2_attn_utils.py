# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#
"""CPU-only tests for the v2 KV cache spec collector and attention-state helper."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.worker.v2.attn_utils import (
    _build_dsa_extra_kwargs,
    _get_attention_kv_cache_dims,
    build_attn_state,
    get_kv_cache_spec,
)

_ATTN_UTILS = "vllm_ascend.worker.v2.attn_utils"


class _PlainKVCacheLayer:
    """An AttentionLayerBase-like layer that is neither Attention nor MLAAttention.

    DeepSeek V4's DSA contributes several of these; the collector has to ask
    them for their own spec instead of dropping them.
    """

    def __init__(self, spec):
        self._spec = spec

    def get_kv_cache_spec(self, vllm_config):
        return self._spec


class _FakeMambaLayer(MambaBase):
    def get_state_shape(self):
        return ((1, 1),)

    @property
    def mamba_type(self):
        return "mamba2"

    def get_state_dtype(self):
        return (torch.bfloat16,)


class _FakeCacheOnlyLayer(CacheOnlyAttentionLayer):
    def __init__(self):
        # The real __init__ needs a live vllm config and a compilation context;
        # only the type matters to the collector.
        nn.Module.__init__(self)


def _full_attention_spec():
    return FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)


def _indexer_spec():
    return AscendSFAIndexerCacheSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)


class TestGetKVCacheSpec(TestBase):
    def _collect(self, layers):
        vllm_config = MagicMock()
        with patch(f"{_ATTN_UTILS}.get_layers_from_vllm_config", return_value=layers):
            return get_kv_cache_spec(vllm_config)

    def test_keeps_non_attention_layer(self):
        spec = _full_attention_spec()
        kv_cache_spec = self._collect({"model.layers.0.dsa": _PlainKVCacheLayer(spec)})

        self.assertEqual(list(kv_cache_spec), ["model.layers.0.dsa"])
        self.assertIs(kv_cache_spec["model.layers.0.dsa"], spec)

    def test_drops_non_attention_layer_without_spec(self):
        kv_cache_spec = self._collect({"model.layers.0.dsa": _PlainKVCacheLayer(None)})

        self.assertEqual(kv_cache_spec, {})

    def test_refuses_indexer_spec_from_any_layer_class(self):
        # The rule is keyed on the spec, so it must fire for a layer class the
        # collector has never heard of, not only for the one indexer class that
        # emits AscendSFAIndexerCacheSpec today.
        with self.assertRaises(NotImplementedError) as ctx:
            self._collect({"model.layers.0.indexer": _PlainKVCacheLayer(_indexer_spec())})

        message = str(ctx.exception)
        self.assertIn("model.layers.0.indexer", message)
        self.assertIn("_PlainKVCacheLayer", message)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0", message)

    def test_refuses_recurrent_state_layer(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self._collect({"model.layers.0.mixer": _FakeMambaLayer()})

        message = str(ctx.exception)
        self.assertIn("model.layers.0.mixer", message)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0", message)

    def test_refuses_hidden_state_cache_layer(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self._collect({"model.layers.0.cache": _FakeCacheOnlyLayer()})

        message = str(ctx.exception)
        self.assertIn("model.layers.0.cache", message)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0", message)


class TestGetAttentionKVCacheDims(TestBase):
    def _refuse(self, spec):
        # A single-latent-vector page holds one head_size vector, so there is no
        # second dimension to report and the pair would double-count the page.
        layer = _PlainKVCacheLayer(None)

        with (
            patch(f"{_ATTN_UTILS}.get_current_vllm_config", return_value=MagicMock()),
            patch(f"{_ATTN_UTILS}.get_layers_from_vllm_config", return_value={"model.layers.0.self_attn": layer}),
            self.assertRaises(NotImplementedError) as ctx,
        ):
            _get_attention_kv_cache_dims("model.layers.0.self_attn", spec)

        message = str(ctx.exception)
        self.assertIn("model.layers.0.self_attn", message)
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=0", message)

    def test_refuses_mla_shaped_spec_from_non_mla_layer(self):
        self._refuse(AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16))

    def test_refuses_upstream_mla_spec_from_non_mla_layer(self):
        # DeepseekV32IndexerCache reports this exact spec, so the refusal must
        # not be keyed on the Ascend subclass.
        self._refuse(MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16))

    def test_refuses_sliding_window_mla_spec_from_non_mla_layer(self):
        # SlidingWindowMLASpec derives from SlidingWindowSpec, not from
        # MLAAttentionSpec: DeepSeek V4's SWA and compressor-state caches reach
        # here through it and must not fall through to the generic pair.
        self._refuse(
            AscendSlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
                sliding_window=128,
            )
        )

    def test_generic_spec_reports_head_size_pair(self):
        spec = _full_attention_spec()

        self.assertEqual(_get_attention_kv_cache_dims("model.layers.0.self_attn", spec), (64, 64))


class TestBuildAttnState(TestBase):
    @staticmethod
    def _pooling_config(kv_cache_groups):
        return SimpleNamespace(
            model_config=SimpleNamespace(runner_type="pooling"),
            kv_cache_config=SimpleNamespace(kv_cache_groups=kv_cache_groups),
        )

    def test_attention_free_model_has_no_kv_cache_group(self):
        # An attention-free pooling model gets an empty group list, so this
        # helper must not index kv_cache_groups[0] to decide the state.
        attn_state = build_attn_state(
            self._pooling_config([]),
            np.array([4], dtype=np.int32),
            1,
            np.array([4], dtype=np.int32),
            np.array([4], dtype=np.int32),
        )

        self.assertIs(attn_state, AscendAttentionState.PrefillNoCache)

    def test_pooling_model_with_cached_group(self):
        group = SimpleNamespace(kv_cache_spec=_full_attention_spec())
        attn_state = build_attn_state(
            self._pooling_config([group]),
            np.array([4], dtype=np.int32),
            1,
            np.array([4], dtype=np.int32),
            np.array([4], dtype=np.int32),
        )

        self.assertIs(attn_state, AscendAttentionState.PrefillCacheHit)


class TestBuildDSAExtraKwargs(TestBase):
    @staticmethod
    def _kwargs(for_cudagraph_capture, memos):
        return _build_dsa_extra_kwargs(
            attn_group=SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
            num_reqs_actual=3,
            prefill_ratio_to_sas_metadata=memos[0],
            decode_ratio_to_sas_metadata=memos[1],
            common_ratio_to_sas_metadata=memos[2],
            for_cudagraph_capture=for_cudagraph_capture,
        )

    def test_shares_the_callers_memos(self):
        memos = ({}, {}, {})
        kwargs = self._kwargs(False, memos)

        self.assertIs(kwargs["prefill_ratio_to_sas_metadata"], memos[0])
        self.assertIs(kwargs["decode_ratio_to_sas_metadata"], memos[1])
        self.assertIs(kwargs["common_ratio_to_sas_metadata"], memos[2])
        # The group's block size, not the builder's rewritten one.
        self.assertEqual(kwargs["block_size"], 128)
        self.assertEqual(kwargs["num_reqs_actual"], 3)

    def test_capture_gets_throwaway_memos(self):
        memos = ({"ratio": "prefill"}, {"ratio": "decode"}, {"ratio": "common"})
        kwargs = self._kwargs(True, memos)

        for key, memo in zip(
            ("prefill_ratio_to_sas_metadata", "decode_ratio_to_sas_metadata", "common_ratio_to_sas_metadata"),
            memos,
        ):
            self.assertEqual(kwargs[key], {})
            self.assertIsNot(kwargs[key], memo)
