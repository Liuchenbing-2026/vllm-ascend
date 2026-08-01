# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, _build_kv_cache
from vllm_ascend.utils import AscendDeviceType


class TestBuildDSAKVCache(TestBase):
    @staticmethod
    def _model(indexer_cache):
        caches = [torch.tensor([value]) for value in range(6)]
        model = SimpleNamespace(
            compress_ratio=4,
            dsa_attn=SimpleNamespace(kv_cache=[caches[0]]),
            swa_cache_layer=SimpleNamespace(kv_cache=[caches[1]]),
            compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=[caches[2]])),
            indexer=SimpleNamespace(
                compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=[caches[3]])),
                k_cache=SimpleNamespace(kv_cache=indexer_cache(caches[4], caches[5])),
            ),
        )
        return model, caches

    def test_accepts_v1_nested_and_v2_direct_indexer_views(self):
        layouts = {
            "v1": lambda key, scale: [[key, scale]],
            "v2": lambda key, scale: [key, scale],
        }

        for layout_name, layout in layouts.items():
            with self.subTest(layout=layout_name):
                model, expected = self._model(layout)
                with patch("vllm_ascend.ops.dsa.get_ascend_device_type", return_value=AscendDeviceType.A2):
                    caches = _build_kv_cache(model, SimpleNamespace(virtual_engine=None))

                self.assertEqual(len(caches), 6)
                for actual, expected_cache in zip(caches, expected):
                    self.assertIs(actual, expected_cache)

    def test_forward_defaults_flash_comm_to_disabled_when_context_omits_flag(self):
        hidden_states = torch.randn(2, 4)
        layer = SimpleNamespace(prefix="model.layers.0.self_attn")

        with (
            patch("vllm_ascend.ops.dsa.get_forward_context", return_value=SimpleNamespace()),
            patch("vllm_ascend.ops.dsa.torch.ops.vllm.dsa_forward") as mock_dsa_forward,
        ):
            output = AscendDeepseekSparseAttention.forward(layer, torch.arange(2), hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertFalse(mock_dsa_forward.call_args.args[1])
