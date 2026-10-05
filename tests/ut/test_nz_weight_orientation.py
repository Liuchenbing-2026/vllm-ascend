# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend import utils
from vllm_ascend.ascend_config import AscendConfig


class TestNzWeightOrientation(TestBase):
    def pack(self, weight, prefix, patterns, enabled=True):
        config = SimpleNamespace(weight_nz_transpose_modules=patterns)
        with (
            patch.object(utils, "_should_trans_nz", return_value=enabled),
            patch.object(utils, "get_ascend_config", return_value=config),
            patch.object(utils.torch_npu, "npu_format_cast", side_effect=lambda value, *a, **k: value.clone()) as cast,
        ):
            packed = utils.maybe_trans_nz(weight, prefix=prefix)
        return packed, cast

    def test_selected_module_preserves_values_and_shape(self):
        for dtype in (torch.float16, torch.bfloat16):
            weight = torch.arange(48).reshape(8, 6).to(dtype)
            packed, cast = self.pack(weight, "model.layers.0.mlp.down_proj", ["*.mlp.down_proj"])
            self.assertEqual(cast.call_args.args[0].shape, (6, 8))
            self.assertTrue(torch.equal(packed, weight))
            self.assertEqual(packed.shape, weight.shape)
            self.assertTrue(packed.t().is_contiguous())

    def test_disabled_by_default(self):
        weight = torch.randn(8, 6).half()
        packed, cast = self.pack(weight, "model.mlp.down_proj", [])
        self.assertEqual(cast.call_args.args[0].shape, weight.shape)
        self.assertTrue(torch.equal(packed, weight))

    def test_nonmatching_module_keeps_orientation(self):
        weight = torch.randn(8, 6).half()
        _, cast = self.pack(weight, "model.self_attn.qkv_proj", ["*.mlp.*"])
        self.assertEqual(cast.call_args.args[0].shape, weight.shape)

    def test_omitted_prefix_keeps_orientation(self):
        weight = torch.randn(8, 6).half()
        _, cast = self.pack(weight, None, ["*"])
        self.assertEqual(cast.call_args.args[0].shape, weight.shape)

    def test_nonfloating_weights_keep_orientation(self):
        weight = torch.ones(8, 6, dtype=torch.int8)
        _, cast = self.pack(weight, "model.mlp.down_proj", ["*"])
        self.assertEqual(cast.call_args.args[0].shape, weight.shape)

    def test_batched_weights_keep_orientation(self):
        weight = torch.ones(2, 8, 6, dtype=torch.bfloat16)
        _, cast = self.pack(weight, "model.mlp.down_proj", ["*"])
        self.assertEqual(cast.call_args.args[0].shape, weight.shape)

    def test_no_nz_policy_returns_original(self):
        weight = torch.randn(8, 6).half()
        packed, cast = self.pack(weight, "model.mlp.down_proj", ["*"], enabled=False)
        self.assertIs(packed, weight)
        cast.assert_not_called()

    def test_copy_reload_keeps_logical_weight_contract(self):
        weight = torch.randn(8, 6).half()
        packed, _ = self.pack(weight, "model.mlp.down_proj", ["*"])
        replacement = torch.randn_like(weight)
        packed.copy_(replacement)
        self.assertTrue(torch.equal(packed, replacement))
        self.assertTrue(packed.t().is_contiguous())

    def test_config_list_defaults_are_independent(self):
        a = AscendConfig(sparse_kv_offload_config=None)
        b = AscendConfig(sparse_kv_offload_config=None)
        a.weight_nz_transpose_modules.append("*.mlp.*")
        self.assertEqual(b.weight_nz_transpose_modules, [])
