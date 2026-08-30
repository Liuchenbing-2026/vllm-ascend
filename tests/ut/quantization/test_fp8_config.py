import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models import mistral3

from tests.ut.base import TestBase
from vllm_ascend.quantization.fp8_config import (
    AscendDeepseekV4FP8Config,
    AscendFp8Config,
)


class TestAscendDeepseekV4FP8Config(TestBase):
    def setUp(self):
        self.config = AscendDeepseekV4FP8Config()
        self.config.weight_block_size = (128, 128)

    def test_get_quant_method_for_linear_base(self):
        """LinearBase layer returns AscendLinearMethod with ds_linear scheme."""
        linear_layer = MagicMock(spec=LinearBase)
        mock_scheme_instance = MagicMock()
        mock_scheme_class = MagicMock(return_value=mock_scheme_instance)
        mock_ascend_linear = MagicMock()

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendLinearMethod",
                return_value=mock_ascend_linear,
            ),
        ):
            method = self.config.get_quant_method(linear_layer, "model.layers.0.mlp.gate_proj")

        mock_scheme_class.assert_called_once_with((128, 128))
        self.assertIs(method, mock_ascend_linear)

    def test_get_quant_method_for_moe_runner_fp4(self):
        """MoERunner layer with fp4 expert_dtype returns AscendFusedMoEMethod."""
        moe_config = MagicMock()
        moe_layer = MagicMock(spec=MoERunner)
        moe_layer.moe_config = moe_config

        mock_scheme_instance = MagicMock()
        mock_scheme_class = MagicMock(return_value=mock_scheme_instance)
        mock_ascend_moe = MagicMock()

        self.config._resolved_expert_dtype = "fp4"

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod",
                return_value=mock_ascend_moe,
            ),
        ):
            method = self.config.get_quant_method(moe_layer, "model.layers.0.mlp")

        mock_scheme_class.assert_called_once_with()
        self.assertIs(method, mock_ascend_moe)

    def test_get_quant_method_for_routed_experts_fp4(self):
        """RoutedExperts layer with fp4 expert_dtype returns AscendFusedMoEMethod."""
        moe_config = MagicMock()
        moe_layer = MagicMock(spec=RoutedExperts)
        moe_layer.moe_config = moe_config

        mock_scheme_instance = MagicMock()
        mock_scheme_class = MagicMock(return_value=mock_scheme_instance)
        mock_ascend_moe = MagicMock()

        self.config._resolved_expert_dtype = "fp4"

        with (
            patch(
                "vllm_ascend.quantization.fp8_config.get_scheme_class",
                return_value=mock_scheme_class,
            ),
            patch(
                "vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod",
                return_value=mock_ascend_moe,
            ),
        ):
            method = self.config.get_quant_method(moe_layer, "model.layers.0.mlp.experts")

        mock_scheme_class.assert_called_once_with()
        self.assertIs(method, mock_ascend_moe)

    def test_get_quant_method_for_moe_non_fp4_raises(self):
        """MoE layer with non-fp4 expert_dtype raises NotImplementedError."""
        moe_config = MagicMock()
        moe_layer = MagicMock(spec=MoERunner)
        moe_layer.moe_config = moe_config

        self.config._resolved_expert_dtype = "fp8"

        with self.assertRaises(NotImplementedError):
            self.config.get_quant_method(moe_layer, "model.layers.0.mlp")

    def test_get_quant_method_for_other_layer_returns_none(self):
        """Non-LinearBase and non-MoE layers return None."""
        other_layer = torch.nn.Linear(10, 10)

        method = self.config.get_quant_method(other_layer, "some.prefix")
        self.assertIsNone(method)


if __name__ == "__main__":
    unittest.main()


def test_mistral4_fp8_config_maps_ignored_layers() -> None:
    config = AscendFp8Config.from_config(
        {
            "activation_scheme": "static",
            "modules_to_not_convert": [
                "model.vision_tower",
                "model.multi_modal_projector",
                "lm_head",
            ],
            "quant_method": "fp8",
            "weight_block_size": None,
        }
    )

    mistral3_model = getattr(
        mistral3,
        "Mistral3For" "ConditionalGeneration",
    )
    config.apply_vllm_mapper(mistral3_model.hf_to_vllm_mapper)

    assert config.is_per_tensor_fp8
    assert not config.mistral4_dynamic_channelwise
    assert config.ignored_layers == [
        "vision_tower",
        "multi_modal_projector",
        "language_model.lm_head",
    ]


def test_block_fp8_keeps_non_mistral4_path() -> None:
    config = AscendFp8Config.from_config(
        {
            "activation_scheme": "dynamic",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    )

    assert not config.is_per_tensor_fp8
