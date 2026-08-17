from unittest.mock import MagicMock, Mock, patch

import torch
from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_mock_ascend_config,
    create_mock_vllm_config,
)
from vllm_ascend.quantization.methods.fp8_per_tensor import (
    AscendFp8PerTensorFusedMoEMethod,
    AscendFp8PerTensorLinearMethod,
)


class TestAscendFp8PerTensorLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendFp8PerTensorLinearMethod("static")

    def test_get_pertensor_param_for_fused_weights(self):
        params = self.method.get_pertensor_param(
            torch.bfloat16,
            output_partition_sizes=[64, 32, 32],
        )
        self.assertEqual(params["weight_scale"].shape, (3,))
        self.assertEqual(params["weight_scale"].dtype, torch.float32)
        self.assertEqual(params["input_scale"].shape, (3,))

    def test_process_expands_logical_scales(self):
        layer = torch.nn.Module()
        layer.logical_widths = [2, 3]
        layer.weight = torch.nn.Parameter(
            torch.zeros(5, 4, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(torch.tensor([0.25, 0.5]), requires_grad=False)

        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (4, 5))
        torch.testing.assert_close(
            layer.weight_scale,
            torch.tensor([0.25, 0.25, 0.5, 0.5, 0.5]),
        )
        torch.testing.assert_close(layer.weight_scale_fp32, layer.weight_scale)


class TestAscendFp8PerTensorFusedMoEMethod(TestBase):
    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    def setUp(self, mock_ascend, mock_mc2, mock_rank):
        with patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_current_vllm_config") as mock_vllm:
            mock_vllm.return_value = create_mock_vllm_config()
            mock_ascend.return_value = create_mock_ascend_config()
            mock_mc2.return_value = MagicMock(
                device_group=Mock(
                    _get_backend=Mock(return_value=Mock(get_hccl_comm_name=Mock(return_value="test_comm")))
                )
            )
            mock_rank.return_value = 0
            self.method = AscendFp8PerTensorFusedMoEMethod("static")

    def test_get_dynamic_quant_param_uses_tensor_scales(self):
        params = self.method.get_dynamic_quant_param(4, 8, 16, torch.bfloat16)
        self.assertEqual(params["w13_weight_scale"].shape, (4, 2))
        self.assertEqual(params["w2_weight_scale"].shape, (4,))
        self.assertEqual(params["w13_input_scale"].shape, (4,))
        self.assertEqual(params["w2_input_scale"].shape, (4,))

    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    def test_process_expands_expert_scales(self, mock_ascend):
        ascend_config = create_mock_ascend_config()
        ascend_config.enable_fused_mc2 = 0
        mock_ascend.return_value = ascend_config

        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 6, 4, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(2, 4, 3, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.tensor([[0.25, 0.5], [0.75, 1.0]]),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(torch.tensor([1.25, 1.5]), requires_grad=False)

        self.method.use_expert_weight_list = False
        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight.shape, (2, 4, 6))
        self.assertEqual(layer.w2_weight.shape, (2, 3, 4))
        torch.testing.assert_close(
            layer.w13_weight_scale,
            torch.tensor(
                [
                    [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 1.0, 1.0, 1.0],
                ]
            ),
        )
        torch.testing.assert_close(
            layer.w2_weight_scale,
            torch.tensor([[1.25, 1.25, 1.25, 1.25], [1.5, 1.5, 1.5, 1.5]]),
        )
