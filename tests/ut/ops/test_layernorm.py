from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw
from vllm_ascend.ops import layernorm as layernorm_ops

enable_custom_op()


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


def mock_add_rms_norm_bias(x, residual, weight, bias, eps):
    if bias is None:
        return 2 * x, None, 2 * residual
    else:
        return 2 * x + bias, None, 2 * residual


@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch(
    "torch.ops._C_ascend.npu_add_rms_norm_bias",
    side_effect=RuntimeError("aclnnAddRmsNormBias not in libopapi.so"),
)
def test_add_rms_norm_bias_falls_back_for_old_cann(mock_custom, mock_fallback, monkeypatch, dummy_tensor):
    monkeypatch.setattr(layernorm_ops, "_ADD_RMS_NORM_BIAS_SUPPORTED", True)
    monkeypatch.setattr(layernorm_ops, "enable_custom_op", lambda: True)
    bias = torch.ones(8, dtype=dummy_tensor.dtype)

    out, residual = layernorm_ops._add_rms_norm_bias_or_fallback(
        dummy_tensor, dummy_tensor, torch.ones_like(bias), bias, 1e-5
    )

    assert torch.allclose(out, 2 * dummy_tensor + bias)
    assert torch.allclose(residual, 2 * dummy_tensor)
    assert layernorm_ops._ADD_RMS_NORM_BIAS_SUPPORTED is False
    mock_custom.assert_called_once()
    mock_fallback.assert_called_once()


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]

    with set_current_vllm_config(mock_config):
        yield mock_config


@pytest.mark.skip("Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=mock_add_rms_norm_bias)
def test_RMSNorm_forward(
    mock_add_rms_norm_bias, mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config
):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rms_norm_bias.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1

        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


@pytest.mark.skipif(not is_310p_hw(), reason="310P device unittest case.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float16)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
def test_RMSNorm_forward_310p(mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1
        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
