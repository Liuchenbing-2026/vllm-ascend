# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


def _input(heads=16, dim=128, dtype=torch.bfloat16, device="npu"):
    return SimpleNamespace(shape=(1, 3500, heads, dim), dtype=dtype, device=SimpleNamespace(type=device))


@pytest.mark.parametrize("available", [False, True])
def test_supported_prefill_respects_operator_availability(available):
    profile = SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.STANDARD)
    with (
        patch("vllm_ascend.ops.gdn.get_current_hardware_profile", return_value=profile),
        patch.object(AscendGatedDeltaNetAttention, "_probe_fused_chunk", return_value=available) as probe,
    ):
        assert (
            AscendGatedDeltaNetAttention._prefer_fused_prefill(
                _input(), _input(), _input(heads=32), SimpleNamespace(keep_meta=None)
            )
            is available
        )
        probe.assert_called_once_with()


@pytest.mark.parametrize(
    "query,key,value,keep,family",
    [
        (_input(device="cpu"), _input(), _input(heads=32), None, DeviceAdaptorFamily.STANDARD),
        (_input(dtype=torch.float16), _input(), _input(heads=32), None, DeviceAdaptorFamily.STANDARD),
        (_input(), _input(dtype=torch.float16), _input(heads=32), None, DeviceAdaptorFamily.STANDARD),
        (_input(), _input(), _input(heads=32, dtype=torch.float16), None, DeviceAdaptorFamily.STANDARD),
        (_input(), _input(heads=8), _input(heads=32), None, DeviceAdaptorFamily.STANDARD),
        (_input(dim=64), _input(dim=64), _input(heads=32), None, DeviceAdaptorFamily.STANDARD),
        (_input(), _input(), _input(heads=32, dim=64), None, DeviceAdaptorFamily.STANDARD),
        (_input(heads=0), _input(heads=0), _input(heads=32), None, DeviceAdaptorFamily.STANDARD),
        (_input(heads=65), _input(heads=65), _input(heads=65), None, DeviceAdaptorFamily.STANDARD),
        (_input(), _input(), _input(heads=80), None, DeviceAdaptorFamily.STANDARD),
        (_input(), _input(), _input(heads=17), None, DeviceAdaptorFamily.STANDARD),
        (_input(), _input(), _input(heads=32), object(), DeviceAdaptorFamily.STANDARD),
        (_input(), _input(), _input(heads=32), None, DeviceAdaptorFamily.FP8_OPTIMIZED),
        (_input(), _input(), _input(heads=32), None, DeviceAdaptorFamily.COMPATIBILITY),
    ],
)
def test_unsupported_prefill_keeps_original_dispatch(query, key, value, keep, family):
    profile = SimpleNamespace(device_adaptor_family=family)
    with (
        patch("vllm_ascend.ops.gdn.get_current_hardware_profile", return_value=profile),
        patch.object(AscendGatedDeltaNetAttention, "_probe_fused_chunk") as probe,
    ):
        assert not AscendGatedDeltaNetAttention._prefer_fused_prefill(
            query, key, value, SimpleNamespace(keep_meta=keep)
        )
        probe.assert_not_called()
