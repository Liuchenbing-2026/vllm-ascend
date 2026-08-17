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

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm_ascend.utils import FP8_METHOD

from .registry import register_scheme
from .w8a8fp8_dynamic import (
    AscendW8A8FP8DynamicFusedMoEMethod,
    AscendW8A8FP8DynamicLinearMethod,
)


@register_scheme(FP8_METHOD, "tensor_linear")
class AscendFp8PerTensorLinearMethod(AscendW8A8FP8DynamicLinearMethod):
    """Load serialized per-tensor FP8 weights for Ascend linear kernels.

    Checkpoint scales remain per logical matrix while loading. They are
    expanded to per-output-channel scales afterwards, which preserves the
    checkpoint dequantization factors and matches ``npu_quant_matmul``.
    Activations are dynamically quantized by the parent Ascend FP8 scheme.
    """

    def __init__(self, activation_scheme: str) -> None:
        super().__init__()
        self.activation_scheme = activation_scheme

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        output_partition_sizes = kwargs.get("output_partition_sizes")
        if not output_partition_sizes:
            raise ValueError("output_partition_sizes is required for per-tensor FP8 scales")

        num_logical_weights = len(output_partition_sizes)
        params = {"weight_scale": torch.ones(num_logical_weights, dtype=torch.float32)}
        if self.activation_scheme == "static":
            params["input_scale"] = torch.ones(num_logical_weights, dtype=torch.float32)
        return params

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logical_widths = tuple(layer.logical_widths)
        weight_scales = layer.weight_scale.data.flatten().to(torch.float32)
        if weight_scales.numel() != len(logical_widths):
            raise ValueError(
                "Expected one FP8 weight scale per logical matrix, but got "
                f"{weight_scales.numel()} scales for {len(logical_widths)} matrices."
            )

        repeats = torch.tensor(logical_widths, device=weight_scales.device)
        expanded_scales = torch.repeat_interleave(weight_scales, repeats)
        output_size = layer.weight.shape[0]
        if expanded_scales.numel() != output_size:
            raise ValueError(
                f"Expanded FP8 scales have {expanded_scales.numel()} values, "
                f"but the linear weight has output size {output_size}."
            )

        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = expanded_scales.contiguous()
        layer.weight_scale_fp32 = layer.weight_scale.data


@register_scheme(FP8_METHOD, "tensor_moe")
class AscendFp8PerTensorFusedMoEMethod(AscendW8A8FP8DynamicFusedMoEMethod):
    """Load serialized per-expert, per-tensor FP8 MoE weights on Ascend."""

    weight_scale_supported = FusedMoeWeightScaleSupported.TENSOR.value

    def __init__(self, activation_scheme: str) -> None:
        super().__init__()
        self.activation_scheme = activation_scheme

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params = {
            "w13_weight_scale": torch.ones(num_experts, 2, dtype=torch.float32),
            "w2_weight_scale": torch.ones(num_experts, dtype=torch.float32),
        }
        if self.activation_scheme == "static":
            params["w13_input_scale"] = torch.ones(num_experts, dtype=torch.float32)
            params["w2_input_scale"] = torch.ones(num_experts, dtype=torch.float32)
        return params

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts, fused_intermediate_size, _ = layer.w13_weight.shape
        num_w13_shards = 2
        if fused_intermediate_size % num_w13_shards != 0:
            raise ValueError(f"The fused w13 size {fused_intermediate_size} is not divisible by 2.")
        intermediate_size = fused_intermediate_size // num_w13_shards

        w13_scales = layer.w13_weight_scale.data
        expected_w13_shape = (num_experts, num_w13_shards)
        if tuple(w13_scales.shape) != expected_w13_shape:
            raise ValueError(f"Expected w13 FP8 scales with shape {expected_w13_shape}, got {tuple(w13_scales.shape)}.")
        layer.w13_weight_scale.data = torch.repeat_interleave(
            w13_scales.to(torch.float32), intermediate_size, dim=1
        ).contiguous()

        w2_scales = layer.w2_weight_scale.data.flatten().to(torch.float32)
        if w2_scales.numel() != num_experts:
            raise ValueError(f"Expected {num_experts} w2 FP8 scales, got {w2_scales.numel()}.")
        hidden_size = layer.w2_weight.shape[1]
        layer.w2_weight_scale.data = w2_scales[:, None].expand(-1, hidden_size).contiguous()

        super().process_weights_after_loading(layer)
