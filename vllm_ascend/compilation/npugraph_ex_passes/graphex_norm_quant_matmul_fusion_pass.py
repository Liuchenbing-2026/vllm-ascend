#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
#

import torch
import torchair
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.npugraph_ex_passes.utils.npugraph_ex_utils_check import extra_stream_scope_check


class GraphEXAddRMSNormQuantMatmulPattern:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        quant_scale = torch.ones(4, device="npu")
        quant_offset = torch.zeros(4, device="npu")
        matmul_weight = torch.ones(8, 4, device="npu", dtype=torch.int8)
        deq_scale = torch.ones(8, device="npu", dtype=torch.float32)
        return [rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset, matmul_weight, deq_scale]

    def register(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            quant_scale: torch.Tensor,
            quant_offset: torch.Tensor,
            matmul_weight: torch.Tensor,
            deq_scale: torch.Tensor,
        ):
            norm_out = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset, epsilon=self.eps)
            int8_x = norm_out[0]
            new_residual = norm_out[2]
            mm_out = torch.ops.npu.npu_quant_matmul(
                int8_x, matmul_weight, deq_scale,
                output_dtype=self.dtype)
            return mm_out, new_residual

        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            quant_scale: torch.Tensor,
            quant_offset: torch.Tensor,
            matmul_weight: torch.Tensor,
            deq_scale: torch.Tensor,
        ):
            result = torch.ops.vllm.npu_add_rms_norm_quant_matmul(
                rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset,
                matmul_weight, deq_scale, self.eps)
            return result[0], result[1]

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXAddRMSNormQuantMatmulPatternWithBias:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        quant_scale = torch.ones(4, device="npu")
        quant_offset = torch.zeros(4, device="npu")
        matmul_weight = torch.ones(8, 4, device="npu", dtype=torch.int8)
        deq_scale = torch.ones(8, device="npu", dtype=torch.float32)
        rmsnorm_bias = torch.randn(4, device="npu")
        return [rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset, matmul_weight, deq_scale,
                rmsnorm_bias]

    def register(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            quant_scale: torch.Tensor,
            quant_offset: torch.Tensor,
            matmul_weight: torch.Tensor,
            deq_scale: torch.Tensor,
            bias: torch.Tensor,
        ):
            norm_out = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset,
                epsilon=self.eps, beta=bias)
            int8_x = norm_out[0]
            new_residual = norm_out[2]
            mm_out = torch.ops.npu.npu_quant_matmul(
                int8_x, matmul_weight, deq_scale,
                output_dtype=self.dtype)
            return mm_out, new_residual

        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            quant_scale: torch.Tensor,
            quant_offset: torch.Tensor,
            matmul_weight: torch.Tensor,
            deq_scale: torch.Tensor,
            bias: torch.Tensor,
        ):
            result = torch.ops.vllm.npu_add_rms_norm_quant_matmul_bias(
                rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset,
                matmul_weight, deq_scale, self.eps, bias)
            return result[0], result[1]

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXAddRMSNormQuantMatmulFusionPass:
    """
    Fuses npu_add_rms_norm_quant + npu_quant_matmul via torchair
    pattern replacement.
    """

    def __init__(self, vllm_config: VllmConfig):
        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug(
                "norm_quant_matmul fusion not enabled: unsupported dtype %s",
                dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            GraphEXAddRMSNormQuantMatmulPattern(
                vllm_config, eps=eps).register()
            GraphEXAddRMSNormQuantMatmulPatternWithBias(
                vllm_config, eps=eps).register()

    def __call__(self, graph: torch.fx.Graph):
        pass

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
