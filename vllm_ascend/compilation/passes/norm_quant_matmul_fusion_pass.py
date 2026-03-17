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

from typing import Tuple

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.utils import enable_custom_op


# ---------------------------------------------------------------------------
# Fused custom op (no bias variant): npu_add_rms_norm_quant_matmul
# ---------------------------------------------------------------------------

def _npu_add_rms_norm_quant_matmul_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    matmul_weight: torch.Tensor,
    deq_scale: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback: sequential npu_add_rms_norm_quant + npu_quant_matmul."""
    norm_out = torch.ops.npu.npu_add_rms_norm_quant(
        x, residual, norm_weight, quant_scale, quant_offset,
        epsilon=epsilon)
    int8_x = norm_out[0]
    new_residual = norm_out[2]
    mm_out = torch.ops.npu.npu_quant_matmul(
        int8_x, matmul_weight, deq_scale, output_dtype=x.dtype)
    return mm_out, new_residual


def _npu_add_rms_norm_quant_matmul_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    matmul_weight: torch.Tensor,
    deq_scale: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape inference for the fused op (no bias)."""
    norm_out = torch.ops.npu.npu_add_rms_norm_quant(
        x, residual, norm_weight, quant_scale, quant_offset,
        epsilon=epsilon)
    int8_x = norm_out[0]
    new_residual = norm_out[2]
    mm_out = torch.ops.npu.npu_quant_matmul(
        int8_x, matmul_weight, deq_scale, output_dtype=x.dtype)
    return mm_out, new_residual


direct_register_custom_op(
    op_name="npu_add_rms_norm_quant_matmul",
    op_func=_npu_add_rms_norm_quant_matmul_impl,
    fake_impl=_npu_add_rms_norm_quant_matmul_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# Fused custom op (with bias variant): npu_add_rms_norm_quant_matmul_bias
# ---------------------------------------------------------------------------

def _npu_add_rms_norm_quant_matmul_bias_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    matmul_weight: torch.Tensor,
    deq_scale: torch.Tensor,
    epsilon: float,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback: sequential npu_add_rms_norm_quant(beta) + npu_quant_matmul."""
    norm_out = torch.ops.npu.npu_add_rms_norm_quant(
        x, residual, norm_weight, quant_scale, quant_offset,
        epsilon=epsilon, beta=beta)
    int8_x = norm_out[0]
    new_residual = norm_out[2]
    mm_out = torch.ops.npu.npu_quant_matmul(
        int8_x, matmul_weight, deq_scale, output_dtype=x.dtype)
    return mm_out, new_residual


def _npu_add_rms_norm_quant_matmul_bias_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    matmul_weight: torch.Tensor,
    deq_scale: torch.Tensor,
    epsilon: float,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape inference for the fused op (with bias)."""
    norm_out = torch.ops.npu.npu_add_rms_norm_quant(
        x, residual, norm_weight, quant_scale, quant_offset,
        epsilon=epsilon, beta=beta)
    int8_x = norm_out[0]
    new_residual = norm_out[2]
    mm_out = torch.ops.npu.npu_quant_matmul(
        int8_x, matmul_weight, deq_scale, output_dtype=x.dtype)
    return mm_out, new_residual


direct_register_custom_op(
    op_name="npu_add_rms_norm_quant_matmul_bias",
    op_func=_npu_add_rms_norm_quant_matmul_bias_impl,
    fake_impl=_npu_add_rms_norm_quant_matmul_bias_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# Pattern 1: AddRMSNormQuant + QuantMatmul  (no norm bias)
# ---------------------------------------------------------------------------

class AddRMSNormQuantMatmulPattern:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        quant_scale = torch.ones(4, device="npu", dtype=self.dtype)
        quant_offset = torch.zeros(4, device="npu", dtype=self.dtype)
        matmul_weight = torch.ones(8, 4, device="npu", dtype=torch.int8)
        deq_scale = torch.ones(8, device="npu", dtype=torch.float32)
        return [rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset, matmul_weight, deq_scale]

    def register(self, pm_pass: PatternMatcherPass):
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

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


# ---------------------------------------------------------------------------
# Pattern 2: AddRMSNormQuant + QuantMatmul  (with norm bias)
# ---------------------------------------------------------------------------

class AddRMSNormQuantMatmulPatternWithBias:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        quant_scale = torch.ones(4, device="npu", dtype=self.dtype)
        quant_offset = torch.zeros(4, device="npu", dtype=self.dtype)
        matmul_weight = torch.ones(8, 4, device="npu", dtype=torch.int8)
        deq_scale = torch.ones(8, device="npu", dtype=torch.float32)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight,
                quant_scale, quant_offset, matmul_weight, deq_scale,
                rmsnorm_bias]

    def register(self, pm_pass: PatternMatcherPass):
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

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------

class AddRMSNormQuantMatmulFusionPass(VllmInductorPass):
    """
    Fuses npu_add_rms_norm_quant + npu_quant_matmul into a single op,
    eliminating the intermediate int8 tensor HBM round-trip.

    This pass runs *after* AddRMSNormQuantFusionPass which produces the
    npu_add_rms_norm_quant nodes that we match here.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="norm_quant_matmul_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug(
                "norm_quant_matmul fusion not enabled: unsupported dtype %s",
                dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            AddRMSNormQuantMatmulPattern(
                vllm_config, eps=eps).register(self.pattern_match_passes)
            if enable_custom_op():
                AddRMSNormQuantMatmulPatternWithBias(
                    vllm_config, eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s norm_quant_matmul patterns",
                     self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
