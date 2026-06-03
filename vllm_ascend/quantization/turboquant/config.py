#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# Adapted from vllm-project/vllm `vllm/model_executor/layers/quantization/turboquant/config.py`.
"""TurboQuant configuration for Ascend NPU.

Differences from upstream vllm TurboQuant:
  - target option: "full" (legacy upstream-compatible) | "latent" (Plan A; DSA/MLA path)
  - slot alignment: 32B for Ascend UB DataCopy (vs 16B on CUDA)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

# Named TurboQuant presets (kept compatible with upstream cache-dtype strings).
TQ_PRESETS: dict[str, dict] = {
    "turboquant_k8v4": {
        "key_quant_bits": 8,
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
}

# Ascend UB DataCopy prefers 32B alignment for max throughput.
_ASCEND_SLOT_ALIGN = 32


def _round_up(x: int, multiple: int) -> int:
    """Round x up to next multiple."""
    return ((x + multiple - 1) // multiple) * multiple


@dataclass
class AscendTurboQuantConfig:
    """TurboQuant config on Ascend NPU.

    Plan A target: quantize MLA-compressed latent KV (kv_lora_rank-dim) for
    DSA models (GLM5 / DeepSeek-V3.2 / DeepSeek-V4). Non-MLA models can use
    the upstream-equivalent "full" target.

    Named presets (use via --kv-cache-dtype):
        turboquant_k8v4:    FP8 keys + 4-bit values, 2.6x, +1.17% PPL
        turboquant_4bit_nc: 4-bit MSE keys + 4-bit values + NC, 3.8x, +2.71%
        turboquant_k3v4_nc: 3-bit MSE keys + 4-bit values + NC, ~3.5x, +10.63%
        turboquant_3bit_nc: 3-bit MSE keys + 3-bit values + NC, 4.9x, +20.59%

    Args:
        head_dim: Quantization vector dimension. For Plan A on GLM5 this is
            kv_lora_rank (typically 512). For "full" target this is the per-head
            head_dim (typically 64/96/128).
        key_quant_bits: 8 = FP8 keys (no rotation/MSE); 3-4 = Lloyd-Max MSE keys.
        value_quant_bits: 3-4 = uniform quantized values.
        norm_correction: Re-normalize centroid vectors to unit norm before
            inverse rotation. Recovers ~0.8% PPL at 4-bit.
        target: "latent" (Plan A; quantize MLA compressed_kv) or
            "full" (legacy; quantize per-head K/V).
    """

    head_dim: int = 128
    key_quant_bits: int = 3
    value_quant_bits: int = 4
    norm_correction: bool = False
    target: str = "latent"  # "latent" | "full"

    @property
    def key_fp8(self) -> bool:
        return self.key_quant_bits == 8

    @property
    def mse_bits(self) -> int:
        if self.key_fp8:
            return self.value_quant_bits
        return self.key_quant_bits

    @property
    def key_mse_bits(self) -> int:
        return 0 if self.key_fp8 else self.key_quant_bits

    @property
    def n_centroids(self) -> int:
        return 2**self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes per key vector.

        FP8: head_dim bytes.
        TQ: ceil(head_dim*key_mse_bits/8) + 2 (vec_norm fp16).
        """
        if self.key_fp8:
            return self.head_dim
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        return mse_bytes + 2

    @property
    def value_packed_size(self) -> int:
        """Packed bytes per value vector: ceil(d*bits/8) + 4 (scale + zero fp16)."""
        return math.ceil(self.head_dim * self.value_quant_bits / 8) + 4

    @property
    def slot_size(self) -> int:
        """Packed bytes per token per layer."""
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        """Slot size aligned to Ascend UB DataCopy boundary (32B)."""
        return _round_up(self.slot_size, _ASCEND_SLOT_ALIGN)

    def compression_ratio(self, original_dtype_bytes: int = 2) -> float:
        """Compression vs baseline that stores K and V separately at original dtype.

        Baseline = head_dim * dtype_bytes * 2 (K + V).
        Compressed = slot_size_aligned (packed key + packed value + alignment pad).

        For Plan A (target='latent'), this metric is also the right one — both K
        and V are projected from the same compressed_kv via separate W_uk / W_uv
        matrices at attention time, so the storage equivalent is still 2×.
        """
        original = self.head_dim * original_dtype_bytes * 2
        return original / max(self.slot_size_aligned, 1)

    @staticmethod
    def from_cache_dtype(
        cache_dtype: str,
        head_dim: int,
        target: str = "latent",
    ) -> "AscendTurboQuantConfig":
        if cache_dtype not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS.keys())
            raise ValueError(
                f"Unknown TurboQuant cache dtype: {cache_dtype!r}. "
                f"Valid presets: {valid}"
            )
        if target not in ("latent", "full"):
            raise ValueError(f"target must be 'latent' or 'full', got {target!r}")
        preset = TQ_PRESETS[cache_dtype]
        return AscendTurboQuantConfig(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            norm_correction=preset["norm_correction"],
            target=target,
        )

    @staticmethod
    def boundary_skip_layers(
        model_config: "ModelConfig",
        n: int = 2,
    ) -> list[str]:
        """Layer indices to skip TQ (first N + last N).

        Empirically required for aggressive presets (k3v4_nc / 3bit_nc)
        to avoid GSM8K accuracy collapse.
        """
        num_layers = model_config.hf_text_config.num_hidden_layers
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        return [str(i) for i in sorted(set(first + last))]
