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
"""TurboQuant KV-cache quantization for vllm-ascend (NPU).

Adapts vllm-project/vllm's TurboQuant (PR #38479, ICLR 2026) to Ascend NPU.

Two targets supported on DSA/GLM5 models:
  - Plan A (target='latent'):  MLA compressed_kv [kv_lora_rank=512]
  - Plan B (target='indexer'): per-layer indexer K cache [n_head=64, head_dim=128]

Plans are orthogonal and can stack (VLLM_ASCEND_TURBOQUANT_TARGET=latent,indexer).

Status:
  - config / centroids: full port from upstream (production)
  - reference quantizer (Plan A): torch fallback, correctness reference (production)
  - reference quantizer (Plan B): per-head wrapper over the Plan A primitives
  - NPU kernel: stub (PoC). Real AscendC / triton-ascend kernel TODO.
  - SFA integration: hook + feature flag (PoC).
"""

from vllm_ascend.quantization.turboquant.config import (
    TQ_PRESETS,
    AscendTurboQuantConfig,
)
from vllm_ascend.quantization.turboquant.indexer import (
    DSA_INDEXER_HEAD_DIM,
    dequantize_indexer_k,
    make_indexer_config,
    quantize_indexer_k,
    topk_rank_correlation,
)

__all__ = [
    "TQ_PRESETS",
    "AscendTurboQuantConfig",
    "DSA_INDEXER_HEAD_DIM",
    "make_indexer_config",
    "quantize_indexer_k",
    "dequantize_indexer_k",
    "topk_rank_correlation",
]
