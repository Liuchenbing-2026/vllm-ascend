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
"""Plan B: TurboQuant on DSA sparse-indexer K cache.

Plan A (sibling: see quantizer.py + npu_ops.py) compresses the MLA latent
`compressed_kv [kv_lora_rank]`. Plan B is orthogonal: it compresses the
**indexer K cache** (the small [n_head=64, head_dim=128] tensor used by
`npu_lightning_indexer` to select top-k tokens).

Why Plan B (in addition to Plan A)
----------------------------------
- Indexer K cache grows linearly with seq length × n_layers × n_head.
  At 64 head × 128 dim × bf16 × num_layers it can be a significant
  fraction of total cache for long-context decode.
- Indexer K is only used to compute Q·K^T scores for top-k selection,
  then discarded. Small quantization noise on those scores is far less
  harmful than on the main attention path -- top-k typically has wide
  margin between selected and unselected scores.
- Can stack with Plan A (latent target) for compound savings.

Why a separate file
-------------------
- Different tensor shape and lifecycle:
    Plan A: [B, kv_lora_rank=512]    written at exec_kv site
    Plan B: [B, n_head=64, head_dim=128] written at indexer pre-process site
- Different decode path:
    Plan A: dequant → W_uk/W_uv → attention
    Plan B: dequant → Q·K^T → top-k selection
- Different precision tolerance and norm_correction tuning.

Status
------
- Config: target='indexer' supported (this file's `make_indexer_config`).
- Quantizer kernel: reuses quantizer.py (Hadamard + Lloyd-Max).
- NPU op: stub. Real AscendC / triton-ascend kernel TODO.
- SFA integration: TODO (hook at `indexer_select_pre_process` in sfa_v1.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_ascend.quantization.turboquant.config import AscendTurboQuantConfig
from vllm_ascend.quantization.turboquant.quantizer import (
    dequantize_key_mse,
    quantize_key_mse,
)

if TYPE_CHECKING:
    pass


# Indexer K head_dim across DSA models in vllm-ascend (verified against
# self.indexer.head_dim in `vllm_ascend/attention/sfa_v1.py:446`).
DSA_INDEXER_HEAD_DIM = 128


def make_indexer_config(
    cache_dtype: str,
    indexer_head_dim: int = DSA_INDEXER_HEAD_DIM,
) -> AscendTurboQuantConfig:
    """Create a Plan B TurboQuant config for the indexer K cache.

    Args:
        cache_dtype: TurboQuant preset name (e.g. "turboquant_4bit_nc").
        indexer_head_dim: K head_dim used by `npu_lightning_indexer`. Default
            128 matches DSA / GLM5 / DSV3.2 / DSV4.

    Returns:
        Config with target='indexer'.
    """
    return AscendTurboQuantConfig.from_cache_dtype(
        cache_dtype, head_dim=indexer_head_dim, target="indexer"
    )


def quantize_indexer_k(
    k_li: torch.Tensor,  # [B, n_head, head_dim] post-RoPE indexer K
    cfg: AscendTurboQuantConfig,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head quantize the indexer K tensor.

    Each head is quantized independently (its own Hadamard rotation +
    Lloyd-Max). This matches the structure of the indexer Q·K^T product
    which is also per-head.

    Returns:
        q_idx:    [B, n_head, head_dim] int64 centroid indices
        vec_norm: [B, n_head]            fp32 per-head norm
    """
    assert cfg.target == "indexer", "quantize_indexer_k requires target='indexer'"
    assert k_li.dim() == 3, f"k_li must be [B, n_head, head_dim]; got {k_li.shape}"
    B, H, D = k_li.shape
    assert D == cfg.head_dim, f"head_dim mismatch: tensor {D} vs cfg {cfg.head_dim}"

    # Reshape to [B*H, D] so the shared quantize_key_mse can process it.
    k_flat = k_li.reshape(B * H, D)
    q_idx_flat, vec_norm_flat = quantize_key_mse(k_flat, cfg, seed=seed)
    q_idx = q_idx_flat.reshape(B, H, D)
    vec_norm = vec_norm_flat.reshape(B, H)
    return q_idx, vec_norm


def dequantize_indexer_k(
    q_idx: torch.Tensor,  # [B, n_head, head_dim] int64
    vec_norm: torch.Tensor,  # [B, n_head] fp32
    cfg: AscendTurboQuantConfig,
    seed: int = 42,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of quantize_indexer_k. Returns reconstructed K [B, n_head, head_dim]."""
    assert cfg.target == "indexer"
    B, H, D = q_idx.shape
    q_flat = q_idx.reshape(B * H, D)
    vn_flat = vec_norm.reshape(B * H)
    k_hat_flat = dequantize_key_mse(q_flat, vn_flat, cfg, seed=seed, out_dtype=out_dtype)
    return k_hat_flat.reshape(B, H, D)


def topk_rank_correlation(
    q_li: torch.Tensor,  # [B, n_head, head_dim] fp32/bf16
    k_li_full: torch.Tensor,  # [B, n_head, head_dim] reference (full precision)
    k_li_quant: torch.Tensor,  # [B, n_head, head_dim] reconstructed from quant
    topk: int = 2048,
) -> float:
    """Spearman-ish rank agreement between full-prec and quantized indexer.

    For Plan B PoC: measure how much top-k selection drifts when we use
    quantized K instead of full-precision K. A value close to 1.0 means
    the selected token sets are nearly identical -- the metric that
    actually matters for end-to-end accuracy.

    Returns the per-batch-head Jaccard similarity averaged.
    """
    # scores: q · k^T per head, summed over head_dim
    score_full = (q_li * k_li_full).sum(dim=-1)  # [B, n_head]
    score_quant = (q_li * k_li_quant).sum(dim=-1)

    # Trivial 1-token case: rank correlation is degenerate; report 1.0.
    if score_full.dim() < 2 or score_full.shape[-1] <= 1:
        return 1.0

    k = min(topk, score_full.shape[-1])
    _, idx_full = score_full.topk(k, dim=-1)
    _, idx_quant = score_quant.topk(k, dim=-1)

    overlap = torch.tensor(0.0)
    total = 0
    for b in range(score_full.shape[0]):
        a = set(idx_full[b].cpu().tolist())
        bset = set(idx_quant[b].cpu().tolist())
        if len(a) > 0:
            overlap += len(a & bset) / len(a)
            total += 1
    return (overlap / max(total, 1)).item()
