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
"""Plan B (target='indexer') reference correctness tests.

Plan B quantizes the DSA sparse-indexer K cache. Critical metric: the
top-k selection should agree with the full-precision reference -- this is
what actually drives end-to-end attention accuracy on DSA models.
"""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.quantization.turboquant import (
    AscendTurboQuantConfig,
    DSA_INDEXER_HEAD_DIM,
    dequantize_indexer_k,
    make_indexer_config,
    quantize_indexer_k,
    topk_rank_correlation,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset", ["turboquant_k8v4", "turboquant_4bit_nc", "turboquant_k3v4_nc"]
)
def test_indexer_config_preset(preset):
    cfg = make_indexer_config(preset)
    assert cfg.target == "indexer"
    assert cfg.head_dim == DSA_INDEXER_HEAD_DIM  # 128


def test_indexer_config_custom_head_dim():
    cfg = make_indexer_config("turboquant_4bit_nc", indexer_head_dim=64)
    assert cfg.head_dim == 64
    assert cfg.target == "indexer"


def test_indexer_target_validated_by_from_cache_dtype():
    """from_cache_dtype must accept target='indexer'."""
    cfg = AscendTurboQuantConfig.from_cache_dtype(
        "turboquant_4bit_nc", head_dim=128, target="indexer"
    )
    assert cfg.target == "indexer"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_indexer_quantize_shapes():
    cfg = make_indexer_config("turboquant_4bit_nc")
    B, H, D = 8, 64, 128
    k_li = torch.randn(B, H, D)
    q_idx, vec_norm = quantize_indexer_k(k_li, cfg)
    assert q_idx.shape == (B, H, D)
    assert vec_norm.shape == (B, H)
    assert q_idx.min().item() >= 0
    assert q_idx.max().item() < cfg.n_centroids


@pytest.mark.parametrize(
    "preset,max_rel_err",
    [
        ("turboquant_4bit_nc", 0.55),
        ("turboquant_k3v4_nc", 0.70),
    ],
)
def test_indexer_roundtrip_bounded_error(preset, max_rel_err):
    """Per-head L2 reconstruction error stays within preset budget.

    Indexer head_dim is small (128 < typical 512), so the per-coord
    variance after rotation is larger and the bound is looser than for
    Plan A. The metric that actually matters is top-k rank agreement
    (see test_indexer_topk_rank_correlation).
    """
    torch.manual_seed(0)
    cfg = make_indexer_config(preset)
    B, H, D = 4, 64, 128
    k = torch.randn(B, H, D)

    q_idx, vec_norm = quantize_indexer_k(k, cfg)
    k_hat = dequantize_indexer_k(q_idx, vec_norm, cfg, out_dtype=torch.float32)
    assert k_hat.shape == (B, H, D)

    err = (k_hat - k).reshape(B * H, D).norm(dim=-1) / k.reshape(B * H, D).norm(
        dim=-1
    ).clamp_min(1e-8)
    assert err.mean().item() < max_rel_err, (
        f"{preset}: per-head mean rel err {err.mean().item():.3f} > {max_rel_err}"
    )


# ---------------------------------------------------------------------------
# Top-k rank correlation (the metric that actually matters)
# ---------------------------------------------------------------------------


def test_indexer_topk_rank_correlation_4bit():
    """At 4-bit, top-k(2048) selection should overlap >= 80% with full-prec."""
    torch.manual_seed(0)
    cfg = make_indexer_config("turboquant_4bit_nc")
    B, H, D = 1, 64, 128
    # Simulate K cache over 4096 past tokens (one head, batch 1 for clarity).
    n_past = 4096
    k_full = torch.randn(B, H, D)
    # Build candidate set: 4096 keys per head, q dot product picks top-k.
    candidates = torch.randn(n_past, H, D)
    q = torch.randn(B, H, D)

    # Reference score:  q · candidates^T  (per head)
    score_full = torch.einsum("bhd,nhd->bnh", q, candidates)  # [B, n_past, H]

    # Quantize each candidate row and dequantize.
    q_idx, vec_norm = quantize_indexer_k(candidates, cfg)
    cand_quant = dequantize_indexer_k(
        q_idx, vec_norm, cfg, out_dtype=torch.float32
    )
    score_quant = torch.einsum("bhd,nhd->bnh", q, cand_quant)

    # Average top-k Jaccard across heads.
    topk = 2048
    per_head_jaccard = []
    for h in range(H):
        sf = score_full[0, :, h]
        sq = score_quant[0, :, h]
        _, idx_f = sf.topk(topk)
        _, idx_q = sq.topk(topk)
        per_head_jaccard.append(
            len(set(idx_f.tolist()) & set(idx_q.tolist())) / topk
        )
    avg = sum(per_head_jaccard) / len(per_head_jaccard)
    assert (
        avg >= 0.80
    ), f"4-bit indexer top-k overlap {avg:.3f} < 0.80 -- expect >= 0.80 for {topk}/{n_past}"


def test_indexer_topk_rank_correlation_3bit():
    """At 3-bit, top-k(2048) overlap should still be >= 65%."""
    torch.manual_seed(0)
    cfg = make_indexer_config("turboquant_k3v4_nc")
    B, H, D = 1, 64, 128
    n_past = 4096
    candidates = torch.randn(n_past, H, D)
    q = torch.randn(B, H, D)
    score_full = torch.einsum("bhd,nhd->bnh", q, candidates)

    q_idx, vec_norm = quantize_indexer_k(candidates, cfg)
    cand_quant = dequantize_indexer_k(
        q_idx, vec_norm, cfg, out_dtype=torch.float32
    )
    score_quant = torch.einsum("bhd,nhd->bnh", q, cand_quant)

    topk = 2048
    per_head_jaccard = []
    for h in range(H):
        sf = score_full[0, :, h]
        sq = score_quant[0, :, h]
        _, idx_f = sf.topk(topk)
        _, idx_q = sq.topk(topk)
        per_head_jaccard.append(
            len(set(idx_f.tolist()) & set(idx_q.tolist())) / topk
        )
    avg = sum(per_head_jaccard) / len(per_head_jaccard)
    assert (
        avg >= 0.65
    ), f"3-bit indexer top-k overlap {avg:.3f} < 0.65"
