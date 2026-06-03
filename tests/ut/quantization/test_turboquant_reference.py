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
"""TurboQuant reference (torch) implementation correctness tests.

These tests run on CPU; no NPU hardware required. They validate the
algorithmic correctness of the reference quantizer. The NPU kernel must
match this reference within tight tolerance.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_ascend.quantization.turboquant import (
    AscendTurboQuantConfig,
    TQ_PRESETS,
)
from vllm_ascend.quantization.turboquant.centroids import (
    get_centroids,
    solve_lloyd_max,
)
from vllm_ascend.quantization.turboquant.quantizer import (
    get_hadamard_matrix,
    quantize_key_mse,
    quantize_value_uniform,
    reference_roundtrip_key,
    reference_roundtrip_value,
)


# ---------------------------------------------------------------------------
# Hadamard matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [64, 128, 256, 512])
def test_hadamard_is_orthogonal(d):
    """H @ H^T must be identity."""
    H = get_hadamard_matrix(d, device=torch.device("cpu"), dtype=torch.float64)
    eye = H @ H.T
    assert torch.allclose(eye, torch.eye(d, dtype=torch.float64), atol=1e-10)


def test_hadamard_non_power_of_two_raises():
    with pytest.raises(ValueError):
        get_hadamard_matrix(100, device=torch.device("cpu"), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Lloyd-Max centroids
# ---------------------------------------------------------------------------


def test_centroids_sorted_and_symmetric():
    centroids, boundaries = solve_lloyd_max(512, 4)
    # Sorted.
    diffs = centroids[1:] - centroids[:-1]
    assert (diffs > 0).all(), "Centroids should be strictly increasing"
    # Boundaries are midpoints between centroids.
    expected_b = (centroids[:-1] + centroids[1:]) / 2
    assert torch.allclose(boundaries, expected_b, atol=1e-6)
    # For symmetric distribution, centroids are anti-symmetric around 0.
    assert abs(centroids.sum().item()) < 1e-3


def test_get_centroids_caches():
    a = get_centroids(512, 4)
    b = get_centroids(512, 4)
    # lru_cache returns the same tensor.
    assert a.data_ptr() == b.data_ptr()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list(TQ_PRESETS.keys()))
def test_config_from_preset(preset):
    cfg = AscendTurboQuantConfig.from_cache_dtype(preset, head_dim=512, target="latent")
    assert cfg.head_dim == 512
    assert cfg.target == "latent"
    if "k8" in preset:
        assert cfg.key_quant_bits == 8
    if "_nc" in preset:
        assert cfg.norm_correction is True


def test_config_slot_size_alignment():
    cfg = AscendTurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", head_dim=512)
    # 4-bit key: ceil(512*4/8) + 2 = 256+2 = 258 B
    # 4-bit val: ceil(512*4/8) + 4 = 256+4 = 260 B
    # total = 518 -> aligned to 32B -> 544
    assert cfg.key_packed_size == 258
    assert cfg.value_packed_size == 260
    assert cfg.slot_size == 518
    assert cfg.slot_size_aligned == 544
    assert cfg.slot_size_aligned % 32 == 0


def test_config_invalid_dtype_raises():
    with pytest.raises(ValueError):
        AscendTurboQuantConfig.from_cache_dtype("turboquant_bogus", head_dim=512)


def test_config_invalid_target_raises():
    with pytest.raises(ValueError):
        AscendTurboQuantConfig.from_cache_dtype(
            "turboquant_4bit_nc", head_dim=512, target="bogus"
        )


# ---------------------------------------------------------------------------
# Key quantization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset,max_rel_err",
    [
        ("turboquant_4bit_nc", 0.30),
        ("turboquant_k3v4_nc", 0.55),  # 3-bit is lossy
        ("turboquant_3bit_nc", 0.55),
    ],
)
def test_key_roundtrip_bounded_error(preset, max_rel_err):
    """Reference quantizer round-trip error bounded vs original on Gaussian K.

    Bound is loose because we test single vectors; aggregated PPL is much
    better than per-vector reconstruction error.
    """
    torch.manual_seed(0)
    cfg = AscendTurboQuantConfig.from_cache_dtype(preset, head_dim=512)
    k = torch.randn(128, 512, dtype=torch.float32)

    k_hat = reference_roundtrip_key(k, cfg).to(torch.float32)

    # Per-vector relative L2 error.
    err = (k_hat - k).norm(dim=-1) / k.norm(dim=-1).clamp_min(1e-8)
    assert err.mean().item() < max_rel_err, (
        f"{preset}: mean rel err {err.mean().item():.3f} > {max_rel_err}"
    )


def test_key_quantize_shapes():
    cfg = AscendTurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", head_dim=512)
    k = torch.randn(64, 512)
    q_idx, vec_norm = quantize_key_mse(k, cfg)
    assert q_idx.shape == (64, 512)
    assert vec_norm.shape == (64,)
    assert q_idx.min().item() >= 0
    assert q_idx.max().item() < cfg.n_centroids


# ---------------------------------------------------------------------------
# Value quantization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [3, 4])
def test_value_roundtrip_bounded(bits):
    cfg = AscendTurboQuantConfig(head_dim=512, value_quant_bits=bits)
    torch.manual_seed(0)
    v = torch.randn(64, 512)
    v_hat = reference_roundtrip_value(v, cfg).to(torch.float32)
    # Uniform quantization error <= scale/2 per element; tight bound on relative L2.
    err = (v_hat - v).norm(dim=-1) / v.norm(dim=-1).clamp_min(1e-8)
    # 4-bit: < ~15% rel error per vector; 3-bit: looser.
    assert err.mean().item() < (0.20 if bits == 4 else 0.40)


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


def test_compression_ratio_matches_preset_expectations():
    """Reported compression ratios should match upstream presets when the
    alignment overhead is amortized (Plan A target head_dim = kv_lora_rank = 512).
    """
    expected_ratio = {
        "turboquant_k8v4": 2.6,
        "turboquant_4bit_nc": 3.8,
        "turboquant_3bit_nc": 4.9,
    }
    for preset, want in expected_ratio.items():
        cfg = AscendTurboQuantConfig.from_cache_dtype(preset, head_dim=512)
        got = cfg.compression_ratio(original_dtype_bytes=2)  # bf16 baseline
        # 25% slack — Ascend 32B slot alignment vs CUDA 16B costs more on the
        # small "vec_norm + scale + zero" overhead bytes.
        assert (
            got > want * 0.75
        ), f"{preset}: got {got:.2f}x, expected ~{want}x (>0.75×)"
