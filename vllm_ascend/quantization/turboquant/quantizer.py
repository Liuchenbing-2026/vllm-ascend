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
"""Reference TurboQuant quantizer (torch-based).

This is the **correctness reference** for the NPU kernel. It runs on any
device (CPU/CUDA/NPU), is slow, and is used:
  1. To validate AscendC / triton-ascend kernel output (bit-for-bit / within tol).
  2. As a fallback when the NPU kernel is not available.
  3. To unit-test the algorithm end-to-end without NPU hardware.

Algorithm (matches upstream vllm `triton_turboquant_store` semantics):
  KEY:
    1. Random sign flip:  s = sign(rand)  (deterministic per seed)
    2. Hadamard rotate:   k_rot = H @ (s ⊙ k)        # H is fixed Hadamard matrix
    3. Per-coord quantize: q_idx = argmin_{c ∈ centroids} (k_rot_i - c * vec_norm)^2
       where vec_norm = ||k_rot|| / sqrt(d)
    4. Pack:               [q_idx (packed bits) | vec_norm (fp16)]

  VALUE:
    1. Compute per-vector scale/zero:  scale = (max-min)/(2^bits - 1), zero = min
    2. Quantize:                       q_idx = round((v - zero) / scale)
    3. Pack:                           [q_idx (packed bits) | scale, zero (fp16)]

Dequant is the symmetric inverse.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm_ascend.quantization.turboquant.centroids import (
    get_centroids_and_boundaries,
)

if TYPE_CHECKING:
    from vllm_ascend.quantization.turboquant.config import AscendTurboQuantConfig


# ---------------------------------------------------------------------------
# Hadamard matrix construction (Sylvester construction; power-of-2 sizes).
# ---------------------------------------------------------------------------

_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def get_hadamard_matrix(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sylvester Hadamard matrix scaled by 1/sqrt(d) so H @ H^T = I.

    d must be a power of 2.
    """
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"Hadamard dim must be power of 2; got {d}")

    key = d
    if key not in _HADAMARD_CACHE:
        # Build on CPU fp32 then cache.
        H = torch.tensor([[1.0]], dtype=torch.float32)
        k = 1
        while k < d:
            H = torch.cat(
                [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
                dim=0,
            )
            k *= 2
        H = H / math.sqrt(d)
        _HADAMARD_CACHE[key] = H

    return _HADAMARD_CACHE[key].to(device=device, dtype=dtype)


def _signs_from_seed(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Deterministic ±1 sign vector. Stays in sync between store / load."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.where(
        torch.rand(d, generator=g) < 0.5,
        torch.full((d,), -1.0),
        torch.full((d,), 1.0),
    ).to(device=device)


# ---------------------------------------------------------------------------
# Key quantization (MSE / Lloyd-Max with Hadamard rotation)
# ---------------------------------------------------------------------------


def quantize_key_mse(
    k: torch.Tensor,  # [B, d] (fp16 / bf16)
    cfg: "AscendTurboQuantConfig",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a batch of key vectors with Hadamard + Lloyd-Max.

    Returns:
        q_idx:    [B, d] int64 centroid index (0..n_centroids-1)
        vec_norm: [B]    fp32 per-vector norm of rotated key (scale factor)
    """
    assert k.dim() == 2, f"k must be [B, d]; got shape {k.shape}"
    B, d = k.shape
    assert d == cfg.head_dim, f"d mismatch: {d} vs cfg.head_dim={cfg.head_dim}"

    device, dtype = k.device, torch.float32
    k_f = k.to(dtype)

    # 1. Random sign flip + Hadamard rotation.
    signs = _signs_from_seed(d, seed, device)
    H = get_hadamard_matrix(d, device, dtype)
    k_rot = (k_f * signs) @ H.T  # [B, d]

    # 2. Per-vector norm (for unit-normalization + storage).
    # Lloyd-Max centroids are solved for N(0, 1/d), which is the per-coord
    # distribution of a UNIT-NORM vector after random rotation in d-D.
    # So we normalize each row to unit L2 norm (vec_norm = ||k_rot||).
    vec_norm = k_rot.norm(dim=-1)  # [B]
    k_n = k_rot / (vec_norm.unsqueeze(-1).clamp_min(1e-8))  # [B, d] unit-norm

    # 3. Lloyd-Max quantize each coordinate independently.
    centroids, boundaries = get_centroids_and_boundaries(d, cfg.key_mse_bits)
    centroids = centroids.to(device=device, dtype=dtype)
    boundaries = boundaries.to(device=device, dtype=dtype)
    # bucketize: returns the index of the first centroid bin k_n[i] falls into.
    q_idx = torch.bucketize(k_n, boundaries)  # [B, d] int64 0..n_centroids-1
    return q_idx, vec_norm


def dequantize_key_mse(
    q_idx: torch.Tensor,  # [B, d] int64
    vec_norm: torch.Tensor,  # [B] fp32
    cfg: "AscendTurboQuantConfig",
    seed: int = 42,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of quantize_key_mse. Returns reconstructed key [B, d]."""
    B, d = q_idx.shape
    device = q_idx.device

    centroids, _ = get_centroids_and_boundaries(d, cfg.key_mse_bits)
    centroids = centroids.to(device=device, dtype=torch.float32)

    # Look up centroid values; result is approximately unit-norm.
    k_n_hat = centroids[q_idx.clamp(0, len(centroids) - 1)]  # [B, d]
    if cfg.norm_correction:
        # Renormalize each reconstructed row to unit L2 norm to fix the bias
        # introduced by per-coord centroid lookup (recovers ~0.8% PPL @ 4-bit).
        post_norm = k_n_hat.norm(dim=-1)
        k_n_hat = k_n_hat / post_norm.unsqueeze(-1).clamp_min(1e-8)

    k_rot_hat = k_n_hat * vec_norm.unsqueeze(-1).to(torch.float32)

    # Inverse Hadamard + sign flip.
    signs = _signs_from_seed(d, seed, device)
    H = get_hadamard_matrix(d, device, torch.float32)
    k_hat = (k_rot_hat @ H) * signs  # H @ H^T = I and H == H^T (symmetric).
    return k_hat.to(out_dtype)


# ---------------------------------------------------------------------------
# Value quantization (uniform / per-vector scale-zero)
# ---------------------------------------------------------------------------


def quantize_value_uniform(
    v: torch.Tensor,  # [B, d]
    cfg: "AscendTurboQuantConfig",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-vector uniform quantization. Returns (q_idx, scale, zero)."""
    assert v.dim() == 2
    B, d = v.shape
    bits = cfg.value_quant_bits
    n_levels = (1 << bits) - 1  # e.g. 4-bit -> 15 levels (signed-asymmetric)

    v_f = v.to(torch.float32)
    v_min = v_f.amin(dim=-1, keepdim=True)
    v_max = v_f.amax(dim=-1, keepdim=True)
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp_min(1e-8)
    zero = v_min

    q_idx = ((v_f - zero) / scale).round().clamp(0, n_levels).to(torch.int64)
    return q_idx, scale.squeeze(-1), zero.squeeze(-1)


def dequantize_value_uniform(
    q_idx: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    v_hat = q_idx.to(torch.float32) * scale.unsqueeze(-1).to(torch.float32) + zero.unsqueeze(-1).to(torch.float32)
    return v_hat.to(out_dtype)


# ---------------------------------------------------------------------------
# Round-trip helpers (correctness self-check)
# ---------------------------------------------------------------------------


def reference_roundtrip_key(
    k: torch.Tensor, cfg: "AscendTurboQuantConfig", seed: int = 42
) -> torch.Tensor:
    """End-to-end quantize → dequantize for a key tensor (correctness check)."""
    q_idx, vec_norm = quantize_key_mse(k, cfg, seed)
    return dequantize_key_mse(q_idx, vec_norm, cfg, seed, out_dtype=k.dtype)


def reference_roundtrip_value(
    v: torch.Tensor, cfg: "AscendTurboQuantConfig"
) -> torch.Tensor:
    """End-to-end quantize → dequantize for a value tensor."""
    q_idx, scale, zero = quantize_value_uniform(v, cfg)
    return dequantize_value_uniform(q_idx, scale, zero, out_dtype=v.dtype)
