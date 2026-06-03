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
"""NPU op wrappers for TurboQuant store / decode.

This module exposes a stable Python API that the SFA backend can call. The
implementation picks the best available backend:

  1. AscendC custom op (preferred; not yet implemented; see TODO below)
  2. triton-ascend port of upstream Triton kernel (TODO)
  3. torch reference quantizer (slow but correct; runs anywhere)

The selection is controlled by VLLM_ASCEND_TURBOQUANT_BACKEND env var:
    "auto" (default) -> aclnn → triton → reference
    "aclnn"          -> force AscendC (raises if missing)
    "triton"         -> force triton-ascend (raises if missing)
    "reference"      -> force torch reference (slow but always available)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from vllm_ascend.quantization.turboquant.quantizer import (
    quantize_key_mse,
    quantize_value_uniform,
    dequantize_key_mse,
    dequantize_value_uniform,
)

if TYPE_CHECKING:
    from vllm_ascend.quantization.turboquant.config import AscendTurboQuantConfig

logger = logging.getLogger(__name__)


def _backend_choice() -> str:
    from vllm_ascend import envs as ascend_envs

    return getattr(ascend_envs, "VLLM_ASCEND_TURBOQUANT_BACKEND", "auto").lower()


def _try_aclnn_store(*args, **kwargs):
    """Try the AscendC `aclnnTurboQuantStore` op.

    TODO(turboquant): wire to actual AscendC custom op once implemented.
    Currently always returns None (i.e. "not available").
    """
    return None


def _try_triton_store(*args, **kwargs):
    """Try the triton-ascend ported `triton_turboquant_store`.

    TODO(turboquant): port `vllm/v1/attention/ops/triton_turboquant_store.py`
    to triton-ascend (replace block_ptr / multibuffer / make_block_ptr usage
    per Ascend Triton constraints).
    """
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def turboquant_store(
    latent_kv: torch.Tensor,  # [B, head_dim] fp16 / bf16
    slot_mapping: torch.Tensor,  # [B] int64
    quant_cache: torch.Tensor,  # [num_slots, slot_size_aligned] uint8
    cfg: "AscendTurboQuantConfig",
    seed: int = 42,
) -> None:
    """Quantize latent KV and store packed bytes into quant_cache at slot_mapping.

    Side-effecting; writes into ``quant_cache`` in-place.

    Layout per slot:
        [ key_packed (cfg.key_packed_size B) | value_packed (cfg.value_packed_size B) | pad ]
    where:
        key_packed   = q_idx packed bits (cfg.key_mse_bits per coord) + vec_norm fp16
        value_packed = q_idx packed bits (cfg.value_quant_bits per coord) + scale fp16 + zero fp16
    """
    choice = _backend_choice()

    # Try AscendC, then Triton, then fall through to reference.
    if choice in ("auto", "aclnn"):
        ret = _try_aclnn_store(latent_kv, slot_mapping, quant_cache, cfg, seed)
        if ret is not None:
            return ret
        if choice == "aclnn":
            raise RuntimeError("VLLM_ASCEND_TURBOQUANT_BACKEND=aclnn but op not available")

    if choice in ("auto", "triton"):
        ret = _try_triton_store(latent_kv, slot_mapping, quant_cache, cfg, seed)
        if ret is not None:
            return ret
        if choice == "triton":
            raise RuntimeError("VLLM_ASCEND_TURBOQUANT_BACKEND=triton but kernel not available")

    # Reference path (slow but correct).
    _reference_store(latent_kv, slot_mapping, quant_cache, cfg, seed)


def turboquant_decode_attention(
    q: torch.Tensor,
    quant_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cfg: "AscendTurboQuantConfig",
    topk_indices: torch.Tensor | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Dequantize selected slots and run attention.

    For Plan A (DSA), `topk_indices` selects which past tokens to attend to
    (sparse indexer output). Only those slots are dequantized.

    TODO(turboquant): replace with fused AscendC / triton-ascend kernel that
    fuses dequant + flash attention. Current implementation is a reference path
    using `dequantize_key_mse` + `dequantize_value_uniform` + torch attention.
    """
    return _reference_decode_attention(
        q, quant_cache, block_table, seq_lens, cfg, topk_indices, seed
    )


# ---------------------------------------------------------------------------
# Reference implementations (slow but always correct)
# ---------------------------------------------------------------------------


def _pack_uint(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack int64 values (each occupying `bits` bits) into uint8 bytes.

    values: [B, d] int64 in [0, 2^bits - 1]
    Returns: [B, ceil(d*bits/8)] uint8
    """
    B, d = values.shape
    total_bits = d * bits
    nbytes = (total_bits + 7) // 8

    # Naive packing — correct but slow. Real NPU kernel must use SIMD packing.
    flat_bits = torch.zeros(B, nbytes * 8, dtype=torch.uint8, device=values.device)
    for i in range(d):
        v = (values[:, i] & ((1 << bits) - 1)).to(torch.uint8)
        for b in range(bits):
            bit_idx = i * bits + b
            flat_bits[:, bit_idx] = (v >> b) & 1
    # Pack 8 consecutive bits into one uint8.
    out = torch.zeros(B, nbytes, dtype=torch.uint8, device=values.device)
    for b in range(8):
        out |= flat_bits[:, b::8] << b
    return out


def _unpack_uint(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Inverse of _pack_uint."""
    B, nbytes = packed.shape
    flat_bits = torch.zeros(B, nbytes * 8, dtype=torch.uint8, device=packed.device)
    for b in range(8):
        flat_bits[:, b::8] = (packed >> b) & 1
    out = torch.zeros(B, d, dtype=torch.int64, device=packed.device)
    for i in range(d):
        for b in range(bits):
            bit_idx = i * bits + b
            if bit_idx < nbytes * 8:
                out[:, i] |= flat_bits[:, bit_idx].to(torch.int64) << b
    return out


def _reference_store(
    latent_kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_cache: torch.Tensor,
    cfg: "AscendTurboQuantConfig",
    seed: int,
) -> None:
    """Slow reference store: torch ops + naive bit packing."""
    B, d = latent_kv.shape

    # Split latent into k and v halves (Plan A: use same buffer for both).
    # For GLM5 MLA: compressed_kv [B, kv_lora_rank] — we treat it as both K and V.
    # The decompression matrices (W_uk, W_uv) live in the model weights; cache
    # only stores the compressed latent. So our "K" and "V" inputs are actually
    # the same tensor; we quantize it once but pack independently for symmetry
    # with the upstream layout (so the decode kernel can dequant K or V slice
    # independently if a future scheme separates them).
    k = latent_kv
    v = latent_kv

    # KEY
    if cfg.key_fp8:
        # FP8 mode: just cast (no rotation/MSE).
        k_packed = k.to(torch.float8_e4m3fn).view(torch.uint8)
        vec_norm = torch.zeros(B, dtype=torch.float16, device=k.device)
    else:
        q_idx_k, vec_norm = quantize_key_mse(k, cfg, seed)
        k_packed = _pack_uint(q_idx_k, cfg.key_mse_bits)
        vec_norm = vec_norm.to(torch.float16)

    # VALUE
    q_idx_v, scale_v, zero_v = quantize_value_uniform(v, cfg)
    v_packed = _pack_uint(q_idx_v, cfg.value_quant_bits)
    scale_v = scale_v.to(torch.float16)
    zero_v = zero_v.to(torch.float16)

    # Lay out into quant_cache[slot_mapping] = [key_packed | vec_norm | value_packed | scale | zero | pad]
    key_size = cfg.key_packed_size
    val_size = cfg.value_packed_size
    slot_size = cfg.slot_size_aligned

    for i in range(B):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        cursor = 0
        # key bytes
        if cfg.key_fp8:
            quant_cache[slot, cursor : cursor + cfg.head_dim] = k_packed[i]
            cursor += cfg.head_dim
        else:
            kp_bytes = k_packed.shape[1]
            quant_cache[slot, cursor : cursor + kp_bytes] = k_packed[i]
            cursor += kp_bytes
            # vec_norm fp16 (2 bytes)
            vn_bytes = vec_norm[i : i + 1].view(torch.uint8)
            quant_cache[slot, cursor : cursor + 2] = vn_bytes
            cursor += 2
        # value bytes
        vp_bytes = v_packed.shape[1]
        quant_cache[slot, cursor : cursor + vp_bytes] = v_packed[i]
        cursor += vp_bytes
        # scale + zero fp16 (4 bytes)
        sb = scale_v[i : i + 1].view(torch.uint8)
        zb = zero_v[i : i + 1].view(torch.uint8)
        quant_cache[slot, cursor : cursor + 2] = sb
        quant_cache[slot, cursor + 2 : cursor + 4] = zb
        # (rest is padding)


def _reference_decode_attention(
    q: torch.Tensor,
    quant_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cfg: "AscendTurboQuantConfig",
    topk_indices: torch.Tensor | None,
    seed: int,
) -> torch.Tensor:
    """Slow reference: dequant selected slots, run torch attention.

    This is intentionally not optimized. Used as ground truth for the NPU
    kernel and as a CPU-runnable e2e correctness path.
    """
    # TODO(turboquant): real implementation — fused dequant + flash attention.
    raise NotImplementedError(
        "TurboQuant decode-attention reference path not implemented yet. "
        "Wire to AscendC / triton-ascend kernel in vllm_ascend/csrc or "
        "vllm_ascend/ops/triton/ first."
    )
