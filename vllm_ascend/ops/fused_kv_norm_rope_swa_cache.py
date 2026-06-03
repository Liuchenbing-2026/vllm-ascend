# SPDX-License-Identifier: Apache-2.0
"""Python wrapper for the DSV4-DSA SWA-kv prep fusion custom op.

Replaces the 3 consecutive ops on the DSA sliding-window KV path
(vllm_ascend/attention/dsa_v1.py:1907-1920 prefill, 2195-2208 decode):
    kv = kv_norm(wkv(hidden))                 # RMSNorm over head_dim
    inplace_partial_rotary_mul(kv, cos, sin, "interleave", [nope, head_dim])
    npu_scatter_nd_update_v2(swa_kv_cache, slot_mapping, kv)

with a single fused AscendC kernel (kv_norm + partial-interleave-RoPE + scatter-nd write).
"""
import torch
from vllm_ascend.utils import enable_custom_op


def fused_kv_norm_rope_swa_cache(
    kv_in: torch.Tensor,        # [nt, head_dim]   fp16/bf16  (post-wkv, pre-norm)
    gamma: torch.Tensor,        # [head_dim]       fp16/bf16  (kv_norm.weight)
    cos: torch.Tensor,          # [nt, rope_dim]   fp32       (pre-gathered, pair-repeated)
    sin: torch.Tensor,          # [nt, rope_dim]   fp32
    slot_mapping: torch.Tensor, # [nt, 2]          int32      (block_idx, block_offset)
    kv_cache: torch.Tensor,     # [num_blocks, block_size, head_dim] fp16/bf16, mutated in place
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Returns kv_out [nt, head_dim] (normed+roped value); kv_cache is updated in place."""
    enable_custom_op()
    return torch.ops._C_ascend.npu_fused_kv_norm_rope_swa_cache(
        kv_in, gamma, cos, sin, slot_mapping, kv_cache, epsilon)
