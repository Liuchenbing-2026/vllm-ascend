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
"""Correctness-first reference attention for Qwen3/GLM DSpark drafts.

This module is the numerical golden for the DSpark draft attention. It makes
the DFlash draft-block visibility contract explicit instead of relying on the
generic maskless paged ``npu_fused_infer_attention_score`` branch (which is a
contract-external operator combination for this shape and was the leading
suspect for the near-zero acceptance rate).

Contract (see ``DFlashQwen3Attention.forward`` and ``precompute_and_store_context_kv``):

* Each request precomputes the target context K/V into the paged KV cache.
* A draft pass runs ``1 + num_speculative_tokens`` query tokens per request.
* Every query token in a draft block attends to:
    - the trailing context window (the full context when the layer is not a
      sliding-window layer), and
    - the **entire** current draft block, strictly **non-causal** (each query
      token sees every other query token in the block, including future ones).
* Standard multi-head / grouped-query attention with per-layer scale
  ``head_dim ** -0.5`` and an optional per-head attention sink bias.

The implementation favors clarity over speed; it is intended for NPU bring-up
and as the reference an optimized FIA/SAS/custom-op path is A/B-tested against.
"""

from __future__ import annotations

import torch

__all__ = [
    "gather_paged_kv",
    "gather_context_kv_from_cache",
    "dspark_mha_reference",
    "qwen3_dspark_reference_attention",
]

_ACL_FORMAT_ND = 2


def _to_nd_format(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize NPU tensors in ND format; leave other devices unchanged."""
    if tensor.device.type != "npu":
        return tensor
    import torch_npu

    converted = torch_npu.npu_format_cast(tensor, _ACL_FORMAT_ND)
    if torch_npu.get_npu_format(converted) == torch_npu.Format.ND:
        return converted

    # Some RoPE/view outputs report NCL even after npu_format_cast(ND). Build
    # the result on a flat ND allocation so ACLNN Cat sees a real ND tensor.
    flat = torch.empty(
        converted.numel(), dtype=converted.dtype, device=converted.device
    )
    flat.copy_(converted.reshape(-1))
    converted = flat.view(converted.shape)
    if torch_npu.get_npu_format(converted) != torch_npu.Format.ND:
        raise RuntimeError("Failed to materialize DSpark attention tensor in ND format")
    return converted


def _copy_concat_tokens(
    context: torch.Tensor, draft: torch.Tensor
) -> torch.Tensor:
    """Concatenate token blocks without ACLNN Cat format constraints."""
    if context.shape[1:] != draft.shape[1:]:
        raise ValueError(
            "DSpark context and draft KV shapes must match after token dim: "
            f"{tuple(context.shape)} vs {tuple(draft.shape)}"
        )
    output = torch.empty(
        (context.shape[0] + draft.shape[0],) + tuple(context.shape[1:]),
        dtype=context.dtype,
        device=context.device,
    )
    context_len = context.shape[0]
    if context_len:
        output[:context_len].copy_(context)
    if draft.shape[0]:
        output[context_len:].copy_(draft)
    return output


def _unwrap_cache(cache: torch.Tensor | list | tuple) -> torch.Tensor:
    """Return the underlying per-layer cache tensor.

    The Ascend attention layer stores its cache as ``[(k_cache, v_cache)]`` (a
    single-element list indexed by virtual engine). Callers may hand us either
    the already-split ``k_cache``/``v_cache`` tensor or one of those wrappers.
    """
    while isinstance(cache, (list, tuple)) and len(cache) == 1:
        cache = cache[0]
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Expected a tensor KV cache, got {type(cache)!r}")
    return cache


def gather_paged_kv(
    cache: torch.Tensor | list | tuple,
    slot_ids: torch.Tensor,
    cache_block_size: int,
) -> torch.Tensor:
    """Gather cache rows for ``slot_ids`` from a paged cache.

    Args:
        cache: Paged cache shaped ``[num_blocks, cache_block_size, num_kv_heads,
            head_dim]`` (a single K or V cache; see
            ``AscendAttentionBackend.get_kv_cache_shape`` which stacks K/V on a
            leading dim of size 2).
        slot_ids: 1-D int tensor of flat slot ids. Negative ids (``-1`` padding)
            are dropped.
        cache_block_size: Tokens per physical cache block.

    Returns:
        Tensor shaped ``[num_valid_slots, num_kv_heads, head_dim]``.
    """
    cache = _unwrap_cache(cache)
    slot_ids = slot_ids.reshape(-1)
    valid = slot_ids[slot_ids >= 0].to(device=cache.device, dtype=torch.long)
    if valid.numel() == 0:
        return cache.new_empty((0,) + tuple(cache.shape[2:]))
    block_ids = valid // cache_block_size
    block_offsets = valid % cache_block_size
    return cache[block_ids, block_offsets]


def gather_context_kv_from_cache(
    key_cache: torch.Tensor | list | tuple,
    value_cache: torch.Tensor | list | tuple,
    block_table_row: torch.Tensor,
    start_pos: int,
    end_pos: int,
    cache_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather context K/V for positions ``[start_pos, end_pos)`` of one request.

    Positions are mapped to physical slots through ``block_table_row``.

    Returns:
        ``(k_ctx, v_ctx)`` each shaped ``[end_pos - start_pos, num_kv_heads,
        head_dim]`` (empty when the range is empty).
    """
    k_cache = _unwrap_cache(key_cache)
    v_cache = _unwrap_cache(value_cache)
    if end_pos <= start_pos:
        empty_k = k_cache.new_empty((0,) + tuple(k_cache.shape[2:]))
        empty_v = v_cache.new_empty((0,) + tuple(v_cache.shape[2:]))
        return empty_k, empty_v

    positions = torch.arange(
        start_pos, end_pos, dtype=torch.long, device=k_cache.device
    )
    block_nums = positions // cache_block_size
    block_offsets = positions % cache_block_size
    block_table_row = block_table_row.to(device=k_cache.device, dtype=torch.long)
    block_ids = block_table_row.index_select(0, block_nums)
    slot_ids = block_ids * cache_block_size + block_offsets
    k_ctx = gather_paged_kv(k_cache, slot_ids, cache_block_size)
    v_ctx = gather_paged_kv(v_cache, slot_ids, cache_block_size)
    return k_ctx, v_ctx


def _expand_kv_heads(kv: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Broadcast KV heads to query heads for grouped-query attention.

    Args:
        kv: ``[tokens, num_kv_heads, head_dim]``.
        num_query_heads: Number of query heads.
    """
    num_kv_heads = kv.shape[1]
    if num_kv_heads == num_query_heads:
        return kv
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be a multiple of num_kv_heads: "
            f"{num_query_heads} vs {num_kv_heads}"
        )
    group_size = num_query_heads // num_kv_heads
    return kv.repeat_interleave(group_size, dim=1)


def dspark_mha_reference(
    q_block: torch.Tensor,
    k_visible: torch.Tensor,
    v_visible: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Non-causal MHA of a draft block over its visible K/V (FP32 softmax).

    Args:
        q_block: ``[query_len, num_heads, head_dim]``.
        k_visible: ``[num_visible, num_kv_heads, head_dim]``.
        v_visible: ``[num_visible, num_kv_heads, head_dim]``.
        scale: Softmax scale (typically ``head_dim ** -0.5``).
        attn_sink: Optional per-query-head sink bias ``[num_heads]`` added as an
            extra logit in the softmax denominator.

    Returns:
        ``[query_len, num_heads, head_dim]`` in ``q_block``'s dtype.
    """
    num_heads = q_block.shape[1]
    out = torch.zeros_like(q_block)
    if k_visible.shape[0] == 0:
        # No visible KV: with a sink the block emits zeros; without a sink the
        # softmax is undefined, so we also emit zeros deterministically.
        return out

    k = _expand_kv_heads(k_visible, num_heads).float()
    v = _expand_kv_heads(v_visible, num_heads).float()
    q = q_block.float()

    # scores: [query_len, num_heads, num_visible]
    scores = torch.einsum("qhd,khd->qhk", q, k) * scale
    scores_max = scores.max(dim=-1, keepdim=True).values
    if attn_sink is not None:
        sink = attn_sink[:num_heads].float().view(1, num_heads, 1)
        scores_max = torch.maximum(scores_max, sink)
    exp_scores = torch.exp(scores - scores_max)
    denom = exp_scores.sum(dim=-1, keepdim=True)
    if attn_sink is not None:
        sink = attn_sink[:num_heads].float().view(1, num_heads, 1)
        denom = denom + torch.exp(sink - scores_max)
    probs = exp_scores / denom
    out = torch.einsum("qhk,khd->qhd", probs, v)
    return out.to(q_block.dtype)


def qwen3_dspark_reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor | list | tuple,
    value_cache: torch.Tensor | list | tuple,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    cache_block_size: int,
    *,
    sliding_window: int | None = None,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference DSpark draft-block attention over a paged context cache.

    Context K/V for positions ``[start, prefix)`` are gathered from the paged
    cache; the current draft block's ``key``/``value`` are concatenated after
    them so the block is fully (non-causally) visible to itself without relying
    on the query K/V having already been written to the cache.

    Args:
        query: ``[num_query_tokens, num_heads, head_dim]``.
        key: ``[num_query_tokens, num_kv_heads, head_dim]`` (draft block K).
        value: ``[num_query_tokens, num_kv_heads, head_dim]`` (draft block V).
        key_cache: Paged K cache ``[num_blocks, cache_block_size, num_kv_heads,
            head_dim]``.
        value_cache: Paged V cache, same shape as ``key_cache``.
        block_table: ``[num_reqs, max_blocks_per_req]`` int tensor.
        query_start_loc: Cumulative query lengths ``[num_reqs + 1]`` (``[0, q0,
            q0 + q1, ...]``).
        seq_lens: Total visible length per request ``[num_reqs]`` (context +
            draft block).
        scale: Softmax scale.
        cache_block_size: Tokens per physical cache block.
        sliding_window: Trailing context window; ``None`` means full context.
        attn_sink: Optional per-head sink bias ``[num_heads]``.

    Returns:
        ``[num_query_tokens, num_heads, head_dim]`` in ``query``'s dtype.
    """
    num_query_tokens = query.shape[0]
    out = torch.zeros_like(query)
    num_reqs = max(int(query_start_loc.numel()) - 1, 0)
    qsl = query_start_loc.to(device="cpu", dtype=torch.long)
    seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.long)

    for req_idx in range(num_reqs):
        if req_idx >= block_table.shape[0] or req_idx >= seq_lens_cpu.numel():
            continue
        row_start = max(int(qsl[req_idx].item()), 0)
        row_end = min(int(qsl[req_idx + 1].item()), num_query_tokens)
        if row_end <= row_start:
            continue

        query_len = row_end - row_start
        seq_len = int(seq_lens_cpu[req_idx].item())
        prefix_len = seq_len - query_len
        if prefix_len < 0:
            raise ValueError(
                "DSpark seq_len is shorter than its draft block: "
                f"seq_len={seq_len}, query_len={query_len}"
            )
        if sliding_window is not None:
            start_pos = max(prefix_len - int(sliding_window), 0)
        else:
            start_pos = 0

        k_ctx, v_ctx = gather_context_kv_from_cache(
            key_cache,
            value_cache,
            block_table[req_idx],
            start_pos,
            prefix_len,
            cache_block_size,
        )

        # Paged-cache gathers and RoPE outputs can carry different internal NPU
        # formats even when get_npu_format reports ND. Materialize one output
        # buffer with copy_ instead of invoking ACLNN Cat's same-format contract.
        k_visible = _copy_concat_tokens(k_ctx, key[row_start:row_end])
        v_visible = _copy_concat_tokens(v_ctx, value[row_start:row_end])

        out[row_start:row_end] = dspark_mha_reference(
            query[row_start:row_end],
            k_visible,
            v_visible,
            scale,
            attn_sink=attn_sink,
        )

    return out
