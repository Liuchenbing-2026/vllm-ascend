# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def draft_tnd_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    causal: bool,
    sliding_window: int | None,
    attn_mask: torch.Tensor | None,
    cache: dict,
) -> torch.Tensor | None:
    """Try paged TND attention with device lengths and device tiling.

    The optional operators must come from a matching, compiled FIA sink
    extension. Native torch-npu FIA v2 has a host-list length schema and is
    deliberately not used as a substitute. Unsupported configurations return
    None before launching work; runtime/precision failures are not swallowed.

    ``cache`` belongs to one attention metadata build, never to a layer or
    graph lifetime. It retains length snapshots until all layers have consumed
    the asynchronous metadata result. Graph capture is handled by the caller.
    """
    ops = torch.ops._C_ascend
    if not hasattr(ops, "_npu_fused_infer_attention_score_v2_sink_metadata") or not hasattr(
        ops, "npu_fused_infer_attention_score_v2_sink"
    ):
        return None
    # Noncausal sliding windows have different alignment semantics from the
    # native band mode; keep the existing masked path for that configuration.
    if sliding_window is not None and not causal:
        return None
    # The pinned sink dependency's CheckGqaDSupport rejects equal Q/K/V
    # head sizes outside this set. In particular, Qwen's D=256 must keep
    # the existing path instead of reaching a failed CANN tiling call.
    supported_gqa_head_sizes = (64, 128, 192)
    if query.shape[-1] not in supported_gqa_head_sizes:
        return None
    if key.ndim != 3 or key.shape != value.shape or key.shape[1] != block_size:
        return None
    if not key.is_contiguous() or not value.is_contiguous() or not query.is_contiguous():
        return None
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        return None
    batch = seq_lens.shape[0]
    if block_table is None or block_table.shape[0] != batch or query_start_loc.numel() != batch + 1:
        return None
    if causal and attn_mask is None:
        return None

    sparse_mode = 4 if sliding_window is not None else 3 if causal else 0
    unlimited_tokens = torch.iinfo(torch.int32).max
    pre_tokens = sliding_window if sliding_window is not None else unlimited_tokens
    next_tokens = 0 if sliding_window is not None else unlimited_tokens
    signature = (num_heads, num_kv_heads, query.shape[-1], sparse_mode, pre_tokens, block_size)
    if signature not in cache:
        qlens = query_start_loc[1:].to(dtype=torch.int64, copy=True)
        kvlens = seq_lens.to(dtype=torch.int64, copy=True)
        limits = torch.npu.get_stream_limit(torch.npu.current_stream())
        metadata = ops._npu_fused_infer_attention_score_v2_sink_metadata(
            num_heads,
            num_kv_heads,
            query.shape[-1],
            query.shape[-1],
            actual_seq_lengths=qlens,
            actual_seq_lengths_kv=kvlens,
            batch_size=batch,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            input_layout="TND",
            input_layout_kv="BnBsH",
            block_size=block_size,
            aic_core_num=limits["cube_core_num"],
            aiv_core_num=limits["vector_core_num"],
        )
        cache[signature] = (qlens, kvlens, metadata)
    qlens, kvlens, metadata = cache[signature]
    output, _ = ops.npu_fused_infer_attention_score_v2_sink(
        query,
        key,
        value,
        actual_seq_qlen=qlens,
        actual_seq_kvlen=kvlens,
        block_table=block_table,
        meta_data=metadata,
        num_query_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        softmax_scale=scale,
        input_layout="TND",
        block_size=block_size,
        sparse_mode=sparse_mode,
        atten_mask=attn_mask if causal else None,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
    )
    return output
