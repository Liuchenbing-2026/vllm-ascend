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
"""Semantic tests for the Qwen3/GLM DSpark reference attention.

These mirror the PR 11196 DSpark attention semantics (full-context visibility,
fully non-causal draft block, multi-request isolation, padding, slot mapping)
but for standard Qwen3/GLM MHA/GQA. They run on CPU only and need no NPU.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.qwen3_dspark_attention import (
    _to_nd_format,
    dspark_mha_reference,
    gather_context_kv_from_cache,
    gather_paged_kv,
    qwen3_dspark_reference_attention,
)


def _build_paged_cache(
    per_req_context,
    block_table,
    num_blocks,
    cache_block_size,
    num_kv_heads,
    head_dim,
    dtype,
):
    """Scatter per-request context K/V into a paged cache.

    Args:
        per_req_context: list of ``(k_ctx, v_ctx)`` with ``k_ctx`` shaped
            ``[ctx_len, num_kv_heads, head_dim]``. Context positions are
            ``0..ctx_len-1`` and map to slots via ``block_table``.
        block_table: ``[num_reqs, max_blocks]`` physical block ids.
    Returns:
        ``(key_cache, value_cache)`` each ``[num_blocks, cache_block_size,
        num_kv_heads, head_dim]``.
    """
    key_cache = torch.zeros(
        num_blocks, cache_block_size, num_kv_heads, head_dim, dtype=dtype
    )
    value_cache = torch.zeros_like(key_cache)
    for req_idx, (k_ctx, v_ctx) in enumerate(per_req_context):
        for pos in range(k_ctx.shape[0]):
            block_id = int(block_table[req_idx, pos // cache_block_size].item())
            offset = pos % cache_block_size
            key_cache[block_id, offset] = k_ctx[pos]
            value_cache[block_id, offset] = v_ctx[pos]
    return key_cache, value_cache


def _sdpa_golden(q_block, k_visible, v_visible, scale):
    """Independent non-causal attention golden via fused SDPA.

    Shapes: q ``[Q, H, D]``, k/v ``[N, Hkv, D]`` -> ``[Q, H, D]``.
    """
    num_heads = q_block.shape[1]
    num_kv_heads = k_visible.shape[1]
    if num_heads != num_kv_heads:
        group = num_heads // num_kv_heads
        k_visible = k_visible.repeat_interleave(group, dim=1)
        v_visible = v_visible.repeat_interleave(group, dim=1)
    q = q_block.permute(1, 0, 2).unsqueeze(0).float()  # [1, H, Q, D]
    k = k_visible.permute(1, 0, 2).unsqueeze(0).float()
    v = v_visible.permute(1, 0, 2).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
    return out.squeeze(0).permute(1, 0, 2).to(q_block.dtype)  # [Q, H, D]


def _rand(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype)


def test_nd_format_normalization_is_noop_on_cpu():
    tensor = _rand(2, 3)
    assert _to_nd_format(tensor) is tensor


# ---------------------------------------------------------------------------
# 1. Reference matches an independent dense (SDPA) implementation.
# ---------------------------------------------------------------------------
def test_reference_matches_dense_full_context():
    torch.manual_seed(0)
    heads, kv_heads, head_dim = 4, 4, 8
    ctx_len, query_len = 10, 8
    scale = head_dim**-0.5
    cache_block_size = 4
    num_blocks = 8

    k_ctx = _rand(ctx_len, kv_heads, head_dim)
    v_ctx = _rand(ctx_len, kv_heads, head_dim)
    block_table = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(k_ctx, v_ctx)], block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.float32
    )

    query = _rand(query_len, heads, head_dim)
    key = _rand(query_len, kv_heads, head_dim)
    value = _rand(query_len, kv_heads, head_dim)
    seq_len = ctx_len + query_len
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )

    k_vis = torch.cat([k_ctx, key], dim=0)
    v_vis = torch.cat([v_ctx, value], dim=0)
    golden = _sdpa_golden(query, k_vis, v_vis, scale)
    assert torch.allclose(out, golden, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Draft block is non-causal: the first query token sees a future token.
# ---------------------------------------------------------------------------
def test_draft_block_non_causal_sees_future_token():
    torch.manual_seed(1)
    heads = kv_heads = 2
    head_dim = 8
    scale = head_dim**-0.5

    # No context; draft block only. If attention were causal, query 0 could not
    # see query 1's value. We make query1's V distinctive and check it leaks in.
    query = _rand(2, heads, head_dim)
    key = _rand(2, kv_heads, head_dim)
    value = torch.zeros(2, kv_heads, head_dim)
    value[1] = 100.0  # future token carries a large, unmistakable signal

    key_cache = torch.zeros(1, 4, kv_heads, head_dim)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)  # prefix_len = 0

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size=4,
    )
    # Query 0's output must be influenced by the future token's value.
    assert out[0].abs().max() > 1.0


# ---------------------------------------------------------------------------
# 3. Two requests are isolated: each attends only to its own cache/block.
# ---------------------------------------------------------------------------
def test_multi_request_kv_isolation():
    torch.manual_seed(2)
    heads = kv_heads = 2
    head_dim = 8
    scale = head_dim**-0.5
    cache_block_size = 4
    num_blocks = 8
    ctx_len = 6
    query_len = 2

    ka, va = _rand(ctx_len, kv_heads, head_dim), _rand(ctx_len, kv_heads, head_dim)
    kb, vb = _rand(ctx_len, kv_heads, head_dim), _rand(ctx_len, kv_heads, head_dim)
    # Distinct, non-overlapping physical blocks per request.
    block_table = torch.tensor([[0, 1], [4, 5]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(ka, va), (kb, vb)], block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.float32
    )

    query = _rand(2 * query_len, heads, head_dim)
    key = _rand(2 * query_len, kv_heads, head_dim)
    value = _rand(2 * query_len, kv_heads, head_dim)
    query_start_loc = torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32)
    seq_lens = torch.tensor([ctx_len + query_len, ctx_len + query_len], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )

    # Request A golden uses only A's context; request B only B's.
    gold_a = _sdpa_golden(
        query[:query_len], torch.cat([ka, key[:query_len]]), torch.cat([va, value[:query_len]]), scale
    )
    gold_b = _sdpa_golden(
        query[query_len:], torch.cat([kb, key[query_len:]]), torch.cat([vb, value[query_len:]]), scale
    )
    assert torch.allclose(out[:query_len], gold_a, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out[query_len:], gold_b, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. Non-contiguous physical blocks in the block table.
# ---------------------------------------------------------------------------
def test_non_contiguous_block_table():
    torch.manual_seed(3)
    heads = kv_heads = 2
    head_dim = 8
    scale = head_dim**-0.5
    cache_block_size = 2
    num_blocks = 16
    ctx_len = 6
    query_len = 4

    k_ctx, v_ctx = _rand(ctx_len, kv_heads, head_dim), _rand(ctx_len, kv_heads, head_dim)
    # Scrambled, non-adjacent physical blocks.
    block_table = torch.tensor([[7, 2, 11, 0]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(k_ctx, v_ctx)], block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.float32
    )

    query = _rand(query_len, heads, head_dim)
    key = _rand(query_len, kv_heads, head_dim)
    value = _rand(query_len, kv_heads, head_dim)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([ctx_len + query_len], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )
    golden = _sdpa_golden(query, torch.cat([k_ctx, key]), torch.cat([v_ctx, value]), scale)
    assert torch.allclose(out, golden, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 5. Padding (-1) slots are never gathered.
# ---------------------------------------------------------------------------
def test_padding_slots_are_dropped():
    kv_heads, head_dim = 2, 4
    cache = torch.arange(4 * 2 * kv_heads * head_dim, dtype=torch.float32).reshape(
        4, 2, kv_heads, head_dim
    )
    slot_ids = torch.tensor([0, -1, 3, -1, 5], dtype=torch.int32)
    gathered = gather_paged_kv(cache, slot_ids, cache_block_size=2)
    assert gathered.shape[0] == 3  # only slots 0, 3, 5 survive
    flat = cache.reshape(-1, kv_heads, head_dim)
    assert torch.equal(gathered[0], flat[0])
    assert torch.equal(gathered[1], flat[3])
    assert torch.equal(gathered[2], flat[5])
    # All-invalid returns an empty, correctly-shaped tensor.
    empty = gather_paged_kv(cache, torch.tensor([-1, -1], dtype=torch.int32), 2)
    assert empty.shape == (0, kv_heads, head_dim)


# ---------------------------------------------------------------------------
# 6. Different context lengths across requests.
# ---------------------------------------------------------------------------
def test_different_context_lengths():
    torch.manual_seed(4)
    heads = kv_heads = 2
    head_dim = 8
    scale = head_dim**-0.5
    cache_block_size = 4
    num_blocks = 16
    query_len = 2
    ctx_a, ctx_b = 3, 9

    ka, va = _rand(ctx_a, kv_heads, head_dim), _rand(ctx_a, kv_heads, head_dim)
    kb, vb = _rand(ctx_b, kv_heads, head_dim), _rand(ctx_b, kv_heads, head_dim)
    block_table = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(ka, va), (kb, vb)], block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.float32
    )

    query = _rand(2 * query_len, heads, head_dim)
    key = _rand(2 * query_len, kv_heads, head_dim)
    value = _rand(2 * query_len, kv_heads, head_dim)
    query_start_loc = torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32)
    seq_lens = torch.tensor([ctx_a + query_len, ctx_b + query_len], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )
    gold_a = _sdpa_golden(query[:query_len], torch.cat([ka, key[:query_len]]), torch.cat([va, value[:query_len]]), scale)
    gold_b = _sdpa_golden(query[query_len:], torch.cat([kb, key[query_len:]]), torch.cat([vb, value[query_len:]]), scale)
    assert torch.allclose(out[:query_len], gold_a, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out[query_len:], gold_b, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 7. Block sizes 1, 2, 8 (varying num_speculative_tokens).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query_len", [1, 2, 8])
def test_various_block_sizes(query_len):
    torch.manual_seed(5 + query_len)
    heads = kv_heads = 2
    head_dim = 8
    scale = head_dim**-0.5
    cache_block_size = 8
    num_blocks = 8
    ctx_len = 5

    k_ctx, v_ctx = _rand(ctx_len, kv_heads, head_dim), _rand(ctx_len, kv_heads, head_dim)
    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(k_ctx, v_ctx)], block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.float32
    )

    query = _rand(query_len, heads, head_dim)
    key = _rand(query_len, kv_heads, head_dim)
    value = _rand(query_len, kv_heads, head_dim)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([ctx_len + query_len], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )
    golden = _sdpa_golden(query, torch.cat([k_ctx, key]), torch.cat([v_ctx, value]), scale)
    assert torch.allclose(out, golden, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 8. BF16 inputs, FP32 softmax stability.
# ---------------------------------------------------------------------------
def test_bf16_inputs_fp32_softmax():
    torch.manual_seed(6)
    heads = kv_heads = 4
    head_dim = 16
    scale = head_dim**-0.5
    cache_block_size = 4
    num_blocks = 8
    ctx_len, query_len = 8, 4

    k_ctx = _rand(ctx_len, kv_heads, head_dim, dtype=torch.bfloat16)
    v_ctx = _rand(ctx_len, kv_heads, head_dim, dtype=torch.bfloat16)
    block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(k_ctx, v_ctx)], block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.bfloat16
    )

    query = _rand(query_len, heads, head_dim, dtype=torch.bfloat16)
    key = _rand(query_len, kv_heads, head_dim, dtype=torch.bfloat16)
    value = _rand(query_len, kv_heads, head_dim, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([ctx_len + query_len], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out.float()).any()
    golden = _sdpa_golden(query, torch.cat([k_ctx, key]), torch.cat([v_ctx, value]), scale)
    # BF16 tolerance.
    assert torch.allclose(out.float(), golden.float(), atol=8e-2, rtol=8e-2)


# ---------------------------------------------------------------------------
# 9. GQA: num_kv_heads < num_heads.
# ---------------------------------------------------------------------------
def test_grouped_query_attention():
    torch.manual_seed(7)
    heads, kv_heads = 8, 2
    head_dim = 8
    scale = head_dim**-0.5
    cache_block_size = 4
    num_blocks = 8
    ctx_len, query_len = 6, 4

    k_ctx, v_ctx = _rand(ctx_len, kv_heads, head_dim), _rand(ctx_len, kv_heads, head_dim)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(k_ctx, v_ctx)], block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.float32
    )

    query = _rand(query_len, heads, head_dim)
    key = _rand(query_len, kv_heads, head_dim)
    value = _rand(query_len, kv_heads, head_dim)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    seq_lens = torch.tensor([ctx_len + query_len], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )
    golden = _sdpa_golden(query, torch.cat([k_ctx, key]), torch.cat([v_ctx, value]), scale)
    assert torch.allclose(out, golden, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 10. Multi-request output equals a per-request loop.
# ---------------------------------------------------------------------------
def test_batched_equals_per_request_loop():
    torch.manual_seed(8)
    heads = kv_heads = 2
    head_dim = 8
    scale = head_dim**-0.5
    cache_block_size = 4
    num_blocks = 24
    query_len = 3
    ctx_lens = [4, 7, 2]

    per_req = [(_rand(c, kv_heads, head_dim), _rand(c, kv_heads, head_dim)) for c in ctx_lens]
    block_table = torch.tensor([[0, 1], [4, 5], [8, 9]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        per_req, block_table, num_blocks, cache_block_size, kv_heads, head_dim, torch.float32
    )

    n = len(ctx_lens)
    query = _rand(n * query_len, heads, head_dim)
    key = _rand(n * query_len, kv_heads, head_dim)
    value = _rand(n * query_len, kv_heads, head_dim)
    query_start_loc = torch.arange(0, (n + 1) * query_len, query_len, dtype=torch.int32)
    seq_lens = torch.tensor([c + query_len for c in ctx_lens], dtype=torch.int32)

    out = qwen3_dspark_reference_attention(
        query, key, value, key_cache, value_cache, block_table,
        query_start_loc, seq_lens, scale, cache_block_size,
    )

    for r in range(n):
        rows = slice(r * query_len, (r + 1) * query_len)
        single = qwen3_dspark_reference_attention(
            query[rows], key[rows], value[rows], key_cache, value_cache,
            block_table[r : r + 1], torch.tensor([0, query_len], dtype=torch.int32),
            seq_lens[r : r + 1], scale, cache_block_size,
        )
        assert torch.allclose(out[rows], single, atol=1e-6)


# ---------------------------------------------------------------------------
# Extra: attention sink bias lowers all attention weights deterministically.
# ---------------------------------------------------------------------------
def test_attention_sink_reduces_weight_mass():
    torch.manual_seed(9)
    heads = kv_heads = 2
    head_dim = 8
    scale = head_dim**-0.5
    q_block = _rand(2, heads, head_dim)
    k_vis = _rand(5, kv_heads, head_dim)
    v_vis = _rand(5, kv_heads, head_dim)

    no_sink = dspark_mha_reference(q_block, k_vis, v_vis, scale, attn_sink=None)
    big_sink = dspark_mha_reference(
        q_block, k_vis, v_vis, scale, attn_sink=torch.full((heads,), 50.0)
    )
    # A huge sink logit dominates the denominator, shrinking the output toward 0.
    assert big_sink.abs().max() < no_sink.abs().max()


# ---------------------------------------------------------------------------
# Extra: gather_context_kv_from_cache round-trips exact rows.
# ---------------------------------------------------------------------------
def test_gather_context_roundtrip():
    kv_heads, head_dim = 2, 4
    cache_block_size = 2
    k_ctx = _rand(5, kv_heads, head_dim)
    v_ctx = _rand(5, kv_heads, head_dim)
    block_table = torch.tensor([[3, 1, 6]], dtype=torch.int32)
    key_cache, value_cache = _build_paged_cache(
        [(k_ctx, v_ctx)], block_table, 8, cache_block_size, kv_heads, head_dim, torch.float32
    )
    k_g, v_g = gather_context_kv_from_cache(
        key_cache, value_cache, block_table[0], 0, 5, cache_block_size
    )
    assert torch.equal(k_g, k_ctx)
    assert torch.equal(v_g, v_ctx)
    # Windowed sub-range.
    k_w, _ = gather_context_kv_from_cache(
        key_cache, value_cache, block_table[0], 2, 5, cache_block_size
    )
    assert torch.equal(k_w, k_ctx[2:5])
