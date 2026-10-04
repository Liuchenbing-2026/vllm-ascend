# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 16])
@pytest.mark.parametrize("strided", [False, True])
def test_draft_tail_mask_matches_exact_attention(causal, window, strided):
    """Compare the real paged BSND operator with FP32 attention over exact KV.

    Reusing the same shapes with different lengths catches stale mask reuse.
    The rejected tail contains large values to expose incomplete masking.
    """
    torch.manual_seed(16271)
    batch, query_len, heads, kv_heads, head_size, block_size = 2, 9, 4, 2, 128, 128
    query = torch.randn(batch, query_len, heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(batch, block_size, kv_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    device_query = query.to("npu")
    block_table = torch.arange(batch, dtype=torch.int32).reshape(batch, 1).to("npu")
    impl = SimpleNamespace(
        sinks=None,
        sliding_window=window,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=head_size,
        scale=head_size**-0.5,
    )
    for lengths in ([31, 55], [55, 31]):
        step_key, step_value = key.clone(), value.clone()
        bounds = [length + query_len - 1 for length in lengths]
        for req, length in enumerate(lengths):
            step_key[req, length:] = 20
            step_value[req, length:] = 100
        backing = torch.stack((step_key, step_value), dim=1).to("npu")
        device_key, device_value = backing[:, 0].flatten(2), backing[:, 1].flatten(2)
        if not strided:
            device_key, device_value = device_key.contiguous(), device_value.contiguous()
        before = backing.clone()
        metadata = SimpleNamespace(
            draft_kv_upper_bound=True,
            draft_query_lens=[query_len] * batch,
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
            draft_tail_mask_cache={},
            causal=causal,
        )
        output = torch.empty_like(device_query).reshape(batch * query_len, heads, head_size)
        AscendAttentionBackendImpl._forward_draft_tail_masked(
            impl,
            device_query.reshape_as(output),
            device_key,
            device_value,
            metadata,
            block_table,
            block_size,
            bounds,
            batch * query_len,
            output,
        )
        references = []
        for req, length in enumerate(lengths):
            keys = step_key[req, :length].float().repeat_interleave(heads // kv_heads, dim=1).permute(1, 0, 2)
            values = step_value[req, :length].float().repeat_interleave(heads // kv_heads, dim=1).permute(1, 0, 2)
            queries = query[req].float().permute(1, 0, 2)
            scores = queries @ keys.transpose(1, 2) * impl.scale
            last = torch.arange(query_len) + length - query_len if causal else torch.full((query_len,), length - 1)
            visible = torch.arange(length).view(1, length) <= last.view(query_len, 1)
            if window is not None:
                visible &= torch.arange(length).view(1, length) >= last.view(query_len, 1) - window
            scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
            references.append((scores.softmax(-1) @ values).permute(1, 0, 2))
        reference = torch.stack(references)
        actual = output.cpu().float().reshape_as(reference)
        # BF16 operator versus FP32 reference: relative L2 bound fixed before running.
        relative_l2_tolerance = 0.02
        assert (actual - reference).norm() / reference.norm() < relative_l2_tolerance
        torch.testing.assert_close(backing, before, rtol=0, atol=0)
