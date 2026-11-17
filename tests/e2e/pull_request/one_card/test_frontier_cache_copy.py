# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device coverage for restored recurrent tails; scheduler contracts use UTs."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.core.kv_cache_interface import AscendFrontierKPoolTailSpec
from vllm_ascend.models.glm5next.cache_views import view_kpool_tail_cache
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace


@pytest.mark.parametrize("padding", [0, 8])
@pytest.mark.parametrize("duplicates", [1, 2])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_frontier_tail_copy_preserves_state_and_padding(padding, duplicates, use_graph):
    spec = AscendFrontierKPoolTailSpec(
        block_size=4,
        compress_ratio=4,
        ring_capacity=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=4,
    )
    tail_elements = spec.unpadded_page_size_bytes // 4
    width = tail_elements + padding
    num_blocks = 4
    storage = torch.zeros(num_blocks * width, dtype=torch.float32, device="npu")
    views = view_kpool_tail_cache("tail", spec, storage.view(torch.int8), num_blocks)
    copies = [KVCacheBlockCopy(src_block_id=1, dst_block_id=2), KVCacheBlockCopy(src_block_id=2, dst_block_id=1)]
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        copy_kv_cache_blocks_inplace(views * duplicates, num_blocks, copies)
    stream.synchronize()
    graph = torch.npu.NPUGraph() if use_graph else None
    if graph is not None:
        with torch.npu.graph(graph, stream=stream):
            copy_kv_cache_blocks_inplace(views * duplicates, num_blocks, copies)
    for offset in (1, 7, 19):
        original = torch.arange(num_blocks * width, dtype=torch.float32).reshape(num_blocks, width) + offset
        storage.copy_(original.flatten())
        if graph is not None:
            graph.replay()
        else:
            copy_kv_cache_blocks_inplace(views * duplicates, num_blocks, copies)
        torch.npu.synchronize()
        expected = original.clone()
        expected[1, :tail_elements] = original[2, :tail_elements]
        expected[2, :tail_elements] = original[1, :tail_elements]
        torch.testing.assert_close(storage.cpu().reshape(num_blocks, width), expected, rtol=0, atol=0)
