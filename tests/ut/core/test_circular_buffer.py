# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendCircularBufferSpec,
    AscendIndexerKPoolTailSpec,
    is_circular_buffer_kv_cache_spec,
    is_circular_kv_cache_spec,
    is_prefix_cacheable,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.block_table import BlockTable


def ring_spec():
    return CircularBufferSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=1024,
        head_size_v=0,
        dtype=torch.float32,
    )


def kpool_tail_spec():
    return AscendIndexerKPoolTailSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
        compress_ratio=16,
        sliding_window=16,
    )


def real_block_table(kv_cache_spec, block_size, kernel_sizes=None):
    """Build a BlockTable off-device by stubbing the distributed DCP lookup."""
    with patch(
        "vllm_ascend.worker.block_table.get_dcp_group",
        return_value=SimpleNamespace(world_size=1, rank_in_group=0),
    ):
        return BlockTable(
            block_size=block_size,
            max_num_reqs=4,
            max_num_blocks_per_req=8,
            max_num_batched_tokens=256,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=kernel_sizes,
            kv_cache_group=KVCacheGroupSpec(["g"], kv_cache_spec),
        )


def test_ring_lifetime_reuse_and_external_tokens():
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=32)
    manager = CircularBufferManager(
        ring_spec(),
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=128,
        needs_kv_cache_zeroing=True,
    )
    free = pool.get_num_free_blocks()
    assert manager.get_num_blocks_to_allocate("a", 129, [], 0, 0, 129) == 1
    a = manager.allocate_new_blocks("a", 129, 129)
    b = manager.allocate_new_blocks("b", 1, 1)
    assert len(a) == len(b) == 1 and a[0].block_id != b[0].block_id
    assert not a[0].is_null and pool.get_num_free_blocks() == free - 2
    for tokens in (130, 1024, 65536):
        assert manager.get_num_blocks_to_allocate("a", tokens, [], tokens - 1, 0, tokens) == 0
        assert manager.allocate_new_blocks("a", tokens, tokens) == []
        manager.allocate_external_computed_blocks("a", 0, tokens)
        manager.remove_skipped_blocks("a", tokens)
        manager.cache_blocks(
            SimpleNamespace(request_id="a"),
            tokens,
            **({} if vllm_version_is("0.29.0") else {"replay_boundaries": [tokens - 1]}),
        )
        assert manager.req_to_blocks["a"] == a
    assert manager.take_new_block_ids() == []
    assert manager.get_num_common_prefix_blocks("a") == manager.get_num_skipped_tokens(65536) == 0
    manager.free("a")  # Finish or preempt, then resume under a new allocation.
    manager.free("b")
    assert pool.get_num_free_blocks() == free
    manager.allocate_external_computed_blocks("a", 0, 65536)
    assert len(manager.req_to_blocks["a"]) == 1
    manager.free("a")
    assert pool.get_num_free_blocks() == free


def test_uniform_properties_and_single_plane_size():
    spec = ring_spec()
    uniform = UniformTypeKVCacheSpecs(block_size=32, kv_cache_specs={"s0": spec, "s1": spec})
    assert spec.page_size_bytes == 131072
    assert spec.max_memory_usage_bytes(None) == 131072
    assert uniform.max_num_blocks_per_req(None, 65536) == 1
    assert is_circular_kv_cache_spec(uniform) and not is_prefix_cacheable(uniform)
    assert not uniform.prefix_cacheable
    assert not is_prefix_cacheable(SimpleNamespace(participates_in_prefix_caching=False))


def test_scratch_groups_do_not_reduce_prefix_hits_or_truncation():
    full = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.float32)
    scratch = ring_spec()
    groups = [KVCacheGroupSpec(["kv"], full), KVCacheGroupSpec(["state"], scratch)]
    coordinator = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        single_type_managers=[SimpleNamespace(), SimpleNamespace()],
        eagle_group_ids=set(),
        scheduler_block_size=128,
        _get_effective_block_size=lambda spec: spec.block_size,
    )
    AscendHybridKVCacheCoordinator.verify_and_split_kv_cache_groups(coordinator)
    assert len(coordinator.attention_groups) == 1
    assert coordinator.attention_groups[0].group_ids == [0]
    host = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        coordinator=SimpleNamespace(
            single_type_managers=[SimpleNamespace(block_size=128), SimpleNamespace(block_size=32)]
        ),
        create_kv_cache_blocks=lambda blocks: blocks,
    )
    assert KVCacheManager.truncate_computed_blocks(host, SimpleNamespace(blocks=([1, 2], [])), 128) == ([1], [])

    mamba = MambaSpec(block_size=128, shapes=((1,),), dtypes=(torch.float32,))
    host.kv_cache_config.kv_cache_groups = [groups[0], KVCacheGroupSpec(["mamba"], mamba), groups[1]]
    host.coordinator.single_type_managers = [
        SimpleNamespace(block_size=128),
        SimpleNamespace(block_size=128),
        SimpleNamespace(block_size=32),
    ]
    assert KVCacheManager.truncate_computed_blocks(host, SimpleNamespace(blocks=([1, 2], [], [])), 128) == ([1], [], [])


@pytest.mark.parametrize("draft", [False, True])
def test_ring_bypasses_position_to_page_mapping(draft):
    mapping = torch.zeros(8, dtype=torch.int64)
    table = SimpleNamespace(is_circular_group=True, slot_mapping=SimpleNamespace(gpu=mapping))
    if draft:
        BlockTable.compute_slot_mapping_draft(table, Mock(), Mock())
    else:
        BlockTable.compute_slot_mapping(table, 1, Mock(), Mock())
    assert mapping.tolist() == [-1] * 8


def test_ring_ownership_predicate_excludes_kpool_tail():
    """Only ``CircularBufferSpec`` groups own their addressing (#17532)."""
    ring = ring_spec()
    tail = kpool_tail_spec()
    # Both report circular addressing, so both keep the ring math.
    assert is_circular_kv_cache_spec(ring) and is_circular_kv_cache_spec(tail)
    # Only the ring buffer wants the all-PAD slot mapping from generic writers.
    assert is_circular_buffer_kv_cache_spec(ring)
    assert not is_circular_buffer_kv_cache_spec(tail)
    assert is_circular_buffer_kv_cache_spec(
        AscendCircularBufferSpec(block_size=32, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32)
    )
    # Uniform wrappers resolve recursively, mirroring ``is_circular_kv_cache_spec``.
    assert is_circular_buffer_kv_cache_spec(UniformTypeKVCacheSpecs(block_size=32, kv_cache_specs={"s": ring}))
    assert not is_circular_buffer_kv_cache_spec(UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"t": tail}))
    assert not is_circular_buffer_kv_cache_spec(
        UniformTypeKVCacheSpecs(block_size=32, kv_cache_specs={"r": ring, "t": tail})
    )
    assert not is_circular_buffer_kv_cache_spec(SimpleNamespace())


def test_kpool_tail_group_keeps_ring_slot_mapping():
    """The KPool tail still consumes the slot mapping, so it must not pad."""
    table = real_block_table(kpool_tail_spec(), block_size=128, kernel_sizes=[128])
    # Circular addressing is retained for the writer...
    assert table.is_circular
    # ...but the PAD short-circuit must not swallow its slot mapping.
    assert not table.is_circular_group

    table.add_row([7], row_idx=0)
    table.slot_mapping.np[:] = -1
    table.compute_slot_mapping_draft(np.array([0, 0]), np.array([0, 1]))
    # Ring math resolves real slots instead of PAD-ing the tail cache dry.
    assert table.slot_mapping.np[:2].tolist() == [7 * 128, 7 * 128 + 1]


def test_circular_buffer_group_still_pads():
    """Genuine ring buffers keep the #16925 all-PAD short-circuit."""
    table = real_block_table(ring_spec(), block_size=32)
    assert table.is_circular and table.is_circular_group
