# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import vllm.v1.kv_cache_interface as kv_cache_interface
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import MambaManager, register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.patch.platform import patch_mamba_midblock_resume as mod
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator

HASH_BLOCK_SIZE = 128
MAMBA_BLOCK_SIZE = 256
CHECKPOINT_ALIGNMENT = 64


def _make_hybrid_coordinator(
    *,
    num_blocks: int = 32,
    retention_interval: int | None = None,
    has_checkpoint_blocks: bool = True,
) -> tuple[AscendHybridKVCacheCoordinator, MambaManager]:
    mod.apply_patch()
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    fa_spec = FullAttentionSpec(block_size=HASH_BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    mamba_spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1 if has_checkpoint_blocks else 0,
        prefill_checkpoint_alignment=CHECKPOINT_ALIGNMENT if has_checkpoint_blocks else None,
    )
    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["fa"], fa_spec),
            KVCacheGroupSpec(["mamba"], mamba_spec),
        ],
        prefix_cache_retention_interval=retention_interval,
    )
    coordinator = AscendHybridKVCacheCoordinator(
        kv_cache_config=kv_config,
        max_model_len=4096,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=HASH_BLOCK_SIZE,
        scheduler_block_size=HASH_BLOCK_SIZE,
        max_in_flight_tokens=4096,
    )
    mamba_mgr = coordinator.single_type_managers[1]
    assert isinstance(mamba_mgr, MambaManager)
    return coordinator, mamba_mgr


def _block_hashes(num_tokens: int) -> list[bytes]:
    return [f"h-{i:030d}".encode() for i in range(num_tokens // HASH_BLOCK_SIZE)]


@pytest.mark.parametrize(
    ("query_start", "query_end", "checkpoint_position", "alignment", "expected"),
    [
        # Mid-block resume at 128 -> chunk end 450 -> checkpoint 384 (col 0 = query_start // 256).
        (128, 450, 384, 64, True),
        # Mid-block resume within single block [128, 300): checkpoint 256 -> col 0.
        (128, 300, 256, 64, True),
        (128, 450, 384, 128, True),
        # Unaligned checkpoint position relative to query_start.
        (128, 300, 256, 256, False),
        # Too short to cover one hash block after query_start.
        (128, 250, 192, 64, False),
        # Disabled backend alignment.
        (128, 450, 384, None, False),
    ],
)
def test_is_mamba_prefill_checkpoint_valid_allows_midblock_resume(
    query_start: int,
    query_end: int,
    checkpoint_position: int,
    alignment: int | None,
    expected: bool,
) -> None:
    mod.apply_mamba_midblock_checkpoint_rule()
    assert (
        kv_cache_interface.is_mamba_prefill_checkpoint_valid(
            query_start=query_start,
            query_end=query_end,
            checkpoint_position=checkpoint_position,
            hash_block_size=HASH_BLOCK_SIZE,
            mamba_block_size=MAMBA_BLOCK_SIZE,
            checkpoint_alignment=alignment,
        )
        is expected
    )


def test_midblock_resume_uses_resumed_block_for_checkpoint_and_masks_hash() -> None:
    coordinator, mamba_mgr = _make_hybrid_coordinator()
    req_b = SimpleNamespace(
        request_id="req-b",
        block_hashes=_block_hashes(512),
        num_prompt_tokens=450,
        num_tokens=450,
        num_computed_tokens=450,
        num_in_flight_tokens=0,
        shared_prefix_boundary=0,
    )
    # Step 1: allocate [0, 128) so block 0 exists as a private resumed block.
    coordinator.allocate_new_blocks(req_b.request_id, 128, 128)
    assert len(mamba_mgr.req_to_blocks[req_b.request_id]) == 1

    # Step 2: resume from 128 (mid-block in block 0) to prefill_end=450.
    needed = mamba_mgr.get_num_blocks_to_allocate(
        request_id=req_b.request_id,
        num_tokens=450,
        new_computed_blocks=(),
        total_computed_tokens=128,
        num_local_computed_tokens=128,
        num_tokens_main_model=450,
        apply_admission_cap=False,
        prefill_end=450,
    )
    assert needed >= 1
    assert mamba_mgr._checkpoints.get(req_b.request_id) == (384, 0)

    coordinator.allocate_new_blocks(req_b.request_id, 450, 450)
    coordinator.cache_blocks(req_b, 450)

    # Block 0 holds an off-boundary checkpoint state (384 != 256), so it must
    # be keyed at checkpoint position 384 and never receive the full-block
    # hash for position 256.
    assert mamba_mgr._mid_block_states[req_b.request_id][0] == 384
    assert mamba_mgr._refine_block_mask(req_b, 0, 1, None) == [False]
    block0 = mamba_mgr.req_to_blocks[req_b.request_id][0]
    assert block0.block_hash_num_tokens == 384
    hit_384 = coordinator.block_pool.get_cached_block(req_b.block_hashes[2], [1])
    assert hit_384 is not None and hit_384[0] is block0
    assert coordinator.block_pool.get_cached_block(req_b.block_hashes[1], [1]) is None

    mamba_mgr.pop_blocks_for_free(req_b.request_id)
    assert req_b.request_id not in mamba_mgr._mid_block_states
    assert req_b.request_id not in mamba_mgr._req_chunk_ranges


def test_midblock_start_run_past_masks_block_without_checkpoint_blocks() -> None:
    coordinator, mamba_mgr = _make_hybrid_coordinator(has_checkpoint_blocks=False)
    req = SimpleNamespace(
        request_id="req-run-past",
        block_hashes=_block_hashes(512),
        num_prompt_tokens=300,
        num_tokens=300,
        num_computed_tokens=300,
        num_in_flight_tokens=0,
        shared_prefix_boundary=0,
    )
    coordinator.allocate_new_blocks(req.request_id, 128, 128)
    mamba_mgr.get_num_blocks_to_allocate(
        request_id=req.request_id,
        num_tokens=300,
        new_computed_blocks=(),
        total_computed_tokens=128,
        num_local_computed_tokens=128,
        num_tokens_main_model=300,
        apply_admission_cap=False,
        prefill_end=300,
    )
    coordinator.allocate_new_blocks(req.request_id, 300, 300)
    coordinator.cache_blocks(req, 300)

    assert mamba_mgr._mid_block_states[req.request_id][0] == 128
    assert mamba_mgr._refine_block_mask(req, 0, 1, None) == [False]
    assert mamba_mgr.req_to_blocks[req.request_id][0].block_hash is None


def test_sparse_decode_boundary_state_held_and_cached_on_free() -> None:
    coordinator, mamba_mgr = _make_hybrid_coordinator(retention_interval=0)
    assert mamba_mgr.keep_decode_resume_state is True

    req = SimpleNamespace(
        request_id="decode-turn-1",
        block_hashes=_block_hashes(512),
        num_prompt_tokens=100,
        num_tokens=260,
        num_computed_tokens=260,
        num_in_flight_tokens=0,
        shared_prefix_boundary=0,
    )
    coordinator.allocate_new_blocks(req.request_id, 100, 100)
    coordinator.new_step_starts()

    # Decode crosses block 0 boundary at 256 (outside prompt=100).
    coordinator.allocate_new_blocks(req.request_id, 260, 260)
    assert len(mamba_mgr.req_to_blocks[req.request_id]) >= 2
    block0 = mamba_mgr.req_to_blocks[req.request_id][0]
    assert not block0.is_null

    mamba_mgr.remove_skipped_blocks(req.request_id, processed_computed_tokens=260, num_prompt_tokens=100)
    assert not mamba_mgr.req_to_blocks[req.request_id][0].is_null

    coordinator.cache_blocks(req, 256)
    assert block0.block_hash is None

    coordinator.free(req)
    assert block0.block_hash is not None


def test_coordinator_and_kv_manager_free_support_request_and_legacy_str() -> None:
    coordinator, mamba_mgr = _make_hybrid_coordinator(retention_interval=512)
    kv_mgr = KVCacheManager.__new__(KVCacheManager)
    kv_mgr.coordinator = coordinator

    req = SimpleNamespace(
        request_id="decode-turn-2",
        block_hashes=_block_hashes(512),
        num_prompt_tokens=64,
        num_tokens=280,
        num_computed_tokens=290,
        num_in_flight_tokens=10,
        shared_prefix_boundary=0,
    )
    coordinator.allocate_new_blocks(req.request_id, 64, 64)
    coordinator.new_step_starts()
    coordinator.allocate_new_blocks(req.request_id, 260, 260)
    target_block = mamba_mgr.req_to_blocks[req.request_id][0]
    assert not target_block.is_null
    mamba_mgr.remove_skipped_blocks(req.request_id, processed_computed_tokens=280, num_prompt_tokens=64)
    coordinator.cache_blocks(req, 280)
    assert target_block.block_hash is None

    kv_mgr.free(req)
    assert target_block.block_hash is not None

    # Legacy str argument remains accepted by coordinator.free / pop_blocks_for_free.
    coordinator.allocate_new_blocks("legacy-req", 128, 128)
    popped = coordinator.pop_blocks_for_free("legacy-req")
    assert isinstance(popped, list)
    coordinator.free("legacy-req-2")
