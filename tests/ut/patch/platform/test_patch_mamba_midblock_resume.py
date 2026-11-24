# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator_mod
import vllm.v1.core.kv_cache_manager as kv_cache_manager_mod
import vllm.v1.core.single_type_kv_cache_manager as st_mgr
import vllm.v1.kv_cache_interface as kv_cache_interface
from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.patch.platform import patch_mamba_midblock_resume as mod
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

HASH_BLOCK_SIZE = 128
MAMBA_BLOCK_SIZE = 256
CHECKPOINT_ALIGNMENT = 64


def _make_hybrid_coordinator(
    *,
    num_blocks: int = 32,
    retention_interval: int | None = None,
    has_checkpoint_blocks: bool = True,
    mamba_cache_mode: str = "align",
    enable_caching: bool = True,
    enable_kv_cache_events: bool = False,
) -> tuple[AscendHybridKVCacheCoordinator, MambaManager]:
    mod.apply_patch()
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    fa_spec = FullAttentionSpec(block_size=HASH_BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    spec_fields = {f.name for f in fields(MambaSpec)}
    mamba_kwargs: dict[str, Any] = {
        "block_size": MAMBA_BLOCK_SIZE,
        "shapes": ((1,),),
        "dtypes": (torch.float32,),
        "num_prefill_checkpoint_blocks": 1 if has_checkpoint_blocks else 0,
        "prefill_checkpoint_alignment": CHECKPOINT_ALIGNMENT if has_checkpoint_blocks else None,
    }
    if "mamba_cache_mode" in spec_fields:
        mamba_kwargs["mamba_cache_mode"] = mamba_cache_mode
    mamba_spec = MambaSpec(**mamba_kwargs)
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
        enable_caching=enable_caching,
        enable_kv_cache_events=enable_kv_cache_events,
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
    assert getattr(mamba_mgr, "keep_decode_resume_state", False) is True
    assert getattr(coordinator, "_ascend_has_decode_retention_mamba", False) is True

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


def test_non_mamba_and_non_align_have_zero_side_effects() -> None:
    mod.apply_patch()
    register_all_kvcache_specs(None)
    # SingleTypeKVCacheManager.__init__ itself must remain unpatched.
    assert not getattr(SingleTypeKVCacheManager.__init__, mod._DECODE_RETENTION_ATTR, False)

    # Full-attention-only coordinator/manager with retention_interval set: zero Mamba retention work.
    fa_spec = FullAttentionSpec(block_size=HASH_BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    fa_cfg = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["fa"], fa_spec)],
        prefix_cache_retention_interval=0,
    )
    fa_coord = UnitaryKVCacheCoordinator(
        kv_cache_config=fa_cfg,
        max_model_len=2048,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=HASH_BLOCK_SIZE,
        scheduler_block_size=HASH_BLOCK_SIZE,
        max_in_flight_tokens=2048,
    )
    fa_mgr = fa_coord.single_type_managers[0]
    assert isinstance(fa_mgr, FullAttentionManager)
    assert not hasattr(fa_mgr, "keep_decode_resume_state")
    assert getattr(fa_coord, "_ascend_has_decode_retention_mamba", False) is False

    # Dense align Mamba (retention_interval=None): zero decode retention work.
    dense_coord, dense_mamba = _make_hybrid_coordinator(retention_interval=None, mamba_cache_mode="align")
    assert getattr(dense_mamba, "keep_decode_resume_state", False) is False
    assert getattr(dense_coord, "_ascend_has_decode_retention_mamba", False) is False

    # Non-align Mamba (mamba_cache_mode="none"): no midblock dicts and no decode retention.
    spec_fields = {f.name for f in fields(MambaSpec)}
    if "mamba_cache_mode" in spec_fields:
        none_coord, none_mamba = _make_hybrid_coordinator(
            retention_interval=0,
            has_checkpoint_blocks=False,
            mamba_cache_mode="none",
            enable_caching=False,
        )
        assert not hasattr(none_mamba, "_req_chunk_ranges")
        assert not hasattr(none_mamba, "_mid_block_states")
        assert getattr(none_mamba, "keep_decode_resume_state", False) is False
        assert getattr(none_coord, "_ascend_has_decode_retention_mamba", False) is False


def test_apply_patch_strict_noop_post_upstream_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    def _native_valid(
        query_start: int,
        query_end: int,
        checkpoint_position: int,
        hash_block_size: int,
        mamba_block_size: int,
        checkpoint_alignment: int | None,
    ) -> bool:
        return bool(checkpoint_alignment)

    class _FakeBaseMamba:
        def __init__(self, kv_cache_spec: Any, block_pool: Any, **kwargs: Any) -> None:
            pass

        def _needs_internal_checkpoint(
            self,
            request_id: str,
            query_start: int,
            query_end: int,
            checkpoint_position: int,
            prefill_end: int,
        ) -> bool:
            reuses_resumed_block = True
            return reuses_resumed_block

        def _record_mid_block_states(self, request: Any, query_start: int, query_end: int) -> None:
            return None

        def _decode_resume_block_idx(
            self,
            request_id: str,
            processed_tokens: int,
            num_prompt_tokens: int,
        ) -> int | None:
            return None

        def get_num_blocks_to_allocate(self, *args: Any, **kwargs: Any) -> int:
            return 0

        def pop_blocks_for_free(self, request_id: str) -> list[Any]:
            return []

        def cache_blocks(self, *args: Any, **kwargs: Any) -> None:
            return None

        def remove_skipped_blocks(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _FakeSingleType:
        def _refine_block_mask(self, request: Any, start_block: int, end_block: int, block_mask: Any) -> Any:
            return block_mask

        def cache_blocks_on_free(self, request: Any) -> None:
            return None

    class _FakeCoordinator:
        def __init__(self, kv_cache_config: Any, *args: Any, **kwargs: Any) -> None:
            pass

        def free(self, request: Any) -> None:
            return None

        def pop_blocks_for_free(self, request: Any) -> list[Any]:
            return []

    class _FakeKVManager:
        def free(self, request: Any) -> None:
            return None

        def pop_blocks_for_free(self, request: Any) -> list[Any]:
            return []

    orig_needs = _FakeBaseMamba._needs_internal_checkpoint
    orig_mamba_init = _FakeBaseMamba.__init__
    orig_cache_blocks = _FakeBaseMamba.cache_blocks
    orig_coord_free = _FakeCoordinator.free
    orig_mgr_free = _FakeKVManager.free

    monkeypatch.setattr(kv_cache_interface, "is_mamba_prefill_checkpoint_valid", _native_valid)
    monkeypatch.setattr(st_mgr, "SingleTypeKVCacheManager", _FakeSingleType)
    monkeypatch.setattr(kv_cache_coordinator_mod, "KVCacheCoordinator", _FakeCoordinator)
    monkeypatch.setattr(kv_cache_manager_mod, "KVCacheManager", _FakeKVManager)

    with patch.object(mod, "_resolve_base_mamba_manager_cls", return_value=_FakeBaseMamba):
        mod.apply_patch()

    assert kv_cache_interface.is_mamba_prefill_checkpoint_valid is _native_valid
    assert _FakeBaseMamba._needs_internal_checkpoint is orig_needs
    assert _FakeBaseMamba.__init__ is orig_mamba_init
    assert _FakeBaseMamba.cache_blocks is orig_cache_blocks
    assert _FakeCoordinator.free is orig_coord_free
    assert _FakeKVManager.free is orig_mgr_free


def test_apply_patch_failsafe_on_compiled_or_raising_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raising_valid(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("unexpected rule probe")

    class _OpaqueBaseMamba:
        _needs_internal_checkpoint = len  # Builtin/C callable without Python __code__

    monkeypatch.setattr(kv_cache_interface, "is_mamba_prefill_checkpoint_valid", _raising_valid)
    mod.apply_mamba_midblock_checkpoint_rule()
    assert kv_cache_interface.is_mamba_prefill_checkpoint_valid is _raising_valid

    mod._patch_mamba_midblock_needs_checkpoint(_OpaqueBaseMamba)
    assert _OpaqueBaseMamba._needs_internal_checkpoint is len


def test_mrv2_sync_align_mamba_cache_geometry_respects_rule4_boundaries() -> None:
    class _SpyCacheConfig:
        writes: list[tuple[str, Any]]

        def __init__(self, hash_block_size: int | None = None, cache_hit_alignment_tokens: int | None = None) -> None:
            object.__setattr__(self, "hash_block_size", hash_block_size)
            object.__setattr__(self, "cache_hit_alignment_tokens", cache_hit_alignment_tokens)
            object.__setattr__(self, "writes", [])

        def __setattr__(self, name: str, value: Any) -> None:
            self.writes.append((name, value))
            object.__setattr__(self, name, value)

    runner: Any = NPUModelRunner.__new__(NPUModelRunner)
    fa_spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)

    # 1. Pure FullAttention config -> zero writes to cache_config.
    spy_fa = _SpyCacheConfig()
    runner.vllm_config = SimpleNamespace(cache_config=spy_fa)
    cfg_fa: Any = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(["fa"], fa_spec)],
        hash_block_size=128,
        cache_hit_alignment_tokens=128,
    )
    NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_fa)
    assert spy_fa.writes == []

    # 2. Non-align Mamba -> zero writes to cache_config.
    spec_fields = {f.name for f in fields(MambaSpec)}
    if "mamba_cache_mode" in spec_fields:
        none_mamba = MambaSpec(block_size=256, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="none")
        spy_none = _SpyCacheConfig()
        runner.vllm_config = SimpleNamespace(cache_config=spy_none)
        cfg_none: Any = SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["mamba"], none_mamba)],
            hash_block_size=128,
            cache_hit_alignment_tokens=256,
        )
        NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_none)
        assert spy_none.writes == []

    # 3. Align Mamba when fields are unset -> syncs resolved geometry once.
    align_kwargs: dict[str, Any] = {"block_size": 256, "shapes": ((1,),), "dtypes": (torch.float32,)}
    if "mamba_cache_mode" in spec_fields:
        align_kwargs["mamba_cache_mode"] = "align"
    align_mamba = MambaSpec(**align_kwargs)

    spy_align = _SpyCacheConfig()
    runner.vllm_config = SimpleNamespace(cache_config=spy_align)
    cfg_align: Any = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(["fa"], fa_spec), KVCacheGroupSpec(["mamba"], align_mamba)],
        hash_block_size=128,
        cache_hit_alignment_tokens=64,
    )
    NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_align)
    assert spy_align.writes == [("hash_block_size", 128), ("cache_hit_alignment_tokens", 64)]

    # 4. Already matching values (e.g. synced by NPUWorker) -> zero redundant writes.
    spy_equal = _SpyCacheConfig(hash_block_size=128, cache_hit_alignment_tokens=64)
    runner.vllm_config = SimpleNamespace(cache_config=spy_equal)
    NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_align)
    assert spy_equal.writes == []


def test_mrv2_sync_align_mamba_cache_geometry_unwraps_uniform_and_kv_cache_specs() -> None:
    class _SpyCacheConfig:
        writes: list[tuple[str, Any]]

        def __init__(self) -> None:
            object.__setattr__(self, "hash_block_size", None)
            object.__setattr__(self, "cache_hit_alignment_tokens", None)
            object.__setattr__(self, "writes", [])

        def __setattr__(self, name: str, value: Any) -> None:
            self.writes.append((name, value))
            object.__setattr__(self, name, value)

    runner: Any = NPUModelRunner.__new__(NPUModelRunner)
    fa_spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    spec_fields = {f.name for f in fields(MambaSpec)}
    align_kwargs: dict[str, Any] = {"block_size": 256, "shapes": ((1,),), "dtypes": (torch.float32,)}
    if "mamba_cache_mode" in spec_fields:
        align_kwargs["mamba_cache_mode"] = "align"
    align_mamba = MambaSpec(**align_kwargs)

    # 1. UniformTypeKVCacheSpecs wrapping an align MambaSpec -> syncs once.
    spy_uniform_align = _SpyCacheConfig()
    runner.vllm_config = SimpleNamespace(cache_config=spy_uniform_align)
    cfg_uniform_align: Any = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(["mamba.0"], UniformTypeKVCacheSpecs.from_specs({"mamba.0": align_mamba})),
        ],
        hash_block_size=128,
        cache_hit_alignment_tokens=64,
    )
    NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_uniform_align)
    assert spy_uniform_align.writes == [("hash_block_size", 128), ("cache_hit_alignment_tokens", 64)]

    # 2. UniformTypeKVCacheSpecs wrapping non-align MambaSpec / FullAttentionSpec / empty dict -> zero writes.
    non_align_specs: list[Any] = [
        UniformTypeKVCacheSpecs.from_specs({"fa.0": fa_spec}),
        UniformTypeKVCacheSpecs(block_size=MAMBA_BLOCK_SIZE, kv_cache_specs={}),
    ]
    if "mamba_cache_mode" in spec_fields:
        none_mamba = MambaSpec(block_size=256, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="none")
        non_align_specs.append(UniformTypeKVCacheSpecs.from_specs({"mamba.none": none_mamba}))
    for wrapped in non_align_specs:
        spy_skip = _SpyCacheConfig()
        runner.vllm_config = SimpleNamespace(cache_config=spy_skip)
        cfg_skip: Any = SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["layer"], wrapped)],
            hash_block_size=128,
            cache_hit_alignment_tokens=64,
        )
        NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_skip)
        assert spy_skip.writes == []

    # 3. Newer group shape exposing `.kv_cache_specs` without `.kv_cache_spec` -> no AttributeError.
    for container in ({"mamba.0": align_mamba}, [align_mamba]):
        spy_group_specs = _SpyCacheConfig()
        runner.vllm_config = SimpleNamespace(cache_config=spy_group_specs)
        cfg_group_specs: Any = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["mamba.0"], kv_cache_specs=container)],
            hash_block_size=128,
            cache_hit_alignment_tokens=64,
        )
        NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_group_specs)
        assert spy_group_specs.writes == [("hash_block_size", 128), ("cache_hit_alignment_tokens", 64)]

    spy_empty_group = _SpyCacheConfig()
    runner.vllm_config = SimpleNamespace(cache_config=spy_empty_group)
    cfg_empty_group: Any = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["fa.0"], kv_cache_specs={"fa.0": fa_spec})],
        hash_block_size=128,
        cache_hit_alignment_tokens=64,
    )
    NPUModelRunner._sync_align_mamba_cache_geometry(runner, cfg_empty_group)
    assert spy_empty_group.writes == []


def test_free_skips_midblock_run_past_state_without_checkpoints() -> None:
    coordinator, mamba_mgr = _make_hybrid_coordinator(has_checkpoint_blocks=False, retention_interval=0)
    req = SimpleNamespace(
        request_id="req-run-past-free",
        block_hashes=_block_hashes(512),
        num_prompt_tokens=100,
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

    block0 = mamba_mgr.req_to_blocks[req.request_id][0]
    assert mamba_mgr._mid_block_states[req.request_id][0] == 128
    assert mamba_mgr._checkpoints.get(req.request_id) is None
    assert block0.block_hash is None
    assert mamba_mgr._decode_resume_block_idx(req.request_id, 300, 100) is None

    coordinator.free(req)
    assert block0.block_hash is None
    assert coordinator.block_pool.get_cached_block(req.block_hashes[1], [1]) is None


def test_free_skips_off_boundary_checkpoint_after_async_checkpoint_pop() -> None:
    coordinator, mamba_mgr = _make_hybrid_coordinator(has_checkpoint_blocks=True, retention_interval=0)
    req = SimpleNamespace(
        request_id="req-async-pop-free",
        block_hashes=_block_hashes(512),
        num_prompt_tokens=100,
        num_tokens=451,
        num_computed_tokens=450,
        num_in_flight_tokens=0,
        shared_prefix_boundary=0,
    )
    coordinator.allocate_new_blocks(req.request_id, 128, 128)
    mamba_mgr.get_num_blocks_to_allocate(
        request_id=req.request_id,
        num_tokens=450,
        new_computed_blocks=(),
        total_computed_tokens=128,
        num_local_computed_tokens=128,
        num_tokens_main_model=450,
        apply_admission_cap=False,
        prefill_end=450,
    )
    assert mamba_mgr._checkpoints.get(req.request_id) == (384, 0)
    coordinator.allocate_new_blocks(req.request_id, 450, 450)
    mamba_mgr._record_mid_block_states(req, 128, 450)
    assert mamba_mgr._mid_block_states[req.request_id][0] == 384

    # Simulate next in-flight decode step popping the transient checkpoint before free().
    mamba_mgr._checkpoints.pop(req.request_id, None)
    block0 = mamba_mgr.req_to_blocks[req.request_id][0]
    assert block0.block_hash is None
    assert mamba_mgr._decode_resume_block_idx(req.request_id, 450, 100) is None

    coordinator.free(req)
    assert block0.block_hash is None
    assert coordinator.block_pool.get_cached_block(req.block_hashes[1], [1]) is None


def test_cache_blocks_on_free_skips_null_block_without_empty_kv_event() -> None:
    coordinator, mamba_mgr = _make_hybrid_coordinator(retention_interval=0, enable_kv_cache_events=True)
    req = SimpleNamespace(
        request_id="req-null-block-free",
        all_token_ids=list(range(512)),
        mm_features=[],
        cache_salt=None,
        lora_request=None,
        block_hashes=_block_hashes(512),
        num_prompt_tokens=100,
        num_tokens=260,
        num_computed_tokens=260,
        num_in_flight_tokens=0,
        shared_prefix_boundary=0,
    )
    coordinator.allocate_new_blocks(req.request_id, 100, 100)
    coordinator.new_step_starts()
    coordinator.allocate_new_blocks(req.request_id, 260, 260)
    coordinator.block_pool.take_events()

    mamba_mgr.req_to_blocks[req.request_id][0] = coordinator.block_pool.null_block
    mamba_mgr.cache_blocks_on_free(req)
    assert coordinator.block_pool.take_events() == []
