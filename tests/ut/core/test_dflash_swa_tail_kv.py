# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for #22692: DFlash/DSpark SWA tail-only KV & target no-padding."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder, group_and_unify_kv_cache_specs
from vllm_ascend.core.kv_cache_interface import (
    AscendDFlashSlidingWindowMLASpec,
    AscendDFlashSlidingWindowSpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    as_dflash_sliding_window_spec,
    build_dflash_sliding_window_layer_config,
    extract_dflash_tail_params,
    is_dflash_family_speculative,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.core.single_type_kv_cache_manager import AscendDFlashSlidingWindowManager
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextChunk,
    DSparkContextDescriptor,
    build_draft_context_slot_mappings,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import (
    DraftKVCacheMetadata,
    build_draft_kv_read_batches,
)
from vllm_ascend.patch.platform import patch_kv_cache_admission, patch_kv_cache_coordinator
from vllm_ascend.worker.v2 import attn_utils


@pytest.fixture(autouse=True)
def _register_specs():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()


def _make_runtime(
    *,
    method: str | None = "dflash",
    num_speculative_tokens: int = 3,
    max_in_flight_steps: int = 1,
    num_lookahead_tokens: int = 3,
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 512,
    disable_hybrid: bool = False,
) -> Any:
    """Build a minimal VllmConfig-compatible double across vLLM pin versions."""
    speculator_cfg = SimpleNamespace(num_lookahead_tokens=num_lookahead_tokens)
    spec_cfg = (
        SimpleNamespace(
            method=method,
            num_speculative_tokens=num_speculative_tokens,
            speculator_config=speculator_cfg,
        )
        if method is not None
        else None
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3"),
            max_model_len=max_model_len,
            supports_kv_cache_grouping=False,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid,
            max_num_batched_tokens=max_num_batched_tokens,
            async_scheduling=max_in_flight_steps > 1,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=False,
            prefix_cache_retention_interval=None,
            cache_dtype="auto",
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=spec_cfg,
        kv_transfer_config=None,
        num_speculative_tokens=num_speculative_tokens if method is not None else 0,
        max_concurrent_batches=max_in_flight_steps,
        max_in_flight_tokens=max_in_flight_steps * max_num_batched_tokens,
        num_lookahead_tokens=num_lookahead_tokens if method is not None else 0,
    )


def _make_dflash_spec(
    *,
    block_size: int = 4,
    sliding_window: int = 8,
    num_speculative_tokens: int = 2,
    max_in_flight_steps: int = 1,
    num_lookahead_tokens: int = 1,
    is_draft: bool = True,
) -> AscendDFlashSlidingWindowSpec:
    return AscendDFlashSlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
        num_speculative_tokens=num_speculative_tokens,
        max_in_flight_steps=max_in_flight_steps,
        num_lookahead_tokens=num_lookahead_tokens,
        is_draft=is_draft,
    )


def _make_dflash_manager(
    spec: AscendDFlashSlidingWindowSpec | None = None,
    *,
    num_gpu_blocks: int = 32,
    max_admission_blocks: int | None = 8,
    enable_caching: bool = False,
) -> AscendDFlashSlidingWindowManager:
    spec = spec or _make_dflash_spec()
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=enable_caching,
        hash_block_size=spec.block_size,
    )
    return AscendDFlashSlidingWindowManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
        max_admission_blocks_per_request=max_admission_blocks,
    )


def test_dflash_spec_tail_bounds_merge_and_registry_isolation(monkeypatch):
    """DFlash specs compute tail bounds, isolate from target SWA, and keep legacy DSpark mapping."""
    spec = _make_dflash_spec(
        block_size=16,
        sliding_window=128,
        num_speculative_tokens=3,
        max_in_flight_steps=1,
        num_lookahead_tokens=3,
    )
    plain_swa = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=128)
    assert spec.tail_tokens == 131
    assert spec.max_admission_blocks_per_request(2048, 32768) < plain_swa.max_admission_blocks_per_request(2048, 32768)

    runtime = _make_runtime(method="dflash", num_speculative_tokens=3, max_in_flight_steps=2, num_lookahead_tokens=3)
    assert is_dflash_family_speculative(runtime)
    assert extract_dflash_tail_params(runtime) == (3, 2, 3)
    expected_bytes = spec.max_admission_blocks_per_request(1024, 4096) * spec.page_size_bytes
    assert spec.max_memory_usage_bytes(runtime) == expected_bytes

    # Merge isolation against regular SlidingWindowSpec and AscendSlidingWindowMLASpec.
    assert AscendDFlashSlidingWindowSpec.merge([spec, spec]) == spec
    with pytest.raises(AssertionError):
        AscendDFlashSlidingWindowSpec.merge([spec, _make_dflash_spec(block_size=16, sliding_window=256)])
    assert not spec.is_uniform_with_collection({"a": spec, "b": plain_swa})

    mla_plain = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
        model_version="deepseek_v4",
    )
    mla_dflash = as_dflash_sliding_window_spec(mla_plain, runtime, is_draft=True)
    assert isinstance(mla_dflash, AscendDFlashSlidingWindowMLASpec)
    assert isinstance(mla_dflash, AscendSlidingWindowMLASpec)
    assert mla_dflash.tail_tokens == 134
    assert not mla_plain.is_uniform_with_collection({"t": mla_plain, "d": mla_dflash})
    assert not mla_dflash.is_uniform_with_collection({"t": mla_plain, "d": mla_dflash})
    with pytest.raises(AssertionError):
        AscendSlidingWindowMLASpec.merge([mla_plain, mla_dflash])

    registrations = {}

    def record(kvcache_spec_cls, manager_class, uniform_type_base_spec):
        registrations[kvcache_spec_cls] = manager_class

    monkeypatch.setattr(KVCacheSpecRegistry, "register", record)
    register_ascend_kv_cache_specs()
    assert registrations[AscendSlidingWindowMLASpec] is SlidingWindowManager
    assert registrations[AscendDFlashSlidingWindowSpec] is AscendDFlashSlidingWindowManager
    assert registrations[AscendDFlashSlidingWindowMLASpec] is AscendDFlashSlidingWindowManager


def test_build_dflash_layer_config_forward_compat(monkeypatch):
    """Forward-compat helper wraps upstream DFlashSlidingWindowManager.layer_config when present."""
    called: dict[str, Any] = {}

    @dataclass(frozen=True)
    class FakeLayerConfig:
        spec: KVCacheGroupSpec | SlidingWindowSpec
        is_draft: bool = False

    class FakeUpstreamDFlash:
        @staticmethod
        def layer_config(spec, *, sliding_window, num_speculative_tokens, max_in_flight_steps, num_lookahead_tokens):
            called.update(
                spec=spec,
                sliding_window=sliding_window,
                num_speculative_tokens=num_speculative_tokens,
                max_in_flight_steps=max_in_flight_steps,
                num_lookahead_tokens=num_lookahead_tokens,
            )
            return FakeLayerConfig(spec=spec, is_draft=False)

    monkeypatch.setattr(
        "vllm_ascend.core.kv_cache_interface.get_upstream_dflash_sliding_window_manager",
        lambda: FakeUpstreamDFlash,
    )
    runtime = _make_runtime(method="dspark", num_speculative_tokens=3, max_in_flight_steps=2, num_lookahead_tokens=3)
    plain = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=128)
    cfg = build_dflash_sliding_window_layer_config(plain, runtime, sliding_window=128, is_draft=True)
    assert isinstance(cfg, FakeLayerConfig)
    assert cfg.is_draft is True
    assert called["num_speculative_tokens"] == 3
    assert called["max_in_flight_steps"] == 2
    assert called["num_lookahead_tokens"] == 3

    # Non-dataclass configs also update in place without tripping dataclasses.replace().
    monkeypatch.setattr(
        FakeUpstreamDFlash,
        "layer_config",
        staticmethod(lambda spec, **kw: SimpleNamespace(spec=spec, is_draft=False)),
    )
    ns_cfg = build_dflash_sliding_window_layer_config(plain, runtime, sliding_window=128, is_draft=True)
    assert ns_cfg is not None
    assert ns_cfg.is_draft is True


def test_dflash_manager_backs_only_each_chunks_tail_and_reserves_bounded():
    """Each chunk backs only its trailing tail, frees past null gaps, and caps reservations."""
    manager = _make_dflash_manager(_make_dflash_spec(), num_gpu_blocks=32, max_admission_blocks=8)
    pool = manager.block_pool
    initial_free = pool.get_num_free_blocks()

    for num_computed, num_target_tokens in ((0, 40), (40, 80)):
        num_tokens = num_target_tokens + 1
        assert (
            manager.get_num_blocks_to_allocate("r", num_tokens, [], num_computed, num_computed, num_target_tokens) == 4
        )
        manager.allocate_new_blocks("r", num_tokens, num_target_tokens)

    resident = [i for i, b in enumerate(manager.req_to_blocks["r"]) if not b.is_null]
    assert resident == [7, 8, 9, 10, 17, 18, 19, 20]
    assert pool.get_num_free_blocks() == initial_free - 8

    # Frees cross the intermediate null gap [11..16] and release [7..10] + 17.
    manager.remove_skipped_blocks("r", 80)
    resident_after = [i for i, b in enumerate(manager.req_to_blocks["r"]) if not b.is_null]
    assert resident_after == [18, 19, 20]
    assert pool.get_num_free_blocks() == initial_free - 3
    assert manager.get_num_blocks_to_reserve("r", 81) == 5

    manager.allocate_new_blocks("r", 121, 120)
    assert pool.get_num_free_blocks() == initial_free - 7
    assert manager.get_num_blocks_to_reserve("r", 121) == 1

    manager.free("r")
    assert pool.get_num_free_blocks() == initial_free


@pytest.mark.parametrize("max_in_flight_steps", [1, 2])
def test_dflash_manager_draft_rejection_and_partial_accept_safety(max_in_flight_steps):
    """Tail backing covers reads below accepted tokens even when all in-flight drafts reject."""
    num_spec = 3
    spec = _make_dflash_spec(
        block_size=4,
        sliding_window=8,
        num_speculative_tokens=num_spec,
        max_in_flight_steps=max_in_flight_steps,
        num_lookahead_tokens=num_spec,
    )
    manager = _make_dflash_manager(spec, num_gpu_blocks=128, max_admission_blocks=32)
    for num_target_tokens in range(40, 48):
        req_id = f"req_{num_target_tokens}"
        manager.allocate_new_blocks(req_id, num_target_tokens, num_target_tokens)
        num_accepted = num_target_tokens - num_spec * max_in_flight_steps
        first_read = num_accepted - spec.sliding_window + 1
        blocks = manager.req_to_blocks[req_id]
        end_blk = (num_target_tokens + spec.block_size - 1) // spec.block_size
        for blk_idx in range(first_read // spec.block_size, end_blk):
            assert not blocks[blk_idx].is_null

    # Worst-case all-draft rejection: committed prefix is 44 - num_spec * max_in_flight_steps
    # (41 for 1 step -> first_read=34 -> block 8; 38 for 2 steps -> first_read=31 -> block 7).
    min_accepted = 44 - num_spec * max_in_flight_steps
    min_first_read_blk = (min_accepted - spec.sliding_window + 1) // spec.block_size
    manager.remove_skipped_blocks("req_44", min_accepted)
    blocks_44 = manager.req_to_blocks["req_44"]
    assert all(b.is_null for b in blocks_44[:min_first_read_blk])
    assert not any(b.is_null for b in blocks_44[min_first_read_blk : 44 // spec.block_size])

    # Partial acceptance (+2 accepted tokens): advances skipped boundary by one full block
    # (43 -> block 9 for 1 step; 40 -> block 8 for 2 steps), freeing the evicted block while
    # keeping the live suffix backed.
    partial_accepted = min_accepted + 2
    partial_first_read_blk = (partial_accepted - spec.sliding_window + 1) // spec.block_size
    assert partial_first_read_blk == min_first_read_blk + 1
    manager.remove_skipped_blocks("req_44", partial_accepted)
    assert all(b.is_null for b in blocks_44[:partial_first_read_blk])
    assert not any(b.is_null for b in blocks_44[partial_first_read_blk : 44 // spec.block_size])


def test_dflash_manager_prefix_hit_sparse_mask_and_external_load():
    """Tail prefix hit ignores lower missing blocks, honors num_request_tokens, and caps sparse mask."""
    spec = _make_dflash_spec(block_size=4, sliding_window=8, num_speculative_tokens=2, num_lookahead_tokens=2)
    manager = _make_dflash_manager(spec, num_gpu_blocks=32, max_admission_blocks=8, enable_caching=True)
    pool = manager.block_pool

    fake_hashes: list[Any] = [f"h{i}".encode() for i in range(10)]
    cached_blocks = [MagicMock(is_null=False, block_id=i + 1) for i in range(10)]

    # Only blocks from index 6 upward exist in the pool; [0..5] are missing.
    def fake_get_cached_block(block_hash, group_ids):
        idx = int(block_hash.decode()[1:])
        return [cached_blocks[idx]] if idx >= 6 else None

    pool.get_cached_block = fake_get_cached_block
    # For a request of 32 tokens, num_retained_tokens=7 -> first_block = (32 - 7)//4 = 6.
    hits, hit_len = AscendDFlashSlidingWindowManager.find_longest_cache_hit(
        block_hashes=fake_hashes,
        max_length=32,
        kv_cache_group_ids=[0],
        block_pool=pool,
        kv_cache_spec=spec,
        drop_eagle_block=True,
        alignment_tokens=4,
        num_request_tokens=32,
    )
    assert hit_len == 32
    assert all(b.is_null for b in hits[0][:6])
    assert hits[0][6:8] == cached_blocks[6:8]

    # Dense retention returns None; sparse retention retains only boundary tails.
    assert AscendDFlashSlidingWindowManager.reachable_block_mask(0, 10, 4, spec, False, None, (28,)) is None
    mask = AscendDFlashSlidingWindowManager.reachable_block_mask(0, 10, 4, spec, False, 16, (28,))
    # boundary=28 -> first_read=28+1-7=22 -> block 5; hit_tokens=28 -> block 7.
    assert mask == [False] * 5 + [True, True] + [False] * 3

    # Int block-size specs (FB-01 regression) and post-#61452/#64109 manager_config calls.
    assert AscendDFlashSlidingWindowManager.reachable_block_mask(0, 1, 16, 16, False, 16, (0,)) == [False]
    assert AscendDFlashSlidingWindowManager.reachable_block_mask(0, 3, 16, 16, False, 16, (32,)) == [False, True, False]
    mgr_cfg = SimpleNamespace(params=SimpleNamespace(sliding_window=8, extra_retained_tokens=0))
    assert (
        AscendDFlashSlidingWindowManager.reachable_block_mask(0, 10, 4, 4, mgr_cfg, False, 16, (28,))
        == [False] * 5 + [True, True] + [False] * 3
    )
    assert (
        AscendDFlashSlidingWindowManager.reachable_block_mask(
            0,
            10,
            4,
            block_size=4,
            manager_config=mgr_cfg,
            use_eagle=False,
            retention_interval=16,
            reachable_boundaries=(28,),
        )
        == [False] * 5 + [True, True] + [False] * 3
    )
    post_hits, post_hit_len = AscendDFlashSlidingWindowManager.find_longest_cache_hit(
        fake_hashes, 32, [0], pool, 4, mgr_cfg, True, 4, num_request_tokens=32
    )
    assert post_hit_len == 32
    assert all(b.is_null for b in post_hits[0][:6])
    assert post_hits[0][6:8] == cached_blocks[6:8]
    kw_hits, kw_len = AscendDFlashSlidingWindowManager.find_longest_cache_hit(
        fake_hashes,
        32,
        [0],
        pool,
        block_size=4,
        manager_config=mgr_cfg,
        drop_eagle_block=True,
        alignment_tokens=4,
        num_request_tokens=32,
    )
    assert kw_len == 32 and kw_hits[0][6:8] == cached_blocks[6:8]

    # External load works both with num_tokens_main_model and pre-#64109 default None.
    ext_mgr = _make_dflash_manager(spec, num_gpu_blocks=32, max_admission_blocks=8)
    ext_mgr.allocate_external_computed_blocks("ext1", 0, 40, num_tokens_main_model=40)
    assert all(b.is_null for b in ext_mgr.req_to_blocks["ext1"][:7])
    assert not any(b.is_null for b in ext_mgr.req_to_blocks["ext1"][7:10])
    ext_mgr.allocate_external_computed_blocks("ext2", 0, 40)
    assert all(b.is_null for b in ext_mgr.req_to_blocks["ext2"][:7])


def test_no_target_padding_for_draft_only_swa_buckets_68533():
    """Muse Spark shape: 5-layer DFlash SWA draft must not pad 16 FullAttention layers to 20."""
    runtime = _make_runtime(method="dflash", num_speculative_tokens=3, num_lookahead_tokens=3)
    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    swa_spec = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, sliding_window=128)
    draft_spec = AscendDFlashSlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=128,
        num_speculative_tokens=3,
        max_in_flight_steps=1,
        num_lookahead_tokens=3,
        is_draft=True,
    )

    specs: dict[str, KVCacheSpec] = {f"target.{i}.attn": (full_spec if i % 4 == 1 else swa_spec) for i in range(62)}
    for i in range(5):
        specs[f"draft.{i}.attn"] = draft_spec

    groups = AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
    full_groups = [g for g in groups if isinstance(g.kv_cache_spec, FullAttentionSpec)]
    sw_groups = [g for g in groups if type(g.kv_cache_spec) is SlidingWindowSpec]
    draft_groups = [g for g in groups if isinstance(g.kv_cache_spec, AscendDFlashSlidingWindowSpec)]
    assert [len(g.layer_names) for g in full_groups] == [16]
    assert sorted(len(g.layer_names) for g in sw_groups) == [15, 15, 16]
    assert [len(g.layer_names) for g in draft_groups] == [5]
    assert len(groups) == 5

    # Counterexample: a full-attention draft bucket is unbounded and still sets group size = 1.
    unbounded_specs: dict[str, KVCacheSpec] = {
        f"target.{i}.attn": (full_spec if i % 4 == 1 else swa_spec) for i in range(62)
    }
    unbounded_specs["draft.0.attn"] = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, dcp_sharded=False, is_draft=True
    )
    assert len(AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, unbounded_specs)) == 63

    # Also verify DeepSeek-V4 mixed unified bucket separation without target SWA merge collisions.
    dsv4_mla_c4 = AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    dsv4_mla_c128 = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    dsv4_swa = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
        model_version="deepseek_v4",
    )
    dsv4_draft = as_dflash_sliding_window_spec(dsv4_swa, runtime, is_draft=True)
    unified = group_and_unify_kv_cache_specs(
        {"t_c4": dsv4_mla_c4, "t_c128": dsv4_mla_c128, "t_swa": dsv4_swa, "d_swa": dsv4_draft}
    )
    assert unified is not None and len(unified) == 4


def test_deepseek_v41_dspark_independent_draft_slots_no_target_padding():
    """DeepSeek-V4.1 DSpark places 3 or >4 draft layers without expanding target groups."""
    pytest.importorskip("vllm.transformers_utils.configs.deepseek_v41", reason="DSV4.1 unavailable on this release")
    from vllm_ascend.models.deepseek_v41.cache_config import (
        group_cache_specs,
        make_cache_groups,
        make_index_cache_spec,
        make_mla_cache_spec,
        make_swa_cache_spec,
    )

    runtime = _make_runtime(method="dspark", num_speculative_tokens=3, num_lookahead_tokens=3)
    mla = make_mla_cache_spec(block_size=64, head_size=512, compress_ratio=2)
    idx = make_index_cache_spec(block_size=64, head_size=128, compress_ratio=1)
    swa = make_swa_cache_spec(
        block_size=64, window_size=128, head_size=512, dtype=torch.bfloat16, cache_dtype="bfloat16"
    )
    draft = as_dflash_sliding_window_spec(swa, runtime, is_draft=True)

    base = {}
    for i in range(4):
        base[f"model.layers.{i}.self_attn.long_kv_cache"] = mla
        base[f"model.layers.{i}.self_attn.indexer.k_cache"] = idx
        base[f"model.layers.{i}.self_attn.swa_cache"] = swa
    for num_draft in (3, 5):
        specs = dict(base)
        for d in range(num_draft):
            specs[f"mtp.{d}.self_attn.swa_cache"] = draft
        groups = make_cache_groups(group_cache_specs(specs))
        target_groups = [g for g in groups if not any(n.startswith("mtp.") for n in g.layer_names)]
        draft_groups = [g for g in groups if all(n.startswith("mtp.") for n in g.layer_names)]
        assert len(target_groups) == 2 and sorted(len(g.layer_names) for g in target_groups) == [4, 8]
        assert len(draft_groups) == (1 if num_draft == 3 else 2)
        assert sum(len(g.layer_names) for g in draft_groups) == num_draft
        assert all(
            isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
            and all(isinstance(s, AscendDFlashSlidingWindowMLASpec) for s in g.kv_cache_spec.kv_cache_specs.values())
            for g in draft_groups
        )

    bad_idx = make_index_cache_spec(block_size=32, head_size=128, compress_ratio=1)
    bad_specs = {**base, "model.layers.0.self_attn.indexer.k_cache": bad_idx}
    with pytest.raises(ValueError, match="Non-uniform DeepSeek V4.1"):
        group_cache_specs(bad_specs)
    with pytest.raises(ValueError, match="None"):
        make_cache_groups([None])


def test_attn_utils_align_and_promote_dflash_draft_specs(monkeypatch):
    """attn_utils aligns AscendDFlashSlidingWindowSpec pages and promotes draft SWA only for dflash/dspark."""
    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    dflash_small = AscendDFlashSlidingWindowSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16, sliding_window=128, is_draft=True
    )
    specs_map: dict[str, KVCacheSpec] = {"target": full_spec, "draft": dflash_small}
    attn_utils._align_hybrid_attention_page_sizes(specs_map)
    assert specs_map["draft"].block_size == 32

    plain_swa = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16, sliding_window=128)
    cases = (
        ("dflash", AscendDFlashSlidingWindowSpec),
        ("dspark", AscendDFlashSlidingWindowSpec),
        ("mtp", SlidingWindowSpec),
    )
    for method, expected_cls in cases:
        runtime = _make_runtime(method=method)
        fake_attn = SimpleNamespace(
            kv_sharing_target_layer_name=None,
            is_draft_layer=True,
            align_kv_cache_with_mamba=True,
            get_kv_cache_spec=lambda cfg, s=plain_swa: s,
            get_attn_backend=lambda: object(),
        )
        monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda cfg, cls, m=fake_attn: {"draft": m})
        monkeypatch.setattr(attn_utils, "customize_attention_spec", lambda backend, spec: spec)
        out = attn_utils.get_kv_cache_spec(runtime)
        assert type(out["draft"]) is expected_cls


def test_admission_and_coordinator_patches_handle_dflash_and_legacy_managers():
    """Admission uses get_num_blocks_to_reserve and coordinator adapts num_request_tokens cleanly."""
    manager = _make_dflash_manager(_make_dflash_spec(), num_gpu_blocks=32, max_admission_blocks=8)
    manager.allocate_new_blocks("r", 41, 40)
    # Call the patched admission function directly with apply_admission_cap=True.
    reserve = patch_kv_cache_admission.get_num_blocks_to_allocate(
        manager, "r", 81, [], 0, 0, 80, apply_admission_cap=True
    )
    assert reserve == 4

    class LegacyMgr:
        supports_fine_grained_hash_lookup = False

        @classmethod
        def find_longest_cache_hit(
            cls,
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size=1,
            pcp_world_size=1,
        ):
            return ([],), 0

    assert not patch_kv_cache_coordinator._accepts_num_request_tokens(LegacyMgr)
    assert patch_kv_cache_coordinator._accepts_num_request_tokens(AscendDFlashSlidingWindowManager)


def test_mla_nope_context_kv_skips_padding_slots():
    """exec_kv_prefill guards MLA-NoPE indexed copies when slot_mapping contains -1."""
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl

    impl: Any = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.use_pcp = False
    impl.use_mla_rope = True
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 0
    impl.num_kv_heads = 1
    impl.kv_a_layernorm = lambda x: x
    impl._uses_strided_kv_adapter = lambda: False

    kv_cache = (torch.zeros(2, 4, 4, dtype=torch.bfloat16), torch.zeros(2, 4, 0, dtype=torch.bfloat16))
    kv_no_split = torch.ones(4, 1, 4, dtype=torch.bfloat16)
    slots = torch.tensor([-1, -1, 4, -1], dtype=torch.int32)
    impl.exec_kv_prefill(kv_no_split, None, None, kv_cache, slots)
    assert torch.count_nonzero(kv_cache[0][0]) == 0
    assert torch.all(kv_cache[0][1, 0] == 1)
    assert torch.count_nonzero(kv_cache[0][1, 1:]) == 0

    # Also verify all -1 slots is a safe no-op.
    all_pad = torch.full((4,), -1, dtype=torch.int32)
    impl.exec_kv_prefill(kv_no_split, None, None, kv_cache, all_pad, skip_padding_slots=True)
    assert torch.count_nonzero(kv_cache[0][0]) == 0


def test_dspark_pd_and_mooncake_null_block_boundaries():
    """SFA P/D context slots, KV read batches, and Mooncake SWA pull handle leading null block 0."""
    descriptor = DSparkContextDescriptor("req-1", "gen-1", 6, (2,), 4)
    chunk = DSparkContextChunk(descriptor, token_offset=0, num_tokens=6)
    default_slots = build_draft_context_slot_mappings(
        chunk,
        draft_group_ids=(0,),
        draft_block_ids_by_group={0: (0, 2)},
        block_sizes_by_group={0: 4},
        layer_group_ids=(0,),
        device=torch.device("cpu"),
    )
    assert default_slots[0].tolist() == [0, 1, 2, 3, 8, 9]

    skipped_slots = build_draft_context_slot_mappings(
        chunk,
        draft_group_ids=(0,),
        draft_block_ids_by_group={0: (0, 2)},
        block_sizes_by_group={0: 4},
        layer_group_ids=(0,),
        device=torch.device("cpu"),
        skip_null_blocks=True,
    )
    assert skipped_slots[0].tolist() == [-1, -1, -1, -1, 8, 9]

    meta_src = DraftKVCacheMetadata(0, 4, 16, (1000,), (64,), (32,), (1,), ((4, 8),), ("torch.bfloat16",))
    meta_dst = DraftKVCacheMetadata(0, 4, 16, (5000,), (64,), (32,), (1,), ((4, 8),), ("torch.bfloat16",))
    batches = list(
        build_draft_kv_read_batches(
            {"l0": meta_src}, {"l0": meta_dst}, {"l0": (0, 0, 5)}, {0: (0, 0, 7)}, prompt_tokens=12
        )
    )
    assert len(batches) == 1 and batches[0] == ([1000 + 5 * 64], [5000 + 7 * 64], [32])
    with pytest.raises(ValueError, match="not resident"):
        list(build_draft_kv_read_batches({"l0": meta_src}, {"l0": meta_dst}, {"l0": (0, 5)}, {0: (4, 7)}, 8))

    # Mooncake pull_worker SWA null filtering.
    worker: Any = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    worker.dcp_size = 1
    worker.dcp_rank = 0
    spec = _make_dflash_spec(block_size=4, sliding_window=8)
    result = worker._compute_group_block_ids(
        "r", [[0]], 1, 0, 4, 4, [0, 7, 8], [0, 7, 8], [0, 17, 18], 12, 12, 0, 1, 1, spec, 0, 4
    )
    assert result == [(0, [7, 8], [17, 18])]
    with pytest.raises(ValueError, match="not resident"):
        worker._compute_group_block_ids(
            "r", [[0]], 1, 0, 4, 4, [0, 7, 8], [0, 7, 8], [0, 0, 18], 12, 12, 0, 1, 1, spec, 0, 4
        )


def _make_layerwise_worker(
    *,
    group_uses_align_state: list[bool],
    group_skips_null_blocks: list[bool],
    grouped_block_size: list[int],
    cache_transfer_granularity: int,
    layerwise_offload: bool = False,
    use_layerwise_transfer: bool = True,
    cache_coordinator: Any = None,
    token_database: Any = None,
) -> Any:
    """Create a minimal LayerwiseKVPoolWorker with all reachable-mask fields initialized."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker

    worker: Any = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.group_uses_align_state = group_uses_align_state
    worker.group_skips_null_blocks = group_skips_null_blocks
    worker.num_kv_cache_groups = len(grouped_block_size)
    worker.grouped_block_size = grouped_block_size
    worker.layerwise_offload = layerwise_offload
    worker.use_layerwise_transfer = use_layerwise_transfer
    worker.block_key_hybrid = False
    worker.cache_coordinator = cache_coordinator
    worker.token_database = token_database
    worker.cache_transfer_granularity = cache_transfer_granularity
    worker._reused_layers_by_group = {}
    return worker


def test_ascend_store_coordinator_and_null_block_workers():
    """AscendStore coordinator skips eagle drop for DFlash & workers exclude block 0 placeholders."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
        AscendStoreCoordinator,
        _find_hit_accepts_num_request_tokens,
    )
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    assert _find_hit_accepts_num_request_tokens(AscendDFlashSlidingWindowManager)

    dflash_spec = _make_dflash_spec(block_size=16, sliding_window=128)
    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    groups = [
        KVCacheGroupSpec(layer_names=["target.0"], kv_cache_spec=full_spec),
        KVCacheGroupSpec(layer_names=["draft.0"], kv_cache_spec=dflash_spec),
    ]
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
        side_effect=lambda s: AscendDFlashSlidingWindowManager if s is dflash_spec else FullAttentionManager,
    ):
        coord = AscendStoreCoordinator(
            groups,
            scheduler_block_size=16,
            hash_block_size=16,
            group_block_sizes=[16, 16],
            group_cache_families=["default", "default"],
            use_eagle=True,
        )
    assert 1 not in coord.eagle_group_ids
    assert 1 not in coord.eagle_reachable_group_ids

    # Worker null-block inference keeps group_uses_align_state focused on Mamba align.
    base_worker: Any = KVPoolWorker.__new__(KVPoolWorker)
    mamba = MambaSpec(shapes=((2, 4),), dtypes=(torch.bfloat16,), block_size=16, mamba_cache_mode="align")
    base_worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["t"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["d"], kv_cache_spec=dflash_spec),
            KVCacheGroupSpec(layer_names=["m"], kv_cache_spec=mamba),
        ]
    )
    assert base_worker._infer_group_uses_align_state() == [False, False, True]
    assert base_worker._infer_group_skips_null_blocks() == [False, True, True]

    # Layerwise worker load/store mask excludes block 0 on SWA/DFlash groups without enabling Mamba checkpoints.
    lw = _make_layerwise_worker(
        group_uses_align_state=[False],
        group_skips_null_blocks=[True],
        grouped_block_size=[16],
        cache_transfer_granularity=16,
    )
    req: Any = SimpleNamespace(
        can_save=True,
        save_end_token=64,
        num_prompt_tokens=64,
        target_token_len=64,
        block_ids_by_group=[[0, 0, 5, 6]],
        load_spec=None,
    )
    assert lw._compute_reachable_store_masks(req) == ([False, False, True, True],)
    assert lw._compute_reachable_load_masks(req, 64) == ([False, False, True, True],)

    # FB-83-03 regression: mutating group_uses_align_state after init (as in test_pool_worker)
    # still activates null-block filtering and preserves aligned recurrent checkpoints.
    lw_align = _make_layerwise_worker(
        group_uses_align_state=[False],
        group_skips_null_blocks=[False],
        grouped_block_size=[4352],
        cache_transfer_granularity=4352,
        layerwise_offload=True,
        use_layerwise_transfer=True,
        cache_coordinator=object(),
        token_database=SimpleNamespace(store_mask=lambda end, prompt: ([False],)),
    )
    lw_align.group_uses_align_state = [True]
    assert lw_align._effective_skip_null_blocks_by_group() == [True]
    aligned_req: Any = SimpleNamespace(
        can_save=True,
        save_end_token=4352,
        target_token_len=4352,
        num_prompt_tokens=8705,
        block_ids_by_group=[[3]],
    )
    null_aligned_req: Any = SimpleNamespace(
        can_save=True,
        save_end_token=4352,
        target_token_len=4352,
        num_prompt_tokens=8705,
        block_ids_by_group=[[0]],
    )
    for use_layerwise_transfer in (True, False):
        lw_align.use_layerwise_transfer = use_layerwise_transfer
        assert lw_align._compute_reachable_store_masks(aligned_req) == ([True],)
        assert lw_align._compute_reachable_store_masks(null_aligned_req) == ([False],)
    lw_align.group_uses_align_state = [False, True]
    assert lw_align._effective_skip_null_blocks_by_group() == [False, True]


def test_dflash_and_dspark_preserve_data_parallel_index(monkeypatch):
    """Independent dense DP DFlash/DFlash2/DSpark specs keep their per-replica compile cache index (FB-03)."""
    from vllm.config import ParallelConfig, replace
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    from vllm_ascend.patch.worker import patch_dspark
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
    from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
    from vllm_ascend.worker.v2.spec_decode.pcp_utils import build_dspark_draft_config

    assert issubclass(AscendDFlash2Speculator, AscendDFlashSpeculator)

    target_pc = ParallelConfig(data_parallel_size=2, data_parallel_rank=1, distributed_executor_backend="mp")
    target_pc.reconfigure_for_independent_dp_rank()
    assert target_pc.data_parallel_rank == 0 and target_pc.data_parallel_index == 1

    def fake_dflash_init(self, vllm_config, device):
        # Upstream DFlashSpeculator.__init__ calls replace(parallel_config, ...),
        # which resets init=False data_parallel_index to 0 in ParallelConfig.__post_init__.
        copied = SimpleNamespace(**vars(vllm_config))
        copied.parallel_config = replace(vllm_config.parallel_config, prefill_context_parallel_size=1)
        self.vllm_config = copied
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                dflash_config={"selector_top_k": 4},
                sample_from_anchor=True,
                dspark_draft_topk=None,
            ),
            get_hidden_size=lambda: 16,
        )
        self.max_num_reqs = 2
        self.max_num_tokens = 8
        self.num_speculative_steps = 3
        self.dtype = torch.bfloat16
        self.draft_logits = None

    monkeypatch.setattr(DFlashSpeculator, "__init__", fake_dflash_init)
    monkeypatch.setattr(LmheadTPDraftSamplingMixin, "_lmhead_tp_validate_draft_sampling", lambda self: None)
    monkeypatch.setattr(AscendDSparkSpeculator, "_init_dcp", lambda self: None)

    dflash_cfg: Any = SimpleNamespace(parallel_config=target_pc, speculative_config=None)
    dflash2 = AscendDFlash2Speculator(dflash_cfg, torch.device("cpu"))
    assert dflash2.vllm_config.parallel_config.data_parallel_index == 1

    target_pcp_pc = ParallelConfig(
        data_parallel_size=4,
        data_parallel_rank=3,
        prefill_context_parallel_size=2,
        distributed_executor_backend="mp",
    )
    target_pcp_pc.reconfigure_for_independent_dp_rank()
    pcp_cfg: Any = SimpleNamespace(parallel_config=target_pcp_pc, engram_config=None)
    dspark_built = build_dspark_draft_config(pcp_cfg)
    assert dspark_built.parallel_config.data_parallel_index == 3

    dspark_cfg: Any = SimpleNamespace(parallel_config=target_pc, engram_config=None)
    dspark = AscendDSparkSpeculator(dspark_cfg, torch.device("cpu"))
    assert dspark.vllm_config.parallel_config.data_parallel_index == 1

    # Also verify patch_dspark wrapper preserves data_parallel_index across _get_dspark_parallel_config.
    def fake_get_dspark_pc(parallel_config, *args, **kwargs):
        return replace(parallel_config, pipeline_parallel_size=1)

    def fake_load_dspark(target_model, cfg):
        return patch_dspark.dspark_utils._get_dspark_parallel_config(cfg.parallel_config)

    monkeypatch.setattr(patch_dspark, "resolve_spec_pp_support", lambda cfg: None)
    monkeypatch.setattr(patch_dspark.eagle_utils, "maybe_share_target_embed", lambda *a: None)
    monkeypatch.setattr(patch_dspark, "_original_load_dspark_model", fake_load_dspark)
    monkeypatch.setattr(patch_dspark.dspark_utils, "_get_dspark_parallel_config", fake_get_dspark_pc, raising=False)
    mock_vllm_cfg: Any = SimpleNamespace(
        parallel_config=target_pc,
        model_config=SimpleNamespace(model="t"),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(model="d")),
    )
    draft_pc = patch_dspark._load_dspark_model_with_target_quant(object(), mock_vllm_cfg)
    assert draft_pc.data_parallel_index == 1
