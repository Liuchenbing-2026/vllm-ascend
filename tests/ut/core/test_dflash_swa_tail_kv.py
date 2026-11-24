# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for #22692: DFlash/DSpark SWA tail-only KV & target no-padding."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    SlidingWindowManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
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
    assert manager.get_num_blocks_to_allocate("r", 200, [], 80, 80, 200, apply_admission_cap=True) == 5
    assert manager.get_num_blocks_to_allocate("r", 81, [], 80, 80, 80, apply_admission_cap=True) == 0

    manager.allocate_new_blocks("r", 121, 120)
    assert pool.get_num_free_blocks() == initial_free - 7
    assert manager.get_num_blocks_to_allocate("r", 200, [], 120, 120, 200, apply_admission_cap=True) == 1
    assert manager.get_num_blocks_to_allocate("r", 121, [], 120, 120, 120, apply_admission_cap=True) == 0

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
    """Admission uses _count_recyclable_held_blocks and coordinator adapts num_request_tokens cleanly."""
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


@pytest.mark.parametrize("return_bf16_kv", [False, True])
def test_mla_nope_context_kv_skips_padding_slots(return_bf16_kv: bool):
    """exec_kv_prefill guards MLA-NoPE indexed copies when slot_mapping contains -1."""
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl

    impl: Any = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.use_pcp = False
    impl.use_mla_rope = True
    impl.use_quant_flashmla = False
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 0
    impl.num_kv_heads = 1
    impl.kv_a_layernorm = lambda x: x
    impl._uses_strided_kv_adapter = lambda: False

    kv_cache = (torch.zeros(2, 4, 4, dtype=torch.bfloat16), torch.zeros(2, 4, 0, dtype=torch.bfloat16))
    kv_no_split = torch.ones(4, 1, 4, dtype=torch.bfloat16)
    slots = torch.tensor([-1, -1, 4, -1], dtype=torch.int32)
    impl.exec_kv_prefill(kv_no_split, None, None, kv_cache, slots, return_bf16_kv=return_bf16_kv)
    assert torch.count_nonzero(kv_cache[0][0]) == 0
    assert torch.all(kv_cache[0][1, 0] == 1)
    assert torch.count_nonzero(kv_cache[0][1, 1:]) == 0

    # Also verify all -1 slots is a safe no-op.
    all_pad = torch.full((4,), -1, dtype=torch.int32)
    impl.exec_kv_prefill(
        kv_no_split, None, None, kv_cache, all_pad, return_bf16_kv=return_bf16_kv, skip_padding_slots=True
    )
    assert torch.count_nonzero(kv_cache[0][0]) == 0


def test_non_speculative_kv_core_fast_path_isolation(monkeypatch):
    """Non-draft SWA and non-PCP full attention preserve the original KV core fast paths."""
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_ascend.attention.mla_v1 import AscendMLAImpl
    from vllm_ascend.core.kv_cache_config_builder import _get_kv_cache_groups_without_windowed_draft_padding
    from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator

    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    swa_spec = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=128)
    specs: dict[str, KVCacheSpec] = {"full": full_spec, "swa": swa_spec}
    assert _get_kv_cache_groups_without_windowed_draft_padding(SimpleNamespace(), specs) is None
    assert group_and_unify_kv_cache_specs(specs) is None

    # Coordinator caches False for legacy managers and invokes find_longest_cache_hit without num_request_tokens.
    coord: AscendHybridKVCacheCoordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    groups = [
        KVCacheGroupSpec(layer_names=["full"], kv_cache_spec=full_spec),
        KVCacheGroupSpec(layer_names=["swa"], kv_cache_spec=swa_spec),
    ]
    seen_kwargs: list[bool] = []

    class PlainLegacyManager:
        supports_fine_grained_hash_lookup = False

        def __init__(self) -> None:
            self.block_size = 16
            self.use_eagle = False

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
            seen_kwargs.append(True)
            return ([[] for _ in kv_cache_group_ids], 0)

    coord.kv_cache_config = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)
    coord.single_type_managers = (PlainLegacyManager(), PlainLegacyManager())
    coord.eagle_group_ids = set()
    coord.skips_eagle_block_drop = True
    coord.dcp_world_size = 1
    coord.hash_block_size = 16
    coord.scheduler_block_size = 16
    coord.enable_partial_hash_hits = False
    coord.block_pool = object()
    coord.verify_and_split_kv_cache_groups()
    assert coord._attn_group_accepts_num_request_tokens == (False, False)
    coord.find_longest_cache_hit([], 0)
    coord.find_longest_cache_hit_per_group([], 0)
    assert len(seen_kwargs) == 4

    # Normal non-PCP prefill with attn_metadata keeps skip_padding_slots=False.
    impl: Any = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.use_pcp = False
    impl.use_mla_rope = True
    impl.use_quant_flashmla = False
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 0
    impl.num_kv_heads = 1
    impl.kv_a_layernorm = lambda x: x
    impl._uses_strided_kv_adapter = lambda: False
    captured: dict[str, bool] = {}

    def fake_nope(kv_no_split, kv_cache, slots, is_prefill: bool, skip_padding_slots: bool = False):
        captured["skip_padding_slots"] = skip_padding_slots
        return torch.empty(0), torch.empty(0)

    monkeypatch.setattr(impl, "_exec_kv_mla_nope", fake_nope)
    kv_cache = (torch.zeros(2, 4, 4, dtype=torch.bfloat16), torch.zeros(2, 4, 0, dtype=torch.bfloat16))
    impl.exec_kv_prefill(
        torch.ones(4, 1, 4, dtype=torch.bfloat16),
        None,
        None,
        kv_cache,
        torch.arange(4, dtype=torch.int32),
        attn_metadata=SimpleNamespace(),
    )
    assert captured["skip_padding_slots"] is False


def test_dflash_manager_external_load_budget_matches_actual_pool_allocation():
    """Legacy external KV load + new prompt tokens must count intermediate blocks in budget."""
    spec = _make_dflash_spec(
        block_size=4,
        sliding_window=8,
        num_speculative_tokens=2,
        max_in_flight_steps=1,
        num_lookahead_tokens=1,
    )
    # Tight pool: 6 usable blocks (+ null block 0) matches exact 2 ext + 4 tail demand.
    tight_mgr = _make_dflash_manager(spec, num_gpu_blocks=7, max_admission_blocks=8)
    pool = tight_mgr.block_pool
    assert pool.get_num_free_blocks() == 6

    predicted = tight_mgr.get_num_blocks_to_allocate("ext-sync", 81, [], 40, 0, 80)
    assert predicted == 6
    assert predicted <= pool.get_num_free_blocks()

    tight_mgr.add_local_computed_blocks("ext-sync", [], 0, 40)
    tight_mgr.allocate_external_computed_blocks("ext-sync", 0, 40)
    tight_mgr.allocate_new_blocks("ext-sync", 81, 80)
    assert pool.get_num_free_blocks() == 0

    tight_mgr.remove_skipped_blocks("ext-sync", 80)
    assert pool.get_num_free_blocks() == 3
    assert [i for i, b in enumerate(tight_mgr.req_to_blocks["ext-sync"]) if not b.is_null] == [18, 19, 20]
    tight_mgr.free("ext-sync")
    assert pool.get_num_free_blocks() == 6

    # One fewer block is cleanly rejected before allocation instead of crashing get_new_blocks().
    smaller_mgr = _make_dflash_manager(spec, num_gpu_blocks=6, max_admission_blocks=8)
    smaller_need = smaller_mgr.get_num_blocks_to_allocate("ext-oom", 81, [], 40, 0, 80)
    assert smaller_need == 6
    assert smaller_need > smaller_mgr.block_pool.get_num_free_blocks()

    # Pure async external load without new prompt tokens still predicts only external blocks.
    assert tight_mgr.get_num_blocks_to_allocate("ext-async", 40, [], 40, 0, 40) == 2

    # Combined local + external + new tokens accounts for evictable local blocks, external block, and tail.
    combo_mgr = _make_dflash_manager(spec, num_gpu_blocks=32, max_admission_blocks=8, enable_caching=True)
    combo_pool = combo_mgr.block_pool
    free_before = combo_pool.get_num_free_blocks()
    local_blocks = list(combo_pool.get_new_blocks(9))
    combo_pool.free_blocks(local_blocks)
    combo_predicted = combo_mgr.get_num_blocks_to_allocate("combo", 81, local_blocks, 40, 36, 80)
    assert combo_predicted == 6
    combo_mgr.add_local_computed_blocks("combo", local_blocks, 36, 4)
    combo_mgr.allocate_external_computed_blocks("combo", 36, 4)
    combo_mgr.allocate_new_blocks("combo", 81, 80)
    assert free_before - combo_pool.get_num_free_blocks() == combo_predicted


def test_deepseek_v41_draft_layer_number_handles_non_mtp_names():
    """Draft layer sorting handles non-.mtp. layer names cleanly without IndexError."""
    pytest.importorskip("vllm.transformers_utils.configs.deepseek_v41", reason="DSV4.1 unavailable on this release")
    from vllm_ascend.models.deepseek_v41.cache_config import (
        _draft_layer_number,
        get_layer_tuples,
        make_index_cache_spec,
        make_mla_cache_spec,
        make_swa_cache_spec,
    )

    assert _draft_layer_number("mtp.2.self_attn") == (0, 2, "mtp.2.self_attn")
    assert _draft_layer_number("draft.layers.10.self_attn") == (1, 10, "draft.layers.10.self_attn")
    assert _draft_layer_number("custom_draft_a") == (2, 0, "custom_draft_a")

    mixed_names = [
        "custom_draft_b",
        "draft.layers.10.self_attn.swa_cache",
        "mtp.1.self_attn.swa_cache",
        "draft.layers.2.self_attn.swa_cache",
        "custom_draft_a",
        "mtp.0.self_attn.swa_cache",
    ]
    assert sorted(mixed_names, key=_draft_layer_number) == [
        "mtp.0.self_attn.swa_cache",
        "mtp.1.self_attn.swa_cache",
        "draft.layers.2.self_attn.swa_cache",
        "draft.layers.10.self_attn.swa_cache",
        "custom_draft_a",
        "custom_draft_b",
    ]

    runtime = _make_runtime(method="dspark", num_speculative_tokens=3, num_lookahead_tokens=3)
    mla = make_mla_cache_spec(block_size=64, head_size=512, compress_ratio=2)
    idx = make_index_cache_spec(block_size=64, head_size=128, compress_ratio=1)
    swa = make_swa_cache_spec(
        block_size=64, window_size=128, head_size=512, dtype=torch.bfloat16, cache_dtype="bfloat16"
    )
    draft = as_dflash_sliding_window_spec(swa, runtime, is_draft=True)
    specs: dict[str, KVCacheSpec] = {
        "model.layers.0.self_attn.long_kv_cache": mla,
        "model.layers.0.self_attn.indexer.k_cache": idx,
        "model.layers.1.self_attn.long_kv_cache": mla,
        "model.layers.1.self_attn.indexer.k_cache": idx,
        "draft.layers.10.self_attn.swa_cache": draft,
        "draft.layers.2.self_attn.swa_cache": draft,
        "custom_draft_b": draft,
        "custom_draft_a": draft,
    }
    _, tuples = get_layer_tuples(specs)
    assert "draft.layers.2.self_attn.swa_cache" in tuples[0]
    assert "draft.layers.10.self_attn.swa_cache" in tuples[1]


def test_dflash_max_memory_usage_bytes_falls_back_to_max_in_flight_steps():
    """Startup memory sizing falls back to max_in_flight_steps rather than 1."""
    spec = _make_dflash_spec(
        block_size=16,
        sliding_window=128,
        num_speculative_tokens=3,
        max_in_flight_steps=3,
        num_lookahead_tokens=3,
    )
    expected = spec.max_admission_blocks_per_request(512 * 3, 4096) * spec.page_size_bytes
    for batch_override in ("missing", None, True, False):
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
            model_config=SimpleNamespace(max_model_len=4096),
        )
        if batch_override != "missing":
            cfg.max_concurrent_batches = batch_override
        assert spec.max_memory_usage_bytes(cfg) == expected


@pytest.mark.parametrize("return_bf16_kv", [False, True])
@pytest.mark.parametrize("skip_padding_slots", [False, True])
def test_exec_kv_prefill_quant_flashmla_coexists_with_skip_padding_slots(
    monkeypatch, return_bf16_kv: bool, skip_padding_slots: bool
):
    """C8 QuantFlashMLA prefill preserves native -1 slot scatter alongside skip_padding_slots."""
    from types import MethodType

    from vllm_ascend.attention import mla_v1

    impl: Any = mla_v1.AscendMLAImpl.__new__(mla_v1.AscendMLAImpl)
    impl.use_pcp = False
    impl.use_mla_rope = True
    impl.use_quant_flashmla = True
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 2
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    impl._validate_mla_cache_layout = lambda _cache: None
    impl._quant_flashmla_writer_scales = (torch.ones(1), torch.ones(1))
    impl._flashmla_writer_buffers = (
        torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16),
        torch.zeros(1, 4, 1, 2, dtype=torch.bfloat16),
        torch.arange(4, dtype=torch.int32),
    )
    impl._exec_quant_flashmla_kv_writer = MethodType(mla_v1.AscendMLAImpl._exec_quant_flashmla_kv_writer, impl)

    fused = torch.zeros(1, 4, 1, 6, dtype=torch.bfloat16)
    kv_cache = (fused[..., :4], fused[..., 4:])
    slots = torch.tensor([-1, 3, -1, 1], dtype=torch.int32)
    current_rope = torch.full((4, 1, 2), 5.0, dtype=torch.bfloat16)
    current_nope = torch.full((4, 1, 4), 3.0, dtype=torch.bfloat16)
    seen_scatters: list[bool] = []

    def fake_writer(*args, **kwargs):
        assert kwargs["is_output_kv"] is return_bf16_kv
        assert torch.equal(args[4], torch.arange(4, dtype=torch.int32))
        rope, nope = args[5], args[6]
        if return_bf16_kv:
            return rope, nope, current_rope, current_nope
        return rope, nope, None, None

    def fake_scatter(*, key, value, key_cache, value_cache, slot_mapping):
        assert key.shape[0] == value.shape[0] == slots.shape[0] == 4
        assert slot_mapping is slots
        seen_scatters.append(True)

    monkeypatch.setattr(mla_v1.torch_npu, "npu_kv_rmsnorm_rope_cache", fake_writer)
    monkeypatch.setattr(mla_v1.DeviceOperator, "reshape_and_cache", fake_scatter)

    out_rope, out_nope = impl.exec_kv_prefill(
        torch.ones(4, 6, dtype=torch.bfloat16),
        torch.ones(4, 2, dtype=torch.bfloat16),
        torch.zeros(4, 2, dtype=torch.bfloat16),
        kv_cache,
        slots,
        attn_metadata=SimpleNamespace(),
        return_bf16_kv=return_bf16_kv,
        skip_padding_slots=skip_padding_slots,
    )
    assert seen_scatters == [True]
    if return_bf16_kv:
        assert out_rope is current_rope and out_nope is current_nope
    else:
        assert out_rope is None and out_nope is None


def test_dflash_admission_cap_bounds_rejection_and_one_block_decode():
    """Capped admission must not over-reserve on spec rejection or 1-block decode tails."""
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    manager = _make_dflash_manager(_make_dflash_spec(), num_gpu_blocks=32, max_admission_blocks=10, enable_caching=True)
    manager.req_to_blocks["req"] = [manager._null_block] * 100 + [
        KVCacheBlock(block_id=i + 1, ref_cnt=1) for i in range(3)
    ]
    manager.num_cached_block["req"] = 103

    for caller in (
        lambda tok: manager.get_num_blocks_to_allocate("req", tok, [], 0, 0, tok, apply_admission_cap=True),
        lambda tok: patch_kv_cache_admission.get_num_blocks_to_allocate(
            manager, "req", tok, [], 0, 0, tok, apply_admission_cap=True
        ),
    ):
        assert caller(408) == 0  # Speculative rejection shrinks below len(blocks).
        assert caller(412) == 0  # Exact match to currently backed slots.
        assert caller(416) == 1  # Single new tail block needed, not cap - 3 == 7.
        assert caller(800) == 7  # Long chunk/in-flight suffix capped at remaining 7.


@pytest.mark.parametrize("sharing", ["exclusive", "shared", "mixed"])
def test_dflash_shared_held_blocks_do_not_offset_admission_capacity(sharing: str):
    """Shared tail blocks (ref_cnt > 1) never offset cap; only exclusive blocks do."""
    spec = _make_dflash_spec()
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager = AscendDFlashSlidingWindowManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=4,
        max_admission_blocks_per_request=6,
    )
    computed = 40
    manager.allocate_external_computed_blocks("request", 0, computed)
    manager.remove_skipped_blocks("request", computed)
    blocks = manager.req_to_blocks["request"]
    loaded_blocks = [block for block in blocks if not block.is_null]
    assert len(loaded_blocks) == 2

    exclusive_blocks = len(loaded_blocks)
    if sharing != "exclusive":
        manager.add_local_computed_blocks("other", list(blocks), computed, 0)
        assert all(block.ref_cnt == 2 for block in loaded_blocks)
        exclusive_blocks = 0
        if sharing == "mixed":
            exclusive_blocks = 2
            manager.allocate_new_blocks("request", 48, 48)
            assert all(block.ref_cnt == 1 for block in blocks[-exclusive_blocks:])

    target_tokens = 120
    # Uncapped tail-only demand trims leading unbacked blocks ((120 - 10) // 4 == 27 -> 30 - 27 == 3).
    assert manager.get_num_blocks_to_allocate("request", target_tokens, [], computed, computed, target_tokens) == 3
    assert (
        manager.get_num_blocks_to_allocate(
            "request", target_tokens, [], computed, computed, target_tokens, apply_admission_cap=True
        )
        == 6 - exclusive_blocks
    )
    assert (
        patch_kv_cache_admission.get_num_blocks_to_allocate(
            manager, "request", target_tokens, [], computed, computed, target_tokens, apply_admission_cap=True
        )
        == 6 - exclusive_blocks
    )
    tail_tokens = (len(blocks) + 1) * 4
    assert (
        manager.get_num_blocks_to_allocate(
            "request", tail_tokens, [], computed, computed, tail_tokens, apply_admission_cap=True
        )
        == 1
    )
    rejected_tokens = (len(blocks) - 1) * 4
    assert (
        manager.get_num_blocks_to_allocate(
            "request", rejected_tokens, [], computed, computed, rejected_tokens, apply_admission_cap=True
        )
        == 0
    )
    manager.max_admission_blocks_per_request = exclusive_blocks
    assert (
        manager.get_num_blocks_to_allocate(
            "request", target_tokens, [], computed, computed, target_tokens, apply_admission_cap=True
        )
        == 0
    )

    free_before = pool.get_num_free_blocks()
    manager.free("request")
    assert pool.get_num_free_blocks() == free_before + exclusive_blocks
    if sharing != "exclusive":
        assert all(block.ref_cnt == 1 for block in loaded_blocks)
        manager.free("other")
    assert pool.get_num_free_blocks() == 31


def test_dflash_max_concurrent_batches_inflight_chunk_reservation():
    """Multi-batch concurrent admission keeps reserving cap - exclusive across null gaps."""
    spec = _make_dflash_spec(
        block_size=4,
        sliding_window=8,
        num_speculative_tokens=2,
        max_in_flight_steps=2,
        num_lookahead_tokens=1,
    )
    cap = spec.max_admission_blocks_per_request(max_in_flight_tokens=16, max_model_len=512)
    manager = _make_dflash_manager(spec, num_gpu_blocks=64, max_admission_blocks=cap, enable_caching=False)
    manager.allocate_new_blocks("req", 41, 40)
    manager.allocate_new_blocks("req", 81, 80)
    manager.remove_skipped_blocks("req", 80)
    exclusive = sum(1 for block in manager.req_to_blocks["req"] if not block.is_null and block.ref_cnt == 1)
    assert exclusive > 0

    for caller in (
        lambda tok: manager.get_num_blocks_to_allocate("req", tok, [], 80, 80, tok, apply_admission_cap=True),
        lambda tok: patch_kv_cache_admission.get_num_blocks_to_allocate(
            manager, "req", tok, [], 80, 80, tok, apply_admission_cap=True
        ),
    ):
        assert caller(200) == cap - exclusive
        assert caller(len(manager.req_to_blocks["req"]) * 4 + 4) == 1
        assert caller(len(manager.req_to_blocks["req"]) * 4 - 4) == 0


@pytest.mark.parametrize("ref_cnt", [0, 1, 2])
def test_dflash_first_time_admission_local_hit_evictable_vs_pinned(ref_cnt: int):
    """First-time capped admission accounts for local hits, evictable blocks, and partial CoW."""
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    spec = _make_dflash_spec(block_size=4, sliding_window=16)
    manager = _make_dflash_manager(spec, num_gpu_blocks=64, max_admission_blocks=6, enable_caching=True)
    assert manager.get_num_blocks_to_allocate("cold", 80, [], 0, 0, 80, apply_admission_cap=True) == 6

    hit_blocks = [manager._null_block] * 26 + [KVCacheBlock(block_id=i + 1, ref_cnt=ref_cnt) for i in range(4)]
    expected_full_hit = 4 if ref_cnt == 0 else 0
    assert (
        manager.get_num_blocks_to_allocate("hit-full", 120, hit_blocks, 120, 120, 120, apply_admission_cap=True)
        == expected_full_hit
    )
    expected_near_hit = 5 if ref_cnt == 0 else 1
    assert (
        manager.get_num_blocks_to_allocate("hit-near", 124, hit_blocks, 120, 120, 124, apply_admission_cap=True)
        == expected_near_hit
    )
    assert manager.get_num_blocks_to_allocate("hit-long", 200, hit_blocks, 120, 120, 200, apply_admission_cap=True) == 6
    if ref_cnt > 0:
        assert (
            manager.get_num_blocks_to_allocate("hit-cow", 124, hit_blocks, 118, 118, 124, apply_admission_cap=True) == 2
        )
