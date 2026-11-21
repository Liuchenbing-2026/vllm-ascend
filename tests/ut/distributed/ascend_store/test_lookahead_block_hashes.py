from __future__ import annotations

import hashlib
import unittest
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import MambaSpec

try:
    from vllm.v1.kv_cache_interface import AttentionSpec as _AttentionSpec
except ImportError:
    _AttentionSpec = None  # type: ignore[assignment,misc]

try:
    from vllm.v1.kv_cache_interface import FullAttentionSpec as _FullAttentionSpec
except ImportError:
    _FullAttentionSpec = None  # type: ignore[assignment,misc]

try:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec as _KVCacheGroupSpec
except ImportError:
    _KVCacheGroupSpec = None  # type: ignore[assignment,misc]

try:
    from vllm.v1.kv_cache_interface import KVCacheGroup as _KVCacheGroup, ManagerConfig as _ManagerConfig
except ImportError:
    _KVCacheGroup = None  # type: ignore[assignment,misc]
    _ManagerConfig = None  # type: ignore[assignment,misc]

try:
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager as _FullAttentionManager,
        MambaManager as _MambaManager,
    )
except ImportError:
    _FullAttentionManager = None  # type: ignore[assignment,misc]
    _MambaManager = None  # type: ignore[assignment,misc]

from vllm.v1.outputs import KVConnectorOutput
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import coordinator as coord_mod
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    group_block_size_signature,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_layout import draft_only_group_ids
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import InFlightSaveBlockTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
    RequestTracker,
    get_reusable_prefix_length,
    get_sharded_mamba_group_ids,
    hashed_prefix_length,
    infer_cacheable_group_ids,
    infer_group_block_sizes,
    resolve_legacy_eagle_drop,
    resolve_num_hash_lookahead_tokens,
    trim_stale_lookahead_hashes,
    uses_hybrid_kv_cache,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

# isort: on


def _hashes(count: int) -> list[bytes]:
    return [f"hash-{i:04d}".encode().ljust(32, b"\x00") for i in range(count)]


def _full_spec(block_size: int = 128):
    if _FullAttentionSpec is not None:
        return _FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=16, dtype=torch.float16)
    if _AttentionSpec is not None:
        return _AttentionSpec(block_size=block_size, num_kv_heads=1, head_size=16, dtype=torch.float16)
    return SimpleNamespace(
        block_size=block_size,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
        dcp_sharded=False,
        prefix_cacheable=True,
    )


def _mamba_spec(block_size: int = 128, num_speculative_blocks: int = 0) -> MambaSpec:
    kwargs: dict[str, object] = {
        "block_size": block_size,
        "shapes": ((1,),),
        "dtypes": (torch.float32,),
    }
    if is_dataclass(MambaSpec):
        spec_fields = {f.name for f in fields(MambaSpec)}
        if "mamba_cache_mode" in spec_fields:
            kwargs["mamba_cache_mode"] = "align"
        if "num_speculative_blocks" in spec_fields:
            kwargs["num_speculative_blocks"] = num_speculative_blocks
    elif _KVCacheGroupSpec is not None:
        kwargs["mamba_cache_mode"] = "align"
        kwargs["num_speculative_blocks"] = num_speculative_blocks
    return MambaSpec(**kwargs)  # type: ignore[arg-type]


def _make_full_group(layer_names: list[str], block_size: int = 128, *, is_eagle_group: bool = False):
    spec = _full_spec(block_size)
    if _KVCacheGroupSpec is not None:
        return _KVCacheGroupSpec(layer_names, spec, is_eagle_group=is_eagle_group)
    if _KVCacheGroup is not None and _ManagerConfig is not None and _FullAttentionManager is not None:
        manager_cfg = _ManagerConfig(_FullAttentionManager, _FullAttentionManager.Params())
        return _KVCacheGroup({name: spec for name in layer_names}, manager_cfg, block_size)
    return SimpleNamespace(
        kv_cache_specs={name: spec for name in layer_names},
        layer_names=layer_names,
        block_size=block_size,
        prefix_cacheable=True,
    )


def _make_mamba_group(layer_names: list[str], block_size: int = 128, num_speculative_blocks: int = 0):
    spec = _mamba_spec(block_size, num_speculative_blocks=num_speculative_blocks)
    if _KVCacheGroupSpec is not None:
        return _KVCacheGroupSpec(layer_names, spec)
    if _KVCacheGroup is not None and _ManagerConfig is not None and _MambaManager is not None:
        manager_cfg = _ManagerConfig(
            _MambaManager,
            _MambaManager.Params(
                mamba_cache_mode="align",
                num_speculative_blocks=num_speculative_blocks,
            ),
        )
        return _KVCacheGroup({name: spec for name in layer_names}, manager_cfg, block_size)
    return SimpleNamespace(
        kv_cache_specs={name: spec for name in layer_names},
        layer_names=layer_names,
        block_size=block_size,
        prefix_cacheable=True,
        manager_config=SimpleNamespace(
            params=SimpleNamespace(
                mamba_cache_mode="align",
                num_speculative_blocks=num_speculative_blocks,
            )
        ),
    )


class _FakeBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 1 if block_id == 0 else 0


class _FakeBlockPool:
    def __init__(self, num_blocks: int = 32) -> None:
        self.num_gpu_blocks = num_blocks
        self.blocks = [_FakeBlock(i) for i in range(num_blocks)]
        self.freed_order: list[int] = []

    def touch(self, blocks: list[_FakeBlock]) -> None:
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks: list[_FakeBlock]) -> None:
        for block in blocks:
            assert block.ref_cnt > 0
            block.ref_cnt -= 1
            self.freed_order.append(block.block_id)

    def get_num_free_blocks(self) -> int:
        return sum(1 for block in self.blocks[1:] if block.ref_cnt == 0)


class _DummyTokenDatabase(ChunkedTokenDatabase):
    def __init__(self, block_size: int = 128, hash_block_size: int | None = None) -> None:
        meta = KeyMetadata(
            model_name="m",
            dcp_rank=0,
            head_or_tp_rank=0,
            pp_rank=0,
        )
        super().__init__(metadata=[meta], block_size=[block_size], partitions=None, hash_block_size=hash_block_size)


class TestConnectorAndConfigResolution(unittest.TestCase):
    def test_supports_lookahead_block_hashes_returns_true(self):
        cfg = SimpleNamespace(kv_connector="AscendStoreConnector", kv_role="kv_both")
        self.assertTrue(AscendStoreConnector.supports_lookahead_block_hashes(cfg))

    def test_multi_connector_propagation_respects_all_children(self):
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
        except ImportError:
            self.skipTest("MultiConnector module not available")
        supports_fn = getattr(MultiConnector, "supports_lookahead_block_hashes", None)
        if not callable(supports_fn):
            self.skipTest("vLLM baseline does not expose MultiConnector.supports_lookahead_block_hashes")

        class _NoLookaheadConnector:
            @classmethod
            def supports_lookahead_block_hashes(cls, kv_transfer_config):
                return False

        parent_cfg = SimpleNamespace(
            engine_id="e0",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "AscendStoreConnector", "kv_role": "kv_both"},
                ]
            },
        )
        with (
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector.KVConnectorConfig",
                side_effect=lambda **kw: SimpleNamespace(**kw),
                create=True,
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector.KVTransferConfig",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector.KVConnectorFactory.get_connector_class",
                return_value=AscendStoreConnector,
            ),
        ):
            self.assertTrue(MultiConnector.supports_lookahead_block_hashes(parent_cfg))

        mixed_cfg = SimpleNamespace(
            engine_id="e0",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "AscendStoreConnector", "kv_role": "kv_both"},
                    {"kv_connector": "LegacyConnector", "kv_role": "kv_both"},
                ]
            },
        )
        with (
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector.KVTransferConfig",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector.KVConnectorFactory.get_connector_class",
                side_effect=[AscendStoreConnector, _NoLookaheadConnector],
            ),
        ):
            self.assertFalse(MultiConnector.supports_lookahead_block_hashes(mixed_cfg))

    def test_resolve_lookahead_and_eagle_across_vllm_baselines(self):
        # 1. Lookahead active (c410543d / #66234 MTP): disables legacy eagle block drop.
        mtp_cfg = SimpleNamespace(
            num_hash_lookahead_tokens=3,
            speculative_config=SimpleNamespace(use_eagle=lambda: True, use_eagle_block_drop=lambda: True),
        )
        lookahead = resolve_num_hash_lookahead_tokens(mtp_cfg)
        self.assertEqual(lookahead, 3)
        self.assertFalse(resolve_legacy_eagle_drop(mtp_cfg, lookahead))

        # 2. v0.33.0 / DSpark with L=0: keeps use_eagle() drop enabled.
        dspark_cfg = SimpleNamespace(
            num_hash_lookahead_tokens=0,
            speculative_config=SimpleNamespace(use_eagle=lambda: True, use_eagle_block_drop=lambda: False),
        )
        self.assertEqual(resolve_num_hash_lookahead_tokens(dspark_cfg), 0)
        self.assertTrue(resolve_legacy_eagle_drop(dspark_cfg, 0))

        # 3. Post-#66234 non-eagle speculative config or mock config.
        mock_cfg = MagicMock()
        self.assertEqual(resolve_num_hash_lookahead_tokens(mock_cfg), 0)
        self.assertFalse(resolve_legacy_eagle_drop(mock_cfg, 0))

    def test_reusable_prefix_and_stale_hash_trimming(self):
        for lookahead, length, align, expected in [
            (0, 128, 128, 0),
            (0, 129, 128, 128),
            (1, 128, 128, 0),
            (1, 129, 128, 128),
            (3, 130, 128, 0),
            (3, 131, 128, 128),
            (5, 260, 128, 128),
            (5, 261, 128, 256),
        ]:
            with self.subTest(lookahead=lookahead, length=length, align=align):
                self.assertEqual(get_reusable_prefix_length(length, lookahead, align), expected)

        hashes = _hashes(4)
        self.assertEqual(hashed_prefix_length(hashes, 128), 512)
        # With N=257 tokens and L=3, only floor((257-3)/128) = 1 hash is valid.
        trimmed = trim_stale_lookahead_hashes(hashes, num_tokens=257, hash_block_size=128, num_hash_lookahead_tokens=3)
        self.assertEqual(trimmed, hashes[:1])
        # With L=0, legacy caller preserves provided list length.
        self.assertEqual(
            trim_stale_lookahead_hashes(hashes, num_tokens=257, hash_block_size=128, num_hash_lookahead_tokens=0),
            hashes,
        )


class TestCoordinatorLookaheadAndBaselines(unittest.TestCase):
    def test_coordinator_clears_eagle_flags_when_lookahead_enabled(self):
        groups = [_make_full_group(["layer.0"], is_eagle_group=True)]
        coord = AscendStoreCoordinator(
            groups,
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["default"],
            use_eagle=True,
            num_hash_lookahead_tokens=3,
        )
        self.assertFalse(coord.use_eagle)
        self.assertEqual(coord.eagle_group_ids, set())
        self.assertEqual(coord.eagle_attn_group_indices, set())

    def test_store_lookup_and_load_masks_pad_unhashed_tail_with_false(self):
        groups = [_make_full_group(["layer.0"])]
        coord = AscendStoreCoordinator(
            groups,
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["default"],
            num_hash_lookahead_tokens=3,
        )
        # 512 aligned tokens = 4 blocks, only 2 committed lookahead hashes.
        store_masks = coord.store_mask(512, num_prompt_tokens=260, num_block_hashes=2)
        self.assertEqual(store_masks, ([True, True, False, False],))

        lookup_masks = coord.lookup_mask(512, num_block_hashes=2)
        self.assertEqual(lookup_masks, ([True, True, False, False],))

        # load_mask clamps unhashed suffix and pads trailing entries with False.
        load_masks = coord.load_mask(_hashes(2), 512)
        self.assertEqual(load_masks, ([True, True, False, False],))

        # Legacy L=0 preserves uncapped store/lookup behavior.
        legacy_coord = AscendStoreCoordinator(
            groups,
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["default"],
            num_hash_lookahead_tokens=0,
        )
        self.assertEqual(legacy_coord.store_mask(512, num_prompt_tokens=260, num_block_hashes=2), ([True] * 4,))

    def test_find_longest_cache_hit_and_reachable_hits_bounded_by_hashes(self):
        groups = [_make_full_group(["layer.0"])]
        hashes = _hashes(2)
        queried: list[tuple[int, int]] = []

        def fake_hit(**kwargs):
            queried.append((kwargs["max_length"], len(kwargs["block_hashes"])))
            count = kwargs["max_length"] // 128
            return ([KVCacheBlock(i + 1) for i in range(count)],), count * 128

        class _HitFA:
            find_longest_cache_hit = staticmethod(fake_hit)

        with patch.object(coord_mod, "_get_group_manager_class", return_value=_HitFA):
            coord2 = AscendStoreCoordinator(
                groups,
                scheduler_block_size=128,
                hash_block_size=128,
                group_block_sizes=[128],
                group_cache_families=["default"],
                num_hash_lookahead_tokens=1,
            )
        pool = ExternalCachedBlockPool(128, {(0, bytes(h)) for h in hashes})
        _, hit_len = coord2.find_longest_cache_hit(hashes, max_length=512, cached_block_pool=pool)
        self.assertEqual(hit_len, 256)
        self.assertEqual(queried[-1], (256, 2))

        probed: list[int] = []

        def query_group(group_id, group_hashes, lookup_mask):
            probed.append(len(group_hashes))
            return group_hashes

        hit = coord2.find_reachable_hit_tokens(hashes, 512, query_group)
        self.assertEqual(hit, 256)
        self.assertEqual(probed, [2])

        # Verify ChunkedTokenDatabase.store_mask forwards num_block_hashes and falls back on 2-arg mocks.
        db = _DummyTokenDatabase(block_size=128)
        db.cache_coordinator = coord2
        self.assertEqual(db.store_mask(512, 260, num_block_hashes=2), ([True, True, False, False],))
        legacy_mock = MagicMock()
        legacy_mock.store_mask.side_effect = [TypeError("unexpected keyword"), ([True, True],)]
        db.cache_coordinator = legacy_mock
        self.assertEqual(db.store_mask(256, 256, num_block_hashes=2), ([True, True],))

    def test_mamba_sparse_retention_receives_lookahead_replay_boundaries(self):
        calls: list[dict[str, object]] = []

        class _Post66234FullManager:
            @classmethod
            def find_longest_cache_hit(
                cls,
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                block_size,
                manager_config,
                alignment_tokens,
                dcp_world_size=1,
                pcp_world_size=1,
            ):
                calls.append({"fn": "hit", "block_size": block_size, "manager_config": manager_config})
                blocks = tuple([KVCacheBlock(1)] * (max_length // block_size) for _ in kv_cache_group_ids)
                return blocks, (max_length // block_size) * block_size

            @classmethod
            def reachable_block_mask(
                cls,
                start_block,
                end_block,
                alignment_tokens,
                block_size,
                manager_config,
                retention_interval=None,
                reachable_boundaries=(),
            ):
                return None

        class _Post66234MambaManager:
            @classmethod
            def find_longest_cache_hit(
                cls,
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                block_size,
                manager_config,
                alignment_tokens,
            ):
                blocks = tuple([KVCacheBlock(1)] * (max_length // block_size) for _ in kv_cache_group_ids)
                return blocks, (max_length // block_size) * block_size

            @classmethod
            def reachable_block_mask(
                cls,
                start_block,
                end_block,
                alignment_tokens,
                block_size,
                manager_config,
                retention_interval=None,
                reachable_boundaries=(),
            ):
                calls.append(
                    {
                        "fn": "mask",
                        "block_size": block_size,
                        "manager_config": manager_config,
                        "reachable_boundaries": tuple(reachable_boundaries),
                    }
                )
                return [i == 1 for i in range(start_block, end_block)]

        # Post-#66234 KVCacheGroup shape: kv_cache_specs + manager_config + block_size, no kv_cache_spec/is_eagle_group.
        fa_group = SimpleNamespace(
            kv_cache_specs={"layers.0.attn": _full_spec(128)},
            manager_config=SimpleNamespace(manager_cls=_Post66234FullManager, params="fa"),
            block_size=128,
            prefix_cacheable=True,
        )
        mamba_group = SimpleNamespace(
            kv_cache_specs={"layers.1.mamba": _mamba_spec(128)},
            manager_config=SimpleNamespace(manager_cls=_Post66234MambaManager, params="mamba"),
            block_size=128,
            prefix_cacheable=True,
        )
        coord = AscendStoreCoordinator(
            [fa_group, mamba_group],
            scheduler_block_size=128,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["default", "default"],
            use_eagle=True,
            retention_interval=0,
            num_hash_lookahead_tokens=3,
        )
        # N=258, L=3 -> resend=255, continuation=258
        masks = coord.store_mask(512, num_prompt_tokens=258, num_block_hashes=2)
        self.assertEqual(masks[0], [True, True, False, False])
        self.assertEqual(masks[1], [False, True, False, False])
        mask_call = [c for c in calls if c["fn"] == "mask"][-1]
        self.assertEqual(mask_call["reachable_boundaries"], (255, 258))
        self.assertEqual(mask_call["manager_config"].params, "mamba")

        _, hit_len = coord.find_longest_cache_hit(_hashes(2), 512, ExternalCachedBlockPool(128))
        self.assertEqual(hit_len, 256)
        hit_call = [c for c in calls if c["fn"] == "hit"][-1]
        self.assertEqual(hit_call["block_size"], 128)
        self.assertEqual(hit_call["manager_config"].params, "fa")


class TestReqMetaAndInFlightSaveTracker(unittest.TestCase):
    def test_from_request_tracker_bounds_unhashed_tail_and_slices_lookahead_tokens(self):
        tracker = RequestTracker(
            req_id="r1",
            token_len=130,
            allocated_block_ids=[1, 2],
            num_saved_tokens=0,
            token_ids=list(range(130)),
            num_prompt_tokens=130,
        )
        all_tokens = list(range(131))
        # With L=3 and only 1 hash (covering 0..128), discard_partial_chunks=False still caps at 128.
        meta = ReqMeta.from_request_tracker(
            tracker,
            cache_transfer_granularity=128,
            discard_partial_chunks=False,
            block_hashes=_hashes(1),
            hash_block_size=128,
            num_hash_lookahead_tokens=3,
            all_token_ids=all_tokens,
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.save_end_token, 128)
        self.assertEqual(meta.token_len_chunk, 128)
        self.assertEqual(meta.num_hash_lookahead_tokens, 3)
        self.assertEqual(meta.token_ids, list(range(131)))

        # Second call with no additional hashes yields None instead of zero-token save.
        meta2 = ReqMeta.from_request_tracker(
            tracker,
            cache_transfer_granularity=128,
            discard_partial_chunks=False,
            block_hashes=_hashes(1),
            hash_block_size=128,
            num_hash_lookahead_tokens=3,
            all_token_ids=all_tokens,
        )
        self.assertIsNone(meta2)

        # Legacy L=0 keeps discard_partial_chunks=False full token_len save.
        legacy_tracker = RequestTracker(
            req_id="r0",
            token_len=20,
            allocated_block_ids=[1, 2],
            num_saved_tokens=0,
        )
        legacy_meta = ReqMeta.from_request_tracker(
            legacy_tracker,
            cache_transfer_granularity=16,
            discard_partial_chunks=False,
            block_hashes=_hashes(1),
            num_hash_lookahead_tokens=0,
        )
        self.assertIsNotNone(legacy_meta)
        self.assertEqual(legacy_meta.save_end_token, 20)

    def test_in_flight_tracker_retain_finished_excludes_mamba_scratch_and_null_blocks(self):
        pool = _FakeBlockPool(16)
        tracker = InFlightSaveBlockTracker(expected_worker_count=2)
        req_meta = ReqMeta(
            req_id="r-mamba",
            save_end_token=256,
            block_ids_by_group=[[0, 2, 3, 2], [4, 5, 6], [7, 8]],
            can_save=True,
        )
        # Group 1 has 1 trailing speculative scratch block (block 6), group 2 is private/non-cacheable.
        retained = tracker.retain_finished(
            req_meta,
            cacheable_group_ids=[0, 1],
            speculative_blocks_by_group={1: 1},
            block_pool=pool,
        )
        self.assertTrue(retained)
        event_id = req_meta.event_id
        self.assertIsNotNone(event_id)
        # Pinned deduplicated blocks: [2, 3, 4, 5] (no 0, no scratch 6, no private 7/8).
        self.assertEqual(tracker.sending_blocks[event_id], [2, 3, 4, 5])
        for bid in (2, 3, 4, 5):
            self.assertEqual(pool.blocks[bid].ref_cnt, 1)
        for bid in (6, 7, 8):
            self.assertEqual(pool.blocks[bid].ref_cnt, 0)

        # First worker report keeps blocks pinned; second worker releases tail-first.
        tracker.complete({event_id: 1}, pool)
        self.assertTrue(tracker.has_pending_work())
        tracker.complete({event_id: 1}, pool)
        self.assertFalse(tracker.has_pending_work())
        self.assertEqual(pool.freed_order, [5, 4, 3, 2])


class _FakeLookupClient:
    def __init__(self, hit_to_return: int) -> None:
        self.closed = False
        self.calls: list[tuple[int, list[bytes]]] = []
        self.hit_to_return = hit_to_return

    def lookup(self, token_len, block_hashes, kv_cache_group_ids, hbm_hit_tokens=0):
        self.calls.append((token_len, list(block_hashes)))
        return min(self.hit_to_return, token_len)

    def reset(self) -> bool:
        return True


def _make_base_scheduler(
    *,
    num_hash_lookahead_tokens: int = 3,
    save_decode_cache: bool = False,
    discard_partial_chunks: bool = True,
    use_layerwise: bool = False,
    enable_kv_events: bool = True,
    world_size: int = 1,
    cls=KVPoolScheduler,
):
    sched = cls.__new__(cls)
    sched.vllm_config = SimpleNamespace(
        use_v2_model_runner=False,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
    )
    sched.use_layerwise = use_layerwise
    sched.layerwise_offload = False
    sched.use_hybrid = False
    sched.kv_cache_config = None
    sched.kv_cache_group_ids = [0]
    sched.kv_cache_group_families = ["default"]
    sched.num_speculative_blocks_by_group = {}
    sched.kv_role = "kv_both"
    sched.consumer_is_to_load = True
    sched.consumer_is_to_put = False
    sched.load_async = False
    sched.enable_kv_events = enable_kv_events
    sched.retention_interval = None
    sched.save_decode_cache = save_decode_cache
    sched.can_put = True
    sched.load_specs = {}
    sched.pcp_size = 1
    sched.dcp_size = 1
    sched.num_hash_lookahead_tokens = num_hash_lookahead_tokens
    sched.use_eagle = False
    sched.original_block_size = [128]
    sched.cacheable_group_ids = [0]
    sched.swa_group_ids = set()
    sched.grouped_block_size = [128]
    sched.hash_block_size = 128
    sched._block_size = 128
    sched.lcm_block_size = 128
    sched.cache_transfer_granularity = 128
    sched.cache_coordinator = None
    sched._request_trackers = {}
    sched._preempted_req_ids = set()
    sched._discard_partial_chunks = discard_partial_chunks
    sched._unfinished_requests = {}
    sched._async_loading_req_ids = set()
    sched._finished_lookahead_metas = {}
    sched._in_flight_save_blocks = InFlightSaveBlockTracker(world_size)
    sched._block_pool = _FakeBlockPool(32)
    sched._kv_cache_manager = None
    sched.tp_mismatch = False
    sched.client = None
    return sched


class TestSchedulerAndWorkerLookaheadLifecycle(unittest.TestCase):
    def test_blockwise_and_layerwise_lookup_enforce_reusable_prefix_and_trim_stale_hashes(self):
        sched = _make_base_scheduler(num_hash_lookahead_tokens=3)
        fake_client = _FakeLookupClient(hit_to_return=512)
        sched.client = fake_client

        # Prompt has 259 tokens, lookahead=3 -> reusable prefix = floor((259-3)/128)*128 = 256.
        # Pass 4 hashes (with 2 stale hashes); lookup must trim to 2 valid hashes and probe 256.
        req_ok = SimpleNamespace(
            request_id="req-ok",
            prompt_token_ids=list(range(259)),
            num_tokens=259,
            block_hashes=_hashes(4),
        )
        new_matched, _ = sched.get_num_new_matched_tokens(req_ok, 0)
        self.assertEqual(new_matched, 256)
        self.assertEqual(fake_client.calls[-1][0], 256)
        self.assertEqual(len(fake_client.calls[-1][1]), 2)

        # Prompt with 258 tokens and L=3 -> reusable prefix = 128 (cannot hit 256).
        req_short = SimpleNamespace(
            request_id="req-short",
            prompt_token_ids=list(range(258)),
            num_tokens=258,
            block_hashes=_hashes(3),
        )
        new_matched_short, _ = sched.get_num_new_matched_tokens(req_short, 0)
        self.assertEqual(new_matched_short, 128)
        self.assertEqual(fake_client.calls[-1][0], 128)
        self.assertEqual(len(fake_client.calls[-1][1]), 1)

        # Layerwise scheduler contiguous query also trims stale hashes and caps by lookahead reusable length.
        lw_sched = _make_base_scheduler(num_hash_lookahead_tokens=3, use_layerwise=True, cls=LayerwiseKVPoolScheduler)
        lw_sched.use_block_key_layerwise = False
        lw_sched.use_layerwise_transfer = True
        lw_sched.sharded_mamba_group_ids = set()
        lw_sched.dspark_prefix_keys = None
        lw_sched.dspark_draft_only_group_ids = frozenset()
        lw_sched.layerwise_max_transfer_blocks = 0
        lw_sched.tp_size = 1
        lw_sched.put_step = 1
        probed_hashes: list[str] = []
        lw_sched.layerwise_keys = SimpleNamespace(
            make_hit_check_keys=lambda gid, bh, ranks: (probed_hashes.append(bh) or [f"{gid}:{bh}"])
        )
        lw_sched.store_scheduler = SimpleNamespace(batch_is_readable=lambda batch: [True] * len(batch))
        lw_hit, _ = lw_sched.get_num_new_matched_tokens(req_short, 0)
        self.assertEqual(lw_hit, 128)
        self.assertEqual(len(probed_hashes), 1)

    def test_first_decode_step_flushes_unsaved_prefill_lookahead_without_save_decode_cache(self):
        sched = _make_base_scheduler(num_hash_lookahead_tokens=1, save_decode_cache=False)
        req_id = "req-decode-flush"
        prompt = list(range(128))
        real_req = SimpleNamespace(
            request_id=req_id,
            prompt_token_ids=prompt,
            all_token_ids=list(prompt),
            block_hashes=[],
            num_computed_tokens=0,
            num_tokens=128,
        )
        sched._unfinished_requests[req_id] = (real_req, [[1]])

        step1 = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id=req_id,
                    num_computed_tokens=0,
                    prefill_token_ids=prompt,
                    prompt_token_ids=prompt,
                    block_ids=([1],),
                )
            ],
            num_scheduled_tokens={req_id: 128},
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        )
        meta1 = sched.build_connector_meta(step1)
        self.assertEqual(meta1.requests, [])
        self.assertTrue(sched._has_unsaved_prefill_lookahead(req_id))

        # Decode step samples token 128, producing hash[0], with new_block_ids=None.
        real_req.num_computed_tokens = 128
        real_req.all_token_ids.append(999)
        real_req.num_tokens = 129
        real_req.block_hashes = _hashes(1)
        step2 = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            num_scheduled_tokens={req_id: 1},
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[req_id],
                new_block_ids=[None],
                num_computed_tokens=[128],
            ),
        )
        meta2 = sched.build_connector_meta(step2)
        self.assertEqual(len(meta2.requests), 1)
        self.assertEqual(meta2.requests[0].save_end_token, 128)
        self.assertEqual(meta2.requests[0].token_ids, list(range(128)) + [999])
        self.assertFalse(sched._has_unsaved_prefill_lookahead(req_id))

    def test_finished_lookahead_pin_save_and_layerwise_no_forward_flush(self):
        lw_sched = _make_base_scheduler(
            num_hash_lookahead_tokens=1,
            save_decode_cache=False,
            use_layerwise=True,
            cls=LayerwiseKVPoolScheduler,
        )
        lw_sched.client = _FakeLookupClient(0)
        req_id = "req-finish-lw"
        prompt = list(range(128))
        real_req = SimpleNamespace(
            request_id=req_id,
            prompt_token_ids=prompt,
            all_token_ids=prompt + [777],
            block_hashes=_hashes(1),
            num_computed_tokens=128,
            num_tokens=129,
            num_in_flight_tokens=0,
            num_stale_output_tokens=0,
        )
        lw_sched._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            token_len=128,
            allocated_block_ids_by_group=[[4]],
            num_saved_tokens=0,
            num_prompt_tokens=128,
            prefill_end_tokens=128,
        )
        lw_sched._unfinished_requests[req_id] = (real_req, [[4]])

        # Aborted/uncomputed request should not retain or queue saves.
        uncomputed = SimpleNamespace(
            request_id="req-abort",
            prompt_token_ids=prompt,
            all_token_ids=prompt,
            block_hashes=[],
            num_computed_tokens=0,
            num_tokens=128,
        )
        lw_sched._request_trackers["req-abort"] = RequestTracker(
            req_id="req-abort",
            token_len=0,
            allocated_block_ids_by_group=[[5]],
            num_saved_tokens=0,
            num_prompt_tokens=128,
            prefill_end_tokens=128,
        )
        lw_sched.request_finished(uncomputed, [5])
        self.assertNotIn("req-abort", lw_sched._finished_lookahead_metas)
        self.assertEqual(lw_sched._block_pool.blocks[5].ref_cnt, 0)

        # Finished request with sampled lookahead token pins block 4 and queues delayed save.
        delay_free, _ = lw_sched.request_finished_all_groups(real_req, ([4],))
        self.assertFalse(delay_free)
        self.assertTrue(lw_sched.has_pending_push_work())
        self.assertEqual(lw_sched._block_pool.blocks[4].ref_cnt, 1)
        self.assertFalse(lw_sched.reset_store())

        empty_step = SimpleNamespace(
            finished_req_ids={req_id},
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            num_scheduled_tokens={},
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        )
        meta = lw_sched.build_connector_meta(empty_step)
        self.assertEqual(len(meta.requests), 1)
        req_meta = meta.requests[0]
        self.assertEqual(req_meta.save_end_token, 128)
        self.assertIsNotNone(req_meta.event_id)

        # Simulate no-forward step on LayerwiseKVPoolWorker flushing pending layerwise saves.
        worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
        worker.num_layers = 2
        worker.use_block_key_layerwise = False
        worker.can_put = True
        worker.kv_send_thread = MagicMock()
        worker.kv_recv_thread = None
        worker.layer_save_tasks = [[MagicMock()], [MagicMock()]]
        worker._pending_last_save_drain = False
        worker._dflash_save_pending = False
        worker.current_layer = 0
        saved_layers: list[int] = []

        def _fake_save_layer(_metadata):
            saved_layers.append(worker.current_layer)
            worker.current_layer += 1

        worker.save_kv_layer = _fake_save_layer
        worker._step_save_event_ids = [req_meta.event_id]
        worker._completed_save_events = {}

        worker.flush_pending_layerwise_saves(meta)
        self.assertEqual(saved_layers, [0, 1])
        worker_meta = worker.build_connector_worker_meta()
        self.assertEqual(worker_meta, AscendStoreKVConnectorWorkerMetadata({req_meta.event_id: 1}))

        lw_sched.update_connector_output(
            KVConnectorOutput(
                finished_sending=set(),
                finished_recving=set(),
                kv_connector_worker_meta=worker_meta,
            )
        )
        self.assertFalse(lw_sched.has_pending_push_work())
        self.assertEqual(lw_sched._block_pool.blocks[4].ref_cnt, 0)
        self.assertTrue(lw_sched.reset_store())

    def test_sending_thread_emits_lookahead_token_ids_and_merges_overlapping_retry(self):
        thread = KVCacheStoreSendingThread.__new__(KVCacheStoreSendingThread)
        thread.done_task_lock = MagicMock()
        thread.done_task_lock.__enter__ = lambda s: None
        thread.done_task_lock.__exit__ = lambda s, *a: None
        thread.enable_kv_event = True
        thread.stored_requests = {"r-ev": {10, 11}}
        thread._retry_token_ids = {}

        # Step 1 (job 10): save [0:128) with lookahead 128..131 failed -> remembered for retry.
        meta1 = ReqMeta(
            req_id="r-ev",
            save_start_token=0,
            save_end_token=128,
            token_len_chunk=128,
            block_ids_by_group=[[1, 2]],
            block_hashes=_hashes(1),
            token_ids=list(range(131)),
            num_hash_lookahead_tokens=3,
            store_job_id=10,
        )
        thread._remember_retry_token_ids(meta1)
        self.assertEqual(thread._retry_token_ids["r-ev"], (0, list(range(131))))

        # Step 2 (job 11): retry adds block [128:256) with lookahead 256..259.
        meta2 = ReqMeta(
            req_id="r-ev",
            save_start_token=128,
            save_end_token=256,
            token_len_chunk=256,
            block_ids_by_group=[[1, 2]],
            block_hashes=_hashes(2),
            token_ids=list(range(128, 259)),
            num_hash_lookahead_tokens=3,
            store_job_id=11,
        )
        tok0, la0 = thread.get_event_token_and_lookahead_ids(meta2, 0, 128)
        self.assertEqual(tok0, list(range(128)))
        self.assertEqual(la0, [[128, 129, 130]])
        tok1, la1 = thread.get_event_token_and_lookahead_ids(meta2, 128, 256)
        self.assertEqual(tok1, list(range(128, 256)))
        self.assertEqual(la1, [[256, 257, 258]])

        # L=0 omits lookahead_token_ids.
        meta_no_la = ReqMeta(
            req_id="r-ev",
            save_start_token=0,
            save_end_token=128,
            token_len_chunk=128,
            block_ids_by_group=[[1]],
            block_hashes=_hashes(1),
            token_ids=list(range(128)),
            num_hash_lookahead_tokens=0,
            store_job_id=10,
        )
        thread._retry_token_ids.clear()
        tok, la = thread.get_event_token_and_lookahead_ids(meta_no_la, 0, 128)
        self.assertEqual(tok, list(range(128)))
        self.assertIsNone(la)

    def test_worker_get_finished_keeps_pending_save_state_and_lookup_clamps_hashes(self):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        send_thread = MagicMock(spec=KVCacheStoreSendingThread)
        send_thread.is_stored_request.return_value = False
        worker.kv_send_thread = send_thread
        worker.kv_recv_thread = None
        worker.tp_rank = 0
        worker.load_async = False
        worker.use_layerwise = False
        worker._retired_store_req_ids = set()

        meta = AscendConnectorMetadata(preempted_req_ids=set())
        meta.add_request(ReqMeta(req_id="r-pending", save_end_token=128, block_ids_by_group=[[1]], can_save=True))
        worker.get_finished({"r-pending", "r-done"}, meta)
        self.assertIn("r-pending", worker._retired_store_req_ids)
        send_thread.reset_saved_request.assert_called_once_with("r-done")

        # Next step with empty meta retires r-pending cleanly.
        send_thread.reset_saved_request.reset_mock()
        worker.get_finished(set(), AscendConnectorMetadata(preempted_req_ids=set()))
        self.assertNotIn("r-pending", worker._retired_store_req_ids)
        send_thread.reset_saved_request.assert_called_once_with("r-pending")

        # Non-coordinator lookup clamps token_len to hashed coverage when L > 0.
        worker.num_hash_lookahead_tokens = 3
        worker.hash_block_size = 128
        worker.max_model_len = 2048
        worker.cache_transfer_granularity = 128
        worker.group_uses_align_state = [False]
        worker._lookup_with_coordinator = MagicMock(return_value=None)
        worker._build_lookup_keys = MagicMock(return_value=(["k0"], [0], [128]))
        worker._expand_lookup_keys_by_rank = MagicMock(return_value=["k0"])
        worker.m_store = SimpleNamespace(exists=MagicMock(return_value=[1]))

        hit = worker.lookup_scheduler(token_len=512, block_hashes=_hashes(1), kv_cache_group_ids=[0])
        self.assertEqual(hit, 128)
        worker._build_lookup_keys.assert_called_once_with(128, _hashes(1), 0)

    def test_tp_mismatch_events_fallback_when_lookahead_zero_and_emit_when_positive(self):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.tp_rank = 0
        worker.pcp_rank = 0
        worker.pcp_size = 1
        worker.kv_role = "kv_both"
        worker.block_size = 4
        worker.num_sub_keys = 1
        worker.enable_kv_events = True
        worker.token_database = _DummyTokenDatabase(block_size=4)
        worker._build_strided_addrs = MagicMock(return_value=([4096], [8]))
        worker._make_sub_key_str = lambda base, rank: f"{base}@tp:{rank}"
        worker.m_store = SimpleNamespace(put=MagicMock(side_effect=lambda keys, addrs, sizes: [True] * len(keys)))

        send_thread = MagicMock(spec=KVCacheStoreSendingThread)
        send_thread.is_live_store_job.return_value = True
        send_thread.get_saved_offset.return_value = 0
        send_thread.lookup.side_effect = lambda keys: [False] * len(keys)
        send_thread.get_event_token_ids.side_effect = lambda req, start, end: req.get_event_token_ids(start, end)
        worker.kv_send_thread = send_thread

        req_l0 = ReqMeta(
            req_id="r-tp-l0",
            token_len_chunk=4,
            block_ids_by_group=[[5]],
            block_hashes=[b"h0"],
            token_ids=[1, 2, 3, 4],
            original_block_size=4,
            num_hash_lookahead_tokens=0,
            store_job_id=1,
        )
        self.assertTrue(worker._store_kv_tp_mismatch(req_l0))
        send_thread.get_event_token_ids.assert_called_once_with(req_l0, 0, 4)
        events_l0 = send_thread.update_kv_event.call_args.args[0]
        self.assertEqual(len(events_l0), 1)
        self.assertIsNone(getattr(events_l0[0], "lookahead_token_ids", None))

        send_thread.update_kv_event.reset_mock()
        send_thread.get_event_token_and_lookahead_ids.side_effect = lambda req, start, end: (
            req.token_ids[start:end],
            [[5, 6]],
        )
        req_l2 = ReqMeta(
            req_id="r-tp-l2",
            token_len_chunk=4,
            block_ids_by_group=[[5]],
            block_hashes=[b"h0"],
            token_ids=[1, 2, 3, 4, 5, 6],
            original_block_size=4,
            num_hash_lookahead_tokens=2,
            store_job_id=2,
        )
        self.assertTrue(worker._store_kv_tp_mismatch(req_l2))
        send_thread.get_event_token_and_lookahead_ids.assert_called_once_with(req_l2, 0, 4)
        events_l2 = send_thread.update_kv_event.call_args.args[0]
        self.assertEqual(len(events_l2), 1)
        self.assertEqual(events_l2[0].lookahead_token_ids, [[5, 6]])

    def test_running_cached_decode_transition_with_new_blocks_flushes_only_prefill(self):
        sched = _make_base_scheduler(num_hash_lookahead_tokens=3, save_decode_cache=False)
        req_id = "req-decode-new-blk"
        prompt = list(range(128))
        real_req = SimpleNamespace(
            request_id=req_id,
            prompt_token_ids=prompt,
            all_token_ids=prompt + [901, 902, 903, 904],
            block_hashes=_hashes(1),
            num_computed_tokens=131,
            num_tokens=132,
        )
        sched._unfinished_requests[req_id] = (real_req, [[1, 2]])
        sched._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            token_len=128,
            allocated_block_ids_by_group=[[1]],
            num_saved_tokens=0,
            token_ids=list(prompt[:64]),
            num_prompt_tokens=128,
            prefill_end_tokens=128,
        )
        step = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            num_scheduled_tokens={req_id: 1},
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[req_id],
                new_block_ids=[[[2]]],
                num_computed_tokens=[128],
            ),
        )
        meta = sched.build_connector_meta(step)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests[0].save_end_token, 128)
        self.assertEqual(meta.requests[0].token_ids, prompt + [901, 902, 903])
        self.assertEqual(sched._request_trackers[req_id].token_len, 128)


class TestPost66234KVCacheGroupCompat(unittest.TestCase):
    def test_mamba_spec_helper_does_not_mutate_frozen_dataclass(self):
        spec = _mamba_spec(block_size=128, num_speculative_blocks=1)
        self.assertEqual(spec.block_size, 128)

    def test_kvc_group_helpers_support_post_66234_shape(self):
        fa_spec = _full_spec(128)
        mamba_spec = _mamba_spec(128, num_speculative_blocks=1)
        fa_group = SimpleNamespace(
            kv_cache_specs={"layers.0.attn": fa_spec},
            layer_names=["layers.0.attn"],
            manager_config=SimpleNamespace(
                manager_cls=_FullAttentionManager,
                params=SimpleNamespace(),
            ),
            block_size=128,
            prefix_cacheable=True,
        )
        mamba_group = SimpleNamespace(
            kv_cache_specs={"layers.1.mamba": mamba_spec},
            layer_names=["layers.1.mamba"],
            manager_config=SimpleNamespace(
                manager_cls=_MambaManager,
                params=SimpleNamespace(mamba_cache_mode="align", num_speculative_blocks=1),
            ),
            block_size=128,
            prefix_cacheable=True,
        )
        draft_group = SimpleNamespace(
            kv_cache_specs={"draft.0.attn": SimpleNamespace(block_size=128, is_draft=True, prefix_cacheable=False)},
            layer_names=["draft.0.attn"],
            block_size=128,
            prefix_cacheable=False,
        )
        cfg = SimpleNamespace(kv_cache_groups=[fa_group, mamba_group, draft_group])
        sched_cfg = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
        self.assertEqual(infer_cacheable_group_ids(cfg.kv_cache_groups), [0, 1])
        self.assertEqual(infer_group_block_sizes(128, cfg.kv_cache_groups), [128, 128, 128])
        self.assertEqual(get_sharded_mamba_group_ids(cfg), {1})
        self.assertEqual(draft_only_group_ids(cfg.kv_cache_groups), (2,))
        self.assertEqual(group_block_size_signature(fa_group), (128,))
        expanded = get_layerwise_kv_cache_specs(cfg)
        self.assertEqual(set(expanded), {"layers.0.attn", "layers.1.mamba", "draft.0.attn"})
        if _FullAttentionSpec is not None or _FullAttentionManager is not None:
            self.assertTrue(uses_hybrid_kv_cache(sched_cfg, [fa_group, mamba_group]))


def _hash_tokens_with_lookahead(
    tokens: list[int],
    hash_block_size: int = 128,
    num_hash_lookahead_tokens: int = 1,
) -> list[bytes]:
    """Compute chained SHA-256 block hashes matching upstream lookahead hashing."""
    usable = len(tokens) - num_hash_lookahead_tokens
    hashes: list[bytes] = []
    prev = b"root-none-hash".ljust(32, b"\x00")
    start = 0
    while start + hash_block_size <= usable:
        end = start + hash_block_size
        hasher = hashlib.sha256()
        hasher.update(prev)
        hasher.update(bytes(f"{start}:{end}:", "ascii"))
        hasher.update(repr(tuple(tokens[start:end])).encode("ascii"))
        if num_hash_lookahead_tokens > 0:
            la_end = end + num_hash_lookahead_tokens
            hasher.update(repr(tuple(tokens[end:la_end])).encode("ascii"))
        prev = hasher.digest()
        hashes.append(prev)
        start = end
    return hashes


class _RefFullManager:
    supports_fine_grained_hash_lookup = True

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        alignment_tokens,
        kv_cache_spec=None,
        block_size=None,
        manager_config=None,
        drop_eagle_block=False,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        bs = block_size or kv_cache_spec.block_size
        computed = tuple([] for _ in kv_cache_group_ids)
        for bh in list(block_hashes)[: max_length // bs]:
            cached = block_pool.get_cached_block(bh, kv_cache_group_ids)
            if not cached:
                break
            for blocks, blk in zip(computed, cached):
                blocks.append(blk)
        if drop_eagle_block and computed[0]:
            for blocks in computed:
                blocks.pop()
        return computed, len(computed[0]) * bs

    @classmethod
    def reachable_block_mask(cls, start_block, end_block, **kwargs):
        return None


class _RefMambaManager:
    supports_fine_grained_hash_lookup = True

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        alignment_tokens,
        kv_cache_spec=None,
        block_size=None,
        manager_config=None,
        drop_eagle_block=False,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        bs = block_size or kv_cache_spec.block_size
        computed = tuple([] for _ in kv_cache_group_ids)
        max_blocks = min(max_length // bs, len(block_hashes))
        for idx in range(max_blocks - 1, -1, -1):
            cached = block_pool.get_cached_block(block_hashes[idx], kv_cache_group_ids)
            if cached:
                for blocks, blk in zip(computed, cached):
                    blocks.extend([block_pool.null_block] * idx)
                    blocks.append(blk)
                return computed, (idx + 1) * bs
        return computed, 0

    @classmethod
    def reachable_block_mask(cls, start_block, end_block, **kwargs):
        return None


def _fake_ref_manager(group, spec):
    if (
        isinstance(spec, MambaSpec)
        or getattr(getattr(group, "manager_config", None), "manager_cls", None) is _MambaManager
    ):
        return _RefMambaManager
    return _RefFullManager


class _HashAwareFakeLookupClient:
    def __init__(self, worker: KVPoolWorker) -> None:
        self.closed = False
        self.worker = worker

    def lookup(self, token_len, block_hashes, kv_cache_group_ids, hbm_hit_tokens=0):
        return self.worker.lookup_scheduler(
            token_len,
            block_hashes,
            kv_cache_group_ids,
            use_layerwise=False,
            hbm_hit_tokens=hbm_hit_tokens,
        )


class TestLookaheadForkBoundaries(unittest.TestCase):
    def _build_hybrid_fork_env(
        self,
        *,
        prefix_tokens: list[int],
        num_hash_lookahead_tokens: int,
        with_coordinator: bool = True,
    ):
        groups = [_make_full_group(["layers.0.attn"], 128), _make_mamba_group(["layers.1.mamba"], 128)]
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=groups,
            prefix_cache_retention_interval=None,
            num_blocks=64,
        )
        prefix_hashes = _hash_tokens_with_lookahead(
            prefix_tokens,
            hash_block_size=128,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        )
        coord = None
        if with_coordinator:
            with patch.object(coord_mod, "_get_group_manager_class", side_effect=_fake_ref_manager):
                coord = AscendStoreCoordinator(
                    groups,
                    scheduler_block_size=128,
                    hash_block_size=128,
                    group_block_sizes=[128, 128],
                    group_cache_families=["default", "default"],
                    use_eagle=False,
                    retention_interval=None,
                    num_hash_lookahead_tokens=num_hash_lookahead_tokens,
                )

        db = ChunkedTokenDatabase(
            metadata=[
                KeyMetadata(model_name="qwen3", dcp_rank=0, head_or_tp_rank=0, pp_rank=0, kv_cache_group_id=0),
                KeyMetadata(model_name="qwen3", dcp_rank=0, head_or_tp_rank=0, pp_rank=0, kv_cache_group_id=1),
            ],
            block_size=[128, 128],
            partitions=None,
            hash_block_size=128,
        )
        db.cache_coordinator = coord
        stored_keys: set[str] = set()
        for gid in (0, 1):
            for _, _, key_str, _ in db.process_token_key_strings(
                len(prefix_hashes) * 128,
                prefix_hashes,
                kv_cache_group_id=gid,
            ):
                stored_keys.add(key_str)
        # Include layerwise keys as well: "<gid>:<hash_str>"
        stored_lw = {(gid, bh) for gid in (0, 1) for bh in prefix_hashes}

        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.num_hash_lookahead_tokens = num_hash_lookahead_tokens
        worker.hash_block_size = 128
        worker.max_model_len = 8192
        worker.cache_transfer_granularity = 128
        worker.num_kv_cache_groups = 2
        worker.cacheable_group_ids = [0, 1]
        worker.group_uses_align_state = [False, True]
        worker.sharded_mamba_group_ids = set()
        worker.use_kvpp = False
        worker.tp_mismatch = False
        worker.use_mla = True
        worker.use_sparse = False
        worker.tp_size = 1
        worker.pp_size = 1
        worker.dcp_size = 1
        worker.pcp_size = 1
        worker.num_kv_head = 1
        worker.token_database = db
        worker.cache_coordinator = coord
        worker.m_store = SimpleNamespace(exists=lambda keys: [int(k in stored_keys) for k in keys])

        sched = _make_base_scheduler(num_hash_lookahead_tokens=num_hash_lookahead_tokens, cls=KVPoolScheduler)
        sched.kv_cache_config = kv_cache_config
        sched.kv_cache_group_ids = [0, 1]
        sched.kv_cache_group_families = ["default", "default"]
        sched.original_block_size = [128, 128]
        sched.grouped_block_size = [128, 128]
        sched.cacheable_group_ids = [0, 1]
        sched.cache_coordinator = coord
        sched.client = _HashAwareFakeLookupClient(worker)

        lw_sched = _make_base_scheduler(
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
            use_layerwise=True,
            cls=LayerwiseKVPoolScheduler,
        )
        lw_sched.kv_cache_config = kv_cache_config
        lw_sched.kv_cache_group_ids = [0, 1]
        lw_sched.kv_cache_group_families = ["default", "default"]
        lw_sched.original_block_size = [128, 128]
        lw_sched.grouped_block_size = [128, 128]
        lw_sched.cacheable_group_ids = [0, 1]
        lw_sched.cache_coordinator = coord
        lw_sched.use_block_key_layerwise = False
        lw_sched.use_layerwise_transfer = True
        lw_sched.sharded_mamba_group_ids = set()
        lw_sched.dspark_prefix_keys = None
        lw_sched.dspark_draft_only_group_ids = frozenset()
        lw_sched.layerwise_max_transfer_blocks = 0
        lw_sched.tp_size = 1
        lw_sched.put_step = 1
        stored_lw_hex = {f"{gid}:{bh.hex()}" for gid, bh in stored_lw}
        lw_sched.block_key_hybrid = False
        lw_sched.layerwise_keys = SimpleNamespace(make_hit_check_keys=lambda gid, bh, ranks: [f"{gid}:{bh}"])
        lw_sched.store_scheduler = SimpleNamespace(
            batch_is_readable=lambda batch: [key in stored_lw_hex for key in batch]
        )
        return sched, worker, lw_sched

    def test_fb01_1536_prefix_fork_256_and_257_exact_hits(self):
        base_prompt = list(range(1539))
        for with_coord in (True, False):
            sched, _, lw_sched = self._build_hybrid_fork_env(
                prefix_tokens=base_prompt,
                num_hash_lookahead_tokens=1,
                with_coordinator=with_coord,
            )
            lw_sched.use_block_key_layerwise = not with_coord
            lw_sched.use_layerwise_transfer = with_coord

            def _lookup(scheduler, req_id: str, fork: int | None):
                toks = list(base_prompt)
                if fork is not None:
                    toks[fork] += 10000
                hashes = _hash_tokens_with_lookahead(toks, hash_block_size=128, num_hash_lookahead_tokens=1)
                req = SimpleNamespace(
                    request_id=req_id,
                    prompt_token_ids=toks,
                    num_tokens=len(toks),
                    block_hashes=hashes,
                )
                hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
                return hit

            for mode_name, target_sched in (("blockwise", sched), ("layerwise", lw_sched)):
                with self.subTest(with_coord=with_coord, mode=mode_name):
                    self.assertEqual(_lookup(target_sched, "warm", None), 1536)
                    fork_256 = _lookup(target_sched, "fork-256", 256)
                    self.assertNotEqual(fork_256, 1536)
                    self.assertEqual(fork_256, 128)
                    fork_257 = _lookup(target_sched, "fork-257", 257)
                    self.assertNotEqual(fork_257, 0)
                    self.assertEqual(fork_257, 256)

    def test_lookahead_fork_sweep_and_legality_invariant(self):
        base_prompt = list(range(1545))
        forks = (0, 1, 127, 128, 129, 255, 256, 257, 258, 259, 383, 384, 385, 512, 1535, 1536, 1537)
        for lookahead in (1, 3):
            for with_coord in (True, False):
                sched, _, lw_sched = self._build_hybrid_fork_env(
                    prefix_tokens=base_prompt,
                    num_hash_lookahead_tokens=lookahead,
                    with_coordinator=with_coord,
                )
                lw_sched.use_block_key_layerwise = not with_coord
                lw_sched.use_layerwise_transfer = with_coord
                for fork in forks:
                    toks = list(base_prompt)
                    toks[fork] += 10000
                    hashes = _hash_tokens_with_lookahead(
                        toks,
                        hash_block_size=128,
                        num_hash_lookahead_tokens=lookahead,
                    )
                    req = SimpleNamespace(
                        request_id=f"fork-{lookahead}-{fork}",
                        prompt_token_ids=toks,
                        num_tokens=len(toks),
                        block_hashes=hashes,
                    )
                    expected = min(1536, max(0, fork - lookahead) // 128 * 128)
                    for target_sched in (sched, lw_sched):
                        hit, _ = target_sched.get_num_new_matched_tokens(req, 0)
                        self.assertEqual(hit, expected)
                        self.assertLessEqual(hit, max(0, fork - lookahead) // 128 * 128)

                # Repeated and interleaved lookups remain isolated and exact.
                repeat_cases = [
                    (256, 128),
                    (257, 256 if lookahead == 1 else 128),
                    (None, 1536),
                    (256, 128),
                ]
                for idx, (fork, expected) in enumerate(repeat_cases):
                    toks = list(base_prompt)
                    if fork is not None:
                        toks[fork] += 10000
                    req = SimpleNamespace(
                        request_id=f"repeat-{idx}",
                        prompt_token_ids=toks,
                        num_tokens=len(toks),
                        block_hashes=_hash_tokens_with_lookahead(toks, 128, lookahead),
                    )
                    self.assertEqual(sched.get_num_new_matched_tokens(req, 0)[0], expected)
                    self.assertEqual(lw_sched.get_num_new_matched_tokens(req, 0)[0], expected)

    def test_stale_untrimmed_hashes_never_overhit(self):
        groups = [_make_full_group(["layers.0.attn"], 128)]
        hashes = _hashes(12)
        with patch.object(coord_mod, "_get_group_manager_class", return_value=_RefFullManager):
            coord = AscendStoreCoordinator(
                groups,
                scheduler_block_size=128,
                hash_block_size=128,
                group_block_sizes=[128],
                group_cache_families=["default"],
                num_hash_lookahead_tokens=1,
            )
        probed_counts: list[int] = []

        def query_all(group_id, group_hashes, lookup_mask):
            probed_counts.append(len(group_hashes))
            return group_hashes

        hit = coord.find_reachable_hit_tokens(hashes, token_len=256, query_group_hits=query_all)
        self.assertEqual(hit, 256)
        self.assertEqual(probed_counts, [2])

        # find_longest_cache_hit with max_length < hashed coverage never leaks past max_length.
        pool = ExternalCachedBlockPool(128, {(0, bytes(h)) for h in hashes})
        _, bounded_hit = coord.find_longest_cache_hit(hashes, max_length=256, cached_block_pool=pool)
        self.assertEqual(bounded_hit, 256)
