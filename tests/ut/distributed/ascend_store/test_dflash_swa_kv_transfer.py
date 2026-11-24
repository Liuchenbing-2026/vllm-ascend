# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for #22692 Part 2: DFlash/DSpark SWA KV transfer & pool boundaries."""

from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
    _find_hit_accepts_num_request_tokens,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key import LookupKeyClient, LookupKeyServer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    StoreLookupResult,
    TailKeyBoundary,
    block_hash_to_bytes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


@dataclass(frozen=True)
class _TailSWAWindowSpec(SlidingWindowSpec):
    is_dflash_sliding_window: bool = True
    tail_tokens: int | None = None


class _TailSWAManager(SlidingWindowManager):
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
        num_request_tokens=None,
    ):
        return ([[] for _ in kv_cache_group_ids], 0)


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


def test_dspark_pd_and_mooncake_null_block_boundaries():
    """SFA P/D context slots, KV read batches, and Mooncake SWA pull handle leading null block 0."""
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
    spec = SlidingWindowSpec(block_size=4, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=8)
    result = worker._compute_group_block_ids(
        "r", [[0]], 1, 0, 4, 4, [0, 7, 8], [0, 7, 8], [0, 17, 18], 12, 12, 0, 1, 1, spec, 0, 4
    )
    assert result == [(0, [7, 8], [17, 18])]
    with pytest.raises(ValueError, match="not resident"):
        worker._compute_group_block_ids(
            "r", [[0]], 1, 0, 4, 4, [0, 7, 8], [0, 7, 8], [0, 0, 18], 12, 12, 0, 1, 1, spec, 0, 4
        )


def test_ascend_store_coordinator_and_null_block_workers():
    """AscendStore coordinator skips eagle drop for DFlash & workers exclude block 0 placeholders."""
    assert _find_hit_accepts_num_request_tokens(_TailSWAManager)
    assert not _find_hit_accepts_num_request_tokens(FullAttentionManager)

    dflash_spec = _TailSWAWindowSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=128
    )
    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    groups = [
        KVCacheGroupSpec(layer_names=["target.0"], kv_cache_spec=full_spec),
        KVCacheGroupSpec(layer_names=["draft.0"], kv_cache_spec=dflash_spec),
    ]
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
        side_effect=lambda s: _TailSWAManager if s is dflash_spec else FullAttentionManager,
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
    assert coord._attn_group_accepts_num_request_tokens == (False, True)

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


def test_ascend_store_align_mutation_and_request_token_forwarding():
    """Verify FB-83-03 align-state mutation and num_request_tokens propagation."""
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
        can_save=True, save_end_token=4352, target_token_len=4352, num_prompt_tokens=8705, block_ids_by_group=[[3]]
    )
    null_req: Any = SimpleNamespace(
        can_save=True, save_end_token=4352, target_token_len=4352, num_prompt_tokens=8705, block_ids_by_group=[[0]]
    )
    for use_lw in (True, False):
        lw_align.use_layerwise_transfer = use_lw
        assert lw_align._compute_reachable_store_masks(aligned_req) == ([True],)
        assert lw_align._compute_reachable_store_masks(null_req) == ([False],)
    lw_align.group_uses_align_state = [False, True]
    assert lw_align._effective_skip_null_blocks_by_group() == [False, True]

    # Scheduler and worker forward num_request_tokens when provided.
    seen_sched_kwargs: list[dict[str, Any]] = []
    sched: Any = LayerwiseKVPoolScheduler.__new__(LayerwiseKVPoolScheduler)
    sched.grouped_block_size = [16]

    def _rec_sched(hashes, token_len, cb, **kw):
        seen_sched_kwargs.append(kw)
        return 32

    sched.cache_coordinator = SimpleNamespace(find_reachable_hit_tokens=_rec_sched)
    req = SimpleNamespace(request_id="req-lw", block_hashes=[b"a", b"b"], num_tokens=33)
    assert sched._lookup_layerwise_with_coordinator(req, token_len=32, num_computed_tokens=0) == 32
    assert seen_sched_kwargs[0]["num_request_tokens"] == 33

    seen_worker_kwargs: list[dict[str, Any]] = []
    pool_worker: Any = KVPoolWorker.__new__(KVPoolWorker)

    def _rec_worker(hashes, token_len, cb, **kw):
        seen_worker_kwargs.append(kw)
        return 16

    pool_worker.num_kv_cache_groups = 1
    pool_worker.cache_coordinator = SimpleNamespace(
        enable_partial_hash_hits=False,
        group_effective_block_sizes=[16],
        find_reachable_hit_tokens=_rec_worker,
    )
    assert pool_worker._lookup_with_coordinator(16, [b"a"], [0], False, num_request_tokens=19) == StoreLookupResult(16)
    assert seen_worker_kwargs[0]["num_request_tokens"] == 19


def test_non_speculative_kv_transfer_fast_path_isolation():
    """Full-attention and legacy KV transfer paths avoid DFlash kwargs and null filtering."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    seen_calls: list[dict[str, Any]] = []

    class LegacyFullManager:
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
        ):
            seen_calls.append({"max_length": max_length})
            return ([[] for _ in kv_cache_group_ids], 0)

        @staticmethod
        def reachable_block_mask(start_block, end_block, alignment_tokens, kv_cache_spec, use_eagle):
            return None

    groups = [KVCacheGroupSpec(layer_names=["target.0"], kv_cache_spec=full_spec)]
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
        return_value=LegacyFullManager,
    ):
        coord = AscendStoreCoordinator(
            groups,
            scheduler_block_size=16,
            hash_block_size=16,
            group_block_sizes=[16],
            group_cache_families=["default"],
            use_eagle=False,
        )
    assert coord._attn_group_accepts_num_request_tokens == (False,)
    assert coord.load_mask([], 0) == ([],)
    assert len(seen_calls) == 1

    # Full-attention worker omits skip_null_blocks and num_request_tokens when None.
    worker: Any = KVPoolWorker.__new__(KVPoolWorker)
    worker.num_kv_cache_groups = 1
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    assert worker._infer_group_skips_null_blocks() == [False]

    seen_lookup_kwargs: list[dict[str, Any]] = []

    def _rec_lookup(hashes, token_len, cb, **kw):
        seen_lookup_kwargs.append(kw)
        return 16

    worker.cache_coordinator = SimpleNamespace(
        enable_partial_hash_hits=False,
        group_effective_block_sizes=[16],
        find_reachable_hit_tokens=_rec_lookup,
    )
    assert worker._lookup_with_coordinator(16, [b"a"], [0], False, num_request_tokens=None) == StoreLookupResult(16)
    assert "num_request_tokens" not in seen_lookup_kwargs[0]

    # Mooncake FullAttention pull path stays on contiguous scaling without SWA null filtering.
    puller: Any = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    puller.dcp_size = 1
    puller.dcp_rank = 0
    assert puller._compute_group_block_ids(
        "r-full", [[2]], 1, 0, 16, 16, [1, 2], [1, 2], [11, 12], 32, 32, 0, 1, 1, full_spec, 0, 16
    ) == [(2, [1, 2], [11, 12])]


def test_nonlayerwise_ascend_store_num_request_tokens_round_trip():
    """Scheduler, ZMQ wire framing, and KVPoolWorker preserve floored num_request_tokens."""
    # 1. KVPoolScheduler forwards num_request_tokens only when it differs from floored token_len.
    sched: Any = KVPoolScheduler.__new__(KVPoolScheduler)
    sched.cacheable_group_ids = [0]
    sched.original_block_size = [16]
    sched._discard_partial_chunks = True
    sched.enable_partial_hash_hits = False
    sched.cache_transfer_granularity = 16
    sched.kv_cache_group_ids = [0]
    sched._floor_to_cache_transfer_granularity = lambda n: n // 16 * 16
    sched.client = MagicMock(closed=False)
    sched.client.lookup.return_value = StoreLookupResult(32)
    h0 = b"\xaa" * 32
    h1 = b"\xbb" * 32

    floored_req: Any = SimpleNamespace(prompt_token_ids=[1] * 35, num_tokens=35, block_hashes=[h0, h1])
    assert sched._lookup_external_hit_tokens(floored_req, 0) == StoreLookupResult(32)
    assert sched.client.lookup.call_args.kwargs["num_request_tokens"] == 35

    sched.client.lookup.reset_mock()
    aligned_req: Any = SimpleNamespace(prompt_token_ids=[1] * 32, num_tokens=32, block_hashes=[h0, h1])
    assert sched._lookup_external_hit_tokens(aligned_req, 0) == StoreLookupResult(32)
    assert "num_request_tokens" not in sched.client.lookup.call_args.kwargs

    sched.client.lookup.reset_mock()
    bool_req: Any = SimpleNamespace(prompt_token_ids=[1] * 35, num_tokens=True, block_hashes=[h0, h1])
    assert sched._lookup_external_hit_tokens(bool_req, 0) == StoreLookupResult(32)
    assert sched.client.lookup.call_args.kwargs["num_request_tokens"] == 35

    # 2. Wire round-trip + KVPoolWorker.lookup + AscendStoreCoordinator tail hit behavior.
    class _HitCheckingTailSWAManager(SlidingWindowManager):
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
            num_request_tokens=None,
        ):
            eff_req = max_length if num_request_tokens is None else num_request_tokens
            block_size = kv_cache_spec.block_size
            first_block = max(eff_req - (kv_cache_spec.sliding_window - 1), 0) // block_size
            max_blocks = max_length // block_size
            hits = tuple([block_pool.null_block] * max_blocks for _ in kv_cache_group_ids)
            idx = min(first_block, max_blocks)
            while idx < max_blocks and (cached := block_pool.get_cached_block(block_hashes[idx], kv_cache_group_ids)):
                for out, blk in zip(hits, cached):
                    out[idx] = blk
                idx += 1
            return hits, idx * block_size

    tail_spec = _TailSWAWindowSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=20)
    groups = [KVCacheGroupSpec(layer_names=["draft.0"], kv_cache_spec=tail_spec)]
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
        return_value=_HitCheckingTailSWAManager,
    ):
        coord = AscendStoreCoordinator(
            groups,
            scheduler_block_size=16,
            hash_block_size=16,
            group_block_sizes=[16],
            group_cache_families=["default"],
            use_eagle=False,
        )

    def fake_key_strings(token_len, block_hashes, mask_num=0, kv_cache_group_id=0, chunk_mask=None):
        del chunk_mask
        num_chunks = token_len // 16
        start_chunk = mask_num // 16
        for i in range(start_chunk, num_chunks):
            bh = block_hashes[i]
            bh_hex = bh if isinstance(bh, str) else block_hash_to_bytes(bh).hex()
            yield i * 16, (i + 1) * 16, f"g{kv_cache_group_id}@{bh_hex}", bh

    h1_hex = h1.hex()
    worker: Any = KVPoolWorker.__new__(KVPoolWorker)
    worker.num_kv_cache_groups = 1
    worker.cache_coordinator = coord
    worker.pp_size = 1
    worker.dcp_size = 1
    worker.tp_size = 1
    worker.pcp_size = 1
    worker.use_mla = True
    worker.use_sparse = False
    worker.use_kvpp = False
    worker.tp_mismatch = False
    worker.sharded_mamba_group_ids = set()
    worker.group_uses_align_state = [False]
    worker.max_model_len = 1024
    worker.token_database = SimpleNamespace(process_token_key_strings=fake_key_strings)
    worker.m_store = SimpleNamespace(exists=lambda keys: [1 if h1_hex in key else 0 for key in keys])

    # Only h1 ([16..32)) is in pool: with num_request_tokens=35 first_block=1 (hit=32);
    # without num_request_tokens it falls back to token_len=32 -> first_block=0 -> 0.
    assert worker.lookup(32, [h0, h1], [0], num_request_tokens=35) == 32
    assert worker.lookup(32, [h0, h1], [0]) == 0

    sent_frames: list[list[bytes]] = []
    client_sock = MagicMock()
    server_sock = MagicMock()
    server_holder: list[Any] = []
    responses: list[list[bytes]] = []

    def client_send(frames, copy=False):
        del copy
        raw_frames = [bytes(frame) for frame in frames]
        sent_frames.append(raw_frames)
        poller_once = [[(server_sock, 1)], []]
        server_sock.recv_multipart.side_effect = [raw_frames]

        def poll_side_effect(timeout=100):
            del timeout
            if poller_once:
                item = poller_once.pop(0)
                if not item:
                    server_holder[0].running = False
                return item
            server_holder[0].running = False
            return []

        mock_poller.poll.side_effect = poll_side_effect
        server_holder[0].running = True
        thread_target()

    client_sock.send_multipart.side_effect = client_send
    server_sock.send_multipart.side_effect = lambda frames: responses.append([bytes(frame) for frame in frames])
    client_sock.recv_multipart.side_effect = lambda: responses[-1]

    mock_poller = MagicMock()
    cfg: Any = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
    )
    lk_mod = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key"
    with (
        patch(f"{lk_mod}.zmq.Context"),
        patch(f"{lk_mod}.make_zmq_socket", side_effect=[server_sock, client_sock]),
        patch(f"{lk_mod}.zmq.Poller", return_value=mock_poller),
        patch(f"{lk_mod}.threading.Thread") as mock_thread,
    ):
        server = LookupKeyServer(worker, cfg)
        server_holder.append(server)
        thread_target = mock_thread.call_args.kwargs["target"]
        client = LookupKeyClient(cfg)

        assert client.lookup(32, [h0, h1], [0], num_request_tokens=35) == StoreLookupResult(32)
        assert len(sent_frames[-1]) == 6
        assert client.lookup(32, [h0, h1], [0]) == StoreLookupResult(0)
        assert len(sent_frames[-1]) == 5
        for invalid in (True, False, -1, 1 << 32):
            with pytest.raises(ValueError, match="uint32"):
                client.lookup(32, [h0, h1], [0], num_request_tokens=invalid)


def test_partial_hash_hits_with_dflash_swa_tail_and_null_prefix():
    """#21167 partial hash hits forward num_request_tokens and tolerate DFlash null-only prefixes."""
    seen_calls: list[tuple[int, int | None]] = []

    class _FullPartialManager:
        supports_fine_grained_hash_lookup = True

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
        ):
            del block_hashes, kv_cache_spec, drop_eagle_block
            num_blocks = min(max_length, 28) // alignment_tokens
            hit = num_blocks * alignment_tokens
            blks = [[block_pool._present_block] * num_blocks for _ in kv_cache_group_ids]
            return blks, hit

    class _RecordingTailSWAManager(SlidingWindowManager):
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
            num_request_tokens=None,
        ):
            del block_hashes, drop_eagle_block, alignment_tokens, dcp_world_size, pcp_world_size
            seen_calls.append((max_length, num_request_tokens))
            hit = min(max_length, 32)
            num_blocks = hit // kv_cache_spec.block_size
            blks = [[block_pool.null_block] * num_blocks for _ in kv_cache_group_ids]
            return blks, hit

    full_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    tail_spec = _TailSWAWindowSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=20)
    groups = [
        KVCacheGroupSpec(layer_names=["target.0"], kv_cache_spec=full_spec),
        KVCacheGroupSpec(layer_names=["draft.0"], kv_cache_spec=tail_spec),
    ]
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
        side_effect=lambda s: _RecordingTailSWAManager if s is tail_spec else _FullPartialManager,
    ):
        coord = AscendStoreCoordinator(
            groups,
            scheduler_block_size=16,
            hash_block_size=4,
            group_block_sizes=[16, 16],
            group_cache_families=["default", "default"],
            use_eagle=False,
            enable_partial_hash_hits=True,
        )

    hashes: list[Any] = [bytes([i + 1]) * 8 for i in range(16)]
    # 1. find_partial_hit_tokens forwards num_request_tokens on both initial call and -1 retry.
    assert (
        coord.find_partial_hit_tokens(hashes, 28, ExternalCachedBlockPool(4), hbm_hit_tokens=16, num_request_tokens=35)
        == 24
    )
    assert seen_calls == [(28, 35), (24, 35)]

    # 2. Group 0 has hashes[6] (token 28), while group 1 has only leading nulls (< block 2 = 32 tokens) at req_len=64.
    stored_hashes = {hashes[i].hex() for i in range(7)}
    worker: Any = KVPoolWorker.__new__(KVPoolWorker)
    worker.hash_block_size = 4
    worker.num_kv_cache_groups = 2
    worker.cache_coordinator = coord
    worker.pp_size = 1
    worker.dcp_size = 1
    worker.tp_size = 1
    worker.pcp_size = 1
    worker.use_mla = True
    worker.use_sparse = False
    worker.use_kvpp = False
    worker.tp_mismatch = False
    worker.sharded_mamba_group_ids = set()
    worker.token_database = SimpleNamespace(_get_key_prefix=lambda g: f"g{g}@")
    worker.m_store = SimpleNamespace(
        exists=lambda keys: [1 if (k.startswith("g0@") and k.split("@", 1)[1] in stored_hashes) else 0 for k in keys]
    )

    result = worker._lookup_with_coordinator(64, hashes, [0, 1], False, hbm_hit_tokens=16, num_request_tokens=64)
    assert result == StoreLookupResult(28, (TailKeyBoundary(0, 28),))

    # 3. Missing FullAttention key or missing non-null DFlash key still raises ValueError.
    empty_pool = ExternalCachedBlockPool(4, set())
    with pytest.raises(ValueError, match="No recoverable tail value for group 0"):
        worker._tail_key_boundaries(coord, hashes, 28, empty_pool, num_request_tokens=64)

    full_only_pool = ExternalCachedBlockPool(4, {(0, hashes[9])})
    with pytest.raises(ValueError, match="No recoverable tail value for group 1"):
        worker._tail_key_boundaries(coord, hashes, 40, full_only_pool, num_request_tokens=64)


def test_mooncake_hybrid_connector_swa_tail_and_null_blocks():
    """Verify hybrid connector uses tail_tokens, strips 0s, and avoids RDMA writes to block 0."""
    import threading

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        KVCacheRecvingThread as HybridKVCacheRecvingThread,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        MooncakeConnectorScheduler as HybridMooncakeConnectorScheduler,
    )

    full_spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    tail_spec = _TailSWAWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        sliding_window=128,
        tail_tokens=288,
    )
    wrapped_spec = UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"draft.0": tail_spec})

    vllm_cfg: Any = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            data_parallel_index=0,
            data_parallel_rank=0,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(kv_port=20000, kv_connector_extra_config={}),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    hybrid_kv_cfg: Any = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["target.0"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["draft.0"], kv_cache_spec=wrapped_spec),
        ]
    )
    single_swa_cfg: Any = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft.0"], kv_cache_spec=tail_spec)]
    )

    mh_mod = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector"
    ascend_cfg = SimpleNamespace(enable_kv_nz=False, enable_transpose_kv_cache_by_block=False)
    with (
        patch(f"{mh_mod}.init_ascend_config"),
        patch(f"{mh_mod}.get_ascend_config", return_value=ascend_cfg),
        patch(f"{mh_mod}.get_ip", return_value="127.0.0.1"),
    ):
        hybrid_sched = HybridMooncakeConnectorScheduler(vllm_cfg, "eng-h", hybrid_kv_cfg)
        single_sched = HybridMooncakeConnectorScheduler(vllm_cfg, "eng-s", single_swa_cfg)

    assert hybrid_sched.num_swa_blocks == [0, 4]
    assert single_sched.num_swa_blocks == [4]
    assert hybrid_sched.get_sw_clipped_blocks(([11, 12], [0, 0, 102, 103])) == ([11, 12], [102, 103])
    assert hybrid_sched.get_sw_clipped_blocks(([11], [0, 101, 102, 103, 104])) == ([11], [101, 102, 103, 104])
    assert single_sched.get_sw_clipped_blocks(([0, 101, 102, 103, 104],)) == ([101, 102, 103, 104],)

    # All-group receiver skips leading 0s for both UniformTypeKVCacheSpecs and plain SlidingWindowSpec.
    for swa_group_spec in (wrapped_spec, tail_spec):
        engine = MagicMock()
        engine.batch_transfer_sync_read.return_value = 0
        recv: Any = HybridKVCacheRecvingThread.__new__(HybridKVCacheRecvingThread)
        recv.remote_metadata_lock = threading.Lock()
        recv.local_engine_id = "local-e"
        recv.local_handshake_port = 20000
        recv.kv_caches_base_addr = {"remote-e": {10000: [50000, 60000]}, "local-e": {20000: [10000, 20000]}}
        recv.remote_te_port = {"remote-e": {10000: 15000}}
        recv._prefill_pp_size = 1
        recv.block_len_per_addr = [64, 64]
        recv.block_stride_per_addr = [64, 64]
        recv.addr_group_idx = [[0], [1]]
        recv.hma_group_size = 2
        recv.kv_cache_specs = [full_spec, swa_group_spec]
        recv.engine = engine
        recv.tp_rank = 0

        base_meta = {
            "remote_request_id": "req-h",
            "remote_engine_id": "remote-e",
            "remote_host": "127.0.0.1",
            "remote_handshake_port": 10000,
        }
        with patch(f"{mh_mod}.get_ip", return_value="127.0.0.1"):
            # Cold D with leading null placeholders paired with backed P blocks.
            recv._transfer_kv_cache_all_groups(
                {
                    **base_meta,
                    "local_block_ids": ([21, 22], [0, 0, 202, 203]),
                    "remote_block_ids": ([11, 12], [102, 103]),
                }
            )
            _, src_list, dst_list, length_list = engine.batch_transfer_sync_read.call_args.args
            assert 20000 not in src_list
            assert src_list == [10000 + 21 * 64, 20000 + 202 * 64]
            assert dst_list == [50000 + 11 * 64, 60000 + 102 * 64]
            assert length_list == [128, 128]

            # Partial D prefix hit + unpatched P leading 0 alignment.
            engine.batch_transfer_sync_read.reset_mock()
            recv._transfer_kv_cache_all_groups(
                {**base_meta, "local_block_ids": ([22], [203]), "remote_block_ids": ([11, 12], [0, 102, 103])}
            )
            _, src_list, dst_list, _ = engine.batch_transfer_sync_read.call_args.args
            assert src_list == [10000 + 22 * 64, 20000 + 203 * 64]
            assert dst_list == [50000 + 12 * 64, 60000 + 103 * 64]

            # Reject missing or null required remote blocks.
            with pytest.raises(ValueError, match="resident|remote"):
                recv._transfer_kv_cache_all_groups(
                    {**base_meta, "local_block_ids": ([21], [202, 203]), "remote_block_ids": ([11], [0, 103])}
                )
            with pytest.raises(ValueError, match="Insufficient|remote"):
                recv._transfer_kv_cache_all_groups(
                    {**base_meta, "local_block_ids": ([21], [202, 203]), "remote_block_ids": ([11], [103])}
                )

            # All-zero local SWA blocks with no full-attention blocks skips empty RDMA call.
            engine.batch_transfer_sync_read.reset_mock()
            recv._transfer_kv_cache_all_groups(
                {**base_meta, "local_block_ids": ([], [0, 0]), "remote_block_ids": ([], [102, 103])}
            )
            engine.batch_transfer_sync_read.assert_not_called()

    # Single-group SWA receiver skips leading 0s and guards empty reads.
    single_engine = MagicMock()
    single_engine.batch_transfer_sync_read.return_value = 0
    srecv: Any = HybridKVCacheRecvingThread.__new__(HybridKVCacheRecvingThread)
    srecv.remote_metadata_lock = threading.Lock()
    srecv.local_engine_id = "local-e"
    srecv.local_handshake_port = 20000
    srecv.kv_caches_base_addr = {"remote-e": {10000: [50000]}, "local-e": {20000: [20000]}}
    srecv.remote_te_port = {"remote-e": {10000: 15000}}
    srecv.pp_layer_indices = {0: (0, 1)}
    srecv._prefill_pp_size = 1
    srecv.vllm_config = SimpleNamespace(speculative_config=None)
    srecv.kv_caches = {"draft.0": [torch.empty(1)]}
    srecv.block_len_per_addr = [64]
    srecv.kv_cache_specs = [tail_spec]
    srecv.engine = single_engine
    srecv.tp_rank = 0
    smeta = {
        "remote_request_id": "req-s",
        "remote_engine_id": "remote-e",
        "remote_host": "127.0.0.1",
        "remote_handshake_port": 10000,
        "offset": 0,
        "tp_num_need_pulls": 1,
    }
    with (
        patch(f"{mh_mod}.get_ascend_config", return_value=ascend_cfg),
        patch(f"{mh_mod}.get_ip", return_value="127.0.0.1"),
    ):
        srecv._transfer_kv_cache(
            {**smeta, "local_block_ids": ([0, 0, 202, 203],), "remote_block_ids": ([0, 102, 103],)}
        )
        _, src_list, dst_list, length_list = single_engine.batch_transfer_sync_read.call_args.args
        assert src_list == [20000 + 202 * 64]
        assert dst_list == [50000 + 102 * 64]
        assert length_list == [128]
        single_engine.batch_transfer_sync_read.reset_mock()
        srecv._transfer_kv_cache({**smeta, "local_block_ids": ([0, 0],), "remote_block_ids": ([102],)})
        single_engine.batch_transfer_sync_read.assert_not_called()


def test_mooncake_connector_swa_tail_and_kernel_null_blocks():
    """Verify MooncakeConnectorV1 uses tail_tokens window and skips null/lookahead blocks."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
        MooncakeConnectorScheduler as MooncakeV1ConnectorScheduler,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
        MooncakeConnectorWorker as MooncakeV1ConnectorWorker,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
        ReqMeta as MooncakeV1ReqMeta,
    )

    tail_spec = _TailSWAWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        sliding_window=128,
        tail_tokens=288,
    )
    sched: Any = MooncakeV1ConnectorScheduler.__new__(MooncakeV1ConnectorScheduler)
    sched.block_size = 128
    group = KVCacheGroupSpec(layer_names=["draft.0"], kv_cache_spec=tail_spec)
    info = sched._get_group_transfer_info(group)
    assert info.blocks_per_window == 4  # cdiv(288, 128) + 1

    worker: Any = MooncakeV1ConnectorWorker.__new__(MooncakeV1ConnectorWorker)
    worker.block_size = 128
    worker.block_size_scale = [[2]]
    group_spec = {
        "kv_cache_spec_type": "_TailSWAWindowSpec",
        "kv_cache_spec": {"sliding_window": 128, "tail_tokens": 288},
        "block_size": 128,
        "kv_cache_group_id": 0,
    }

    def _meta(local_ids, remote_ids, *, full_ids=(), computed=0, external=512):
        return MooncakeV1ReqMeta(
            local_block_ids=local_ids,
            num_external_tokens=external,
            num_computed_tokens=computed,
            remote_block_ids=remote_ids,
            remote_host="127.0.0.1",
            remote_port=10000,
            remote_engine_id="p",
            remote_request_id="req-v1",
            remote_pcp_size=1,
            remote_dcp_size=1,
            remote_ptp_size=1,
            remote_multi_nodes_meta_mapping={},
            num_prompt_blocks=4,
            remote_block_size=128,
            local_full_block_ids=full_ids,
        )

    # Full block table path drops leading 0s and trailing speculative lookahead beyond end_token=512.
    local_k, remote_k = worker._get_kernel_block_ids(
        [0],
        _meta(([0, 0, 202, 203, 204],), ([102, 103],), full_ids=([0, 0, 202, 203, 204],)),
        0,
        group_spec,
    )
    assert local_k == [404, 405, 406, 407]
    assert remote_k == [204, 205, 206, 207]

    # Partial D prefix hit aligns suffix onto remote blocks.
    local_k_hit, remote_k_hit = worker._get_kernel_block_ids(
        [0],
        _meta(([203],), ([102, 103],), full_ids=([0, 0, 202, 203],), computed=384, external=128),
        0,
        group_spec,
    )
    assert local_k_hit == [406, 407]
    assert remote_k_hit == [206, 207]

    # Legacy path without local_full_block_ids drops leading 0s cleanly.
    local_k_legacy, remote_k_legacy = worker._get_kernel_block_ids(
        [0],
        _meta(([0, 0, 202, 203],), ([102, 103],)),
        0,
        group_spec,
    )
    assert local_k_legacy == [404, 405, 406, 407]
    assert remote_k_legacy == [204, 205, 206, 207]

    # Insufficient remote suffix or null required remote block raises ValueError.
    with pytest.raises(ValueError, match="Insufficient|remote"):
        worker._get_kernel_block_ids(
            [0],
            _meta(([0, 0, 202, 203],), ([103],), full_ids=([0, 0, 202, 203],)),
            0,
            group_spec,
        )
    with pytest.raises(ValueError, match="resident|remote"):
        worker._get_kernel_block_ids(
            [0],
            _meta(([0, 0, 202, 203],), ([0, 103],), full_ids=([0, 0, 202, 203],)),
            0,
            group_spec,
        )


def test_mooncake_push_worker_swa_tail_tokens_window():
    """Verify PushWorker retains tail_tokens window rather than clipping at sliding_window."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import (
        MooncakePushConnectorWorker,
        RemoteReqInfo,
        SendReqInfo,
    )

    tail_spec = _TailSWAWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        sliding_window=1024,
        tail_tokens=1054,
    )
    worker: Any = MooncakePushConnectorWorker.__new__(MooncakePushConnectorWorker)
    worker._layer_indices = {"draft.0": 0}
    worker.layer_name_to_spec_index = {"draft.0": 0}
    worker.kv_cache_specs = [tail_spec]
    worker.transfer_metadata = SimpleNamespace(
        group_indices=[0],
        block_size_scales=[[1]],
        layer_block_sizes=[128],
        cp_kv_cache_interleave_size=128,
    )
    worker.dcp_rank = 0
    worker._get_layer_dcp_sizes = lambda *args: (1, 1)

    remote = SimpleNamespace(
        pcp_size=1,
        metadata_by_pp_rank={
            0: SimpleNamespace(
                layer_names=["draft.0"],
                group_indices=[0],
                block_size_scales=[[1]],
                layer_block_sizes=[128],
                cp_kv_cache_interleave_size=128,
            )
        },
    )
    meta_future: Future[Any] = Future()
    meta_future.set_result(remote)
    state = SendReqInfo(
        local_request_id="req-push",
        metadata=SimpleNamespace(
            num_prompt_tokens=2048,
            local_block_ids=([0] * 7 + list(range(107, 116)),),
        ),
        available_layers={"draft.0": (2048, object())},
    )
    dest = RemoteReqInfo(
        transfer_params={
            "consumer_rank": [0, 0, 0],
            "num_computed_tokens": 0,
            "num_external_tokens": 2048,
            "block_ids": ([0] * 7 + list(range(207, 216)),),
        },
        metadata_future=meta_future,
    )

    built = worker._build_layer_transfer_request("req-push", state, dest, "draft.0")
    assert built is not None
    req, _ = built
    # end=2048, tail_tokens=1054 -> start=994 -> block 7 (107->207) must be transferred.
    assert req.local_block_ids == tuple(range(107, 116))
    assert req.remote_block_ids == tuple(range(207, 216))


def test_mooncake_layerwise_connector_swa_null_blocks():
    """Verify layerwise connector preserves 0 on scale and skips D-side null blocks cleanly."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        LayerMetadata as LayerwiseLayerMetadata,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        MooncakeLayerwiseConnectorMetadata,
        MooncakeLayerwiseConnectorWorker,
        SendTask,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        ReqMeta as LayerwiseReqMeta,
    )

    tail_spec = _TailSWAWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        sliding_window=128,
        tail_tokens=288,
    )
    wrapped_spec = UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs={"draft.0": tail_spec})
    # LayerwiseReqMeta.remote_block_size is annotated list[list[int]] upstream while runtime stores list[int].
    lw_remote_block_size: Any = [128]

    for swa_spec in (tail_spec, wrapped_spec):
        worker: Any = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
        worker.kv_cache_specs = [swa_spec]
        worker.num_kv_cache_groups = 1
        worker.block_size = [64]
        worker.kernel_block_size_scale = [1]
        worker.pcp_size = 1
        worker.dcp_size = 1
        worker.tp_size = 1
        worker.pcp_rank = 0
        worker.dcp_rank = 0
        worker.tp_rank = 0
        worker.pd_head_ratio = 1
        worker.use_mla = True
        worker.total_num_kv_heads = 1
        worker.total_layers = 1
        worker.attn_resharding_group_idx = set()
        worker.vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False)
        )

        # _align_remote_block_ids preserves placeholder 0 as [0, 0] instead of [0, 1].
        align_meta = LayerwiseReqMeta(
            local_block_ids=[[0, 0, 102]],
            token_ids=[],
            remote_block_ids=[[0, 202]],
            remote_block_size=lw_remote_block_size,
            remote_engine_id="d",
            remote_host="127.0.0.1",
            remote_port=9000,
            remote_te_rpc_port=None,
            remote_layer_metadata=None,
            metaserver=None,
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
        )
        worker._align_remote_block_ids(align_meta)
        assert align_meta.remote_block_ids[0] == [0, 0, 404, 405]

        # prepare_layerwise_step filters (0, 0) placeholders before _get_kernel_block_ids.
        worker.block_size = [128]
        meta = MooncakeLayerwiseConnectorMetadata()
        meta.requests = {
            "req-lw": LayerwiseReqMeta(
                local_block_ids=[[0, 0, 102, 103]],
                token_ids=[],
                remote_block_ids=[[0, 0, 202, 203]],
                remote_block_size=lw_remote_block_size,
                remote_engine_id="d",
                remote_host="127.0.0.1",
                remote_port=9000,
                remote_te_rpc_port=None,
                remote_layer_metadata=None,
                metaserver=None,
                remote_tp_size=1,
                remote_pcp_size=1,
                remote_dcp_size=1,
                chunk_finish=True,
                prompt_len=512,
                local_computed_tokens=512,
            )
        }
        worker.prepare_layerwise_step(meta)
        assert meta.requests["req-lw"].local_block_ids == [[102, 103]]
        assert meta.requests["req-lw"].remote_block_ids == [[202, 203]]

        # Non-resident P source block paired with resident D destination block raises ValueError.
        bad_meta = MooncakeLayerwiseConnectorMetadata()
        bad_meta.requests = {
            "req-lw-bad": LayerwiseReqMeta(
                local_block_ids=[[0, 0, 0, 103]],
                token_ids=[],
                remote_block_ids=[[0, 0, 202, 203]],
                remote_block_size=lw_remote_block_size,
                remote_engine_id="d",
                remote_host="127.0.0.1",
                remote_port=9000,
                remote_te_rpc_port=None,
                remote_layer_metadata=None,
                metaserver=None,
                remote_tp_size=1,
                remote_pcp_size=1,
                remote_dcp_size=1,
                chunk_finish=True,
                prompt_len=512,
                local_computed_tokens=512,
            )
        }
        with pytest.raises(ValueError, match="not resident"):
            worker.prepare_layerwise_step(bad_meta)

    # All-null chunk sets group_num_blocks=[0] and must skip npu_gather_pa_kv_cache even with quant enabled.
    quant_worker: Any = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
    quant_worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=True),
        quant_config=SimpleNamespace(c8_quant_layers={0}, kvcache_quant_layers=set()),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "draft.0": SimpleNamespace(
                    _c8_k_inv_scale=1.0,
                    _c8_k_offset=0.0,
                    _c8_v_inv_scale=1.0,
                    _c8_v_offset=0.0,
                )
            }
        ),
    )
    quant_worker.current_layer = 0
    quant_worker.total_layers = 1
    quant_worker.index_to_name = {0: ["draft.0"]}
    quant_worker._cache_write_events = [MagicMock()]
    quant_worker.layer_metadata = {"draft.0": LayerwiseLayerMetadata([0], [1000], [64], [1])}
    quant_worker.sparse_sfa_c8_layers = set()
    quant_worker.pd_head_ratio = 1
    quant_worker.enable_c8_quant = True
    quant_worker.enable_kv_quant = False
    quant_worker.kv_cache_specs = [tail_spec]
    quant_worker.resharding_stream = MagicMock()
    quant_worker.k_buffer = torch.empty(1)
    quant_worker.get_nz_cache = lambda x, _: x
    quant_worker.kv_send_layer_thread = MagicMock()
    null_req_meta = LayerwiseReqMeta(
        local_block_ids=[[]],
        token_ids=[],
        remote_block_ids=[[]],
        remote_block_size=lw_remote_block_size,
        remote_engine_id="d",
        remote_host="127.0.0.1",
        remote_port=9000,
        remote_te_rpc_port=None,
        remote_layer_metadata=None,
        metaserver=None,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
    )
    quant_meta = MooncakeLayerwiseConnectorMetadata()
    quant_meta.requests = {"req-null": null_req_meta}
    quant_meta.send_task = SendTask(
        group_num_blocks=[0],
        group_num_tokens=[0],
        group_rearrange_block_ids=[[]],
        group_block_table=[torch.empty((1, 0), dtype=torch.int32)],
        group_block_len_tensor=[torch.tensor([0], dtype=torch.int32)],
        group_seq_start_tensor=[torch.tensor([0], dtype=torch.int32)],
    )
    lw_mod = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector"
    with (
        patch(f"{lw_mod}.npu_stream_switch", return_value=nullcontext()),
        patch(f"{lw_mod}.torch_npu.npu_gather_pa_kv_cache") as mock_gather,
    ):
        quant_worker.save_kv_layer("draft.0", [torch.empty(1), torch.empty(1)], None, quant_meta)
    mock_gather.assert_not_called()
