#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    RSWASpec,
    SlidingWindowSpec,
)
# isort: on

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend_map, get_backend_class
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key import (
    DEFAULT_ASYNC_LOOKUP_TTL_S,
    DEFAULT_LOOKUP_ASYNC,
    DEFAULT_LOOKUP_BATCH_SIZE,
    DEFAULT_LOOKUP_MAX_INFLIGHT,
    DEFAULT_LOOKUP_TIMEOUT_MS,
    AsyncLookupManager,
    LookupKeyClient,
    LookupKeyServer,
    make_lookup_signature,
    parse_lookup_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def _make_hashes(num_blocks: int) -> list[bytes]:
    return [index.to_bytes(32, "big") for index in range(num_blocks)]


def _make_request(req_id: str, num_tokens: int, block_size: int = 16):
    num_blocks = num_tokens // block_size
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        num_tokens=num_tokens,
        block_hashes=_make_hashes(num_blocks),
    )


def _make_scheduler_config(extra_config: dict | None = None, block_size: int = 16, kv_role: str = "kv_both"):
    config = MagicMock()
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_connector_extra_config = extra_config if extra_config is not None else {}
    config.kv_transfer_config.get_from_extra_config.return_value = True
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_index = 0
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = 1
    config.cache_config.block_size = block_size
    config.cache_config.prefix_match_unit = block_size
    config.cache_config.prefix_cache_retention_interval = None
    config.model_config.model = "org/llama-7b"
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 2
    config.speculative_config = None
    config.kv_events_config = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    return config


class _PrefixManager:
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=16,
        **kwargs,
    ):
        del drop_eagle_block, kwargs
        block_size = kv_cache_spec.block_size
        scale = block_size // block_pool.hash_block_size
        hashes = list(block_hashes[scale - 1 :: scale]) if scale > 1 else list(block_hashes)
        computed = tuple([] for _ in kv_cache_group_ids)
        for block_hash in hashes[: max_length // block_size]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for group_blocks, block in zip(computed, cached):
                group_blocks.append(block)
        hit_len = len(computed[0]) * block_size
        aligned_len = hit_len // alignment_tokens * alignment_tokens
        return computed, aligned_len


class _SparseCheckpointManager:
    @classmethod
    def reachable_block_mask(cls, start_block, end_block, alignment_tokens, kv_cache_spec, use_eagle, **kwargs):
        del alignment_tokens, kv_cache_spec, use_eagle, kwargs
        return [(idx % 2 == 1) for idx in range(start_block, end_block)]

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=16,
        **kwargs,
    ):
        del drop_eagle_block, kwargs
        block_size = kv_cache_spec.block_size
        max_blocks = max_length // block_size
        for idx in range(max_blocks - 1, -1, -1):
            if block_pool.get_cached_block(block_hashes[idx], kv_cache_group_ids):
                end_pos = (idx + 1) * block_size
                if end_pos % alignment_tokens == 0:
                    return tuple([] for _ in kv_cache_group_ids), end_pos
        return tuple([] for _ in kv_cache_group_ids), 0


def _make_worker_stub(
    *,
    block_size: int = 16,
    tp_size: int = 1,
    pp_size: int = 1,
    extra_config: dict | None = None,
    coordinator: AscendStoreCoordinator | None = None,
    group_uses_align_state: list[bool] | None = None,
) -> KVPoolWorker:
    worker = object.__new__(KVPoolWorker)
    worker._lookup_config = parse_lookup_config(extra_config)
    worker.block_size = block_size
    worker.hash_block_size = block_size
    worker.grouped_block_size = [block_size] * (len(coordinator.kv_cache_groups) if coordinator else 1)
    worker.cache_transfer_granularity = coordinator.lcm_block_size if coordinator else block_size
    worker.max_model_len = 1048576
    worker.tp_size = tp_size
    worker.pp_size = pp_size
    worker.dcp_size = 1
    worker.pcp_size = 1
    worker.use_kvpp = False
    worker.tp_mismatch = False
    worker.use_mla = False
    worker.use_sparse = False
    worker.num_kv_head = tp_size
    worker.sharded_mamba_group_ids = set()
    worker.num_layers = 2
    worker.group_uses_align_state = group_uses_align_state or (
        [False] * (len(coordinator.kv_cache_groups) if coordinator else 1)
    )
    worker.num_kv_cache_groups = len(worker.grouped_block_size)
    worker.cache_coordinator = coordinator
    worker.m_store = MagicMock()
    worker.token_database = MagicMock()

    def process_token_key_strings(
        token_len,
        block_hashes,
        mask_num=0,
        kv_cache_group_id=0,
        chunk_mask=None,
    ):
        g_bs = worker.grouped_block_size[kv_cache_group_id]
        for start in range(mask_num, (token_len // g_bs) * g_bs, g_bs):
            idx = start // g_bs
            if chunk_mask is not None and not (idx < len(chunk_mask) and chunk_mask[idx]):
                continue
            bh = block_hashes[(start + g_bs) // worker.hash_block_size - 1]
            key = f"model@pp_rank:0@dcp:0@head_or_tp_rank:0@group:{kv_cache_group_id}@{bh.hex()}"
            yield start, start + g_bs, key, bh

    def build_lookup_keys(token_len, block_hashes, group_id):
        keys, starts, ends = [], [], []
        for start, end, key, _ in process_token_key_strings(token_len, block_hashes, kv_cache_group_id=group_id):
            keys.append(key)
            starts.append(start)
            ends.append(end)
        return keys, starts, ends

    worker.token_database.process_token_key_strings.side_effect = process_token_key_strings
    worker._build_lookup_keys = build_lookup_keys  # type: ignore[method-assign]
    return worker


class TestLookupConfigParsing(unittest.TestCase):
    def test_defaults_and_non_mapping_fallback(self):
        for raw in (None, {}, MagicMock()):
            cfg = parse_lookup_config(raw)
            self.assertEqual(cfg.lookup_batch_size, DEFAULT_LOOKUP_BATCH_SIZE)
            self.assertEqual(cfg.lookup_timeout_ms, DEFAULT_LOOKUP_TIMEOUT_MS)
            self.assertEqual(cfg.lookup_async, DEFAULT_LOOKUP_ASYNC)
            self.assertEqual(cfg.lookup_max_inflight, DEFAULT_LOOKUP_MAX_INFLIGHT)

    def test_valid_custom_values(self):
        cfg = parse_lookup_config(
            {
                "lookup_batch_size": 0,
                "lookup_timeout_ms": 2500,
                "lookup_async": True,
                "lookup_max_inflight": 16,
            }
        )
        self.assertEqual(cfg.lookup_batch_size, 0)
        self.assertEqual(cfg.lookup_timeout_ms, 2500)
        self.assertTrue(cfg.lookup_async)
        self.assertEqual(cfg.lookup_max_inflight, 16)

    def test_invalid_values_raise_value_error(self):
        invalid_configs = [
            {"lookup_batch_size": -1},
            {"lookup_batch_size": True},
            {"lookup_batch_size": "512"},
            {"lookup_timeout_ms": 0},
            {"lookup_timeout_ms": -5},
            {"lookup_timeout_ms": False},
            {"lookup_timeout_ms": 1 << 31},
            {"lookup_async": "true"},
            {"lookup_async": 1},
            {"lookup_max_inflight": 0},
            {"lookup_max_inflight": True},
        ]
        for bad in invalid_configs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_lookup_config(bad)


class TestBoundedEarlyStopWorkerLookup(unittest.TestCase):
    def test_151k_long_context_early_miss_stops_in_single_batch(self):
        # Reproduce Issue #22107: 151,625 tokens, low local HBM hit (4992 = 312 blocks),
        # miss at the first external block (block 312).
        total_blocks = 9476
        token_len = total_blocks * 16
        hashes = _make_hashes(total_blocks)
        worker = _make_worker_stub(extra_config={"lookup_batch_size": 512})
        worker.m_store.exists.side_effect = lambda keys: [0] * len(keys)

        hit = worker.lookup_scheduler(token_len, hashes, [0], hbm_hit_tokens=4992)

        self.assertEqual(hit, 4992)
        worker.m_store.exists.assert_called_once()
        probed_keys = worker.m_store.exists.call_args.args[0]
        self.assertEqual(len(probed_keys), 512)

    def test_304k_long_context_partial_hit_matches_unchunked_oracle(self):
        # 304,338 tokens -> 19,021 blocks; external pool hits blocks [312..1200) -> hit=19200.
        total_blocks = 19021
        token_len = total_blocks * 16
        hashes = _make_hashes(total_blocks)
        hit_hashes = {hashes[idx].hex() for idx in range(312, 1200)}

        def fake_exists(keys):
            return [1 if key.rsplit("@", 1)[-1] in hit_hashes else 0 for key in keys]

        chunked_worker = _make_worker_stub(extra_config={"lookup_batch_size": 512})
        chunked_worker.m_store.exists.side_effect = fake_exists
        oracle_worker = _make_worker_stub(extra_config={"lookup_batch_size": 0})
        oracle_worker.m_store.exists.side_effect = fake_exists

        chunked_hit = chunked_worker.lookup_scheduler(token_len, hashes, [0], hbm_hit_tokens=4992)
        oracle_hit = oracle_worker.lookup_scheduler(token_len, hashes, [0], hbm_hit_tokens=4992)

        self.assertEqual(chunked_hit, oracle_hit)
        self.assertEqual(chunked_hit, 1200 * 16)
        # 888 external hit blocks span two 512-key batches (1024 keys vs 18,709 oracle keys).
        self.assertEqual(chunked_worker.m_store.exists.call_count, 2)
        self.assertTrue(all(len(call.args[0]) <= 512 for call in chunked_worker.m_store.exists.call_args_list))

    def test_full_hit_multi_tp_pp_respects_batch_size_and_matches_oracle(self):
        total_blocks = 300
        token_len = total_blocks * 16
        hashes = _make_hashes(total_blocks)

        worker = _make_worker_stub(tp_size=2, pp_size=2, extra_config={"lookup_batch_size": 128})
        worker.m_store.exists.side_effect = lambda keys: [1] * len(keys)

        hit = worker.lookup_scheduler(token_len, hashes, [0], hbm_hit_tokens=0)

        self.assertEqual(hit, token_len)
        # 4 ranks per block -> 32 blocks per 128-key batch -> ceil(300 / 32) = 10 calls.
        self.assertEqual(worker.m_store.exists.call_count, 10)
        for call in worker.m_store.exists.call_args_list:
            self.assertLessEqual(len(call.args[0]), 128)

    def test_coordinator_non_divisible_variants_never_exceed_batch_size(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            return_value=_PrefixManager,
        ):
            coord = AscendStoreCoordinator(
                [KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1))],
                scheduler_block_size=16,
                hash_block_size=16,
                group_block_sizes=[16],
                group_cache_families=["default"],
            )
        # 3 ranks per block with batch_size=4 must flush 1 chunk (3 keys) per call.
        worker = _make_worker_stub(tp_size=3, coordinator=coord, extra_config={"lookup_batch_size": 4})
        worker.m_store.exists.side_effect = lambda keys: [1] * len(keys)
        hit = worker.lookup_scheduler(64, _make_hashes(4), [0])
        self.assertEqual(hit, 64)
        self.assertEqual([len(c.args[0]) for c in worker.m_store.exists.call_args_list], [3, 3, 3, 3])

    def test_non_coordinator_hbm_prefix_skipped_and_partial_tail_loaded(self):
        hashes = _make_hashes(8)
        # Blocks 0..3 are only in HBM (missing in external pool), blocks 4..5 exist externally.
        pool_hashes = {hashes[4].hex(), hashes[5].hex()}
        worker = _make_worker_stub(extra_config={"lookup_batch_size": 2})
        worker.cache_transfer_granularity = 32
        worker.m_store.exists.side_effect = lambda keys: [
            1 if key.rsplit("@", 1)[-1] in pool_hashes else 0 for key in keys
        ]

        hit = worker.lookup_scheduler(128, hashes, [0], hbm_hit_tokens=64)

        self.assertEqual(hit, 96)
        queried = [key for call in worker.m_store.exists.call_args_list for key in call.args[0]]
        for idx in range(4):
            self.assertFalse(any(key.endswith(hashes[idx].hex()) for key in queried))

    def test_hybrid_coordinator_full_attn_miss_caps_swa_group(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            side_effect=lambda spec: (
                _SparseCheckpointManager if isinstance(spec, SlidingWindowSpec) else _PrefixManager
            ),
        ):
            coord = AscendStoreCoordinator(
                [
                    KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1)),
                    KVCacheGroupSpec(
                        ["layer.1"],
                        SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=1, sliding_window=32),
                    ),
                ],
                scheduler_block_size=32,
                hash_block_size=16,
                group_block_sizes=[16, 16],
                group_cache_families=["default", "c1"],
            )
        worker = _make_worker_stub(coordinator=coord, extra_config={"lookup_batch_size": 128})
        worker.m_store.exists.side_effect = lambda keys: [0] * len(keys)

        hit = worker.lookup_scheduler(9600 * 16, _make_hashes(9600), [0, 1], hbm_hit_tokens=4992)

        self.assertEqual(hit, 4992)
        # Only group 0 queried once; group 1 capped to 4992 <= lookup_start and emits 0 RPCs.
        worker.m_store.exists.assert_called_once()
        self.assertTrue(all("@group:0@" in k for k in worker.m_store.exists.call_args.args[0]))

    def test_rswa_and_discontinuous_mamba_do_not_early_stop(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            side_effect=lambda spec: _SparseCheckpointManager if isinstance(spec, RSWASpec) else _PrefixManager,
        ):
            coord = AscendStoreCoordinator(
                [
                    KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1)),
                    KVCacheGroupSpec(["layer.1"], RSWASpec(block_size=16, num_kv_heads=1, head_size=1)),
                ],
                scheduler_block_size=32,
                hash_block_size=16,
                group_block_sizes=[16, 16],
                group_cache_families=["default", "c1"],
            )
        worker = _make_worker_stub(
            coordinator=coord,
            extra_config={"lookup_batch_size": 2},
            group_uses_align_state=[False, False],
        )
        hashes = _make_hashes(6)
        # Group 0 hits all 6 blocks; Group 1 (RSWA) hits only block 3 and block 5.
        present_keys = {f"@group:0@{hashes[i].hex()}" for i in range(6)} | {
            f"@group:1@{hashes[i].hex()}" for i in (3, 5)
        }
        worker.m_store.exists.side_effect = lambda keys: [
            1 if any(key.endswith(suffix) for suffix in present_keys) else 0 for key in keys
        ]
        # Force lookup_mask=None on group 1 so RSWASpec guard itself is exercised.
        with patch.object(coord, "lookup_mask", return_value=(None, None)):
            hit = worker.lookup_scheduler(96, hashes, [0, 1])
        self.assertEqual(hit, 96)

        # Discontinuous Mamba group in non-coordinator path must also scan past block 0 miss.
        mamba_worker = _make_worker_stub(extra_config={"lookup_batch_size": 2}, group_uses_align_state=[True])
        mamba_worker.m_store.exists.side_effect = lambda keys: [
            1 if key.endswith(hashes[3].hex()) else 0 for key in keys
        ]
        self.assertEqual(mamba_worker.lookup_scheduler(96, hashes, [0]), 64)

    def test_single_rank_lookup_bounded_early_stop(self):
        hashes = _make_hashes(100)
        hit_hashes = {hashes[i].hex() for i in range(5)}
        worker = _make_worker_stub(extra_config={"lookup_batch_size": 4})
        worker.m_store.exists.side_effect = lambda keys: [
            1 if key.rsplit("@", 1)[-1] in hit_hashes else 0 for key in keys
        ]

        hit = worker.lookup(1600, hashes, [0])

        self.assertEqual(hit, 80)
        self.assertEqual(worker.m_store.exists.call_count, 2)


class TestLookupDeadlineBudget(unittest.TestCase):
    def test_worker_deadline_stops_slow_backend_without_overclaiming(self):
        hashes = _make_hashes(200)
        worker = _make_worker_stub(extra_config={"lookup_batch_size": 4, "lookup_timeout_ms": 50})
        clock = {"now": 0.0}

        def slow_exists(keys):
            clock["now"] += 0.03
            return [1] * len(keys)

        worker.m_store.exists.side_effect = slow_exists
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.monotonic",
            side_effect=lambda: clock["now"],
        ):
            hit = worker.lookup_scheduler(3200, hashes, [0], deadline=0.05)

        # Stops cleanly after 2 batches (0.03 -> 0.06 >= 0.05), returning 8 blocks = 128 tokens.
        self.assertEqual(worker.m_store.exists.call_count, 2)
        self.assertEqual(hit, 128)

    def test_coordinator_deadline_stops_across_groups(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            side_effect=lambda spec: _SparseCheckpointManager if isinstance(spec, MambaSpec) else _PrefixManager,
        ):
            coord = AscendStoreCoordinator(
                [
                    KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1)),
                    KVCacheGroupSpec(["layer.1"], MambaSpec(block_size=16, shapes=((1,),), dtypes=("float32",))),
                ],
                scheduler_block_size=32,
                hash_block_size=16,
                group_block_sizes=[16, 16],
                group_cache_families=["default", "default"],
            )
        worker = _make_worker_stub(
            coordinator=coord,
            extra_config={"lookup_batch_size": 4, "lookup_timeout_ms": 50},
            group_uses_align_state=[False, True],
        )
        clock = {"now": 0.0}

        def slow_exists(keys):
            clock["now"] += 0.03
            return [1] * len(keys)

        worker.m_store.exists.side_effect = slow_exists
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.monotonic",
            side_effect=lambda: clock["now"],
        ):
            hit = worker.lookup_scheduler(3200, _make_hashes(200), [0, 1], hbm_hit_tokens=32, deadline=0.05)

        # Group 0 used 2 batches (0.06s >= 0.05s); group 1 must not query at all.
        self.assertEqual(worker.m_store.exists.call_count, 2)
        self.assertLessEqual(hit, 32)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.threading.Thread")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.zmq.Poller")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.zmq.Context")
    def test_lookup_server_propagates_timeout_deadline(
        self, mock_ctx, mock_make_socket, mock_poller_cls, mock_thread_cls
    ):
        del mock_ctx
        config = _make_scheduler_config(extra_config={"lookup_timeout_ms": 250})
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        mock_poller = MagicMock()
        mock_poller.poll.side_effect = [[(mock_socket, 1)], [(mock_socket, 1)]]
        mock_poller_cls.return_value = mock_poller
        pool_worker = MagicMock()
        pool_worker.lookup_scheduler.return_value = 64

        server = LookupKeyServer(pool_worker, config)
        server.decoder = MagicMock()
        server.decoder.decode.side_effect = [[0], ["00" * 32]]

        calls = {"n": 0}

        def recv_frames(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return [b"lookup", (64).to_bytes(4, "big"), b"g", (0).to_bytes(4, "big"), b"h"]
            server.running = False
            return [b"reset"]

        mock_socket.recv_multipart.side_effect = recv_frames
        before = time.monotonic()
        mock_thread_cls.call_args.kwargs["target"]()
        after = time.monotonic()

        deadline = pool_worker.lookup_scheduler.call_args.kwargs["deadline"]
        self.assertGreaterEqual(deadline, before + 0.20)
        self.assertLessEqual(deadline, after + 0.30)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.make_zmq_socket")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.zmq")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key.MsgpackEncoder")
    def test_lookup_client_deadline_clamping_and_error_reply(self, mock_encoder, mock_zmq, mock_make_socket):
        mock_zmq.REQ = 3
        mock_zmq.RCVTIMEO = 27
        mock_zmq.ZMQError = RuntimeError
        mock_encoder.return_value.encode.side_effect = lambda x: [b"encoded"]
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket

        config = _make_scheduler_config(extra_config={"lookup_timeout_ms": 500})
        client = LookupKeyClient(config)
        mock_socket.setsockopt.assert_called_once_with(27, 500)

        # Expired deadline skips ZMQ RPC entirely.
        self.assertEqual(client.lookup(64, _make_hashes(4), deadline=time.monotonic() - 0.1), 0)
        mock_socket.send_multipart.assert_not_called()

        # Tight remaining deadline clamps RCVTIMEO; RESP_ERR fails open to 0 without closing client.
        mock_socket.recv.return_value = b"\x00"
        self.assertEqual(client.lookup(64, _make_hashes(4), deadline=time.monotonic() + 0.08), 0)
        self.assertFalse(client.closed)
        clamped_timeout = mock_socket.setsockopt.call_args_list[-1].args[1]
        self.assertGreaterEqual(clamped_timeout, 1)
        self.assertLessEqual(clamped_timeout, 85)
        client.close()


class TestLayerwiseBoundedLookup(unittest.TestCase):
    def setUp(self):
        self._patches = []
        for backend_name in backend_map:
            backend_class = get_backend_class(backend_name)
            assert backend_class is not None
            p = patch.object(backend_class, "create_scheduler_client")
            p.start()
            self._patches.append(p)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()

    def test_layerwise_contiguous_early_stops_and_respects_budget(self):
        config = _make_scheduler_config(
            extra_config={
                "backend": "memcache",
                "lookup_batch_size": 4,
                "layerwise_max_transfer_blocks": 10,
            }
        )
        scheduler = LayerwiseKVPoolScheduler(config, use_layerwise=True)
        request = _make_request("lw-early", 1600)
        scheduler.store_scheduler.batch_is_readable.side_effect = lambda keys: [True, True, False, True][: len(keys)]

        hit = scheduler._lookup_external_hit_tokens(request, 0)

        self.assertEqual(hit, 32)
        scheduler.store_scheduler.batch_is_readable.assert_called_once()
        self.assertEqual(len(scheduler.store_scheduler.batch_is_readable.call_args.args[0]), 4)

    def test_layerwise_deadline_stops_before_all_batches(self):
        config = _make_scheduler_config(
            extra_config={
                "backend": "memcache",
                "lookup_batch_size": 4,
                "lookup_timeout_ms": 50,
            }
        )
        scheduler = LayerwiseKVPoolScheduler(config, use_layerwise=True)
        request = _make_request("lw-timeout", 1600)
        clock = {"now": 0.0}

        def slow_readable(keys):
            clock["now"] += 0.03
            return [True] * len(keys)

        scheduler.store_scheduler.batch_is_readable.side_effect = slow_readable
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler.time.monotonic",
            side_effect=lambda: clock["now"],
        ):
            hit = scheduler._get_layerwise_hit_tokens(request, 1600, 0, deadline=0.05)

        self.assertEqual(scheduler.store_scheduler.batch_is_readable.call_count, 2)
        self.assertEqual(hit, 128)


class TestAsyncNonBlockingLookup(unittest.TestCase):
    def setUp(self):
        self._patches = []
        for backend_name in backend_map:
            backend_class = get_backend_class(backend_name)
            assert backend_class is not None
            p = patch.object(backend_class, "create_scheduler_client")
            p.start()
            self._patches.append(p)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_async_lookup_does_not_block_scheduler_and_completes(self, mock_client_cls):
        started = threading.Event()
        release = threading.Event()

        def slow_lookup(token_len, block_hashes, group_ids, hbm_hit_tokens=0, *, deadline=None):
            del token_len, block_hashes, group_ids, hbm_hit_tokens, deadline
            started.set()
            release.wait(timeout=2.0)
            return 48

        mock_client = MagicMock()
        mock_client.closed = False
        mock_client.lookup.side_effect = slow_lookup
        mock_client_cls.return_value = mock_client

        config = _make_scheduler_config(extra_config={"lookup_async": True, "lookup_timeout_ms": 2000})
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        req_a = _make_request("req-a", 64)
        req_short = _make_request("req-short", 8)

        t0 = time.monotonic()
        self.assertEqual(scheduler.get_num_new_matched_tokens(req_a, 0), (None, False))
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.2)
        self.assertTrue(started.wait(timeout=1.0))

        # A short request below granularity is not blocked while req-a lookup is in flight.
        self.assertEqual(scheduler.get_num_new_matched_tokens(req_short, 0), (0, False))

        release.set()
        deadline = time.monotonic() + 2.0
        result = (None, False)
        while time.monotonic() < deadline:
            result = scheduler.get_num_new_matched_tokens(req_a, 0)
            if result[0] is not None:
                break
            time.sleep(0.005)

        self.assertEqual(result, (48, False))
        self.assertEqual(scheduler.load_specs["req-a"].kvpool_cached_tokens, 48)
        scheduler.shutdown_async_lookup()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_async_lookup_timeout_fails_open_without_blocking(self, mock_client_cls):
        release = threading.Event()

        def hung_lookup(token_len, block_hashes, group_ids, hbm_hit_tokens=0, *, deadline=None):
            del token_len, block_hashes, group_ids, hbm_hit_tokens, deadline
            release.wait(timeout=2.0)
            return 48

        mock_client = MagicMock()
        mock_client.closed = False
        mock_client.lookup.side_effect = hung_lookup
        mock_client_cls.return_value = mock_client

        config = _make_scheduler_config(extra_config={"lookup_async": True, "lookup_timeout_ms": 60})
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        req = _make_request("req-hung", 64)

        self.assertEqual(scheduler.get_num_new_matched_tokens(req, 0), (None, False))
        time.sleep(0.09)
        t0 = time.monotonic()
        self.assertEqual(scheduler.get_num_new_matched_tokens(req, 0), (0, False))
        self.assertLess(time.monotonic() - t0, 0.05)

        release.set()
        scheduler.shutdown_async_lookup()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_async_lookup_inflight_cap_and_stale_queued_skip(self, mock_client_cls):
        gate = threading.Event()
        started = threading.Event()
        called_tokens: list[int] = []

        def gated_lookup(token_len, block_hashes, group_ids, hbm_hit_tokens=0, *, deadline=None):
            del block_hashes, group_ids, hbm_hit_tokens, deadline
            called_tokens.append(token_len)
            started.set()
            gate.wait(timeout=2.0)
            return 32

        mock_client = MagicMock()
        mock_client.closed = False
        mock_client.lookup.side_effect = gated_lookup
        mock_client_cls.return_value = mock_client

        config = _make_scheduler_config(
            extra_config={"lookup_async": True, "lookup_timeout_ms": 60, "lookup_max_inflight": 2}
        )
        scheduler = KVPoolScheduler(config, use_layerwise=False)
        req1 = _make_request("req-1", 64)
        req2 = _make_request("req-2", 96)
        req3 = _make_request("req-3", 128)

        self.assertEqual(scheduler.get_num_new_matched_tokens(req1, 0), (None, False))
        self.assertTrue(started.wait(timeout=1.0))
        self.assertEqual(scheduler.get_num_new_matched_tokens(req2, 0), (None, False))
        # Third concurrent request exceeds max_inflight=2 -> fail-open (0, False).
        self.assertEqual(scheduler.get_num_new_matched_tokens(req3, 0), (0, False))

        # Wait past req2's 60ms deadline before unblocking req1; queued req2 must skip RPC.
        time.sleep(0.09)
        gate.set()
        scheduler.shutdown_async_lookup()
        self.assertEqual(called_tokens, [64])

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_async_lookup_lifecycle_signature_ttl_and_connector_shutdown(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.closed = False
        mock_client.lookup.return_value = 32
        mock_client.reset.return_value = True
        mock_client_cls.return_value = mock_client

        # Verify TTL expiry and signature mismatch handling directly on AsyncLookupManager.
        self.assertGreater(DEFAULT_ASYNC_LOOKUP_TTL_S, 0)
        mgr = AsyncLookupManager(timeout_ms=500, max_inflight=4, result_ttl_s=0.05)
        calls = {"count": 0}

        def fast_factory(val: int):
            def factory():
                calls["count"] += 1
                return lambda _deadline: val

            return factory

        def poll_done(req_id: str, sig, factory) -> int:
            end = time.monotonic() + 2.0
            res = mgr.poll_or_submit(req_id, sig, factory)
            while res is None and time.monotonic() < end:
                time.sleep(0.005)
                res = mgr.poll_or_submit(req_id, sig, factory)
            assert res is not None
            return res

        sig1 = make_lookup_signature(64, 0, _make_hashes(4))
        sig2 = make_lookup_signature(64, 16, _make_hashes(4))
        self.assertEqual(poll_done("r1", sig1, fast_factory(32)), 32)
        self.assertEqual(mgr.poll_or_submit("r1", sig1, fast_factory(48)), 32)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(poll_done("r1", sig2, fast_factory(48)), 48)
        self.assertEqual(calls["count"], 2)
        time.sleep(0.07)
        self.assertEqual(poll_done("r1", sig2, fast_factory(64)), 64)
        self.assertEqual(calls["count"], 3)
        mgr.shutdown()

        # Verify scheduler cleanup hooks and AscendStoreConnector.shutdown().
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        cache_cfg = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1))
            ]
        )
        connector = AscendStoreConnector(
            vllm_config=_make_scheduler_config(extra_config={"lookup_async": True, "lookup_timeout_ms": 500}),
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=cache_cfg,  # type: ignore[arg-type]
        )
        scheduler = connector.connector_scheduler
        assert scheduler is not None
        scheduler.client = mock_client
        req = _make_request("r-clean", 64)
        scheduler.get_num_new_matched_tokens(req, 0)
        scheduler.update_state_after_alloc(req, MagicMock(), 0)
        self.assertNotIn("r-clean", scheduler._async_lookup._entries)

        sched_out = SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
            finished_req_ids={"r-fin"},
            preempted_req_ids={"r-preempt"},
        )
        scheduler._async_lookup.poll_or_submit("r-fin", sig1, fast_factory(32))
        scheduler._async_lookup.poll_or_submit("r-preempt", sig1, fast_factory(32))
        scheduler.build_connector_meta(sched_out)  # type: ignore[arg-type]
        self.assertNotIn("r-fin", scheduler._async_lookup._entries)
        self.assertNotIn("r-preempt", scheduler._async_lookup._entries)

        scheduler._block_pool = MagicMock(num_gpu_blocks=10, get_num_free_blocks=MagicMock(return_value=9))
        self.assertTrue(scheduler.reset_store())
        connector.shutdown()
        self.assertTrue(scheduler._async_lookup._closed)

    def test_layerwise_async_lookup_nonblocking(self):
        started = threading.Event()
        release = threading.Event()

        def slow_readable(keys):
            started.set()
            release.wait(timeout=2.0)
            return [True] * len(keys)

        config = _make_scheduler_config(
            extra_config={
                "backend": "memcache",
                "lookup_async": True,
                "lookup_timeout_ms": 2000,
                "lookup_batch_size": 16,
            }
        )
        scheduler = LayerwiseKVPoolScheduler(config, use_layerwise=True)
        scheduler.store_scheduler.batch_is_readable.side_effect = slow_readable
        request = _make_request("lw-async", 48)

        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (None, False))
        self.assertTrue(started.wait(timeout=1.0))
        release.set()

        deadline = time.monotonic() + 2.0
        result = (None, False)
        while time.monotonic() < deadline:
            result = scheduler.get_num_new_matched_tokens(request, 0)
            if result[0] is not None:
                break
            time.sleep(0.005)
        self.assertEqual(result, (47, False))
        scheduler.shutdown_async_lookup()


if __name__ == "__main__":
    unittest.main()
