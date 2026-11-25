# SPDX-License-Identifier: Apache-2.0
"""Focused tests for bounded KV pool lookup probing and non-blocking async lookup."""

import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
import zmq
from msgspec import ValidationError
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, RSWASpec
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
# isort: on

import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lookup_key as lk
from tests.ut.distributed.ascend_store.cache_config_helpers import make_resolved_cache_config
from tests.ut.distributed.ascend_store.test_coordinator import _FakePrefixManager, _full_spec
from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend_map, get_backend_class
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import StoreLookupResult, TailKeyBoundary
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

_COORD_MGR = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class"
_SCHED_CLIENT = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
_h = lambda n: [i.to_bytes(32, "big") for i in range(n)]
_req = lambda r, n: SimpleNamespace(request_id=r, prompt_token_ids=range(n), num_tokens=n, block_hashes=_h(n // 16))


class _SparseMgr:
    @classmethod
    def reachable_block_mask(cls, s, e, *_a, **_kw):
        return [(i % 2 == 1) for i in range(s, e)]

    @classmethod
    def find_longest_cache_hit(cls, **kw):
        bhs, gids, pool = kw["block_hashes"], kw["kv_cache_group_ids"], kw["block_pool"]
        bs = kw["kv_cache_spec"].block_size
        n = max((i + 1 for i in range(kw["max_length"] // bs) if pool.get_cached_block(bhs[i], gids)), default=0)
        return tuple([] for _ in gids), n * bs


def _worker(extra_cfg=None, coord=None, hash_bs=16) -> KVPoolWorker:
    w, n = object.__new__(KVPoolWorker), len(coord.kv_cache_groups) if coord else 1
    w.__dict__.update({k: False for k in ("use_kvpp", "tp_mismatch", "use_mla", "use_sparse")})
    w.__dict__.update({k: 1 for k in ("tp_size", "pp_size", "dcp_size", "pcp_size", "num_kv_head")})
    w._lookup_config, w.block_size, w.hash_block_size = lk.parse_lookup_config(extra_cfg), 16, hash_bs
    w.grouped_block_size, w.cache_transfer_granularity = [16] * n, coord.lcm_block_size if coord else 16
    w.max_model_len, w.num_layers, w.sharded_mamba_group_ids, w._group_pp_ranks = 1048576, 2, set(), None
    w.group_uses_align_state, w.num_kv_cache_groups = [False] * n, n
    w.cache_coordinator, w.m_store, w.token_database = coord, MagicMock(), MagicMock()
    w.token_database._get_key_prefix.side_effect = lambda g: f"m@pp_rank:0@dcp:0@head_or_tp_rank:0@group:{g}@"

    def proc(tl, bh, mask_num=0, kv_cache_group_id=0, chunk_mask=None):
        for st in range(mask_num, (tl // 16) * 16, 16):
            if chunk_mask is None or (st // 16 < len(chunk_mask) and chunk_mask[st // 16]):
                h = bh[(st + 16) // hash_bs - 1]
                hx = h if isinstance(h, str) else h.hex()
                yield st, st + 16, f"m@pp_rank:0@dcp:0@head_or_tp_rank:0@group:{kv_cache_group_id}@{hx}", h

    w.token_database.process_token_key_strings.side_effect = proc
    return w


def _gated(started: threading.Event, gate: threading.Event, val):
    def _fn(*args, **_kw):
        started.set()
        gate.wait(timeout=2.0)
        return val(args[0]) if callable(val) else val

    return _fn


def _poll_ready(fn):
    dl = time.monotonic() + 2.0
    while time.monotonic() < dl:
        if (val := fn()[0]) is not None:
            return val
        time.sleep(0.005)
    raise AssertionError("timed out waiting for async lookup")


@contextmanager
def _rpc(port: int, s_ms: int = 1000, c_ms: int = 500):
    with ExitStack() as st, tempfile.TemporaryDirectory(prefix="lk-") as d:
        st.enter_context(patch.object(lk.envs, "VLLM_RPC_BASE_PATH", d))
        ex = {"lookup_rpc_port": port, "lookup_batch_size": 4}
        mk = lambda ms: make_config(kv_role="kv_both", extra_config={**ex, "lookup_timeout_ms": ms})
        yield mk(s_ms), mk(c_ms), st


class TestLookupNonblocking(unittest.TestCase):
    def setUp(self):
        st = ExitStack()
        self.addCleanup(st.close)
        for name in backend_map:
            st.enter_context(patch.object(get_backend_class(name), "create_scheduler_client"))

    def test_config_and_bounded_worker_lookup(self):
        cfg = lk.parse_lookup_config(None)
        self.assertEqual((cfg.lookup_batch_size, cfg.lookup_timeout_ms, cfg.lookup_async), (512, 1000, False))
        for k, v in dict(lookup_batch_size=True, lookup_timeout_ms=0, lookup_async=1, lookup_max_inflight=0).items():
            with self.assertRaises(ValueError):
                lk.parse_lookup_config({k: v})
        hashes = _h(200)
        hits = {hashes[i].hex() for i in range(4, 15)}
        exists_fn = lambda ks: [1 if k.rsplit("@", 1)[-1] in hits else 0 for k in ks]
        wc, wo = _worker({"lookup_batch_size": 8}), _worker({"lookup_batch_size": 0})
        wc.m_store.exists.side_effect = wo.m_store.exists.side_effect = exists_fn
        self.assertEqual(wc.lookup_scheduler(3200, hashes, [0], hbm_hit_tokens=64), StoreLookupResult(240))
        self.assertEqual(wo.lookup_scheduler(3200, hashes, [0], hbm_hit_tokens=64), StoreLookupResult(240))
        self.assertEqual(wc.m_store.exists.call_count, 2)

        rswa = RSWASpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32, rswa_window=32)
        with patch(_COORD_MGR, side_effect=lambda s: _SparseMgr if isinstance(s, RSWASpec) else _FakePrefixManager):
            gs = [KVCacheGroupSpec(["l0"], _full_spec(16)), KVCacheGroupSpec(["l1"], rswa)]
            coord = AscendStoreCoordinator(gs, 32, 16, [16, 16], ["default", "c1"])
        wh, h6 = _worker({"lookup_batch_size": 2}, coord), _h(6)
        present = {f"@group:0@{h6[i].hex()}" for i in range(6)} | {f"@group:1@{h6[i].hex()}" for i in (3, 5)}
        wh.m_store.exists.side_effect = lambda ks: [1 if any(k.endswith(s) for s in present) else 0 for k in ks]
        with patch.object(coord, "lookup_mask", return_value=(None, None)):
            self.assertEqual(wh.lookup_scheduler(96, h6, [0, 1]), StoreLookupResult(96))
        pc = AscendStoreCoordinator([gs[0]], 16, 4, [16], ["default"], enable_partial_hash_hits=True)
        wp = _worker({"lookup_batch_size": 4}, pc, hash_bs=4)
        wp.m_store.exists.side_effect = lambda ks: (threading.Event().wait(0.05), [1] * len(ks))[1]
        hit = wp.lookup_scheduler(65, _h(16), [0], deadline=time.monotonic() + 0.08)
        n = wp.m_store.exists.call_count
        self.assertTrue(1 <= n <= 2 and hit == StoreLookupResult(n * 16, (TailKeyBoundary(0, n * 16),)))

    def test_real_zmq_roundtrip_deadline_and_reset(self):
        with _rpc(201) as (s_cfg, c_cfg, st):
            expected, worker, hashes = StoreLookupResult(32, (TailKeyBoundary(0, 48),)), MagicMock(), _h(3)
            worker.lookup_scheduler.return_value = expected
            st.callback(lk.LookupKeyServer(worker, s_cfg).close)
            client = lk.LookupKeyClient(c_cfg)
            st.callback(client.close)
            t0 = time.monotonic()
            self.assertEqual(client.lookup(64, hashes, deadline=t0 + 0.08), expected)
            self.assertTrue(1 <= client._current_rcvtimeo_ms <= 85)
            self.assertTrue(t0 <= worker.lookup_scheduler.call_args.kwargs["deadline"] <= time.monotonic() + 0.14)
            self.assertTrue(client.reset() and client._current_rcvtimeo_ms == lk.DEFAULT_RESET_TIMEOUT_MS)
            worker.lookup_scheduler.side_effect = RuntimeError("boom")
            with self.assertRaises(ValidationError):
                client.lookup(64, hashes)
            self.assertFalse(client.closed)

        with _rpc(202, s_ms=100, c_ms=30) as (s_cfg, c_cfg, st):
            ctx = zmq.Context()  # type: ignore[attr-defined]
            st.callback(ctx.destroy, 0)
            rep, req = ctx.socket(zmq.REP), ctx.socket(zmq.REQ)  # type: ignore[attr-defined]
            rep.bind(lk.get_zmq_rpc_path_lookup(c_cfg))
            c = lk.LookupKeyClient(c_cfg)
            st.callback(c.close)
            self.assertEqual((c.lookup(64, _h(2)), c.closed), (StoreLookupResult(0), True))
            w = _worker({"lookup_batch_size": 4, "lookup_timeout_ms": 100})
            w.m_store.exists.side_effect = lambda ks: [1] * len(ks)
            st.callback(lk.LookupKeyServer(w, s_cfg).close)
            req.setsockopt(zmq.RCVTIMEO, 2000)  # type: ignore[attr-defined]
            req.connect(lk.get_zmq_rpc_path_lookup(s_cfg))
            enc, dec = MsgpackEncoder(), MsgpackDecoder(tuple[int, tuple[TailKeyBoundary, ...]])
            hdr = [b"lookup", (64).to_bytes(4, "big"), *enc.encode([0]), (0).to_bytes(4, "big")]
            req.send_multipart([*hdr, *enc.encode([h.hex() for h in _h(4)]), lk._encode_lookup_budget_frame(80)])
            self.assertEqual(StoreLookupResult(*dec.decode(req.recv_multipart())), StoreLookupResult(64))
            req.send_multipart([b"unknown"])
            self.assertEqual(req.recv_multipart(), [lk.RESP_ERR])

    @patch(_SCHED_CLIENT)
    def test_async_lookup_completion_cap_and_timeout_fallback(self, mock_client_cls):
        started, release, bounds = threading.Event(), threading.Event(), (TailKeyBoundary(0, 56),)
        self.addCleanup(release.set)
        client = mock_client_cls.return_value
        client.closed, client.lookup.side_effect = False, _gated(started, release, StoreLookupResult(48, bounds))
        mk = lambda ms, cap=64: KVPoolScheduler(
            make_config(extra_config={"lookup_async": True, "lookup_timeout_ms": ms, "lookup_max_inflight": cap}),
            False,
            kv_cache_config=make_resolved_cache_config(make_config()),
        )
        sched, slow = mk(2000, 1), mk(40)
        self.addCleanup(sched.shutdown_async_lookup)
        self.addCleanup(slow.shutdown_async_lookup)
        self.assertEqual(sched.get_num_new_matched_tokens(_req("r1", 64), 0), (None, False))
        self.assertTrue(started.wait(timeout=1.0))
        self.assertEqual(sched.get_num_new_matched_tokens(_req("r2", 64), 0), (0, False))
        release.set()
        self.assertEqual(_poll_ready(lambda: sched.get_num_new_matched_tokens(_req("r1", 64), 0)), 48)
        self.assertEqual(sched.load_specs["r1"].tail_key_boundaries, bounds)
        release.clear()
        client.lookup.side_effect = _gated(started, release, StoreLookupResult(48))
        self.assertEqual(slow.get_num_new_matched_tokens(_req("r-slow", 64), 0), (None, False))
        time.sleep(0.06)
        self.assertEqual(slow.get_num_new_matched_tokens(_req("r-slow", 64), 0), (0, False))

    @patch(_SCHED_CLIENT)
    def test_async_abort_error_and_shutdown_drain_safety(self, mock_client_cls):
        gate, started = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        client = mock_client_cls.return_value
        client.closed, client.reset.return_value, client.lookup.side_effect = False, True, ValidationError("bad")
        cfg = make_config(extra_config={"lookup_async": True, "lookup_timeout_ms": 30})
        groups = SimpleNamespace(kv_cache_groups=[KVCacheGroupSpec(["l0"], _full_spec(16))])
        conn = AscendStoreConnector(cfg, KVConnectorRole.SCHEDULER, make_resolved_cache_config(cfg, groups))
        self.addCleanup(conn.shutdown)
        sched = conn.connector_scheduler
        assert sched is not None and sched._async_lookup is not None
        sched._block_pool = MagicMock(num_gpu_blocks=10, get_num_free_blocks=MagicMock(return_value=9))
        sched.store_scheduler = MagicMock()
        self.assertEqual(_poll_ready(lambda: sched.get_num_new_matched_tokens(_req("r-err", 64), 0)), 0)
        r_alloc = _req("r-alloc", 64)
        client.lookup.side_effect = _gated(started, gate, StoreLookupResult(32))
        self.assertEqual(sched.get_num_new_matched_tokens(r_alloc, 0), (None, False))
        self.assertTrue(started.wait(timeout=1.0))
        sched.update_state_after_alloc(r_alloc, MagicMock(), 0)
        self.assertNotIn("r-alloc", sched._async_lookup._entries)
        self.assertFalse(sched.reset_store())
        conn.shutdown()
        client.close.assert_not_called()
        gate.set()
        self.assertTrue(sched._async_lookup.wait_and_clear(timeout_s=1.0))
        conn.shutdown()
        client.close.assert_called_once_with()

    def test_layerwise_bounded_early_stop_and_async_lookup(self):
        extra = {"backend": "memcache", "lookup_batch_size": 4, "layerwise_max_transfer_blocks": 10}
        cfg = make_config(kv_role="kv_both", extra_config={**extra, "lookup_async": True, "lookup_timeout_ms": 2000})
        sched = LayerwiseKVPoolScheduler(cfg, use_layerwise=True, kv_cache_config=make_resolved_cache_config(cfg))
        self.addCleanup(sched.shutdown_async_lookup)
        sched.store_scheduler.batch_is_readable.side_effect = lambda ks: [True, True, False, True][: len(ks)]
        self.assertEqual(sched._lookup_external_hit_tokens(_req("lw-early", 1600), 0), StoreLookupResult(32))
        sched.store_scheduler.batch_is_readable.assert_called_once()
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        sched.store_scheduler.batch_is_readable.side_effect = _gated(started, release, lambda ks: [True] * len(ks))
        req = _req("lw-async", 48)
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0), (None, False))
        self.assertTrue(started.wait(timeout=1.0))
        release.set()
        self.assertEqual(_poll_ready(lambda: sched.get_num_new_matched_tokens(req, 0)), 47)
