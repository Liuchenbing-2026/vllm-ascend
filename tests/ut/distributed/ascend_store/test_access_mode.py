# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Behavioral unit tests for access_mode across AscendStore components."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import CachedRequestData, KVConnectorBlockState, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker, start_patch
from tests.ut.kv_offload.utils import create_request
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import backend, kv_transfer, metadata, pool_scheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler

_MOD = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"


def _cfg(ex: dict[str, Any]) -> MagicMock:
    c = make_config(kv_role="kv_both", extra_config=dict(ex), block_size=16)
    c.kv_transfer_config.kv_connector, c.model_config.max_model_len = "AscendStoreConnector", 1024
    c.model_config.hf_config, c.scheduler_config.disable_hybrid_kv_cache_manager = MagicMock(spec=[]), False
    c.speculative_config, c.use_v2_model_runner = None, False
    return c


def _kv(reuse: dict[str, Any] | None = None) -> KVCacheConfig:
    sp = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    c = KVCacheConfig(32, [], [KVCacheGroupSpec(["model.layers.0.self_attn"], sp)])
    c.hash_block_size = c.cache_hit_alignment_tokens = 16  # type: ignore[attr-defined]
    if reuse is not None:
        tc = KVTransferConfig(kv_connector="AscendStoreConnector", kv_role="kv_both", kv_connector_extra_config=reuse)
        c.kv_transfer_config = tc  # type: ignore[attr-defined]
    return c


def _sout(req: Any = None, blks: KVCacheBlocks | None = None, pre: Any = None) -> SchedulerOutput:
    out = SchedulerOutput.make_empty()
    out.preempted_req_ids = pre or set()
    if req is not None and blks is not None:
        out.scheduled_new_reqs = [NewRequestData.from_request(req, blks.get_block_ids(), req.prompt_token_ids)]
        out.num_scheduled_tokens = {req.request_id: req.num_tokens - req.num_computed_tokens}
        out.kv_connector_block_state = KVConnectorBlockState({req.request_id}, lambda _: blks.get_block_ids(), {})
    return out


def test_scheduler_and_connector_access_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in backend.backend_map:
        monkeypatch.setattr(backend.get_backend_class(name), "create_scheduler_client", MagicMock())
    for mode in ("readwrite", "read", "write"):
        ex = {"backend": "mooncake", "access_mode": mode, "enable_lookup": True, "load_async": True}
        conn = AscendStoreConnector(_cfg(ex), KVConnectorRole.SCHEDULER, _kv())
        sched = conn.connector_scheduler
        assert isinstance(sched, pool_scheduler.KVPoolScheduler)
        with patch(f"{_MOD}.pool_scheduler.LookupKeyClient") as c_cls:
            ht = 0 if mode == "readwrite" else 16
            c = c_cls.return_value
            c.closed, c.lookup.return_value, c.reset.return_value = False, metadata.StoreLookupResult(ht), True
            assert conn.reset_cache() is True and c.reset.called is (mode != "read")
            r1, b1 = create_request(1, 48, 16), KVCacheBlocks(([KVCacheBlock(i) for i in (1, 2, 3)],))
            hit, is_async = sched.get_num_new_matched_tokens(r1, 0)
            assert (hit, is_async) == ((ht, bool(ht)) if mode != "write" else (0, False))
            sched.update_state_after_alloc(r1, b1, hit)
            assert sched.build_connector_meta(_sout(r1, b1)).requests[0].can_save is (mode != "read")
            if mode == "read":
                sched.build_connector_meta(_sout(pre={r1.request_id}))
                assert sched.get_num_new_matched_tokens(r1, 0)[0] == 16
                sched.update_state_after_alloc(r1, b1, 16)
                cout = _sout()
                cout.scheduled_cached_reqs = CachedRequestData([r1.request_id], [[]], [([1, 2, 3],)], [0], [0])
                cout.num_scheduled_tokens = {r1.request_id: 16}
                res = sched.build_connector_meta(cout).requests
                assert res[0].load_spec is not None and res[0].load_spec.can_load and not res[0].can_save
    with pytest.raises(ValueError, match="layerwise_offload"):
        LayerwiseKVPoolScheduler(
            _cfg({"backend": "memcache", "access_mode": "read"}), True, _kv({"_layerwise_reuse_backing_bytes": 1024})
        )


class TestAccessModeWorkers(unittest.TestCase):
    def test_worker_and_data_planes_access_mode(self) -> None:
        t_start = start_patch(self, f"{_MOD}.kv_transfer.KVTransferThread.start", autospec=True)
        t_start.side_effect = lambda t: t.ready_event.set()
        mw = lambda ex, lw=False: make_worker(
            self, kv_role="kv_both", extra_config=ex, use_layerwise=lw, kv_cache_config=_kv()
        )
        kv = {"model.layers.0.self_attn": torch.zeros((32, 16, 1, 8), dtype=torch.float16)}
        wr, ww = (mw({"backend": "mooncake", "access_mode": m, "load_async": True}) for m in ("read", "write"))
        wr.register_kv_caches(kv)
        ww.register_kv_caches(kv)
        assert wr.kv_send_thread is None and isinstance(wr.kv_recv_thread, kv_transfer.KVCacheStoreRecvingThread)
        assert isinstance(ww.kv_send_thread, kv_transfer.KVCacheStoreSendingThread) and ww.kv_recv_thread is None
        wr.reset_store()
        wr.m_store.remove_all.assert_not_called()
        assert ww.lookup_scheduler(32, [b"h" * 32]) == metadata.StoreLookupResult(0)
        ww.start_load_kv(metadata.AscendConnectorMetadata(set(), {"r1"}))
        ww.m_store.get.assert_not_called()
        lwr, lww = (mw({"backend": "memcache", "access_mode": m}, True) for m in ("read", "write"))
        bk_w = mw({"backend": "mooncake", "access_mode": "write"}, True)
        req, lwr.m_store, lww.m_store = metadata.ReqMeta("r", load_block_keys=["stale"]), MagicMock(), MagicMock()
        lwr._data_plane._alloc_gvas_for_save([req])
        lwr.m_store.batch_alloc.assert_not_called()
        lww._data_plane._prepare_load_gvas([req])
        lww.m_store.batch_get_key_info.assert_not_called()
        assert (bk_w._data_plane._prepare_single_get_session(req), req.load_block_keys) == ([], [])
