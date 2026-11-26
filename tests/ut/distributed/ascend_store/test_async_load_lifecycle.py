# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from tests.ut.distributed.ascend_store.cache_config_helpers import make_resolved_cache_config
from tests.ut.distributed.ascend_store.test_kv_transfer import FakeStore, FakeTokenDatabase
from tests.ut.kv_offload.utils import create_vllm_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer, metadata, pool_scheduler, pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend_map

_BLOCKS = KVCacheBlocks(blocks=(tuple(KVCacheBlock(i) for i in (1, 2, 3)),))


class _DummyBackend(FakeStore):
    def __init__(self, *_: Any, get_fn: Any = None, **__: Any):
        super().__init__([1, 1, 1, 1])
        self._get_fn, self.remove_all_calls = get_fn, 0

    @classmethod
    def create_scheduler_client(cls, parallel_config: Any) -> Any:
        return None

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int]:
        return self._get_fn(keys, addrs, sizes) if self._get_fn is not None else super().get(keys, addrs, sizes)

    def remove_all(self) -> None:
        self.remove_all_calls += 1


def _out(**kw: Any) -> SchedulerOutput:
    c = SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[])
    ns = SimpleNamespace(finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[])
    ns.scheduled_cached_reqs, ns.num_scheduled_tokens, ns.kv_connector_block_state = c, {}, None
    vars(ns).update(kw)
    return cast(SchedulerOutput, ns)


def _req(rid: str) -> Request:
    sp, bh = SamplingParams(max_tokens=4), get_request_block_hasher(16, sha256)
    return Request(rid, list(range(48)), sp, pooling_params=None, block_hasher=bh)


def _meta(rid: str, blks: list[list[int]], grps: list[int] | None = None) -> metadata.ReqMeta:
    n, bh = len(blks[0]) * 16, [b"h0", b"h1"][: len(blks[0])]
    sp = metadata.LoadSpec(0, n, can_load=True, token_len=n)
    return metadata.ReqMeta(rid, n, block_ids_by_group=blks, block_hashes=bh, kv_cache_group_ids=grps, load_spec=sp)


def _wait(pred) -> None:
    dl = time.monotonic() + 3.0
    while not pred():
        assert time.monotonic() < dl
        time.sleep(0.005)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    init_none_hash(sha256)
    monkeypatch.setitem(backend_map, "dummy", {"path": __name__, "name": "_DummyBackend"})
    for attr in ("get_tensor_model_parallel_rank", "get_decode_context_model_parallel_rank"):
        monkeypatch.setattr(pool_worker, attr, lambda: 0)
    for attr in ("get_tensor_model_parallel_world_size", "get_decode_context_model_parallel_world_size"):
        monkeypatch.setattr(pool_worker, attr, lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    started: list[kv_transfer.KVCacheStoreRecvingThread] = []

    def start_recv(backend: Any, worker: Any = None, bsz: Any = 16, inv: set[int] | None = None):
        inv_s = inv if inv is not None else getattr(worker, "_invalid_block_ids", set())
        lk = getattr(worker, "_invalid_block_ids_lock", None) or threading.Lock()
        db, ev = FakeTokenDatabase(), threading.Event()
        t = kv_transfer.KVCacheStoreRecvingThread(backend, db, bsz, 0, 1, 1, ev, inv_s, lk, worker=worker)
        t.start()
        assert t.ready_event.wait(timeout=2.0)
        started.append(t)
        return t

    def sched_worker():
        mod = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"
        ex = {"backend": "dummy", "load_async": True, "use_layerwise": False}
        kw = {"kv_connector": "AscendStoreConnector", "kv_role": "kv_both", "kv_connector_module_path": mod}
        cfg = create_vllm_config(block_size=16, kv_transfer_config=KVTransferConfig(**kw, kv_connector_extra_config=ex))
        cfg.cache_config.prefix_match_unit = None  # type: ignore[attr-defined]
        cfg.cache_config.prefix_cache_retention_interval = None  # type: ignore[attr-defined]
        kc = make_resolved_cache_config(cfg)
        sched = pool_scheduler.KVPoolScheduler(cfg, use_layerwise=False, kv_cache_config=kc)
        sched.bind_gpu_block_pool(cast(Any, SimpleNamespace(num_gpu_blocks=16, get_num_free_blocks=lambda: 12)))
        worker = pool_worker.KVPoolWorker(cfg, use_layerwise=False, kv_cache_config=kc)
        worker.token_database = FakeTokenDatabase()
        worker.kv_recv_thread = start_recv(worker.m_store, worker)
        return sched, worker, cast(kv_transfer.KVCacheStoreRecvingThread, worker.kv_recv_thread)

    with patch.object(pool_scheduler, "LookupKeyClient") as c:
        c.return_value.closed, c.return_value.reset.return_value = False, True
        c.return_value.lookup.return_value = metadata.StoreLookupResult(32)
        yield SimpleNamespace(start_recv=start_recv, sched_worker=sched_worker)
    for t in started:
        t.stop()
        t.join(timeout=2.0)


def test_async_load_abort_and_reset_lifecycle(env):
    gate, started = threading.Event(), threading.Event()
    sched, worker, recv = env.sched_worker()

    def slow_get(k, *_):
        started.set()
        gate.wait(3.0)
        return [0] * len(k)

    cast(_DummyBackend, worker.m_store)._get_fn = slow_get
    req = _req("r-in")
    sched.get_num_new_matched_tokens(req, 0)
    sched.update_state_after_alloc(req, _BLOCKS, 32)
    worker.start_load_kv(sched.build_connector_meta(_out()))
    assert started.wait(timeout=2.0)
    step2 = sched.build_connector_meta(_out(finished_req_ids={"r-in"}))
    assert (step2.loading_req_ids, step2.cancelled_load_req_ids) == ({"r-in"}, set()) and not sched.reset_store()
    gate.set()
    _wait(lambda: "r-in" in recv.finished_requests)
    assert worker.get_finished(set(), step2) == (set(), {"r-in"})
    sched.update_connector_output(KVConnectorOutput(finished_recving={"r-in"}))
    assert (sched._async_loading_req_ids, sched.load_specs) == (set(), {})
    recv.set_finished_request("stale")
    worker.kv_send_thread = kv_transfer.KVTransferThread(worker.m_store, FakeTokenDatabase(), 16, tp_rank=0)
    worker.reset_store()
    assert recv.finished_requests == set() and cast(_DummyBackend, worker.m_store).remove_all_calls == 1


@pytest.mark.parametrize("mode", ["async_readmit", "sync_readmit", "ordinary_preemption"])
def test_same_step_preemption_and_ordinary_preemption(mode, env):
    sched, worker, recv = env.sched_worker()
    worker.kv_send_thread, rid, req = None, f"r-{mode}", _req(f"r-{mode}")
    if mode == "sync_readmit":
        sched.update_state_after_alloc(req, _BLOCKS, 0)
        t, d = req.prompt_token_ids, {"num_computed_tokens": 0, "block_ids": [[1, 2, 3]]}
        nr = SimpleNamespace(req_id=rid, prefill_token_ids=t, prompt_token_ids=t, **d)
        bs = KVConnectorBlockState({rid}, lambda r: tuple(sched._request_trackers[r].allocated_block_ids_by_group), {})
        kw = {"preempted_req_ids": {rid}, "scheduled_new_reqs": [nr], "num_scheduled_tokens": {rid: 32}}
        meta = sched.build_connector_meta(_out(**kw, kv_connector_block_state=bs))
        assert [m.req_id for m in meta.requests] == [rid] and rid not in sched._preempted_req_ids
        return
    sched.get_num_new_matched_tokens(req, 0)
    sched.update_state_after_alloc(req, _BLOCKS, 32)
    if mode == "ordinary_preemption":
        worker.start_load_kv(sched.build_connector_meta(_out()))
        _wait(lambda: rid in recv.finished_requests)
    meta = sched.build_connector_meta(_out(preempted_req_ids={rid}))
    if mode == "async_readmit":
        assert rid not in sched._preempted_req_ids and meta.loading_req_ids == {rid}
        worker.start_load_kv(meta)
        _wait(lambda: rid in recv.finished_requests)
        assert worker.get_finished(set(), meta) == (set(), {rid})
    else:
        assert (meta.loading_req_ids, sched.load_specs, sched._async_loading_req_ids) == (set(), {}, set())
        assert worker.get_finished(set(), meta) == (set(), set()) and recv.finished_requests == set()


@pytest.mark.parametrize("mode", ["single_group", "hybrid", "tp_mismatch", "overlapping"])
def test_recving_thread_exceptions_and_overlapping_req_ids(mode, env):
    gate, started, calls = threading.Event(), threading.Event(), 0

    def get_fn(keys, *_):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            if mode == "overlapping":
                assert gate.wait(timeout=3.0)
            raise TimeoutError("backend read failed")
        return [0] * len(keys)

    invalid: set[int] = set()
    hybrid = mode in ("hybrid", "overlapping")
    w_tp = SimpleNamespace(tp_mismatch=True, _load_kv_tp_mismatch=lambda *_: get_fn([], [], []))
    w, bs = w_tp if mode == "tp_mismatch" else None, [16, 16] if hybrid else 16
    thread = env.start_recv(_DummyBackend(get_fn=get_fn), w, bs, invalid)
    if mode == "overlapping":
        thread.set_finished_request("r-re")
        for _ in range(2):
            thread.add_request(_meta("r-re", [[5], [6]], [0]))
            assert started.wait(timeout=2.0)
        assert (thread.finished_requests, thread.failed_requests) == (set(), set())
        gate.set()
        _wait(lambda: "r-re" in thread.finished_requests)
        assert thread.get_and_clear_receive_results({"r-re"}, set()) == ({"r-re"}, set())
        thread.stop()
        thread.join(timeout=2.0)
        thread.add_request(_meta("r-stop", [[5], [6]], [0]))
        assert thread.get_and_clear_receive_results({"r-stop"}, set()) == ({"r-stop"}, {"r-stop"})
        return
    thread.add_request(_meta("r-err", [[5], [6]] if hybrid else [[1, 2]], [0] if hybrid else None))
    _wait(lambda: "r-err" in thread.finished_requests)
    assert thread.is_alive() and thread._fatal_error is None
    assert thread.get_and_clear_receive_results({"r-err"}, set()) == ({"r-err"}, {"r-err"} if hybrid else set())
    assert invalid == (set() if hybrid else {1, 2})
    thread.add_request(_meta("r-ok", [[7], [8]] if hybrid else [[3]], [0] if hybrid else None))
    _wait(lambda: "r-ok" in thread.finished_requests)
    assert thread.get_and_clear_receive_results({"r-ok"}, set()) == ({"r-ok"}, set())
