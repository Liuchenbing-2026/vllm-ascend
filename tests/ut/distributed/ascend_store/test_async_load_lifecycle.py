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
from typing import Any, cast
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend_map, get_backend_class
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def _make_vllm_config(kv_role="kv_both", extra_config=None, block_size=16, world_size=1, tp_size=1):
    config = MagicMock()
    config.additional_config = {}
    config.kv_events_config = None
    config.speculative_config = None
    config.use_v2_model_runner = False
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_connector_extra_config = {
        "backend": "mooncake",
        "load_async": True,
        "use_layerwise": False,
        **(extra_config or {}),
    }
    config.kv_transfer_config.get_from_extra_config.return_value = True
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_index = 0
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = world_size
    config.cache_config.block_size = block_size
    config.cache_config.hash_block_size = block_size
    config.cache_config.prefix_match_unit = block_size
    config.cache_config.prefix_cache_retention_interval = None
    config.cache_config.use_kda_recoverssm = False
    config.model_config.model = "org/llama-7b"
    config.model_config.max_model_len = 1024
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = max(1, tp_size)
    config.model_config.get_num_layers.return_value = 2
    return config


def _make_scheduler_output(
    *,
    finished_req_ids=None,
    preempted_req_ids=None,
    scheduled_new_reqs=None,
    cached_req_ids=None,
):
    return SimpleNamespace(
        finished_req_ids=set(finished_req_ids or ()),
        preempted_req_ids=set(preempted_req_ids or ()),
        scheduled_new_reqs=list(scheduled_new_reqs or ()),
        scheduled_cached_reqs=SimpleNamespace(req_ids=list(cached_req_ids or ())),
        num_scheduled_tokens={},
    )


def _make_request(req_id: str, num_tokens: int = 48, block_size: int = 16):
    num_blocks = (num_tokens + block_size - 1) // block_size
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        all_token_ids=list(range(num_tokens)),
        num_tokens=num_tokens,
        num_computed_tokens=0,
        block_hashes=[f"{req_id}-h{i}".encode() for i in range(num_blocks)],
    )


class _DummyTokenDatabase(ChunkedTokenDatabase):
    def __init__(self, block_size: int = 16):
        super().__init__([KeyMetadata("model", 0, 0, 0)], [block_size], None)
        self.set_group_buffers({0: [4096]}, {0: [block_size]}, {0: [block_size]}, group_num_layers={0: 1})


class _DummyBackend(Backend):
    def __init__(self, get_fn=None):
        self.get_calls = []
        self.remove_all_calls = 0
        self._get_fn = get_fn

    def set_device(self):
        return None

    def register_buffer(self, ptrs, lengths):
        return None

    def exists(self, keys):
        return [1] * len(keys)

    def put(self, keys, addrs, sizes):
        return [True] * len(keys)

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))
        if self._get_fn is not None:
            return self._get_fn(keys, addrs, sizes)
        return [0] * len(keys)

    def remove_all(self):
        self.remove_all_calls += 1


class _DummyMemcacheBackend(MemcacheBackend):
    def __init__(self, get_fn=None):
        self._get_fn = get_fn
        self.get_calls = []

    def set_device(self):
        return None

    def register_buffer(self, ptrs, lengths):
        return None

    def exists(self, keys):
        return [1] * len(keys)

    def put(self, keys, addrs, sizes):
        return [True] * len(keys)

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))
        if self._get_fn is not None:
            return self._get_fn(keys, addrs, sizes)
        return [0] * len(keys)


def _start_patch(test: unittest.TestCase, target: str, **kwargs):
    patcher = patch(target, **kwargs)
    mocked = patcher.start()
    test.addCleanup(patcher.stop)
    return mocked


def _patch_scheduler_deps(test: unittest.TestCase, hit_tokens: int = 32):
    for backend_name in backend_map:
        backend_class = get_backend_class(backend_name)
        assert backend_class is not None
        patcher = patch.object(backend_class, "create_scheduler_client")
        patcher.start()
        test.addCleanup(patcher.stop)
    client_cls = _start_patch(
        test,
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient",
    )
    client_cls.return_value.closed = False
    client_cls.return_value.lookup.return_value = hit_tokens
    client_cls.return_value.reset.return_value = True
    return client_cls


def _make_worker(test: unittest.TestCase, *, tp_rank: int = 0, tp_size: int = 1, use_layerwise: bool = False):
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
    _start_patch(test, f"{module}.get_tensor_model_parallel_rank", return_value=tp_rank)
    _start_patch(test, f"{module}.get_tensor_model_parallel_world_size", return_value=tp_size)
    _start_patch(test, f"{module}.get_pp_group", return_value=SimpleNamespace(rank_in_group=0))
    pcp_group = _start_patch(test, f"{module}.get_pcp_group")
    pcp_group.return_value.world_size = 1
    pcp_group.return_value.rank_in_group = 0
    _start_patch(test, f"{module}.get_decode_context_model_parallel_world_size", return_value=1)
    _start_patch(test, f"{module}.get_decode_context_model_parallel_rank", return_value=0)
    importlib_mock = _start_patch(test, f"{module}.importlib")
    importlib_mock.import_module.return_value = MagicMock()

    config = _make_vllm_config(
        kv_role="kv_consumer",
        extra_config={"load_async": True, "use_layerwise": use_layerwise},
        world_size=tp_size,
        tp_size=tp_size,
    )
    if use_layerwise:
        return LayerwiseKVPoolWorker(config, use_layerwise=True)
    return KVPoolWorker(config, use_layerwise=False)


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for async condition")
        time.sleep(0.005)


class TestAscendStoreAsyncLoadLifecycle(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.client_cls = _patch_scheduler_deps(self, hit_tokens=32)

    def _start_recv_thread(self, backend: Backend, worker: KVPoolWorker | None = None) -> KVCacheStoreRecvingThread:
        ready = threading.Event()
        thread = KVCacheStoreRecvingThread(
            m_store=backend,
            token_database=_DummyTokenDatabase(),
            block_size=16,
            tp_rank=worker.tp_rank if worker is not None else 0,
            tp_size=worker.tp_size if worker is not None else 1,
            dcp_size=1,
            ready_event=ready,
            invalid_block_ids=worker._invalid_block_ids if worker is not None else set(),
            invalid_block_ids_lock=worker._invalid_block_ids_lock if worker is not None else threading.Lock(),
        )
        thread.start()
        self.assertTrue(ready.wait(timeout=2.0))

        def _cleanup():
            thread.stop()
            thread.join(timeout=2.0)

        self.addCleanup(_cleanup)
        return thread

    def test_normal_async_load_lifecycle_clears_all_state(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        worker = _make_worker(self)
        backend = _DummyBackend()
        worker.m_store = backend
        worker.token_database = _DummyTokenDatabase()
        worker.kv_recv_thread = self._start_recv_thread(backend, worker)

        req = _make_request("req-normal", num_tokens=48)
        need_alloc, load_async = scheduler.get_num_new_matched_tokens(req, 0)
        self.assertEqual((need_alloc, load_async), (32, True))
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 32)

        meta = scheduler.build_connector_meta(_make_scheduler_output())
        self.assertEqual(meta.loading_req_ids, {"req-normal"})
        self.assertEqual(meta.cancelled_loading_req_ids, set())
        self.assertEqual([m.req_id for m in meta.requests], ["req-normal"])

        worker.start_load_kv(meta)
        _wait_until(lambda: "req-normal" in worker.kv_recv_thread.finished_requests)

        done_s, done_r = worker.get_finished(set(), meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, {"req-normal"})
        scheduler.update_connector_output(KVConnectorOutput(finished_recving=done_r))

        self.assertEqual(scheduler._async_loading_req_ids, set())
        self.assertEqual(scheduler.load_specs, {})
        self.assertEqual(worker.kv_recv_thread.finished_requests, set())
        follow_up_meta = scheduler.build_connector_meta(_make_scheduler_output())
        self.assertEqual(worker.get_finished(set(), follow_up_meta), (set(), set()))

    def test_in_flight_abort_after_dispatch_retains_until_completion(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        worker = _make_worker(self)
        gate = threading.Event()
        started = threading.Event()

        def slow_get(keys, addrs, sizes):
            started.set()
            gate.wait(timeout=3.0)
            return [0] * len(keys)

        backend = _DummyBackend(get_fn=slow_get)
        worker.m_store = backend
        worker.token_database = _DummyTokenDatabase()
        worker.kv_recv_thread = self._start_recv_thread(backend, worker)

        req = _make_request("req-inflight", num_tokens=48)
        scheduler.get_num_new_matched_tokens(req, 0)
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 32)

        step1_meta = scheduler.build_connector_meta(_make_scheduler_output())
        self.assertEqual(step1_meta.loading_req_ids, {"req-inflight"})
        worker.start_load_kv(step1_meta)
        self.assertTrue(started.wait(timeout=2.0))

        # Request is aborted after dispatch while backend get() is still in flight.
        step2_meta = scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids={"req-inflight"}))
        self.assertEqual(step2_meta.loading_req_ids, {"req-inflight"})
        self.assertEqual(step2_meta.cancelled_loading_req_ids, set())
        self.assertEqual(step2_meta.requests, [])
        self.assertEqual(worker.get_finished({"req-inflight"}, step2_meta), (set(), set()))

        # Repeated poll before DMA finishes keeps registration without premature release.
        step3_meta = scheduler.build_connector_meta(_make_scheduler_output())
        self.assertEqual(step3_meta.loading_req_ids, {"req-inflight"})
        self.assertEqual(step3_meta.cancelled_loading_req_ids, set())
        self.assertEqual(worker.get_finished(set(), step3_meta), (set(), set()))

        gate.set()
        _wait_until(lambda: "req-inflight" in worker.kv_recv_thread.finished_requests)

        done_s, done_r = worker.get_finished(set(), step3_meta)
        self.assertEqual((done_s, done_r), (set(), {"req-inflight"}))
        scheduler.update_connector_output(KVConnectorOutput(finished_recving=done_r))
        self.assertEqual(scheduler._async_loading_req_ids, set())

        step4_meta = scheduler.build_connector_meta(_make_scheduler_output())
        self.assertEqual(step4_meta.loading_req_ids, set())
        self.assertEqual(worker.get_finished(set(), step4_meta), (set(), set()))

    def test_deferred_load_drained_then_aborted_before_dispatch(self):
        scheduler = KVPoolScheduler(_make_vllm_config(tp_size=2, world_size=2), use_layerwise=False)
        workers = [_make_worker(self, tp_rank=r, tp_size=2) for r in range(2)]
        for worker in workers:
            backend = _DummyBackend()
            worker.m_store = backend
            worker.kv_recv_thread = self._start_recv_thread(backend, worker)

        req = _make_request("req-deferred", num_tokens=48)
        scheduler.get_num_new_matched_tokens(req, 0)
        # Step 1 finishes without update_state_after_alloc (deferred_loads path).
        step1_meta = scheduler.build_connector_meta(_make_scheduler_output())
        self.assertEqual(step1_meta.loading_req_ids, set())
        self.assertEqual(step1_meta.requests, [])

        # _drain_deferred_loads() calls update_state_after_alloc at step 1 output processing.
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 32)
        self.assertIn("req-deferred", scheduler._async_loading_req_ids)
        self.assertTrue(scheduler.load_specs["req-deferred"].can_load)

        # Request is aborted before step 2 build_connector_meta dispatches it.
        step2_meta = scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids={"req-deferred"}))
        self.assertEqual(step2_meta.requests, [])
        self.assertEqual(step2_meta.loading_req_ids, {"req-deferred"})
        self.assertEqual(step2_meta.cancelled_loading_req_ids, {"req-deferred"})
        self.assertNotIn("req-deferred", scheduler.load_specs)
        self.assertNotIn("req-deferred", scheduler._unfinished_requests)

        # All TP ranks must acknowledge without executing a backend get().
        per_rank_recving = []
        for worker in workers:
            worker.start_load_kv(step2_meta)
            self.assertEqual(worker.m_store.get_calls, [])
            _, done_r = worker.get_finished({"req-deferred"}, step2_meta)
            per_rank_recving.append(done_r)
        aggregated_recving = set.intersection(*per_rank_recving)
        self.assertEqual(aggregated_recving, {"req-deferred"})

        # Even before update_connector_output arrives, a follow-up step must not repeat cancelled completion.
        step3_meta = scheduler.build_connector_meta(_make_scheduler_output())
        self.assertEqual(step3_meta.cancelled_loading_req_ids, set())
        for worker in workers:
            self.assertEqual(worker.get_finished(set(), step3_meta), (set(), set()))

        scheduler.update_connector_output(KVConnectorOutput(finished_recving=aggregated_recving))
        self.assertEqual(scheduler._async_loading_req_ids, set())

    def test_abort_before_update_state_after_alloc_cleans_load_spec_without_spurious_completion(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        worker = _make_worker(self)
        backend = _DummyBackend()
        worker.m_store = backend
        worker.kv_recv_thread = self._start_recv_thread(backend, worker)

        req = _make_request("req-pre-alloc", num_tokens=48)
        scheduler.get_num_new_matched_tokens(req, 0)
        self.assertIn("req-pre-alloc", scheduler.load_specs)
        self.assertFalse(scheduler.load_specs["req-pre-alloc"].can_load)

        # Aborted before update_state_after_alloc (still in deferred_loads or alloc failed).
        meta = scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids={"req-pre-alloc"}))
        self.assertEqual(scheduler.load_specs, {})
        self.assertEqual(meta.loading_req_ids, set())
        self.assertEqual(meta.cancelled_loading_req_ids, set())
        self.assertEqual(worker.get_finished({"req-pre-alloc"}, meta), (set(), set()))

    def test_repeated_abort_signal_does_not_emit_duplicate_completion(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        req = _make_request("req-repeat", num_tokens=48)
        scheduler.get_num_new_matched_tokens(req, 0)
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 32)

        first_meta = scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids={"req-repeat"}))
        self.assertEqual(first_meta.cancelled_loading_req_ids, {"req-repeat"})

        second_meta = scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids={"req-repeat"}))
        self.assertEqual(second_meta.cancelled_loading_req_ids, set())
        self.assertEqual(second_meta.loading_req_ids, {"req-repeat"})

        scheduler.update_connector_output(KVConnectorOutput(finished_recving={"req-repeat"}))
        self.assertEqual(scheduler._async_loading_req_ids, set())

    def test_same_step_preemption_and_async_readmit_preserves_load_state(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        worker = _make_worker(self)
        recv_thread = MagicMock()
        recv_thread.get_and_clear_finished_requests.return_value = {"r-same-step"}
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = None

        req = _make_request("r-same-step", num_tokens=48)
        scheduler.get_num_new_matched_tokens(req, 0)
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 32)

        meta = scheduler.build_connector_meta(_make_scheduler_output(preempted_req_ids={"r-same-step"}))
        self.assertEqual(meta.loading_req_ids, {"r-same-step"})
        self.assertEqual([m.req_id for m in meta.requests], ["r-same-step"])
        self.assertIsNotNone(meta.requests[0].load_spec)
        self.assertTrue(meta.requests[0].load_spec.can_load)

        _, done_r = worker.get_finished(set(), meta)
        recv_thread.discard_finished_requests.assert_called_once_with(set())
        recv_thread.get_and_clear_finished_requests.assert_called_once_with({"r-same-step"})
        self.assertEqual(done_r, {"r-same-step"})

    def test_same_step_preemption_and_sync_readmit_preserves_unfinished_request(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        req = _make_request("r-sync-readmit", num_tokens=32)
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2]]), 0)

        new_req = SimpleNamespace(
            req_id="r-sync-readmit",
            prefill_token_ids=list(range(32)),
            prompt_token_ids=list(range(32)),
            num_computed_tokens=0,
            block_ids=[[1, 2]],
        )
        sched_out = _make_scheduler_output(
            preempted_req_ids={"r-sync-readmit"},
            scheduled_new_reqs=[new_req],
        )
        sched_out.num_scheduled_tokens = {"r-sync-readmit": 32}

        meta = scheduler.build_connector_meta(sched_out)
        self.assertEqual([m.req_id for m in meta.requests], ["r-sync-readmit"])
        self.assertEqual(meta.loading_req_ids, set())

    def test_ordinary_preemption_clears_load_state_and_discards_worker_completion(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        worker = _make_worker(self)
        recv_thread = MagicMock()
        recv_thread.get_and_clear_finished_requests.return_value = set()
        worker.kv_recv_thread = recv_thread
        worker.kv_send_thread = None

        req = _make_request("r-preempt", num_tokens=48)
        scheduler.get_num_new_matched_tokens(req, 0)
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 32)
        scheduler.build_connector_meta(_make_scheduler_output())
        self.assertIn("r-preempt", scheduler._async_loading_req_ids)

        # Ordinary preemption in a later step without new allocation in load_specs.
        meta = scheduler.build_connector_meta(_make_scheduler_output(preempted_req_ids={"r-preempt"}))
        self.assertEqual(meta.loading_req_ids, set())
        self.assertEqual(scheduler.load_specs, {})
        self.assertEqual(scheduler._unfinished_requests, {})
        self.assertEqual(scheduler._async_loading_req_ids, set())

        worker.get_finished(set(), meta)
        recv_thread.discard_finished_requests.assert_called_once_with({"r-preempt"})

    def test_recving_thread_single_group_exception_keeps_thread_alive_and_marks_finished(self):
        call_count = 0

        def flaky_get(keys, addrs, sizes):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("backend read timed out")
            return [0] * len(keys)

        backend = _DummyBackend(get_fn=flaky_get)
        invalid_blocks: set[int] = set()
        thread = KVCacheStoreRecvingThread(
            m_store=backend,
            token_database=_DummyTokenDatabase(),
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=invalid_blocks,
            invalid_block_ids_lock=threading.Lock(),
        )
        thread.start()
        self.assertTrue(thread.ready_event.wait(timeout=2.0))

        def _cleanup_thread() -> None:
            thread.stop()
            thread.join(timeout=2.0)

        self.addCleanup(_cleanup_thread)

        req1 = ReqMeta(
            req_id="r-timeout",
            token_len_chunk=32,
            block_ids_by_group=[[1, 2]],
            block_hashes=[b"h0", b"h1"],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32),
        )
        req2 = ReqMeta(
            req_id="r-ok",
            token_len_chunk=16,
            block_ids_by_group=[[3]],
            block_hashes=[b"h2"],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16),
        )
        thread.add_request(req1)
        _wait_until(lambda: "r-timeout" in thread.finished_requests)
        self.assertIsNone(thread._fatal_error)
        self.assertTrue(thread.is_alive())
        self.assertEqual(invalid_blocks, {1, 2})
        self.assertEqual(thread.get_and_clear_finished_requests({"r-timeout"}), {"r-timeout"})

        thread.add_request(req2)
        _wait_until(lambda: "r-ok" in thread.finished_requests)
        self.assertEqual(invalid_blocks, {1, 2})
        self.assertEqual(thread.get_and_clear_finished_requests({"r-ok"}), {"r-ok"})

    def test_recving_thread_hybrid_exception_records_failed_request_and_marks_finished(self):
        def failing_get(keys, addrs, sizes):
            raise RuntimeError("rdma failure")

        backend = _DummyBackend(get_fn=failing_get)
        invalid_blocks: set[int] = set()
        thread = KVCacheStoreRecvingThread(
            m_store=backend,
            token_database=_DummyTokenDatabase(),
            block_size=[16, 16],
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=invalid_blocks,
            invalid_block_ids_lock=threading.Lock(),
        )
        req = ReqMeta(
            req_id="r-hybrid-err",
            token_len_chunk=16,
            block_ids_by_group=[[5], [6]],
            block_hashes=[b"h0"],
            kv_cache_group_ids=[0],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16),
        )
        thread.add_request(req)
        dequeued = thread.request_queue.get()
        thread._handle_request(dequeued)

        self.assertEqual(invalid_blocks, set())
        self.assertEqual(thread.get_and_clear_failed_requests(), {"r-hybrid-err"})
        self.assertEqual(thread.get_and_clear_finished_requests({"r-hybrid-err"}), {"r-hybrid-err"})
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_recving_thread_memcache_exception_marks_failed_and_save_disabled(self):
        def failing_get(keys, addrs, sizes):
            raise RuntimeError("memcache error")

        backend = _DummyMemcacheBackend(get_fn=failing_get)
        invalid_blocks: set[int] = set()
        thread = KVCacheStoreRecvingThread(
            m_store=backend,
            token_database=_DummyTokenDatabase(),
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=invalid_blocks,
            invalid_block_ids_lock=threading.Lock(),
        )
        req = ReqMeta(
            req_id="r-memcache-err",
            token_len_chunk=16,
            block_ids_by_group=[[9]],
            block_hashes=[b"h0"],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16),
        )
        thread.add_request(req)
        dequeued = thread.request_queue.get()
        thread._handle_request(dequeued)

        self.assertTrue(thread.save_disabled.is_set())
        self.assertEqual(invalid_blocks, {9})
        self.assertEqual(thread.get_and_clear_finished_requests({"r-memcache-err"}), {"r-memcache-err"})

    def test_recving_thread_handle_request_exception_drains_current_and_queued_loads(self):
        backend = _DummyBackend()
        invalid_blocks: set[int] = set()
        thread = KVCacheStoreRecvingThread(
            m_store=backend,
            token_database=_DummyTokenDatabase(),
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=invalid_blocks,
            invalid_block_ids_lock=threading.Lock(),
        )
        req_current = ReqMeta(req_id="r-curr", block_ids_by_group=[[0, 11]], block_hashes=[])
        req_queued = ReqMeta(req_id="r-queued", block_ids_by_group=[[12], [13]], block_hashes=[])
        thread.add_request(req_current)
        thread.add_request(req_queued)

        current_item = thread.request_queue.get()
        thread._handle_request_exception(current_item)

        self.assertEqual(thread.request_queue.unfinished_tasks, 0)
        self.assertTrue(thread.request_queue.empty())
        self.assertEqual(invalid_blocks, {11})
        self.assertEqual(thread.get_and_clear_failed_requests(), {"r-queued"})
        self.assertEqual(thread.get_and_clear_finished_requests(), {"r-curr", "r-queued"})

    def test_recving_thread_add_request_after_fatal_or_stop_marks_failed_and_finished(self):
        for mode in ("fatal", "stopping"):
            with self.subTest(mode=mode):
                invalid_blocks: set[int] = set()
                thread = KVCacheStoreRecvingThread(
                    m_store=_DummyBackend(),
                    token_database=_DummyTokenDatabase(),
                    block_size=16,
                    tp_rank=0,
                    dcp_size=1,
                    ready_event=threading.Event(),
                    invalid_block_ids=invalid_blocks,
                    invalid_block_ids_lock=threading.Lock(),
                )
                if mode == "fatal":
                    thread._fatal_error = RuntimeError("fatal")
                else:
                    thread._stopping.set()
                req = ReqMeta(req_id=f"r-{mode}", block_ids_by_group=[[0, 14]], block_hashes=[])
                thread.add_request(req)
                self.assertTrue(thread.request_queue.empty())
                self.assertEqual(thread.request_queue.unfinished_tasks, 0)
                self.assertEqual(invalid_blocks, {14})
                self.assertEqual(thread.get_and_clear_finished_requests(), {f"r-{mode}"})

    def test_reused_req_id_discards_stale_completion_and_counts_overlapping_loads(self):
        thread = KVCacheStoreRecvingThread(
            m_store=_DummyBackend(),
            token_database=_DummyTokenDatabase(),
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        # Leftover completion from an earlier lifecycle must not finish a reused req_id early.
        thread.set_finished_request("r-reuse")
        req1 = ReqMeta(
            req_id="r-reuse",
            token_len_chunk=16,
            block_ids_by_group=[[1]],
            block_hashes=[b"h0"],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16),
        )
        req2 = ReqMeta(
            req_id="r-reuse",
            token_len_chunk=16,
            block_ids_by_group=[[2]],
            block_hashes=[b"h1"],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=16, can_load=True, token_len=16),
        )
        thread.add_request(req1)
        self.assertEqual(thread.finished_requests, set())
        thread.add_request(req2)

        first = thread.request_queue.get()
        thread._handle_request(first)
        self.assertEqual(thread.get_and_clear_finished_requests({"r-reuse"}), set())

        second = thread.request_queue.get()
        thread._handle_request(second)
        self.assertEqual(thread.get_and_clear_finished_requests({"r-reuse"}), {"r-reuse"})

    def test_scheduler_reset_store_blocks_while_aborted_async_load_holds_blocks_then_succeeds(self):
        scheduler = KVPoolScheduler(_make_vllm_config(), use_layerwise=False)
        pool = MagicMock()
        pool.num_gpu_blocks = 16
        free_blocks = 12
        pool.get_num_free_blocks.side_effect = lambda: free_blocks
        scheduler.bind_gpu_block_pool(pool)

        req = _make_request("req-reset", num_tokens=48)
        scheduler.get_num_new_matched_tokens(req, 0)
        scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 32)
        scheduler.build_connector_meta(_make_scheduler_output())
        scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids={"req-reset"}))

        assert scheduler.client is not None
        reset_mock = cast(MagicMock, scheduler.client.reset)
        self.assertFalse(scheduler.reset_store())
        reset_mock.assert_not_called()

        scheduler._block_pool = None
        self.assertFalse(scheduler.reset_store())
        scheduler._block_pool = pool

        scheduler.update_connector_output(KVConnectorOutput(finished_recving={"req-reset"}))
        free_blocks = 15  # Only null block remains used (used == 1).
        self.assertTrue(scheduler.reset_store())
        reset_mock.assert_called_once_with()

    def test_worker_reset_store_drains_recv_thread_and_clears_stale_completions(self):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        order: list[str] = []

        send_queue = MagicMock()
        send_queue.join.side_effect = lambda: order.append("send_join")
        worker.kv_send_thread = cast(Any, SimpleNamespace(request_queue=send_queue))

        recv_queue = MagicMock()
        recv_queue.join.side_effect = lambda: order.append("recv_join")
        recv_thread = MagicMock(_fatal_error=None, request_queue=recv_queue)

        def _recv_clear() -> set[str]:
            order.append("recv_clear")
            return {"stale"}

        recv_thread.get_and_clear_finished_requests.side_effect = _recv_clear
        worker.kv_recv_thread = recv_thread

        worker.m_store = MagicMock()
        worker.m_store.remove_all.side_effect = lambda: order.append("remove_all")

        worker.reset_store()
        self.assertEqual(order, ["send_join", "recv_join", "recv_clear", "remove_all"])

        # With a fatal error on recv_thread, skip queue.join() to avoid deadlock.
        order.clear()
        recv_thread._fatal_error = RuntimeError("dead")
        worker.reset_store()
        self.assertEqual(order, ["send_join", "recv_clear", "remove_all"])

    def test_layerwise_mode_disables_async_load_and_keeps_done_recving_empty(self):
        lw_scheduler = LayerwiseKVPoolScheduler(
            _make_vllm_config(extra_config={"backend": "memcache", "load_async": True, "use_layerwise": True}),
            use_layerwise=True,
        )
        self.assertFalse(lw_scheduler.load_async)
        lw_scheduler.load_specs["r-lw"] = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=32,
            can_load=True,
            kvpool_store_skip_tokens=32,
        )
        meta = lw_scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids={"r-lw"}))
        self.assertEqual(lw_scheduler.load_specs, {})
        self.assertEqual(meta.loading_req_ids, set())
        self.assertEqual(meta.cancelled_loading_req_ids, set())

        lw_scheduler.load_specs["r-lw-p"] = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=32,
            can_load=True,
            kvpool_store_skip_tokens=32,
        )
        preempt_meta = lw_scheduler.build_connector_meta(_make_scheduler_output(preempted_req_ids={"r-lw-p"}))
        self.assertEqual(lw_scheduler.load_specs, {})
        self.assertEqual(preempt_meta.loading_req_ids, set())
        self.assertEqual(preempt_meta.cancelled_loading_req_ids, set())

        lw_worker = _make_worker(self, use_layerwise=True)
        lw_worker.kv_recv_thread = MagicMock()
        forged_meta = AscendConnectorMetadata(set(), loading_req_ids={"r-lw"}, cancelled_loading_req_ids={"r-lw"})
        done_s, done_r = lw_worker.get_finished(set(), forged_meta)
        self.assertEqual(done_s, set())
        self.assertEqual(done_r, set())
        lw_worker.kv_recv_thread.get_and_clear_finished_requests.assert_not_called()

    def test_concurrent_mixed_async_load_lifecycle_no_block_leak_or_admission_stall(self):
        # 12 GPU blocks total: block 0 is null block, blocks 1..11 are allocatable.
        free_block_pool = set(range(1, 12))
        req_blocks: dict[str, list[int]] = {}
        double_free_violations: list[str] = []
        scheduler_requests: dict[str, dict] = {}
        aborted_kv_loads: set[str] = set()
        deferred_loads: dict[str, tuple[object, list[int], int]] = {}
        finished_req_ids_queue: set[str] = set()
        recv_remaining_count: dict[str, int] = {}
        duplicate_completions: list[str] = []
        completed_once: set[str] = set()

        scheduler = KVPoolScheduler(_make_vllm_config(tp_size=2, world_size=2), use_layerwise=False)
        block_pool_stub = SimpleNamespace(
            num_gpu_blocks=12,
            get_num_free_blocks=lambda: len(free_block_pool),
        )
        scheduler.bind_gpu_block_pool(block_pool_stub)

        gate = threading.Event()
        gated_tokens = tuple(f"{gid}-h0".encode().hex() for gid in ("r2", "r3"))
        r7_token = b"r7-h0".hex()

        def backend_get_rank0(keys, addrs, sizes):
            key_text = " ".join(str(k) for k in keys)
            if any(tok in key_text for tok in gated_tokens):
                gate.wait(timeout=3.0)
            if r7_token in key_text:
                raise TimeoutError("simulated transient backend timeout on r7")
            return [0] * len(keys)

        def backend_get_rank1(keys, addrs, sizes):
            key_text = " ".join(str(k) for k in keys)
            if any(tok in key_text for tok in gated_tokens):
                gate.wait(timeout=3.0)
            return [0] * len(keys)

        workers = [_make_worker(self, tp_rank=r, tp_size=2) for r in range(2)]
        for idx, worker in enumerate(workers):
            backend = _DummyBackend(get_fn=backend_get_rank0 if idx == 0 else backend_get_rank1)
            worker.m_store = backend
            worker.token_database = _DummyTokenDatabase()
            worker.kv_recv_thread = self._start_recv_thread(backend, worker)

        def try_admit(req_id: str, num_tokens: int = 48, defer_alloc_update: bool = False) -> bool:
            req = _make_request(req_id, num_tokens=num_tokens)
            need_alloc, is_async = scheduler.get_num_new_matched_tokens(req, 0)
            self.assertEqual((need_alloc, is_async), (32, True))
            needed = (need_alloc + 15) // 16
            if len(free_block_pool) < needed:
                return False
            allocated = [free_block_pool.pop() for _ in range(needed)]
            req_blocks[req_id] = allocated
            scheduler_requests[req_id] = {
                "req": req,
                "status": "WAITING_FOR_REMOTE_KVS",
                "delay_free": False,
            }
            if defer_alloc_update:
                deferred_loads[req_id] = (req, allocated, need_alloc)
            else:
                scheduler.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda ids=allocated: [ids]), 32)
            return True

        def abort_request(req_id: str) -> None:
            entry = scheduler_requests[req_id]
            if deferred_loads.pop(req_id, None) is not None:
                # Never handed to connector via update_state_after_alloc: free blocks immediately.
                for bid in req_blocks.pop(req_id):
                    if bid in free_block_pool:
                        double_free_violations.append(req_id)
                    free_block_pool.add(bid)
                scheduler_requests.pop(req_id)
                finished_req_ids_queue.add(req_id)
                return
            entry["status"] = "FINISHED_ABORTED"
            entry["delay_free"] = True
            aborted_kv_loads.add(req_id)
            finished_req_ids_queue.add(req_id)

        def run_step():
            nonlocal finished_req_ids_queue
            step_finished = finished_req_ids_queue
            finished_req_ids_queue = set()
            meta = scheduler.build_connector_meta(_make_scheduler_output(finished_req_ids=step_finished))
            for worker in workers:
                worker.start_load_kv(meta)
            # Drain deferred loads right before worker outputs are collected (matches Scheduler.update_from_output).
            for d_req_id, (d_req, d_blocks, d_tokens) in list(deferred_loads.items()):
                del deferred_loads[d_req_id]
                scheduler.update_state_after_alloc(
                    d_req,
                    SimpleNamespace(get_block_ids=lambda ids=d_blocks: [ids]),
                    d_tokens,
                )
            agg_recving: set[str] = set()
            for worker in workers:
                _, done_r = worker.get_finished(step_finished, meta)
                for done_id in done_r:
                    remaining = recv_remaining_count.get(done_id, len(workers)) - 1
                    if remaining == 0:
                        recv_remaining_count.pop(done_id, None)
                        agg_recving.add(done_id)
                    else:
                        recv_remaining_count[done_id] = remaining
            scheduler.update_connector_output(KVConnectorOutput(finished_recving=agg_recving or None))
            for done_id in agg_recving:
                if done_id in completed_once:
                    duplicate_completions.append(done_id)
                completed_once.add(done_id)
                entry = scheduler_requests.get(done_id)
                self.assertIsNotNone(entry, f"Completed unknown req {done_id}")
                assert entry is not None, f"Completed unknown req {done_id}"
                if entry["status"] == "WAITING_FOR_REMOTE_KVS":
                    finished_req_ids_queue.add(done_id)
                aborted_kv_loads.discard(done_id)
                for bid in req_blocks.pop(done_id):
                    if bid in free_block_pool:
                        double_free_violations.append(done_id)
                    free_block_pool.add(bid)
                scheduler_requests.pop(done_id)
            return agg_recving

        # Round 1: admit r0, r1 (normal), r7 (timeout exception) ahead of gated r2, r3.
        for rid in ("r0", "r1", "r7", "r2", "r3"):
            self.assertTrue(try_admit(rid))
        run_step()

        # Wait for r0, r1, r7 to settle while r2, r3 are held on gate.
        _wait_until(lambda: all(r not in scheduler_requests for r in ("r0", "r1", "r7")) or (run_step() and False))
        # Abort r2 and r3 while their async loads are in flight.
        abort_request("r2")
        abort_request("r3")
        run_step()
        # Blocks of r2 and r3 must still be retained until backend finishes.
        self.assertEqual(aborted_kv_loads, {"r2", "r3"})
        self.assertEqual(len(free_block_pool), 7)  # 11 - 4 blocks held by r2, r3

        # Admit r4, r5 with deferred loads; drain then abort before next build_connector_meta.
        self.assertTrue(try_admit("r4", defer_alloc_update=True))
        self.assertTrue(try_admit("r5", defer_alloc_update=True))
        # Also admit r6 with deferred load and abort BEFORE drain.
        self.assertTrue(try_admit("r6", defer_alloc_update=True))
        abort_request("r6")
        run_step()  # Drains r4, r5 into update_state_after_alloc; r6 cleaned immediately.
        self.assertNotIn("r6", scheduler.load_specs)

        # Abort r4 and r5 right after deferred drain and before their dispatch step.
        abort_request("r4")
        abort_request("r5")
        run_step()
        self.assertNotIn("r4", scheduler_requests)
        self.assertNotIn("r5", scheduler_requests)

        # Now release r2, r3 backend transfers; they must finish and release their 4 blocks.
        gate.set()
        _wait_until(lambda: (run_step() or True) and not scheduler_requests)
        self.assertEqual(len(free_block_pool), 11)

        # Reuse ID "r0" and admit 5 concurrent requests (10 of 11 blocks); none should stall.
        if finished_req_ids_queue:
            run_step()
        completed_once.discard("r0")
        for rid in ("r0", "r8", "r9", "r10", "r11"):
            self.assertTrue(try_admit(rid))
        _wait_until(lambda: (run_step() or True) and not scheduler_requests)
        if finished_req_ids_queue:
            run_step()

        self.assertEqual(len(free_block_pool), 11)
        self.assertEqual(aborted_kv_loads, set())
        self.assertEqual(recv_remaining_count, {})
        self.assertEqual(double_free_violations, [])
        self.assertEqual(duplicate_completions, [])
        self.assertEqual(scheduler._async_loading_req_ids, set())
        self.assertEqual(scheduler.load_specs, {})
        self.assertEqual(scheduler._unfinished_requests, {})
        self.assertEqual(scheduler._request_trackers, {})
        for worker in workers:
            self.assertEqual(worker.kv_recv_thread.finished_requests, set())
        self.assertTrue(scheduler.reset_store())


if __name__ == "__main__":
    unittest.main()
