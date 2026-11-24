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
#

"""Unit tests for Mooncake pull fast-fail on lost prefill peers."""

from __future__ import annotations

import queue
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import msgspec
import pytest
import zmq
from cachetools import TTLCache
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_connector as v1_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_hybrid_connector as hybrid_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import (
    pull_scheduler,
    pull_worker,
    utils,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadataGroups,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    DEFAULT_PULL_ZMQ_TIMEOUT_MS,
    PULL_HEARTBEAT_IVL_MS,
    PULL_HEARTBEAT_TIMEOUT_MS,
    PULL_LOST_PEER_EVENTS,
    MooncakePeerLostError,
    SizedDict,
    close_zmq_socket,
    ensure_zmq_recv,
    lost_peer_monitor,
    make_mooncake_client_socket,
    recv_unless_lost,
    zmq_ctx,
)

from .helpers import make_metadata_groups

_HAS_REAL_ZMQ = bool(getattr(zmq, "__file__", None)) and callable(
    getattr(getattr(zmq, "Socket", None), "get_monitor_socket", None)
)

_SILENT_ROUTER_SCRIPT = """
import sys
import zmq

mode = sys.argv[1]
ctx = zmq.Context()
router = ctx.socket(zmq.ROUTER)
router.setsockopt(zmq.LINGER, 0)
port = router.bind_to_random_port("tcp://127.0.0.1")
print(port, flush=True)
if mode == "live":
    ident, delim, payload = router.recv_multipart()
    router.send_multipart([ident, delim, b"PONG"])
    router.recv_multipart()
elif mode in ("killed", "frozen"):
    router.recv_multipart()
    print("pulled", flush=True)
    router.recv_multipart()
"""


def _monitor_event_frame(event_id: int) -> list[bytes]:
    return [struct.pack("=HI", event_id, 0)]


def _make_fake_poller_factory(outcomes: list[list[tuple[Any, int]]]) -> type[Any]:
    queue_of_outcomes = list(outcomes)

    class FakePoller:
        def __init__(self) -> None:
            self.registered: list[Any] = []
            self.unregistered: list[Any] = []

        def register(self, sock: Any, _flags: int) -> None:
            self.registered.append(sock)

        def unregister(self, sock: Any) -> None:
            self.unregistered.append(sock)

        def poll(self, _timeout_ms: int | None = None) -> list[tuple[Any, int]]:
            if not queue_of_outcomes:
                return []
            return queue_of_outcomes.pop(0)

    return FakePoller


def _make_req_meta(
    *,
    engine_id: str = "engine-p",
    host: str = "192.0.2.1",
    port: int = 6000,
    request_id: str = "remote-req-1",
    local_blocks: tuple[list[int], ...] = ([10, 11],),
) -> ReqMeta:
    return ReqMeta(
        local_block_ids=local_blocks,
        local_num_prompt_tokens=16,
        num_external_tokens=16,
        num_computed_tokens=0,
        remote_block_ids=([1, 2],),
        remote_host=host,
        remote_port=port,
        remote_engine_id=engine_id,
        remote_request_id=request_id,
        remote_num_prompt_tokens=16,
        local_full_block_ids=local_blocks,
    )


def _make_v2_pull_thread(*, can_report_invalid_block_ids: bool = True) -> pull_worker.MooncakePullRecvingThread:
    thread = pull_worker.MooncakePullRecvingThread.__new__(pull_worker.MooncakePullRecvingThread)
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
    thread.remote_metadata = SizedDict()
    thread.remote_tp_rank_groups = SizedDict()
    thread.remote_layer_index_pairs = SizedDict()
    thread.finished_requests = queue.SimpleQueue()
    thread.invalid_block_ids = set()
    thread.invalid_block_ids_lock = threading.Lock()
    thread.can_report_invalid_block_ids = can_report_invalid_block_ids
    thread.layer_names = ["model.layers.0.self_attn"]
    thread.spec_indices = [0]
    thread.kv_cache_specs = []
    thread.executor = MagicMock()
    return thread


def _make_v2_scheduler(*, policy: str = "fail") -> pull_scheduler.MooncakePullConnectorScheduler:
    sched = pull_scheduler.MooncakePullConnectorScheduler.__new__(pull_scheduler.MooncakePullConnectorScheduler)
    sched.kv_transfer_config = SimpleNamespace(
        is_kv_producer=False,
        is_kv_consumer=True,
        kv_load_failure_policy=policy,
    )
    sched._recving_thread = MagicMock()
    sched._sending_thread = None
    sched._heartbeat_thread = None
    sched._reqs_recv_info = {}
    sched._validated_producer_leases = TTLCache(maxsize=16, ttl=60)
    return sched


def _make_v1_recving_thread(*, use_request_level_recv_errors: bool = False) -> v1_connector.KVCacheRecvingThread:
    thread = v1_connector.KVCacheRecvingThread.__new__(v1_connector.KVCacheRecvingThread)
    thread.local_engine_id = "local-d"
    thread.local_handshake_port = 5555
    thread.side_channel_port = 5555
    thread.tp_rank = 0
    thread.tp_size = 1
    thread.kv_group2layeridx = {}
    thread.remote_metadata_lock = threading.Lock()
    thread.kv_caches_base_addr = SizedDict()
    thread.remote_te_port = SizedDict()
    thread.remote_block_size_scale = SizedDict()
    thread.remote_block_stride_per_addr = SizedDict()
    thread.remote_kv_group2layeridx = SizedDict()
    thread.remote_sockets_lock = threading.Lock()
    thread.remote_sockets = defaultdict(deque)
    thread.timeout = DEFAULT_PULL_ZMQ_TIMEOUT_MS / 1000.0
    thread._zmq_context = None
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = MagicMock()
    thread.request_task_counts = defaultdict(int)
    thread.finished_request_markers = set()
    thread.request_task_counts_lock = threading.Lock()
    thread.proc_not_transfer_request = {}
    thread.proc_not_transfer_request_lock = threading.Lock()
    thread.pending_reformat = defaultdict(dict)
    thread.pending_reformat_lock = threading.Lock()
    thread.request_queue = MagicMock()
    thread.task_tracker = v1_connector.KVCacheTaskTracker()
    thread.use_request_level_recv_errors = use_request_level_recv_errors
    thread.failed_recv_requests = set()
    thread.peer_lost_requests = set()
    thread.invalid_block_ids = set()
    thread.failed_recv_requests_lock = threading.Lock()
    return thread


def _make_hybrid_recving_thread(
    *,
    use_hybrid: bool = False,
    use_request_level_recv_errors: bool = False,
) -> hybrid_connector.KVCacheRecvingThread:
    thread = hybrid_connector.KVCacheRecvingThread.__new__(hybrid_connector.KVCacheRecvingThread)
    thread.local_engine_id = "local-d"
    thread.local_handshake_port = 5555
    thread.side_channel_port = 5555
    thread.tp_rank = 0
    thread.tp_size = 1
    thread.use_hybrid = use_hybrid
    thread.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()] * (2 if use_hybrid else 1))
    thread.remote_metadata_lock = threading.Lock()
    thread.kv_caches_base_addr = SizedDict()
    thread.remote_te_port = SizedDict()
    thread.remote_sockets_lock = threading.Lock()
    thread.remote_sockets = defaultdict(deque)
    thread.timeout = DEFAULT_PULL_ZMQ_TIMEOUT_MS / 1000.0
    thread._zmq_context = None
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = MagicMock()
    thread.request_task_counts = defaultdict(int)
    thread.finished_request_markers = set()
    thread.request_task_counts_lock = threading.Lock()
    thread.proc_not_transfer_request = {}
    thread.proc_not_transfer_request_lock = threading.Lock()
    thread.request_queue = MagicMock()
    thread.task_tracker = hybrid_connector.KVCacheTaskTracker()
    thread.use_request_level_recv_errors = use_request_level_recv_errors
    thread.failed_recv_requests = set()
    thread.peer_lost_requests = set()
    thread.invalid_block_ids = set()
    thread.failed_recv_requests_lock = threading.Lock()
    return thread


# ---------------------------------------------------------------------------
# Group 1: Dual-version make_mooncake_client_socket & zmq_ctx lifecycle
# ---------------------------------------------------------------------------


def test_make_mooncake_client_socket_legacy_signature_sets_pre_connect_options() -> None:
    """Older vLLM pins without heartbeat args get LINGER and heartbeats pre-connect."""
    events: list[tuple[Any, ...]] = []
    raw_socket = MagicMock()
    raw_socket.setsockopt.side_effect = lambda opt, val: events.append(("setsockopt", opt, val))
    raw_socket.connect.side_effect = lambda addr: events.append(("connect", addr))
    ctx = MagicMock()
    ctx.socket.return_value = raw_socket
    seen_calls: list[dict[str, Any]] = []

    def legacy_make_zmq_socket(
        ctx: Any,
        path: str,
        socket_type: Any,
        *,
        bind: bool = False,
        linger: int | None = None,
    ) -> Any:
        sock = ctx.socket(socket_type)
        sock.connect(path)
        seen_calls.append({"bind": bind, "linger": linger, "path": path})
        return sock

    sock = make_mooncake_client_socket(
        ctx,
        "tcp://192.0.2.1:6000",
        zmq.REQ,  # type: ignore[attr-defined]
        linger=0,
        heartbeat_ivl_ms=PULL_HEARTBEAT_IVL_MS,
        heartbeat_timeout_ms=PULL_HEARTBEAT_TIMEOUT_MS,
        make_socket_fn=legacy_make_zmq_socket,
    )

    assert sock is raw_socket
    assert seen_calls == [{"bind": False, "linger": 0, "path": "tcp://192.0.2.1:6000"}]
    connect_idx = next(i for i, item in enumerate(events) if item[0] == "connect")
    pre_connect_opts = {item[1]: item[2] for item in events[:connect_idx] if item[0] == "setsockopt"}
    assert pre_connect_opts[zmq.LINGER] == 0  # type: ignore[attr-defined]
    if hasattr(zmq, "HEARTBEAT_IVL"):
        assert pre_connect_opts[zmq.HEARTBEAT_IVL] == PULL_HEARTBEAT_IVL_MS  # type: ignore[attr-defined]
    if hasattr(zmq, "HEARTBEAT_TIMEOUT"):
        assert pre_connect_opts[zmq.HEARTBEAT_TIMEOUT] == PULL_HEARTBEAT_TIMEOUT_MS  # type: ignore[attr-defined]

    # Also verify a legacy factory without `linger` parameter receives no unexpected `linger` kwarg.
    no_linger_calls: list[dict[str, Any]] = []

    def minimal_make_zmq_socket(ctx: Any, path: str, socket_type: Any, *, bind: bool = False) -> Any:
        no_linger_calls.append({"path": path, "socket_type": socket_type, "bind": bind})
        return ctx.socket(socket_type)

    make_mooncake_client_socket(
        ctx,
        "tcp://192.0.2.1:6001",
        zmq.REQ,  # type: ignore[attr-defined]
        make_socket_fn=minimal_make_zmq_socket,
    )
    assert len(no_linger_calls) == 1


def test_make_mooncake_client_socket_new_signature_passes_heartbeat_kwargs() -> None:
    """Newer make_zmq_socket signatures receive native heartbeat kwargs directly."""
    ctx = MagicMock()
    expected_socket = MagicMock()
    captured: dict[str, Any] = {}

    def modern_make_zmq_socket(
        ctx: Any,
        path: str,
        socket_type: Any,
        *,
        bind: bool = False,
        linger: int | None = None,
        heartbeat_ivl_ms: int | None = None,
        heartbeat_timeout_ms: int | None = None,
    ) -> Any:
        captured.update(
            {
                "ctx": ctx,
                "path": path,
                "socket_type": socket_type,
                "bind": bind,
                "linger": linger,
                "heartbeat_ivl_ms": heartbeat_ivl_ms,
                "heartbeat_timeout_ms": heartbeat_timeout_ms,
            }
        )
        return expected_socket

    sock = make_mooncake_client_socket(
        ctx,
        "tcp://192.0.2.2:6000",
        zmq.DEALER,  # type: ignore[attr-defined]
        linger=0,
        heartbeat_ivl_ms=150,
        heartbeat_timeout_ms=600,
        make_socket_fn=modern_make_zmq_socket,
    )

    assert sock is expected_socket
    assert captured == {
        "ctx": ctx,
        "path": "tcp://192.0.2.2:6000",
        "socket_type": zmq.DEALER,  # type: ignore[attr-defined]
        "bind": False,
        "linger": 0,
        "heartbeat_ivl_ms": 150,
        "heartbeat_timeout_ms": 600,
    }


def test_close_zmq_socket_and_zmq_ctx_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """close_zmq_socket sets linger=0 cleanly and zmq_ctx always destroys context."""
    close_zmq_socket(None)

    linger_sock = MagicMock()
    close_zmq_socket(linger_sock)
    linger_sock.close.assert_called_once_with(linger=0)

    legacy_sock = MagicMock()
    legacy_sock.close.side_effect = [TypeError("unexpected keyword argument 'linger'"), None]
    close_zmq_socket(legacy_sock)
    assert legacy_sock.close.call_count == 2

    context = MagicMock()
    client_sock = MagicMock()
    router_sock = MagicMock()
    monkeypatch.setattr(utils.zmq, "Context", MagicMock(return_value=context))
    client_helper = MagicMock(return_value=client_sock)
    router_helper = MagicMock(return_value=router_sock)
    monkeypatch.setattr(utils, "make_mooncake_client_socket", client_helper)
    monkeypatch.setattr(utils, "make_zmq_socket", router_helper)

    with zmq_ctx(zmq.REQ, "tcp://192.0.2.1:6000") as sock:  # type: ignore[attr-defined]
        assert sock is client_sock
    context.destroy.assert_called_once_with(linger=0)
    client_helper.assert_called_once()

    context.reset_mock()
    with pytest.raises(RuntimeError, match="boom"), zmq_ctx(zmq.ROUTER, "tcp://192.0.2.1:6001") as sock:  # type: ignore[attr-defined]
        assert sock is router_sock
        raise RuntimeError("boom")
    context.destroy.assert_called_once_with(linger=0)
    router_helper.assert_called_once_with(
        ctx=context,
        path="tcp://192.0.2.1:6001",
        socket_type=zmq.ROUTER,  # type: ignore[attr-defined]
        bind=True,
        linger=0,
    )


# ---------------------------------------------------------------------------
# Group 2: Live-P zero-regression
# ---------------------------------------------------------------------------


def test_recv_unless_lost_prefers_ready_reply_and_ignores_monitor_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed replies win over simultaneous monitor events; STOP frame is ignored."""
    sock = MagicMock()
    sock.getsockopt.return_value = 500
    monitor = MagicMock()
    poller_cls = _make_fake_poller_factory(
        [
            [(monitor, zmq.POLLIN), (sock, zmq.POLLIN)],  # type: ignore[attr-defined]
        ]
    )
    monkeypatch.setattr(utils.zmq, "Poller", poller_cls)
    sock.recv.return_value = b"reply-payload"
    monitor.recv_multipart.return_value = _monitor_event_frame(int(getattr(zmq, "EVENT_DISCONNECTED", 512)))

    assert recv_unless_lost(sock, monitor, "tcp://192.0.2.1:6000") == b"reply-payload"
    sock.recv.assert_called_once_with(flags=zmq.NOBLOCK)  # type: ignore[attr-defined]

    # EVENT_MONITOR_STOPPED does not abort polling; next reply is returned normally.
    stopped_id = int(getattr(zmq, "EVENT_MONITOR_STOPPED", 1024))
    sock.recv.reset_mock()
    sock.recv.return_value = b"after-stop"
    monitor.recv_multipart.return_value = _monitor_event_frame(stopped_id)
    poller_stop_then_reply = _make_fake_poller_factory(
        [
            [(monitor, zmq.POLLIN)],  # type: ignore[attr-defined]
            [(sock, zmq.POLLIN)],  # type: ignore[attr-defined]
        ]
    )
    monkeypatch.setattr(utils.zmq, "Poller", poller_stop_then_reply)
    assert recv_unless_lost(sock, monitor, "tcp://192.0.2.1:6000") == b"after-stop"


def test_lost_peer_monitor_lifecycle_and_mock_bypass() -> None:
    """lost_peer_monitor cleans up real sockets and skips NonCallableMock."""

    class FakeSocket:
        def __init__(self) -> None:
            self.monitor = MagicMock()
            self.events_seen: list[int] = []
            self.disabled = 0

        def get_monitor_socket(self, events: int) -> Any:
            self.events_seen.append(events)
            return self.monitor

        def disable_monitor(self) -> None:
            self.disabled += 1

    fake = FakeSocket()
    with pytest.raises(RuntimeError, match="inner"), lost_peer_monitor(fake) as mon:
        assert mon is fake.monitor
        raise RuntimeError("inner")

    assert fake.events_seen == [PULL_LOST_PEER_EVENTS]
    assert fake.disabled == 1
    fake.monitor.close.assert_called_once_with(linger=0)

    mock_sock = MagicMock()
    with lost_peer_monitor(mock_sock) as mon:
        assert mon is None
    mock_sock.get_monitor_socket.assert_not_called()


def test_v2_pull_worker_reuses_live_cached_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cached metadata for the same (host, port) avoids any ZMQ call."""
    thread = _make_v2_pull_thread()
    cached = make_metadata_groups(engine_id="engine-p")
    thread.remote_metadata["engine-p"] = cached
    zmq_spy = MagicMock(side_effect=AssertionError("Should reuse cached metadata"))
    monkeypatch.setattr(pull_worker, "zmq_ctx", zmq_spy)

    got = thread._get_remote_metadata("engine-p", cached.scheduler_host, cached.scheduler_port)
    assert got is cached


# ---------------------------------------------------------------------------
# Group 3: Deterministic fast-fail via fake socket/Poller/monitor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event_attr", ["EVENT_DISCONNECTED", "EVENT_CONNECT_RETRIED"])
def test_ensure_zmq_recv_fails_fast_on_peer_lost_event(
    monkeypatch: pytest.MonkeyPatch,
    event_attr: str,
) -> None:
    """Disconnect / connect-retried events raise MooncakePeerLostError with 0 sleep."""
    event_id = int(getattr(zmq, event_attr, 512))
    monitor = MagicMock()
    monitor.recv_multipart.return_value = _monitor_event_frame(event_id)

    class FakeSocket:
        def get_monitor_socket(self, _events: int) -> Any:
            return monitor

        def disable_monitor(self) -> None:
            pass

        def getsockopt(self, _opt: int) -> int:
            return 500

        def recv(self, *_args: Any, **_kwargs: Any) -> bytes:
            raise AssertionError("sock.recv should not run when monitor reports lost peer")

    sleep_spy = MagicMock()
    monkeypatch.setattr(utils.time, "sleep", sleep_spy)
    monkeypatch.setattr(
        utils.zmq,
        "Poller",
        _make_fake_poller_factory([[(monitor, zmq.POLLIN)]]),  # type: ignore[attr-defined]
    )

    with pytest.raises(MooncakePeerLostError, match="Lost connection to Mooncake peer"):
        ensure_zmq_recv(
            FakeSocket(),  # type: ignore[arg-type]
            "tcp://192.0.2.1:6000",
            max_retries=3,
            detect_peer_lost=True,
        )

    sleep_spy.assert_not_called()
    monitor.close.assert_called_once_with(linger=0)


def test_recv_unless_lost_times_out_when_peer_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty poll events bound waiting by timeout_ms / RCVTIMEO cleanly."""
    sock = MagicMock()
    monitor = MagicMock()
    monkeypatch.setattr(utils.zmq, "Poller", _make_fake_poller_factory([[]]))

    with pytest.raises(zmq.Again):  # type: ignore[attr-defined]
        recv_unless_lost(sock, monitor, "tcp://192.0.2.1:6000", timeout_ms=20)

    monkeypatch.setattr(utils.zmq, "Poller", _make_fake_poller_factory([[]]))
    with pytest.raises(RuntimeError, match="Failed to receive data after 1 retries"):
        ensure_zmq_recv(
            sock,
            "tcp://192.0.2.1:6000",
            max_retries=1,
            monitor=monitor,
            timeout_ms=20,
        )


# ---------------------------------------------------------------------------
# Group 4: V2, V1, and Hybrid integration & eviction
# ---------------------------------------------------------------------------


def test_v2_pull_worker_evicts_metadata_on_peer_lost_and_endpoint_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V2 worker evicts stale metadata on MooncakePeerLostError, failure, or restart."""
    thread = _make_v2_pull_thread()
    stale = make_metadata_groups(engine_id="engine-p")
    thread.remote_metadata["engine-p"] = stale
    thread.remote_tp_rank_groups["engine-p"] = {0: {(0, 0): [[0]]}}
    thread.remote_layer_index_pairs["engine-p"] = {0: [(0, 0)]}

    # Case 1: Same engine_id moved to a new endpoint (restarted P) -> evicts old & re-fetches.
    restarted_port = stale.scheduler_port + 1
    restarted_meta = MooncakeTransferMetadataGroups(
        engine_id="engine-p",
        scheduler_host=stale.scheduler_host,
        scheduler_port=restarted_port,
        pp_size=stale.pp_size,
        pcp_size=stale.pcp_size,
        dcp_size=stale.dcp_size,
        tp_size=stale.tp_size,
        use_kv_pp=stale.use_kv_pp,
        metadata_by_pp_rank=stale.metadata_by_pp_rank,
    )
    fake_sock = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_sock
    monkeypatch.setattr(pull_worker, "zmq_ctx", MagicMock(return_value=fake_ctx))
    monkeypatch.setattr(pull_worker, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(
        pull_worker,
        "ensure_zmq_recv",
        MagicMock(return_value=thread.encoder.encode(restarted_meta)),
    )
    thread._build_remote_transfer_layout = MagicMock(  # type: ignore[method-assign]
        return_value=({0: {(0, 0): [[0]]}}, {0: [(0, 0)]})
    )

    got = thread._get_remote_metadata("engine-p", stale.scheduler_host, restarted_port)
    assert got.scheduler_port == restarted_port
    assert thread.remote_metadata["engine-p"].scheduler_port == restarted_port

    # Case 2: Cache miss / new endpoint query hits MooncakePeerLostError -> clears cache for new query.
    monkeypatch.setattr(
        pull_worker,
        "ensure_zmq_recv",
        MagicMock(side_effect=MooncakePeerLostError("lost peer")),
    )
    with pytest.raises(MooncakePeerLostError):
        thread._get_remote_metadata("engine-p", stale.scheduler_host, restarted_port + 1)
    assert "engine-p" not in thread.remote_metadata
    assert "engine-p" not in thread.remote_tp_rank_groups
    assert "engine-p" not in thread.remote_layer_index_pairs

    # Case 3: _handle_requests bucket failure evicts cached metadata so next request re-queries.
    cached_meta = make_metadata_groups(engine_id="engine-p")
    thread.remote_metadata["engine-p"] = cached_meta
    thread.remote_tp_rank_groups["engine-p"] = {0: {(0, 0): [[0]]}}
    thread.remote_layer_index_pairs["engine-p"] = {0: [(0, 0)]}
    thread._build_transfer_block_buckets = MagicMock(  # type: ignore[method-assign]
        return_value=(
            {(0, 0): {0: {(0, 0): [("req-1", [10, 11], [1, 2])]}}},
            {(0, 0): {"req-1"}},
        )
    )
    failed_future: Future[None] = Future()
    failed_future.set_exception(RuntimeError("rdma failed"))
    mock_executor = MagicMock()
    mock_executor.submit.return_value = failed_future
    thread.executor = mock_executor

    req = _make_req_meta(host=cached_meta.scheduler_host, port=cached_meta.scheduler_port)
    failed_ids = thread._handle_requests(
        "engine-p", cached_meta.scheduler_host, cached_meta.scheduler_port, {"req-1": req}
    )
    assert failed_ids == {"req-1"}
    assert "engine-p" not in thread.remote_metadata


def test_v2_pull_scheduler_evicts_socket_pool_and_lease_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler drops dead socket pool on PeerLost and invalidates lease on failed_recving."""
    thread = pull_scheduler.MooncakeSchedulerRecvingThread(threading.Event())
    host, port = "192.0.2.1", 7000
    path = make_zmq_path("tcp", host, port)
    active_sock = MagicMock()
    extra_pooled_sock = MagicMock()
    other_path = make_zmq_path("tcp", "192.0.2.2", port)
    healthy_other_sock = MagicMock()
    thread.remote_sockets[path].extend([active_sock, extra_pooled_sock])
    thread.remote_sockets[other_path].append(healthy_other_sock)

    monkeypatch.setattr(pull_scheduler, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(
        pull_scheduler,
        "ensure_zmq_recv",
        MagicMock(side_effect=MooncakePeerLostError("lost")),
    )

    with pytest.raises(MooncakePeerLostError):
        thread._send_done_recving(host, port, "remote-req-1")

    active_sock.close.assert_called_once_with(linger=0)
    extra_pooled_sock.close.assert_called_once_with(linger=0)
    assert path not in thread.remote_sockets
    assert list(thread.remote_sockets[other_path]) == [healthy_other_sock]

    sched = _make_v2_scheduler()
    recving_spy = MagicMock()
    sched._recving_thread = recving_spy
    sched._reqs_recv_info["req-1"] = (host, port, "remote-req-1")
    sched._validated_producer_leases[(host, port)] = ("engine-p", 30)
    sched._validated_producer_leases[("192.0.2.2", port)] = ("engine-p-2", 30)
    connector_output = SimpleNamespace(
        finished_sending=set(),
        finished_recving={"req-1"},
        failed_recving={"req-1"},
    )
    sched.update_connector_output(connector_output)  # type: ignore[arg-type]

    recving_spy.add_request.assert_called_once_with(host, port, "remote-req-1")
    assert (host, port) not in sched._validated_producer_leases
    assert ("192.0.2.2", port) in sched._validated_producer_leases


def test_v1_connector_peer_lost_isolation_and_context_sharing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1 connector shares one ZMQ context and isolates lost peer without cross-host eviction."""
    thread = _make_v1_recving_thread()

    created_contexts: list[MagicMock] = []

    def fake_context() -> MagicMock:
        ctx = MagicMock()
        created_contexts.append(ctx)
        return ctx

    monkeypatch.setattr(v1_connector.zmq, "Context", fake_context)
    monkeypatch.setattr(
        v1_connector,
        "make_mooncake_client_socket",
        MagicMock(side_effect=[MagicMock(), MagicMock()]),
    )
    thread._get_remote_socket("192.0.2.1", 6000)
    thread._get_remote_socket("192.0.2.2", 6000)
    assert len(created_contexts) == 1

    # Populate metadata via _get_remote_metadata for two hosts sharing port 6000.
    meta_a = SimpleNamespace(
        engine_id="engine-a",
        te_rpc_port=9001,
        kv_caches_base_addr=[[0x1000]],
        block_size_scale=[[1]],
        block_strides=[64],
        kv_group2layeridx={},
    )
    meta_b = SimpleNamespace(
        engine_id="engine-b",
        te_rpc_port=9002,
        kv_caches_base_addr=[[0x2000]],
        block_size_scale=[[1]],
        block_strides=[64],
        kv_group2layeridx={},
    )
    decoder_mock = MagicMock()
    decoder_mock.decode.side_effect = [meta_a, meta_b]
    thread.decoder = decoder_mock
    sock_a = MagicMock()
    sock_b = MagicMock()
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.1", 6000)].append(sock_a)
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.2", 6000)].append(sock_b)
    monkeypatch.setattr(v1_connector, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(v1_connector, "ensure_zmq_recv", MagicMock(return_value=b"meta"))

    thread._get_remote_metadata("192.0.2.1", 6000)
    thread._get_remote_metadata("192.0.2.2", 6000)

    # Now lose 192.0.2.1:6000 during _handle_request; 192.0.2.2:6000 stays untouched.
    extra_sock_a = MagicMock()
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.1", 6000)].append(extra_sock_a)
    done_port_spy = MagicMock()
    done_recv_spy = MagicMock()
    monkeypatch.setattr(thread, "_send_done_signal_to_free_remote_port", done_port_spy)
    monkeypatch.setattr(thread, "_send_done_recv_signal", done_recv_spy)
    monkeypatch.setattr(
        thread,
        "_transfer_kv_cache_all_groups",
        MagicMock(side_effect=MooncakePeerLostError("lost A")),
    )
    thread.task_tracker.add_req_to_process("req-1")

    thread._handle_request(
        {
            "request_id": "req-1",
            "remote_request_id": "p-req-1",
            "remote_engine_id": "engine-a",
            "remote_host": "192.0.2.1",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": True,
            "local_block_ids": ([10, 11],),
        }
    )

    assert "engine-a" not in thread.kv_caches_base_addr
    assert "engine-b" in thread.kv_caches_base_addr
    assert make_zmq_path("tcp", "192.0.2.1", 6000) not in thread.remote_sockets
    assert list(thread.remote_sockets[make_zmq_path("tcp", "192.0.2.2", 6000)]) == [sock_b]
    sock_a.close.assert_called_once_with()
    extra_sock_a.close.assert_called_once_with()
    done_port_spy.assert_not_called()
    done_recv_spy.assert_not_called()
    assert thread.get_and_clear_invalid_block_ids() == {10, 11}

    # Verify fallback port scan respects other_host_eids when remote_engine_id is omitted.
    thread.kv_caches_base_addr["engine-c"][6000] = [[0x3000]]
    thread._invalidate_remote_metadata(remote_host="192.0.2.3", remote_handshake_port=6000)
    assert "engine-c" not in thread.kv_caches_base_addr
    assert "engine-b" in thread.kv_caches_base_addr


@pytest.mark.parametrize("use_request_level", [False, True])
def test_hybrid_connector_propagates_errors_and_isolates_lost_peer(
    monkeypatch: pytest.MonkeyPatch,
    use_request_level: bool,
) -> None:
    """Hybrid connector records errors instead of swallowing them and isolates lost peers."""
    thread = _make_hybrid_recving_thread(
        use_hybrid=use_request_level,
        use_request_level_recv_errors=use_request_level,
    )
    sock_b = MagicMock()
    decoder_mock = MagicMock()
    decoder_mock.decode.side_effect = [
        SimpleNamespace(engine_id="engine-a", te_rpc_port=9001, kv_caches_base_addr=[0x1000]),
        SimpleNamespace(engine_id="engine-b", te_rpc_port=9002, kv_caches_base_addr=[0x2000]),
    ]
    thread.decoder = decoder_mock
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.1", 6000)].append(MagicMock())
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.2", 6000)].append(sock_b)
    monkeypatch.setattr(hybrid_connector, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(hybrid_connector, "ensure_zmq_recv", MagicMock(return_value=b"meta"))
    thread._get_remote_metadata("192.0.2.1", 6000)
    thread._get_remote_metadata("192.0.2.2", 6000)

    done_recv_spy = MagicMock()
    monkeypatch.setattr(thread, "_send_done_recv_signal", done_recv_spy)
    monkeypatch.setattr(thread, "_send_done_signal_to_free_remote_port", MagicMock())
    target_method = "_transfer_kv_cache_all_groups" if use_request_level else "_transfer_kv_cache"
    monkeypatch.setattr(thread, target_method, MagicMock(side_effect=MooncakePeerLostError("lost A")))

    thread.task_tracker.add_req_to_process("req-1")
    thread._handle_request(
        {
            "request_id": "req-1",
            "remote_request_id": "p-req-1",
            "remote_engine_id": "engine-a",
            "remote_host": "192.0.2.1",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": True,
            "local_block_ids": ([21, 22],),
        }
    )

    done_recv_spy.assert_not_called()
    assert "engine-a" not in thread.kv_caches_base_addr
    assert "engine-b" in thread.kv_caches_base_addr

    worker = hybrid_connector.MooncakeConnectorWorker.__new__(hybrid_connector.MooncakeConnectorWorker)
    worker.kv_role = "kv_consumer"
    worker.tp_rank = 0
    worker.kv_recv_thread = thread
    worker.kv_send_thread = None
    connector = hybrid_connector.MooncakeConnector.__new__(hybrid_connector.MooncakeConnector)
    connector._role = KVConnectorRole.WORKER
    connector.connector_worker = worker
    connector.connector_scheduler = None

    results = connector.get_transfer_results({"req-1"})
    invalid_blocks = connector.get_block_ids_with_load_errors()
    assert results == KVConnectorTransferResults(
        finished_sending=set(),
        finished_recving={"req-1"},
        failed_recving={"req-1"} if use_request_level else set(),
    )
    assert invalid_blocks == (set() if use_request_level else {21, 22})


# ---------------------------------------------------------------------------
# Group 5: kv_load_failure_policy contract propagation
# ---------------------------------------------------------------------------


def _assert_legacy_connector_failure_surface(
    recv_thread: Any,
    worker: Any,
    conn: Any,
    *,
    request_level: bool,
) -> None:
    recv_thread.task_tracker.add_req_to_process("req-lost")
    recv_thread.task_tracker.update_done_task_count("req-lost")
    recv_thread._mark_failed_recv_request("req-lost", ([41, 42],))

    worker.kv_role = "kv_consumer"
    worker.tp_rank = 0
    worker.kv_recv_thread = recv_thread
    worker.kv_send_thread = None

    conn._role = KVConnectorRole.WORKER
    conn.connector_worker = worker
    conn.connector_scheduler = None

    res = conn.get_transfer_results({"req-lost"})
    blocks = conn.get_block_ids_with_load_errors()
    assert res.finished_recving == {"req-lost"}
    if request_level:
        assert res.failed_recving == {"req-lost"}
        assert blocks == set()
    else:
        assert res.failed_recving == set()
        assert blocks == {41, 42}


@pytest.mark.parametrize("policy", ["fail", "recompute"])
def test_kv_load_failure_policy_surfaces_failed_recving_and_invalid_blocks(policy: str) -> None:
    """V2, V1, and Hybrid expose failed_recving and invalid_block_ids under both policies."""
    # V2 worker + scheduler
    v2_thread = _make_v2_pull_thread(can_report_invalid_block_ids=True)
    v2_thread._mark_request_failed(_make_req_meta(local_blocks=([31, 32],)))
    v2_thread.finished_requests.put(("req-lost", True))
    v2_worker = pull_worker.MooncakePullConnectorWorker.__new__(pull_worker.MooncakePullConnectorWorker)
    v2_worker._recving_thread = v2_thread

    v2_results = v2_worker.get_transfer_results()
    assert v2_results == KVConnectorTransferResults(finished_recving={"req-lost"}, failed_recving={"req-lost"})
    assert v2_worker.get_block_ids_with_load_errors() == {31, 32}

    v2_sched = _make_v2_scheduler(policy=policy)
    v2_sched._reqs_recv_info["req-lost"] = ("192.0.2.1", 6000, "remote-req-lost")
    v2_sched._validated_producer_leases[("192.0.2.1", 6000)] = ("engine-p", 30)
    v2_sched.update_connector_output(v2_results)  # type: ignore[arg-type]
    assert ("192.0.2.1", 6000) not in v2_sched._validated_producer_leases

    # V1 & Hybrid connector worker surfaces
    _assert_legacy_connector_failure_surface(
        _make_v1_recving_thread(use_request_level_recv_errors=False),
        v1_connector.MooncakeConnectorWorker.__new__(v1_connector.MooncakeConnectorWorker),
        v1_connector.MooncakeConnector.__new__(v1_connector.MooncakeConnector),
        request_level=False,
    )
    _assert_legacy_connector_failure_surface(
        _make_hybrid_recving_thread(use_hybrid=True, use_request_level_recv_errors=True),
        hybrid_connector.MooncakeConnectorWorker.__new__(hybrid_connector.MooncakeConnectorWorker),
        hybrid_connector.MooncakeConnector.__new__(hybrid_connector.MooncakeConnector),
        request_level=True,
    )


# ---------------------------------------------------------------------------
# Group 6: Real pyzmq subprocess tests (skipped when pyzmq is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_ZMQ, reason="Requires real pyzmq")
@pytest.mark.parametrize("scenario", ["live", "killed", "frozen", "not_listening"])
def test_real_pyzmq_pull_fast_fail_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    """Real pyzmq ROUTER subprocess verifies killed/not_listening/frozen fast fail."""
    prefill = subprocess.Popen(
        [sys.executable, "-c", _SILENT_ROUTER_SCRIPT, scenario],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert prefill.stdout is not None
        port = int(prefill.stdout.readline().strip())
        path = f"tcp://127.0.0.1:{port}"
        if scenario == "not_listening":
            prefill.kill()
            prefill.wait(timeout=5)

        scale = 10
        ivl_ms = max(1, PULL_HEARTBEAT_IVL_MS // scale)
        hb_timeout_ms = max(1, PULL_HEARTBEAT_TIMEOUT_MS // scale)
        rcv_timeout_ms = max(1, DEFAULT_PULL_ZMQ_TIMEOUT_MS // scale)
        monkeypatch.setattr(utils, "PULL_HEARTBEAT_IVL_MS", ivl_ms)
        monkeypatch.setattr(utils, "PULL_HEARTBEAT_TIMEOUT_MS", hb_timeout_ms)

        ctx = zmq.Context()  # type: ignore[attr-defined]
        sock = None
        try:
            sock = make_mooncake_client_socket(
                ctx,
                path,
                zmq.REQ,  # type: ignore[attr-defined]
                linger=0,
                heartbeat_ivl_ms=ivl_ms,
                heartbeat_timeout_ms=hb_timeout_ms,
                make_socket_fn=make_zmq_socket,
            )
            sock.setsockopt(zmq.SNDTIMEO, rcv_timeout_ms)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, rcv_timeout_ms)  # type: ignore[attr-defined]
            start = time.monotonic()
            sock.send(b"PING")
            if scenario in ("killed", "frozen"):
                assert prefill.stdout.readline().strip() == "pulled"
                if scenario == "killed":
                    prefill.kill()
                    prefill.wait(timeout=5)
                else:
                    prefill.send_signal(signal.SIGSTOP)

            with lost_peer_monitor(sock) as monitor:
                if scenario == "live":
                    reply = ensure_zmq_recv(sock, path, monitor=monitor)
                    assert reply == b"PONG"
                    assert time.monotonic() - start < 2.0
                else:
                    with pytest.raises(MooncakePeerLostError):
                        ensure_zmq_recv(sock, path, monitor=monitor)
                    elapsed = time.monotonic() - start
                    limit = 5.0 if scenario == "frozen" else 3.0
                    assert elapsed < limit
        finally:
            close_zmq_socket(sock)
            ctx.destroy(linger=0)
    finally:
        if prefill.poll() is None:
            prefill.kill()
            prefill.wait(timeout=5)


def test_default_pull_timeout_exceeds_heartbeat_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default pull recv timeout stays strictly above ZMTP heartbeat detection time."""
    assert PULL_HEARTBEAT_IVL_MS > 0
    assert PULL_HEARTBEAT_TIMEOUT_MS > 0
    assert DEFAULT_PULL_ZMQ_TIMEOUT_MS > PULL_HEARTBEAT_IVL_MS + PULL_HEARTBEAT_TIMEOUT_MS
    assert v1_connector.DEFAULT_PULL_ZMQ_TIMEOUT_MS == DEFAULT_PULL_ZMQ_TIMEOUT_MS
    assert hybrid_connector.DEFAULT_PULL_ZMQ_TIMEOUT_MS == DEFAULT_PULL_ZMQ_TIMEOUT_MS

    captured: dict[str, Any] = {}

    def fake_make_zmq_socket(
        ctx: Any,
        path: str,
        socket_type: Any,
        *,
        bind: bool = False,
        linger: int | None = None,
        heartbeat_ivl_ms: int | None = None,
        heartbeat_timeout_ms: int | None = None,
    ) -> MagicMock:
        captured.update(
            {
                "heartbeat_ivl_ms": heartbeat_ivl_ms,
                "heartbeat_timeout_ms": heartbeat_timeout_ms,
            }
        )
        return MagicMock()

    fake_ctx = MagicMock()
    monkeypatch.setattr(utils, "PULL_HEARTBEAT_IVL_MS", 123)
    monkeypatch.setattr(utils, "PULL_HEARTBEAT_TIMEOUT_MS", 456)
    monkeypatch.setattr(utils.zmq, "Context", MagicMock(return_value=fake_ctx))
    monkeypatch.setattr(utils, "make_zmq_socket", fake_make_zmq_socket)

    with zmq_ctx(zmq.REQ, "tcp://192.0.2.1:6000"):  # type: ignore[attr-defined]
        pass

    assert captured == {"heartbeat_ivl_ms": 123, "heartbeat_timeout_ms": 456}
    fake_ctx.destroy.assert_called_once_with(linger=0)


@pytest.mark.skipif(not _HAS_REAL_ZMQ, reason="Requires real pyzmq")
def test_real_pyzmq_frozen_peer_fails_fast_with_production_defaults() -> None:
    """Unmodified production timeouts detect a SIGSTOP peer as MooncakePeerLostError."""
    prefill = subprocess.Popen(
        [sys.executable, "-c", _SILENT_ROUTER_SCRIPT, "frozen"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert prefill.stdout is not None
        port = int(prefill.stdout.readline().strip())
        path = f"tcp://127.0.0.1:{port}"
        ctx = zmq.Context()  # type: ignore[attr-defined]
        sock = None
        try:
            sock = make_mooncake_client_socket(
                ctx,
                path,
                zmq.REQ,  # type: ignore[attr-defined]
                linger=0,
                make_socket_fn=make_zmq_socket,
            )
            sock.setsockopt(zmq.SNDTIMEO, DEFAULT_PULL_ZMQ_TIMEOUT_MS)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, DEFAULT_PULL_ZMQ_TIMEOUT_MS)  # type: ignore[attr-defined]
            start = time.monotonic()
            sock.send(b"PING")
            assert prefill.stdout.readline().strip() == "pulled"
            prefill.send_signal(signal.SIGSTOP)
            with lost_peer_monitor(sock) as monitor, pytest.raises(MooncakePeerLostError):
                ensure_zmq_recv(sock, path, monitor=monitor)
            elapsed = time.monotonic() - start
            assert elapsed < (2 * DEFAULT_PULL_ZMQ_TIMEOUT_MS) / 1000.0
        finally:
            close_zmq_socket(sock)
            ctx.destroy(linger=0)
    finally:
        if prefill.poll() is None:
            prefill.kill()
            prefill.wait(timeout=5)


def test_v2_pull_scheduler_done_recving_timeout_discards_pooled_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler drops all pooled sockets for an endpoint on non-PeerLost timeout."""
    thread = pull_scheduler.MooncakeSchedulerRecvingThread(threading.Event())
    host, port = "192.0.2.1", 7000
    path = make_zmq_path("tcp", host, port)
    active_sock = MagicMock()
    extra_pooled_sock = MagicMock()
    other_path = make_zmq_path("tcp", "192.0.2.2", port)
    healthy_other_sock = MagicMock()
    thread.remote_sockets[path].extend([active_sock, extra_pooled_sock])
    thread.remote_sockets[other_path].append(healthy_other_sock)

    monkeypatch.setattr(pull_scheduler, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(
        pull_scheduler,
        "ensure_zmq_recv",
        MagicMock(side_effect=RuntimeError("Failed to receive data after 3 retries: timed out")),
    )

    with pytest.raises(RuntimeError, match="Failed to receive data after 3 retries"):
        thread._send_done_recving(host, port, "remote-req-1")

    active_sock.close.assert_called_once_with(linger=0)
    extra_pooled_sock.close.assert_called_once_with(linger=0)
    assert path not in thread.remote_sockets
    assert list(thread.remote_sockets[other_path]) == [healthy_other_sock]


def _setup_two_connector_peers(
    connector_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_spec: bool = True,
) -> tuple[Any, Any, MagicMock, MagicMock]:
    """Populate cached metadata and pooled sockets for 192.0.2.1 and 192.0.2.2."""
    is_v1 = connector_kind == "v1"
    module: Any = v1_connector if is_v1 else hybrid_connector
    thread: Any = (
        _make_v1_recving_thread()
        if is_v1
        else _make_hybrid_recving_thread(
            use_hybrid=(connector_kind == "hybrid_all_groups"),
            use_request_level_recv_errors=(connector_kind == "hybrid_all_groups"),
        )
    )
    meta_a = SimpleNamespace(
        engine_id="engine-a",
        te_rpc_port=9001,
        kv_caches_base_addr=[[0x1000]] if is_v1 else [0x1000],
        block_size_scale=[[1]],
        block_strides=[64],
        kv_group2layeridx={},
    )
    meta_b = SimpleNamespace(
        engine_id="engine-b",
        te_rpc_port=9002,
        kv_caches_base_addr=[[0x2000]] if is_v1 else [0x2000],
        block_size_scale=[[1]],
        block_strides=[64],
        kv_group2layeridx={},
    )
    decoder = MagicMock()
    decoder.decode.side_effect = [meta_a, meta_b]
    thread.decoder = decoder
    sock_a = MagicMock(spec=zmq.Socket) if use_spec else MagicMock()  # type: ignore[attr-defined]
    sock_b = MagicMock(spec=zmq.Socket) if use_spec else MagicMock()  # type: ignore[attr-defined]
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.1", 6000)].append(sock_a)
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.2", 6000)].append(sock_b)
    monkeypatch.setattr(module, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(module, "ensure_zmq_recv", MagicMock(return_value=b"meta"))
    thread._get_remote_metadata("192.0.2.1", 6000)
    thread._get_remote_metadata("192.0.2.2", 6000)
    return module, thread, sock_a, sock_b


@pytest.mark.parametrize("connector_kind", ["v1", "hybrid_sync", "hybrid_all_groups"])
def test_connector_handle_request_general_failure_evicts_metadata(
    monkeypatch: pytest.MonkeyPatch,
    connector_kind: str,
) -> None:
    """Backend transfer RuntimeError evicts cached metadata so restarts rediscover P."""
    _, thread, _, _ = _setup_two_connector_peers(connector_kind, monkeypatch, use_spec=False)
    done_port_spy = MagicMock()
    done_recv_spy = MagicMock()
    monkeypatch.setattr(thread, "_send_done_signal_to_free_remote_port", done_port_spy)
    monkeypatch.setattr(thread, "_send_done_recv_signal", done_recv_spy)
    transfer_method = "_transfer_kv_cache" if connector_kind == "hybrid_sync" else "_transfer_kv_cache_all_groups"
    monkeypatch.setattr(thread, transfer_method, MagicMock(side_effect=RuntimeError("transfer failed, ret: -1")))
    thread.task_tracker.add_req_to_process("req-1")

    thread._handle_request(
        {
            "request_id": "req-1",
            "remote_request_id": "p-req-1",
            "remote_engine_id": "engine-a",
            "remote_host": "192.0.2.1",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": True,
            "local_block_ids": ([10, 11],),
        }
    )

    assert "engine-a" not in thread.kv_caches_base_addr
    assert "engine-b" in thread.kv_caches_base_addr
    done_port_spy.assert_called_once()
    done_recv_spy.assert_called_once()
    is_request_level = connector_kind == "hybrid_all_groups"
    assert thread.get_and_clear_failed_recv_requests({"req-1"}) == ({"req-1"} if is_request_level else set())
    assert thread.get_and_clear_invalid_block_ids() == (set() if is_request_level else {10, 11})


@pytest.mark.parametrize("connector_kind", ["v1", "hybrid"])
@pytest.mark.parametrize("failure_mode", ["get_meta", "done_recv_runtime", "done_recv_zmq"])
def test_connector_zmq_failure_evicts_metadata_and_pooled_sockets(
    monkeypatch: pytest.MonkeyPatch,
    connector_kind: str,
    failure_mode: str,
) -> None:
    """ZMQ timeout or ZMQError evicts both cached metadata and pooled dead sockets."""
    module, thread, sock_a, sock_b = _setup_two_connector_peers(connector_kind, monkeypatch, use_spec=True)
    extra_sock_a = MagicMock(spec=zmq.Socket)  # type: ignore[attr-defined]
    path_a = make_zmq_path("tcp", "192.0.2.1", 6000)
    path_b = make_zmq_path("tcp", "192.0.2.2", 6000)
    thread.remote_sockets[path_a].append(extra_sock_a)

    exc: Exception = (
        zmq.ZMQError("reset")  # type: ignore[attr-defined]
        if failure_mode == "done_recv_zmq"
        else RuntimeError("Failed to receive data after 3 retries: timed out")
    )
    monkeypatch.setattr(module, "ensure_zmq_recv", MagicMock(side_effect=exc))

    if failure_mode == "get_meta":
        with pytest.raises(RuntimeError, match="Failed to receive data"):
            thread._get_remote_metadata("192.0.2.1", 6000)
    else:
        thread._send_done_recv_signal("req-1", "192.0.2.1", 6000, {})

    sock_a.close.assert_called_once_with()
    extra_sock_a.close.assert_called_once_with()
    assert path_a not in thread.remote_sockets
    assert list(thread.remote_sockets[path_b]) == [sock_b]
    assert "engine-a" not in thread.kv_caches_base_addr
    assert "engine-b" in thread.kv_caches_base_addr


def test_ensure_zmq_recv_skips_monitor_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared callers without detect_peer_lost opt-in keep standard transient retry semantics."""

    class SharedSocket:
        def __init__(self) -> None:
            self.recv_calls = 0

        def get_monitor_socket(self, _events: int) -> Any:
            raise AssertionError("monitor should not attach unless opted in")

        def recv(self, *_args: Any, **_kwargs: Any) -> bytes:
            self.recv_calls += 1
            if self.recv_calls == 1:
                raise zmq.ZMQError("transient connect retry")  # type: ignore[attr-defined]
            return b"push-ack"

    sleep_spy = MagicMock()
    monkeypatch.setattr(utils.time, "sleep", sleep_spy)
    sock = SharedSocket()

    assert ensure_zmq_recv(sock, "tcp://192.0.2.1:6000", max_retries=3) == b"push-ack"  # type: ignore[arg-type]
    assert sock.recv_calls == 2
    sleep_spy.assert_called_once_with(0.1)


@pytest.mark.parametrize("bad_timeout", [True, False])
def test_ensure_zmq_recv_rejects_bool_timeout_ms(bad_timeout: bool) -> None:
    """Boolean timeout_ms arguments must be rejected explicitly rather than coerced to int."""
    sock = MagicMock()
    with pytest.raises(ValueError, match="timeout_ms must be an int or None"):
        ensure_zmq_recv(sock, "tcp://192.0.2.1:6000", timeout_ms=bad_timeout)
    with pytest.raises(ValueError, match="timeout_ms must be an int or None"):
        recv_unless_lost(sock, MagicMock(), "tcp://192.0.2.1:6000", timeout_ms=bad_timeout)


@pytest.mark.parametrize("connector_kind", ["v1", "hybrid_sync", "hybrid_all_groups"])
def test_connector_multi_subtask_peer_lost_skips_done_for_dead_host_and_clears_state(
    monkeypatch: pytest.MonkeyPatch,
    connector_kind: str,
) -> None:
    """Multi-subtask pulls skip DONE on every subtask for a lost host and clean up on drain."""
    is_v1 = connector_kind == "v1"
    thread: Any = (
        _make_v1_recving_thread(use_request_level_recv_errors=False)
        if is_v1
        else _make_hybrid_recving_thread(
            use_hybrid=(connector_kind == "hybrid_all_groups"),
            use_request_level_recv_errors=(connector_kind == "hybrid_all_groups"),
        )
    )
    monkeypatch.setattr(thread, "_invalidate_remote_metadata", MagicMock())
    monkeypatch.setattr(thread, "_discard_remote_sockets", MagicMock())
    done_port_spy = MagicMock()
    done_recv_spy = MagicMock()
    monkeypatch.setattr(thread, "_send_done_signal_to_free_remote_port", done_port_spy)
    monkeypatch.setattr(thread, "_send_done_recv_signal", done_recv_spy)
    transfer_method = "_transfer_kv_cache" if connector_kind == "hybrid_sync" else "_transfer_kv_cache_all_groups"
    monkeypatch.setattr(thread, transfer_method, MagicMock(side_effect=MooncakePeerLostError("dead 192.0.2.10")))

    tasks = [
        {
            "request_id": "req-multi",
            "remote_request_id": "p-req-multi",
            "remote_engine_id": "engine-a",
            "remote_host": "192.0.2.10",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": False,
            "local_block_ids": ([10, 11],),
        },
        {
            "request_id": "req-multi",
            "remote_request_id": "p-req-multi",
            "remote_engine_id": "engine-a",
            "remote_host": "192.0.2.10",
            "remote_handshake_port": 6001,
            "remote_port_send_num": {},
            "all_task_done": False,
            "local_block_ids": ([10, 11],),
        },
        {
            "request_id": "req-multi",
            "remote_request_id": "p-req-multi",
            "remote_engine_id": "engine-b",
            "remote_host": "192.0.2.11",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": True,
            "local_block_ids": ([10, 11],),
        },
    ]
    thread.task_tracker.add_req_to_process("req-multi")
    for task in tasks:
        thread._mark_request_task_submitted(task)

    # Task 0 loses 192.0.2.10:6000; Task 1 on 192.0.2.10:6001 skips transfer AND skips DONE.
    # Task 2 targets a different host 192.0.2.11:6000, so DONE still runs for the live host.
    for task in tasks:
        thread._handle_request(task)

    assert done_port_spy.call_args_list == [(("p-req-multi", "192.0.2.11", {}),)]
    assert done_recv_spy.call_args_list == [(("p-req-multi", "192.0.2.11", 6000, {}),)]
    assert thread.peer_lost_requests == set()

    # Also verify get_and_clear_failed_recv_requests drains any retained peer_lost_requests.
    thread._mark_peer_lost_request("req-orphan", "192.0.2.12")
    assert thread.get_and_clear_failed_recv_requests({"req-orphan"}) == set()
    assert thread.peer_lost_requests == set()
