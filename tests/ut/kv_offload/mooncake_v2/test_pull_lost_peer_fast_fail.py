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

import contextlib
import importlib
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
    mooncake_d2rh_connector as d2rh_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_hybrid_connector as hybrid_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import (
    pull_scheduler,
    pull_worker,
    push_worker,
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
    ident, delim, _ = router.recv_multipart()
    router.send_multipart([ident, delim, b"PONG"])
    router.recv_multipart()
elif mode in ("killed", "frozen"):
    router.recv_multipart()
    print("pulled", flush=True)
    router.recv_multipart()
"""


def _monitor_event_frame(event_id: int) -> list[bytes]:
    return [struct.pack("=HI", event_id, 0)]


def _make_fake_poller_cls(outcomes: list[list[tuple[Any, int]]]) -> type[Any]:
    pending = list(outcomes)

    class FakePoller:
        def register(self, sock: Any, _flags: int) -> None:
            pass

        def unregister(self, sock: Any) -> None:
            pass

        def poll(self, _timeout_ms: int | None = None) -> list[tuple[Any, int]]:
            return pending.pop(0) if pending else []

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


def _get_layerwise_connector() -> Any:
    # Lazy import avoids populating sys.modules before test_mooncake_layerwise_connector resets it.
    return importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector")


def _make_connector_thread(
    kind: str,
    *,
    use_request_level_recv_errors: bool = False,
    timeout: float = 0.8,
) -> Any:
    """Create a minimal thread/worker instance for V1/Hybrid/D2RH/Layerwise/V2 tests."""
    if kind == "d2rh":
        obj: Any = d2rh_connector.D2RHThread.__new__(d2rh_connector.D2RHThread)
        obj.local_engine_id = "local-d"
        obj.kv_caches_base_addr = SizedDict()
        obj.remote_te_port = SizedDict()
        obj.remote_block_size_scale = SizedDict()
        obj.remote_block_stride_per_addr = SizedDict()
        obj.remote_kv_group2layeridx = SizedDict()
        obj.remote_metadata_lock = threading.Lock()
        obj.remote_sockets_lock = threading.Lock()
        obj.remote_sockets = defaultdict(deque)
        obj.timeout = timeout
        obj.encoder = msgspec.msgpack.Encoder()
        obj.decoder = msgspec.msgpack.Decoder(v1_connector.MooncakeAgentMetadata)
        return obj

    if kind == "layerwise":
        layerwise_connector = _get_layerwise_connector()
        worker: Any = layerwise_connector.MooncakeLayerwiseConnectorWorker.__new__(
            layerwise_connector.MooncakeLayerwiseConnectorWorker
        )
        worker.local_engine_id = "local-d"
        worker.engine_id = "local-d"
        worker.side_channel_host = "127.0.0.1"
        worker.handshake_port = 5556
        worker.pd_head_ratio = 1
        worker.layer_metadata = {}
        worker.remote_metadata_lock = threading.Lock()
        worker.remote_layer_metadata = SizedDict()
        worker.remote_te_port = SizedDict()
        worker.remote_sockets_lock = threading.Lock()
        worker.remote_sockets = defaultdict(deque)
        worker.remote_poller = zmq.Poller()  # type: ignore[attr-defined]
        worker.timeout = timeout
        worker.encoder = msgspec.msgpack.Encoder()
        worker.decoder = msgspec.msgpack.Decoder(layerwise_connector.MooncakeAgentMetadata)
        return worker

    is_hybrid = kind.startswith("hybrid")
    cls: Any = hybrid_connector.KVCacheRecvingThread if is_hybrid else v1_connector.KVCacheRecvingThread
    thread = cls.__new__(cls)
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
    thread.timeout = timeout
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = msgspec.msgpack.Decoder(
        hybrid_connector.MooncakeAgentMetadata if is_hybrid else v1_connector.MooncakeAgentMetadata
    )
    thread.request_task_counts = defaultdict(int)
    thread.finished_request_markers = set()
    thread.request_task_counts_lock = threading.Lock()
    thread.proc_not_transfer_request = {}
    thread.proc_not_transfer_request_lock = threading.Lock()
    thread.pending_reformat = defaultdict(dict)
    thread.pending_reformat_lock = threading.Lock()
    thread.request_queue = MagicMock()
    thread.task_tracker = hybrid_connector.KVCacheTaskTracker() if is_hybrid else v1_connector.KVCacheTaskTracker()
    if is_hybrid:
        thread.use_hybrid = use_request_level_recv_errors
        thread.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[object()] * (2 if use_request_level_recv_errors else 1)
        )
    thread.use_request_level_recv_errors = use_request_level_recv_errors
    thread.failed_recv_requests = set()
    thread.peer_lost_requests = set()
    thread.invalid_block_ids = set()
    thread.failed_recv_requests_lock = threading.Lock()
    return thread


def _make_v2_pull_thread(
    *,
    mode: str = "request",
    can_report_invalid_block_ids: bool = True,
) -> pull_worker.MooncakePullRecvingThread:
    thread = pull_worker.MooncakePullRecvingThread.__new__(pull_worker.MooncakePullRecvingThread)
    thread.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16))
    thread.pull_transfer_mode = mode
    thread._process_next = thread._process_next_request if mode == "request" else thread._process_next_batch
    thread.encoder = msgspec.msgpack.Encoder()
    thread.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
    thread.remote_metadata = SizedDict()
    thread.remote_tp_rank_groups = SizedDict()
    thread.remote_layer_index_pairs = SizedDict()
    thread.request_queue = queue.Queue()
    thread.finished_requests = queue.SimpleQueue()
    thread.invalid_block_ids = set()
    thread.invalid_block_ids_lock = threading.Lock()
    thread._remote_metadata_lock = threading.Lock()
    thread.can_report_invalid_block_ids = can_report_invalid_block_ids
    thread.layer_names = ["model.layers.0.self_attn"]
    thread.spec_indices = [0]
    thread.kv_cache_specs = []
    thread.executor = MagicMock()
    return thread


def _start_router_that_replies_once_then_dies(
    reply_payload: bytes,
    *,
    replies_before_death: int = 1,
    wait_for_second_request: bool = True,
) -> tuple[int, threading.Event, threading.Thread]:
    """Start a real in-process ZMQ ROUTER returning `replies_before_death` replies, then dying."""
    ready = threading.Event()
    port_holder: list[int] = []

    def serve() -> None:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        router = ctx.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        router.setsockopt(zmq.LINGER, 0)  # type: ignore[attr-defined]
        router.setsockopt(zmq.RCVTIMEO, 5000)  # type: ignore[attr-defined]
        port = router.bind_to_random_port("tcp://127.0.0.1")
        port_holder.append(port)
        ready.set()
        try:
            for _ in range(replies_before_death):
                ident, delim, _ = router.recv_multipart()
                router.send_multipart([ident, delim, reply_payload])
            if wait_for_second_request:
                # Wait for the next request, then close immediately to trigger EVENT_DISCONNECTED.
                with contextlib.suppress(zmq.Again):  # type: ignore[attr-defined]
                    router.recv_multipart()
        finally:
            router.close(linger=0)
            ctx.destroy(linger=0)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    assert ready.wait(3)
    return port_holder[0], ready, server


# ---------------------------------------------------------------------------
# 1. Dual-version socket options, zmq_ctx, monitor & timeout validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory_kind", ["legacy_linger", "legacy_minimal", "modern"])
def test_make_mooncake_client_socket_and_zmq_ctx_lifecycle(factory_kind: str) -> None:
    """Client socket factory sets pre-connect LINGER/heartbeats across vLLM signatures."""
    assert 0 < PULL_HEARTBEAT_IVL_MS < PULL_HEARTBEAT_TIMEOUT_MS < DEFAULT_PULL_ZMQ_TIMEOUT_MS
    assert v1_connector.DEFAULT_PULL_ZMQ_TIMEOUT_MS == DEFAULT_PULL_ZMQ_TIMEOUT_MS
    assert hybrid_connector.DEFAULT_PULL_ZMQ_TIMEOUT_MS == DEFAULT_PULL_ZMQ_TIMEOUT_MS

    events: list[tuple[Any, ...]] = []
    raw_sock = MagicMock()
    raw_sock.setsockopt.side_effect = lambda opt, val: events.append(("setsockopt", opt, val))
    raw_sock.connect.side_effect = lambda addr: events.append(("connect", addr))
    fake_ctx = MagicMock()
    fake_ctx.socket.return_value = raw_sock
    seen: dict[str, Any] = {}

    def build_legacy_linger(
        ctx: Any, path: str, socket_type: Any, *, bind: bool = False, linger: int | None = None
    ) -> Any:
        s = ctx.socket(socket_type)
        s.connect(path)
        seen.update({"bind": bind, "linger": linger, "path": path})
        return s

    def build_legacy_minimal(ctx: Any, path: str, socket_type: Any, *, bind: bool = False) -> Any:
        s = ctx.socket(socket_type)
        s.connect(path)
        seen.update({"bind": bind, "path": path})
        return s

    def build_modern(
        ctx: Any,
        path: str,
        socket_type: Any,
        *,
        bind: bool = False,
        linger: int | None = None,
        heartbeat_ivl_ms: int | None = None,
        heartbeat_timeout_ms: int | None = None,
    ) -> Any:
        seen.update(
            {
                "bind": bind,
                "linger": linger,
                "heartbeat_ivl_ms": heartbeat_ivl_ms,
                "heartbeat_timeout_ms": heartbeat_timeout_ms,
                "path": path,
            }
        )
        return raw_sock

    builders: dict[str, Any] = {
        "legacy_linger": build_legacy_linger,
        "legacy_minimal": build_legacy_minimal,
        "modern": build_modern,
    }
    got = make_mooncake_client_socket(
        fake_ctx,
        "tcp://192.0.2.1:6000",
        zmq.REQ,  # type: ignore[attr-defined]
        linger=0,
        heartbeat_ivl_ms=150,
        heartbeat_timeout_ms=600,
        make_socket_fn=builders[factory_kind],
    )
    assert got is raw_sock and seen["bind"] is False and seen["path"] == "tcp://192.0.2.1:6000"
    if factory_kind == "modern":
        assert (seen["heartbeat_ivl_ms"], seen["heartbeat_timeout_ms"], seen["linger"]) == (150, 600, 0)
    else:
        connect_idx = next(i for i, item in enumerate(events) if item[0] == "connect")
        pre = {item[1]: item[2] for item in events[:connect_idx] if item[0] == "setsockopt"}
        assert pre[zmq.LINGER] == 0  # type: ignore[attr-defined]
        if hasattr(zmq, "HEARTBEAT_IVL"):
            assert pre[zmq.HEARTBEAT_IVL] == 150  # type: ignore[attr-defined]

    close_zmq_socket(None)
    legacy_close = MagicMock()
    legacy_close.close.side_effect = [TypeError("no linger"), None]
    close_zmq_socket(legacy_close)
    assert legacy_close.close.call_count == 2
    if _HAS_REAL_ZMQ:
        port, _, server = _start_router_that_replies_once_then_dies(
            b"PONG", replies_before_death=1, wait_for_second_request=False
        )
        try:
            with zmq_ctx(zmq.REQ, f"tcp://127.0.0.1:{port}") as req, lost_peer_monitor(req) as mon:  # type: ignore[attr-defined]
                req.setsockopt(zmq.RCVTIMEO, 2000)  # type: ignore[attr-defined]
                req.send(b"PING")
                assert ensure_zmq_recv(req, f"tcp://127.0.0.1:{port}", monitor=mon) == b"PONG"
        finally:
            server.join(timeout=2)


@pytest.mark.parametrize("event_attr", ["EVENT_DISCONNECTED", "EVENT_CONNECT_RETRIED"])
def test_ensure_zmq_recv_monitor_validation_and_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
    event_attr: str,
) -> None:
    """Validate monitor opt-in, bool timeout guard, event precedence, and zero-sleep abort."""
    sock = MagicMock()
    for bad in (True, False):
        with pytest.raises(ValueError, match="timeout_ms must be an int or None"):
            ensure_zmq_recv(sock, "tcp://192.0.2.1:6000", timeout_ms=bad)
        with pytest.raises(ValueError, match="timeout_ms must be an int or None"):
            recv_unless_lost(sock, MagicMock(), "tcp://192.0.2.1:6000", timeout_ms=bad)

    # Without detect_peer_lost opt-in, shared callers keep standard transient retry semantics.
    sleep_spy = MagicMock()
    monkeypatch.setattr(utils.time, "sleep", sleep_spy)
    shared_sock = MagicMock()
    shared_sock.recv.side_effect = [zmq.ZMQError("retry"), b"push-ack"]  # type: ignore[attr-defined]
    assert ensure_zmq_recv(shared_sock, "tcp://192.0.2.1:6000", max_retries=3) == b"push-ack"
    shared_sock.get_monitor_socket.assert_not_called()
    sleep_spy.assert_called_once_with(0.1)
    sleep_spy.reset_mock()

    # Completed reply wins over simultaneous disconnect, and EVENT_MONITOR_STOPPED is ignored.
    monitor = MagicMock()
    stopped_id = int(getattr(zmq, "EVENT_MONITOR_STOPPED", 1024))
    event_id = int(getattr(zmq, event_attr, 512))
    monitor.recv_multipart.side_effect = [_monitor_event_frame(stopped_id), _monitor_event_frame(event_id)]
    sock.recv.return_value = b"ready-reply"
    monkeypatch.setattr(
        utils.zmq,
        "Poller",
        _make_fake_poller_cls(
            [
                [(monitor, zmq.POLLIN), (sock, zmq.POLLIN)],  # type: ignore[attr-defined]
                [(monitor, zmq.POLLIN)],  # type: ignore[attr-defined]
                [(sock, zmq.POLLIN)],  # type: ignore[attr-defined]
            ]
        ),
    )
    assert recv_unless_lost(sock, monitor, "tcp://192.0.2.1:6000", timeout_ms=500) == b"ready-reply"
    assert recv_unless_lost(sock, monitor, "tcp://192.0.2.1:6000", timeout_ms=500) == b"ready-reply"
    assert monitor.recv_multipart.call_count == 1

    # NonCallableMock bypasses monitor; real monitor raises MooncakePeerLostError with 0 sleep.
    with lost_peer_monitor(sock) as mon:
        assert mon is None

    class MonitoredSock:
        def get_monitor_socket(self, _events: int) -> Any:
            return monitor

        def disable_monitor(self) -> None:
            pass

        def getsockopt(self, _opt: int) -> int:
            return 500

    monitor.recv_multipart.side_effect = None
    monitor.recv_multipart.return_value = _monitor_event_frame(event_id)
    monkeypatch.setattr(utils.zmq, "Poller", _make_fake_poller_cls([[(monitor, zmq.POLLIN)]]))  # type: ignore[attr-defined]
    with pytest.raises(MooncakePeerLostError, match="Lost connection to Mooncake peer"):
        ensure_zmq_recv(MonitoredSock(), "tcp://192.0.2.1:6000", max_retries=3, detect_peer_lost=True)  # type: ignore[arg-type]
    sleep_spy.assert_not_called()
    monitor.close.assert_called_once_with(linger=0)


# ---------------------------------------------------------------------------
# 2. Real pyzmq subprocess & real in-process TCP fast-fail across all connectors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_ZMQ, reason="Requires real pyzmq")
@pytest.mark.parametrize(
    ("scenario", "scaled"),
    [("live", True), ("killed", True), ("frozen", True), ("not_listening", True), ("frozen", False)],
)
def test_real_pyzmq_pull_fast_fail_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    scaled: bool,
) -> None:
    """Real pyzmq ROUTER subprocess verifies live, killed, not_listening, and SIGSTOP fast-fail."""
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

        scale = 10 if scaled else 1
        ivl_ms = max(1, PULL_HEARTBEAT_IVL_MS // scale)
        hb_timeout_ms = max(1, PULL_HEARTBEAT_TIMEOUT_MS // scale)
        rcv_timeout_ms = max(1, DEFAULT_PULL_ZMQ_TIMEOUT_MS // scale)
        if scaled:
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
                prefill.kill() if scenario == "killed" else prefill.send_signal(signal.SIGSTOP)
                if scenario == "killed":
                    prefill.wait(timeout=5)

            with lost_peer_monitor(sock) as monitor:
                if scenario == "live":
                    assert ensure_zmq_recv(sock, path, monitor=monitor) == b"PONG"
                else:
                    with pytest.raises(MooncakePeerLostError, match="Lost connection to Mooncake peer"):
                        ensure_zmq_recv(sock, path, monitor=monitor)
                    limit = (2 * DEFAULT_PULL_ZMQ_TIMEOUT_MS) / 1000.0 if not scaled else 5.0
                    assert (time.monotonic() - start) < limit
        finally:
            close_zmq_socket(sock)
            ctx.destroy(linger=0)
    finally:
        if prefill.poll() is None:
            prefill.kill()
            prefill.wait(timeout=5)


def _agent_metadata_payload(kind: str, engine_id: str) -> bytes:
    if kind == "layerwise":
        layerwise_connector = _get_layerwise_connector()
        meta: Any = layerwise_connector.MooncakeAgentMetadata(te_rpc_port=9001, layer_metadata={})
    elif kind == "hybrid":
        meta = hybrid_connector.MooncakeAgentMetadata(
            engine_id=engine_id,
            te_rpc_port=9001,
            block_size=16,
            kv_caches_base_addr=[0x1000],
            num_blocks=2,
            block_lens=[1024],
            ssm_sizes=(0, 0),
        )
    else:
        meta = v1_connector.MooncakeAgentMetadata(
            engine_id=engine_id,
            te_rpc_port=9001,
            kv_group2layeridx={},
            block_size=16,
            kv_caches_base_addr=[[0x1000]],
            block_size_scale=[[1]],
            num_blocks=2,
            block_lens=[[1024]],
            block_strides=[[1024]],
        )
    return msgspec.msgpack.Encoder().encode(meta)


def _assert_real_pyzmq_layerwise_and_push(kind: str, port: int, dead_path: str) -> None:
    if kind == "layerwise":
        layerwise_connector = _get_layerwise_connector()
        worker = _make_connector_thread("layerwise", timeout=0.8)
        worker.engine = MagicMock()
        meta = layerwise_connector.ReqMeta(
            local_block_ids=[[1]],
            token_ids=None,
            remote_block_ids=[[1]],
            remote_block_size=[[1]],
            remote_engine_id="engine-a",
            remote_host="127.0.0.1",
            remote_port=port,
            remote_te_rpc_port=None,
            remote_layer_metadata={},
            metaserver=None,
            local_computed_tokens=0,
            prompt_len=16,
            remote_pcp_size=1,
            remote_dcp_size=1,
            remote_tp_size=1,
        )
        try:
            worker.update_decoder_info("req-1", meta)
            assert port in worker.remote_layer_metadata["engine-a"]
            worker._invalidate_remote_metadata("engine-a")
            t0 = time.monotonic()
            with pytest.raises(layerwise_connector.MooncakePeerLostError):
                worker.update_decoder_info("req-2", meta)
            assert (time.monotonic() - t0) < 2.0
            assert dead_path not in worker.remote_sockets and "engine-a" not in worker.remote_layer_metadata
        finally:
            if getattr(worker, "_zmq_context", None) is not None:
                worker._zmq_context.destroy(linger=0)
        return

    pw = push_worker.MooncakePushConnectorWorker.__new__(push_worker.MooncakePushConnectorWorker)
    pw.pp_size = pw.pcp_size = pw.tp_size = 1
    pw._zmq_context = zmq.Context()  # type: ignore[attr-defined]
    pw._remote_metadata = SizedDict(max_size=16)
    pw._remote_metadata_lock = threading.Lock()
    params = {
        "request_id": "r1",
        "producer_engine_id": "p",
        "producer_rank": [0, 0, 0],
        "consumer_engine_id": "engine-a",
        "consumer_rank": [0, 0, 0],
        "scheduler_address": dead_path,
        "completion_address": dead_path,
    }
    pw._remote_metadata[("engine-a", dead_path)] = Future()
    try:
        assert pw._get_remote_metadata(params).engine_id == "engine-a"
        with pytest.raises(MooncakePeerLostError):
            pw._get_remote_metadata(params)
        assert ("engine-a", dead_path) not in pw._remote_metadata
        pw._remote_metadata[("engine-a", dead_path)] = Future()
        assert pw._send_done(params, None) is False
        assert ("engine-a", dead_path) not in pw._remote_metadata
    finally:
        pw._zmq_context.destroy(linger=0)


@pytest.mark.skipif(not _HAS_REAL_ZMQ, reason="Requires real pyzmq")
@pytest.mark.parametrize("kind", ["v1", "hybrid", "layerwise", "d2rh", "d2rh_sched", "v2_sched", "push"])
def test_real_pyzmq_shared_mixin_and_sibling_connectors(kind: str) -> None:
    """Real in-process TCP/ZMQ servers verify fast-fail, socket eviction & metadata isolation."""
    if kind in ("v2_sched", "d2rh_sched"):
        reply_bytes = b"ACK"
    elif kind == "push":
        reply_bytes = msgspec.msgpack.Encoder().encode(make_metadata_groups(engine_id="engine-a"))
    else:
        reply_bytes = _agent_metadata_payload(kind, "engine-a")

    port, _, srv = _start_router_that_replies_once_then_dies(reply_bytes, replies_before_death=1)
    dead_path = make_zmq_path("tcp", "127.0.0.1", port)

    try:
        if kind in ("v1", "hybrid", "d2rh"):
            thread = _make_connector_thread(kind, timeout=0.8)
            try:
                thread._get_remote_metadata("127.0.0.1", port)
                assert "engine-a" in thread.kv_caches_base_addr and len(thread.remote_sockets[dead_path]) == 1
                shared_ctx = thread._zmq_context
                t0 = time.monotonic()
                with pytest.raises(MooncakePeerLostError):
                    thread._invalidate_remote_metadata("engine-a")
                    thread._get_remote_metadata("127.0.0.1", port)
                assert (time.monotonic() - t0) < 2.0 and thread._zmq_context is shared_ctx
                assert dead_path not in thread.remote_sockets and "engine-a" not in thread.kv_caches_base_addr
            finally:
                if getattr(thread, "_zmq_context", None) is not None:
                    thread._zmq_context.destroy(linger=0)
        elif kind in ("layerwise", "push"):
            _assert_real_pyzmq_layerwise_and_push(kind, port, dead_path)
        elif kind == "d2rh_sched":
            sched = d2rh_connector.MooncakeConnectorScheduler.__new__(d2rh_connector.MooncakeConnectorScheduler)
            sched.local_host = "127.0.0.1"
            sched.timeout = 0.8
            sched.encoder = msgspec.msgpack.Encoder()
            sched.remote_poller = zmq.Poller()  # type: ignore[attr-defined]
            sched.remote_sockets = defaultdict(deque)
            sched.remote_sockets_lock = threading.Lock()
            try:
                assert sched._send_start_pull("req-1", {}, port) == b"ACK" and len(sched.remote_sockets[dead_path]) == 1
                with pytest.raises(MooncakePeerLostError):
                    sched._send_start_pull("req-2", {}, port)
                assert dead_path not in sched.remote_sockets
            finally:
                if getattr(sched, "_zmq_context", None) is not None:
                    sched._zmq_context.destroy(linger=0)
        else:
            sched_thread = pull_scheduler.MooncakeSchedulerRecvingThread(threading.Event())
            try:
                sched_thread._send_done_recving("127.0.0.1", port, "req-1")
                assert len(sched_thread.remote_sockets[dead_path]) == 1
                with pytest.raises(MooncakePeerLostError):
                    sched_thread._send_done_recving("127.0.0.1", port, "req-2")
                assert dead_path not in sched_thread.remote_sockets
            finally:
                if sched_thread.zmq_context is not None:
                    sched_thread.zmq_context.destroy(linger=0)
    finally:
        srv.join(timeout=2)


# ---------------------------------------------------------------------------
# 3. Shared Mixin multi-subtask isolation, HMA error surfaces & V2 pull modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["v1_single", "v1_hma", "hybrid_sync", "hybrid_hma"])
@pytest.mark.parametrize("failure_mode", ["peer_lost", "transfer_runtime"])
def test_recv_thread_multi_subtask_isolation_and_error_surface(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    failure_mode: str,
) -> None:
    """Shared Mixin handles multi-subtask DONE skip, single-host isolation & HMA surfaces."""
    is_hma = kind.endswith("_hma")
    is_v1 = kind.startswith("v1")
    thread = _make_connector_thread(kind, use_request_level_recv_errors=is_hma)
    mod: Any = v1_connector if is_v1 else hybrid_connector

    # Populate two hosts sharing port 6000 to verify per-host isolation & fallback scan.
    thread._record_remote_endpoint_engine_id("192.0.2.1", 6000, "engine-a")
    thread._record_remote_endpoint_engine_id("192.0.2.2", 6000, "engine-b")
    for eid, port in (("engine-a", 6000), ("engine-b", 6000), ("engine-c", 6000)):
        thread.kv_caches_base_addr[eid][port] = [[0x1000]] if is_v1 else [0x1000]
        thread.remote_te_port[eid][port] = 9001
    sock_a = MagicMock(spec=zmq.Socket)  # type: ignore[attr-defined]
    sock_b = MagicMock(spec=zmq.Socket)  # type: ignore[attr-defined]
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.1", 6000)].append(sock_a)
    thread.remote_sockets[make_zmq_path("tcp", "192.0.2.2", 6000)].append(sock_b)

    thread._invalidate_remote_metadata(remote_host="192.0.2.3", remote_handshake_port=6000)
    assert "engine-c" not in thread.kv_caches_base_addr and "engine-b" in thread.kv_caches_base_addr

    done_port_spy = MagicMock()
    done_recv_spy = MagicMock()
    monkeypatch.setattr(thread, "_send_done_signal_to_free_remote_port", done_port_spy)
    monkeypatch.setattr(thread, "_send_done_recv_signal", done_recv_spy)
    transfer_method = "_transfer_kv_cache" if kind == "hybrid_sync" else "_transfer_kv_cache_all_groups"
    exc: Exception = (
        MooncakePeerLostError("dead 192.0.2.1") if failure_mode == "peer_lost" else RuntimeError("rdma failed")
    )
    monkeypatch.setattr(thread, transfer_method, MagicMock(side_effect=exc))

    tasks = [
        {
            "request_id": "req-multi",
            "remote_request_id": "p-req-multi",
            "remote_engine_id": "engine-a",
            "remote_host": "192.0.2.1",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": False,
            "local_block_ids": ([10, 11],),
        },
        {
            "request_id": "req-multi",
            "remote_request_id": "p-req-multi",
            "remote_engine_id": "engine-a",
            "remote_host": "192.0.2.1",
            "remote_handshake_port": 6001,
            "remote_port_send_num": {},
            "all_task_done": False,
            "local_block_ids": ([10, 11],),
        },
        {
            "request_id": "req-multi",
            "remote_request_id": "p-req-multi",
            "remote_engine_id": "engine-b",
            "remote_host": "192.0.2.2",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": True,
            "local_block_ids": ([10, 11],),
        },
    ]
    thread.task_tracker.add_req_to_process("req-multi")
    for task in tasks:
        thread._mark_request_task_submitted(task)
    for task in tasks:
        thread._handle_request(task)

    assert "engine-a" not in thread.kv_caches_base_addr and "engine-b" in thread.kv_caches_base_addr
    if failure_mode == "peer_lost":
        assert done_port_spy.call_args_list == [(("p-req-multi", "192.0.2.2", {}),)]
        assert done_recv_spy.call_args_list == [(("p-req-multi", "192.0.2.2", 6000, {}),)]
        sock_a.close.assert_called_once_with()
        assert make_zmq_path("tcp", "192.0.2.1", 6000) not in thread.remote_sockets
    else:
        assert done_port_spy.call_args_list == [
            (("p-req-multi", "192.0.2.1", {}),),
            (("p-req-multi", "192.0.2.1", {}),),
            (("p-req-multi", "192.0.2.2", {}),),
        ]
        assert done_recv_spy.call_args_list == [
            (("p-req-multi", "192.0.2.1", 6000, {}),),
            (("p-req-multi", "192.0.2.1", 6001, {}),),
            (("p-req-multi", "192.0.2.2", 6000, {}),),
        ]
        sock_a.close.assert_not_called()
    assert list(thread.remote_sockets[make_zmq_path("tcp", "192.0.2.2", 6000)]) == [sock_b]
    assert thread.peer_lost_requests == set()

    worker = mod.MooncakeConnectorWorker.__new__(mod.MooncakeConnectorWorker)
    worker.kv_role, worker.tp_rank, worker.kv_recv_thread, worker.kv_send_thread = "kv_consumer", 0, thread, None
    conn = mod.MooncakeConnector.__new__(mod.MooncakeConnector)
    conn._role, conn.connector_worker, conn.connector_scheduler = KVConnectorRole.WORKER, worker, None
    assert conn.get_transfer_results({"req-multi"}) == KVConnectorTransferResults(
        finished_sending=set(),
        finished_recving={"req-multi"},
        failed_recving={"req-multi"} if is_hma else set(),
    )
    assert conn.get_block_ids_with_load_errors() == (set() if is_hma else {10, 11})


@pytest.mark.parametrize("num_groups", [1, 2])
def test_layerwise_hma_get_transfer_results_contract(
    monkeypatch: pytest.MonkeyPatch,
    num_groups: int,
) -> None:
    """Layerwise reports request-level failed_recving under HMA and invalid blocks for 1 group."""
    layerwise_connector = _get_layerwise_connector()
    monkeypatch.setattr(layerwise_connector, "get_ip", lambda: "127.0.0.1")
    agent_meta = layerwise_connector.MooncakeAgentMetadata(te_rpc_port=9001, layer_metadata={})
    recv_thread = layerwise_connector.KVCacheRecvingLayerThread(
        tp_rank=0,
        side_channel_port=5555,
        tp_size=1,
        pd_head_ratio=1,
        local_engine_id="local-d",
        metadata=agent_meta,
        ready_event=threading.Event(),
    )
    recv_thread.update_failed_task("ext-req-fail")
    recv_thread.update_done_task("ext-req-ok", 1, "127.0.0.1:5555")

    worker: Any = layerwise_connector.MooncakeLayerwiseConnectorWorker.__new__(
        layerwise_connector.MooncakeLayerwiseConnectorWorker
    )
    worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False))
    worker.num_kv_cache_groups = num_groups
    worker.kv_recv_layer_thread = recv_thread
    worker.virtual_request = set()
    worker._invalid_block_ids = set()
    worker.request_map = {"ext-req-fail": "ext-req-fail-suffix01", "ext-req-ok": "ext-req-ok-suffix01"}
    worker._recving_metadata = {
        "ext-req-fail-suffix01": SimpleNamespace(local_block_ids=[[11, 12], [21, 22]]),
        "ext-req-ok-suffix01": SimpleNamespace(local_block_ids=[[13], [23]]),
    }

    conn: Any = layerwise_connector.MooncakeLayerwiseConnector.__new__(layerwise_connector.MooncakeLayerwiseConnector)
    conn._role = layerwise_connector.KVConnectorRole.WORKER
    conn.connector_worker = worker
    conn.connector_scheduler = None

    results = conn.get_transfer_results({"ext-req-fail-suffix01", "ext-req-ok-suffix01"})
    blocks = conn.get_block_ids_with_load_errors()
    assert results == layerwise_connector.KVConnectorTransferResults(
        finished_sending=set(),
        finished_recving={"ext-req-fail-suffix01", "ext-req-ok-suffix01"},
        failed_recving={"ext-req-fail-suffix01"} if num_groups > 1 else set(),
    )
    assert blocks == (set() if num_groups > 1 else {11, 12})


@pytest.mark.parametrize("mode", ["request", "batch"])
@pytest.mark.parametrize("policy", ["fail", "recompute"])
def test_v2_pull_modes_evict_metadata_and_scheduler_lease_without_race(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    policy: str,
) -> None:
    """V2 request/batch modes evict metadata cleanly on failure and avoid SizedDict race."""
    thread = _make_v2_pull_thread(mode=mode, can_report_invalid_block_ids=True)
    cached = make_metadata_groups(engine_id="engine-p")
    thread.remote_metadata["engine-p"] = cached
    thread.remote_tp_rank_groups["engine-p"] = {0: {(0, 0): [[0]]}}
    thread.remote_layer_index_pairs["engine-p"] = {0: [(0, 0)]}

    # Concurrent failure on req-1 evicts metadata; req-2 in request mode must fail fast, not silently succeed.
    req1 = _make_req_meta(host=cached.scheduler_host, port=cached.scheduler_port, local_blocks=([31],))
    req2 = _make_req_meta(host=cached.scheduler_host, port=cached.scheduler_port, local_blocks=([32],))
    thread._build_transfer_block_buckets = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda _pp, _lp, _tp, _pcp, _dcp, reqs, *_a, **_kw: (
            {(0, 0): {0: {(0, 0): [(rid, [31], [1]) for rid in reqs]}}},
            {(0, 0): set(reqs)},
        )
    )
    failed_future: Future[None] = Future()
    failed_future.set_exception(RuntimeError("rdma failed"))
    thread.executor = MagicMock()
    thread.executor.submit.return_value = failed_future

    thread.add_requests("engine-p", {"req-1": req1, "req-2": req2})
    thread._process_next()
    results = thread.get_and_clear_finished_requests()
    assert results == KVConnectorTransferResults(finished_recving={"req-1", "req-2"}, failed_recving={"req-1", "req-2"})
    assert thread.get_and_clear_invalid_block_ids() == {31, 32}
    assert "engine-p" not in thread.remote_metadata

    # Endpoint change forces metadata eviction and re-query; scheduler clears failed lease only.
    thread.remote_metadata["engine-p"] = cached
    restarted = MooncakeTransferMetadataGroups(
        engine_id="engine-p",
        scheduler_host=cached.scheduler_host,
        scheduler_port=cached.scheduler_port + 1,
        pp_size=cached.pp_size,
        pcp_size=cached.pcp_size,
        dcp_size=cached.dcp_size,
        tp_size=cached.tp_size,
        use_kv_pp=cached.use_kv_pp,
        metadata_by_pp_rank=cached.metadata_by_pp_rank,
    )
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = MagicMock()
    monkeypatch.setattr(pull_worker, "zmq_ctx", MagicMock(return_value=fake_cm))
    monkeypatch.setattr(pull_worker, "ensure_zmq_send", MagicMock())
    monkeypatch.setattr(pull_worker, "ensure_zmq_recv", MagicMock(return_value=thread.encoder.encode(restarted)))
    thread._build_remote_transfer_layout = MagicMock(  # type: ignore[method-assign]
        return_value=({0: {(0, 0): [[0]]}}, {0: [(0, 0)]})
    )
    assert (
        thread._get_remote_metadata("engine-p", cached.scheduler_host, restarted.scheduler_port).scheduler_port
        == restarted.scheduler_port
    )

    sched = pull_scheduler.MooncakePullConnectorScheduler.__new__(pull_scheduler.MooncakePullConnectorScheduler)
    sched.kv_transfer_config = SimpleNamespace(is_kv_producer=False, is_kv_consumer=True, kv_load_failure_policy=policy)
    sched._recving_thread = MagicMock()
    sched._sending_thread = None
    sched._heartbeat_thread = None
    sched._reqs_recv_info = {"req-1": ("192.0.2.1", 6000, "remote-req-1")}
    sched._validated_producer_leases = TTLCache(maxsize=16, ttl=60)
    sched._validated_producer_leases[("192.0.2.1", 6000)] = ("engine-p", 30)
    sched._validated_producer_leases[("192.0.2.2", 6000)] = ("engine-p-2", 30)
    sched.update_connector_output(
        SimpleNamespace(finished_sending=set(), finished_recving={"req-1"}, failed_recving={"req-1"})  # type: ignore[arg-type]
    )
    sched._recving_thread.add_request.assert_called_once_with("192.0.2.1", 6000, "remote-req-1")
    assert ("192.0.2.1", 6000) not in sched._validated_producer_leases
    assert ("192.0.2.2", 6000) in sched._validated_producer_leases
