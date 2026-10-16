# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Failure-path tests for the NPU simple-KV-offload copy backend.

``NPUDmaCopyBackend._copy_loop`` used to run the whole copy pipeline with no
exception handling. One raising job killed the worker thread, so the ordered
event list stopped growing, the high-water mark froze and every later transfer
was stranded with no log line. The thread must survive, the failure must be
recorded, and ``get_finished`` must not report the missing transfer as done.

Everything here runs on CPU with the ``torch.npu`` mocks that
``tests/ut/conftest.py`` installs.
"""

import contextlib
import queue
import time
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple import (
    copy_backend as copy_backend_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.copy_backend import (
    NPUDmaCopyBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.worker import (
    SimpleCPUOffloadNPUWorker,
)


class FakeEvent:
    def __init__(self) -> None:
        self.recorded_streams: list[object] = []

    def record(self, stream) -> None:
        self.recorded_streams.append(stream)

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class FakeStream:
    def __init__(self) -> None:
        self.waited_events: list[object] = []

    def wait_event(self, event) -> None:
        self.waited_events.append(event)


@contextlib.contextmanager
def _fake_stream_context(stream):
    yield stream


@pytest.fixture
def fake_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda device: None,
            stream=_fake_stream_context,
            Event=FakeEvent,
            current_stream=lambda: FakeStream(),
        ),
        raising=False,
    )


def _build_backend(monkeypatch: pytest.MonkeyPatch) -> NPUDmaCopyBackend:
    monkeypatch.setattr(copy_backend_module, "build_params", lambda *args: object())
    backend = NPUDmaCopyBackend()
    backend.init({}, {}, torch.device("cpu"), FakeStream(), FakeStream())
    return backend


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_copy_loop_survives_a_failing_job(
    monkeypatch: pytest.MonkeyPatch,
    fake_npu: None,
) -> None:
    """A raising job must not kill the thread or hide the later transfers."""
    backend = _build_backend(monkeypatch)
    attempts: list[int] = []

    def flaky_copy_blocks(src_blocks, dst_blocks, params) -> None:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise RuntimeError("simulated DMA failure")

    monkeypatch.setattr(copy_backend_module, "copy_blocks", flaky_copy_blocks)

    events: list[tuple[int, object]] = []
    try:
        backend.launch_copy([], [], is_store=False, event_idx=1, events_list=events)
        backend.launch_copy([], [], is_store=False, event_idx=2, events_list=events)

        assert _wait_for(lambda: len(events) == 1), "the job after the failing one never ran"
    finally:
        backend.shutdown()

    assert attempts == [0, 1], "the loop stopped after the first failure"
    assert backend.failed_event_indices == {1}
    assert [event_idx for event_idx, _ in events] == [2]
    assert isinstance(events[0][1], FakeEvent)


def test_copy_loop_keeps_running_after_every_job_fails(
    monkeypatch: pytest.MonkeyPatch,
    fake_npu: None,
) -> None:
    """Repeated failures are all recorded and none of them is reported as done."""
    backend = _build_backend(monkeypatch)

    def always_failing(src_blocks, dst_blocks, params) -> None:
        raise RuntimeError("simulated DMA failure")

    monkeypatch.setattr(copy_backend_module, "copy_blocks", always_failing)

    events: list[tuple[int, object]] = []
    try:
        for event_idx in (1, 2, 3):
            backend.launch_copy([], [], is_store=False, event_idx=event_idx, events_list=events)
        assert _wait_for(lambda: backend.failed_event_indices == {1, 2, 3})
    finally:
        backend.shutdown()

    assert events == []


def test_get_finished_raises_when_a_copy_job_failed() -> None:
    """The worker must fail loudly instead of releasing an unfinished load."""

    class FailingBackend:
        @property
        def failed_event_indices(self) -> set[int]:
            return {7}

    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = FailingBackend()
    worker._connector_metadata = None
    worker._pending_load_event_indices = set()
    worker._pending_store_event_indices = set()
    worker._load_events = []
    worker._store_events = []
    worker._completed_store_events = {}

    with pytest.raises(RuntimeError, match=r"event indices \[7\]"):
        worker.get_finished(set())


def test_get_finished_unchanged_for_a_clean_backend() -> None:
    """A clean backend keeps the existing behaviour."""

    class CleanBackend:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def launch_copy(self, *args, **kwargs) -> None:
            self.calls.append(kwargs)

        @property
        def failed_event_indices(self) -> set[int]:
            return set()

    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    worker._backend = CleanBackend()
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        load_event=1,
        load_gpu_blocks=[2],
        load_cpu_blocks=[3],
        store_event=4,
        store_gpu_blocks=[5],
        store_cpu_blocks=[6],
    )
    worker._store_compute_done = None
    worker._load_events = []
    worker._store_events = []
    worker._pending_load_event_indices = set()
    worker._pending_store_event_indices = set()
    worker._completed_store_events = {}

    assert worker.get_finished(set()) == (None, None)


def test_shutdown_warns_when_the_thread_outlives_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker thread that ignores the sentinel must be visible in the logs."""

    class StuckThread:
        joined_with: float | None = None

        def join(self, timeout=None) -> None:
            self.joined_with = timeout

        def is_alive(self) -> bool:
            return True

    warnings: list[str] = []
    monkeypatch.setattr(copy_backend_module.logger, "warning", lambda message, *args: warnings.append(message))

    backend = NPUDmaCopyBackend()
    backend._queue = queue.SimpleQueue()
    backend._thread = StuckThread()
    backend.shutdown()

    assert backend._thread.joined_with == 5.0
    assert warnings == ["NPU KV-offload copy thread did not stop within the shutdown timeout."]
