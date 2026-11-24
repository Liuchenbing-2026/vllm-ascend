# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared utilities for Mooncake KV transfer connectors."""

import contextlib
import hashlib
import inspect
import math
import queue
import struct
import threading
import time
import unittest.mock
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast

import httpx
import numpy as np
import numpy.typing as npt
import torch
import zmq
from vllm import envs
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

from vllm_ascend.distributed.kv_transfer.utils.utils import (
    RegisterRegions,
    tensor_storage_key,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers, vllm_version_is

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


KV_CACHE_BUFFER_ALIGNMENT = 2 * 1024 * 1024
# A pull reply can only arrive on the connection that sent the request: a
# closed/disconnected peer drops pending replies and refused reconnects mean
# nothing is listening at that endpoint.
PULL_LOST_PEER_EVENTS: Final[int] = int(getattr(zmq, "EVENT_DISCONNECTED", 0)) | int(
    getattr(zmq, "EVENT_CONNECT_RETRIED", 0)
)
# ZMTP heartbeats detect frozen processes or unreachable hosts.
# Keep a single pull receive timeout strictly above IVL + TIMEOUT so a frozen
# peer triggers EVENT_DISCONNECTED (MooncakePeerLostError) before recv times
# out with a generic RuntimeError.
PULL_HEARTBEAT_IVL_MS: Final[int] = 1000
PULL_HEARTBEAT_TIMEOUT_MS: Final[int] = 3000
DEFAULT_PULL_ZMQ_TIMEOUT_MS: Final[int] = 5000
_ZMQ_MONITOR_EVENT_STRUCT_SIZE: Final[int] = 6


class _DefaultOption(Enum):
    UNSET = "unset"


class MooncakePeerLostError(RuntimeError):
    """Raised when a prefill ZMQ peer disconnects or refuses connections."""


class MooncakeMetaserverClient:
    """Submit D metadata in the background and surface failures to the scheduler."""

    def __init__(self) -> None:
        # The metaserver responds after P finishes, not when it accepts D's
        # metadata. Allow the request lifetime for that response.
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                connect=envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT,
            )
        )
        # Each D DP engine has its own pool: at most 32 metaserver POSTs can
        # run concurrently, and each occupies a thread until P finishes.
        # TODO: decouple metadata acceptance from P completion. A multi-prompt
        # batch can exhaust this pool while the proxy waits for its remaining
        # queued metadata, causing all involved POSTs to wait until timeout.
        self.executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="MooncakeMetaserver")
        self._results: queue.SimpleQueue = queue.SimpleQueue()

    def submit(self, request_id: str, url: str | None, message: dict[str, Any]) -> None:
        try:
            if not isinstance(url, str):
                raise ValueError("Metaserver URL is missing or is not a string")
            parsed_url = httpx.URL(url)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.host:
                raise ValueError("Metaserver requires an absolute HTTP(S) URL")
        except (TypeError, ValueError, httpx.InvalidURL) as error:
            self._results.put((request_id, error, True))
            return

        # submit() can enqueue work before starting a thread raises. Gate the
        # HTTP call so a rejected submission can never publish D block addresses.
        submitted = threading.Event()
        accepted = False

        def send() -> None:
            submitted.wait()
            if accepted:
                self._access_metaserver(url, message)

        try:
            future = self.executor.submit(send)
        except RuntimeError as error:
            self._results.put((request_id, error, True))
        else:
            accepted = True
            future.add_done_callback(lambda result: self._on_done(request_id, result))
        finally:
            submitted.set()

    def _access_metaserver(self, url: str, message: dict[str, Any]) -> None:
        # Only connection establishment is retried: a lost response may follow
        # a successful P dispatch, so do not replay the request after a timeout.
        for attempt in range(3):
            try:
                response = self.client.post(url, json=message)
                response.raise_for_status()
                return
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if attempt == 2:
                    raise

    def _on_done(self, request_id: str, future: Future) -> None:
        if not future.cancelled() and (error := future.exception()) is not None:
            not_dispatched = isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
            self._results.put((request_id, error, not_dispatched))

    def get_failed_requests(self) -> dict[str, bool]:
        """Return failed D requests and whether P dispatch can be ruled out."""
        failed_requests: dict[str, bool] = {}
        while True:
            try:
                request_id, error, not_dispatched = self._results.get_nowait()
            except queue.Empty:
                return failed_requests
            logger.error("Mooncake metaserver dispatch failed for %s: %s", request_id, error)
            # Preserve uncertainty if more than one failure was recorded.
            failed_requests[request_id] = failed_requests.get(request_id, True) and not_dispatched

    def shutdown(self) -> None:
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self.client.close()


class _PreConnectOptionContext:
    """Wrap a ZMQ context to set socket options before connect() runs.

    Older vLLM pins (`v0.33.0` and `c410543d`) do not expose heartbeat
    parameters on `make_zmq_socket`, and libzmq ignores heartbeat options set
    after `connect()`.
    """

    def __init__(
        self,
        ctx: Any,
        *,
        linger: int | None,
        heartbeat_ivl_ms: int | None,
        heartbeat_timeout_ms: int | None,
    ) -> None:
        self._ctx = ctx
        self._linger = linger
        self._heartbeat_ivl_ms = heartbeat_ivl_ms
        self._heartbeat_timeout_ms = heartbeat_timeout_ms

    def socket(self, socket_type: Any) -> Any:
        sock = self._ctx.socket(socket_type)
        _apply_pre_connect_options(
            sock,
            linger=self._linger,
            heartbeat_ivl_ms=self._heartbeat_ivl_ms,
            heartbeat_timeout_ms=self._heartbeat_timeout_ms,
        )
        return sock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)


def _apply_pre_connect_options(
    sock: Any,
    *,
    linger: int | None,
    heartbeat_ivl_ms: int | None,
    heartbeat_timeout_ms: int | None,
) -> None:
    """Apply LINGER and ZMTP heartbeat options prior to connect()."""
    if linger is not None and hasattr(zmq, "LINGER"):
        sock.setsockopt(zmq.LINGER, linger)  # type: ignore[attr-defined]
    if heartbeat_ivl_ms is not None and hasattr(zmq, "HEARTBEAT_IVL"):
        sock.setsockopt(zmq.HEARTBEAT_IVL, heartbeat_ivl_ms)  # type: ignore[attr-defined]
    if heartbeat_timeout_ms is not None and hasattr(zmq, "HEARTBEAT_TIMEOUT"):
        sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, heartbeat_timeout_ms)  # type: ignore[attr-defined]


def _make_zmq_socket_supports_heartbeat(make_socket_fn: Any) -> bool:
    """Check whether make_zmq_socket explicitly accepts heartbeat kwargs."""
    try:
        params = inspect.signature(make_socket_fn).parameters
    except (TypeError, ValueError):
        return False
    return "heartbeat_ivl_ms" in params and "heartbeat_timeout_ms" in params


def _make_zmq_socket_accepts_param(make_socket_fn: Any, param_name: str) -> bool:
    """Check whether a socket factory accepts a named keyword argument."""
    try:
        params = inspect.signature(make_socket_fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(param.name == param_name or param.kind == inspect.Parameter.VAR_KEYWORD for param in params)


def make_mooncake_client_socket(
    ctx: Any,
    path: str,
    socket_type: Any,
    *,
    linger: int | None = 0,
    heartbeat_ivl_ms: int | None | _DefaultOption = _DefaultOption.UNSET,
    heartbeat_timeout_ms: int | None | _DefaultOption = _DefaultOption.UNSET,
    make_socket_fn: Any | None = None,
) -> Any:
    """Create a client ZMQ socket with pre-connect LINGER and heartbeat options."""
    resolved_hb_ivl = PULL_HEARTBEAT_IVL_MS if heartbeat_ivl_ms is _DefaultOption.UNSET else heartbeat_ivl_ms
    resolved_hb_timeout = (
        PULL_HEARTBEAT_TIMEOUT_MS if heartbeat_timeout_ms is _DefaultOption.UNSET else heartbeat_timeout_ms
    )
    factory = make_zmq_socket if make_socket_fn is None else make_socket_fn
    if _make_zmq_socket_supports_heartbeat(factory):
        return factory(
            ctx=ctx,
            path=path,
            socket_type=socket_type,
            bind=False,
            linger=linger,
            heartbeat_ivl_ms=resolved_hb_ivl,
            heartbeat_timeout_ms=resolved_hb_timeout,
        )

    wrapped_ctx = _PreConnectOptionContext(
        ctx,
        linger=linger,
        heartbeat_ivl_ms=resolved_hb_ivl,
        heartbeat_timeout_ms=resolved_hb_timeout,
    )
    kwargs: dict[str, Any] = {
        "ctx": wrapped_ctx,
        "path": path,
        "socket_type": socket_type,
        "bind": False,
    }
    if linger is not None and _make_zmq_socket_accepts_param(factory, "linger"):
        kwargs["linger"] = linger
    return factory(**kwargs)


def close_zmq_socket(sock: Any) -> None:
    """Close a ZMQ socket with LINGER=0 without blocking on dead peers."""
    if sock is None:
        return
    try:
        sock.close(linger=0)
    except TypeError:
        sock.close()


@contextlib.contextmanager
def zmq_ctx(
    socket_type: Any,
    addr: str,
    *,
    linger: int = 0,
    heartbeat_ivl_ms: int | None | _DefaultOption = _DefaultOption.UNSET,
    heartbeat_timeout_ms: int | None | _DefaultOption = _DefaultOption.UNSET,
) -> Iterator[Any]:
    """Create a Mooncake ROUTER, REQ, or DEALER socket and clean up its context."""
    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore[attr-defined]
        raise ValueError(f"Unexpected socket type: {socket_type}")

    context: Any | None = None
    try:
        context = zmq.Context()  # type: ignore[attr-defined]
        if socket_type == zmq.ROUTER:  # type: ignore[attr-defined]
            yield make_zmq_socket(
                ctx=context,
                path=addr,
                socket_type=socket_type,
                bind=True,
                linger=linger,
            )
        else:
            yield make_mooncake_client_socket(
                context,
                addr,
                socket_type,
                linger=linger,
                heartbeat_ivl_ms=heartbeat_ivl_ms,
                heartbeat_timeout_ms=heartbeat_timeout_ms,
                make_socket_fn=make_zmq_socket,
            )
    finally:
        if context is not None:
            context.destroy(linger=0)


def as_kv_cache_tensors(cache_or_caches: Any) -> tuple[torch.Tensor, ...]:
    """Normalize one layer's KV cache into a tuple of tensors."""
    if isinstance(cache_or_caches, torch.Tensor):
        return (cache_or_caches,)
    if isinstance(cache_or_caches, (list, tuple)) and all(isinstance(cache, torch.Tensor) for cache in cache_or_caches):
        return tuple(cache_or_caches)
    raise TypeError(
        f"A layer KV cache must be a tensor or a list/tuple of tensors, but got {type(cache_or_caches).__name__}."
    )


def _get_storage_nbytes(tensor: torch.Tensor) -> int:
    """Return the byte size of the allocation backing tensor."""
    try:
        return tensor.untyped_storage().nbytes()
    except Exception:
        try:
            return tensor.storage().nbytes()
        except Exception:
            return tensor.nbytes


def _get_tensor_span_nbytes(tensor: torch.Tensor) -> int:
    """Return the physical byte span touched by a strided tensor view."""
    if tensor.numel() == 0:
        return 0
    return tensor.element_size() + sum(
        (size - 1) * stride * tensor.element_size() for size, stride in zip(tensor.shape, tensor.stride())
    )


def collect_configured_register_regions(
    kv_cache_config: "KVCacheConfig",
    kv_caches: dict[str, Any],
) -> RegisterRegions:
    """Collect one registration range per configured backing storage."""
    ranges_by_storage: dict[int, tuple[int, int]] = {}
    configured_tensor_count = 0

    def merge_storage_range(
        storage_key: int,
        register_start: int,
        register_end: int,
    ) -> None:
        previous_range = ranges_by_storage.get(storage_key)
        if previous_range is None:
            ranges_by_storage[storage_key] = (register_start, register_end)
        else:
            ranges_by_storage[storage_key] = (
                min(previous_range[0], register_start),
                max(previous_range[1], register_end),
            )

    for tensor_config in kv_cache_config.kv_cache_tensors:
        layer_names = get_kv_cache_tensor_layers(tensor_config)
        if not layer_names:
            continue

        cache_tensors: list[torch.Tensor] = []
        for layer_name in layer_names:
            cache_tensors.extend(cache for cache in as_kv_cache_tensors(kv_caches.get(layer_name)) if cache.numel() > 0)

        caches_by_storage: dict[int, list[torch.Tensor]] = {}
        for cache in cache_tensors:
            storage_key = tensor_storage_key(cache)
            caches_by_storage.setdefault(storage_key, []).append(cache)

        if len(caches_by_storage) == 1:
            # The model runner over-allocates by one alignment unit and returns
            # views into the aligned suffix of that raw storage. ``offset`` is
            # relative to the standardized backing, not necessarily to the
            # first logical view (for example, compressor scale follows K in a
            # padded page), so recover the backing from the storage boundary.
            storage_key, storage_caches = next(iter(caches_by_storage.items()))
            storage_end = storage_key + _get_storage_nbytes(storage_caches[0])
            aligned_start = (
                (storage_key + KV_CACHE_BUFFER_ALIGNMENT - 1) // KV_CACHE_BUFFER_ALIGNMENT * KV_CACHE_BUFFER_ALIGNMENT
            )
            configured_end = aligned_start + tensor_config.size
            views_fit_configured_range = all(
                aligned_start <= cache.data_ptr()
                and cache.data_ptr() + _get_tensor_span_nbytes(cache) <= configured_end
                for cache in storage_caches
            )
            if configured_end <= storage_end and views_fit_configured_range:
                merge_storage_range(
                    storage_key,
                    aligned_start,
                    configured_end,
                )
            else:
                # Cache types that allocate components independently do not
                # expose the standardized shared backing. Register the actual
                # allocation range instead.
                register_start = min(cache.data_ptr() for cache in storage_caches)
                if storage_end <= register_start:
                    raise ValueError(f"Invalid KV cache storage range: start={register_start}, end={storage_end}.")
                merge_storage_range(storage_key, register_start, storage_end)
        else:
            # Some cache types initialize component tensors independently.
            # The config size cannot be applied to every storage, so register
            # the actual range of each allocation.
            for storage_key, storage_caches in caches_by_storage.items():
                register_start = min(cache.data_ptr() for cache in storage_caches)
                register_end = max(storage_key + _get_storage_nbytes(cache) for cache in storage_caches)
                if register_end <= register_start:
                    raise ValueError(f"Invalid KV cache storage range: start={register_start}, end={register_end}.")
                merge_storage_range(storage_key, register_start, register_end)

        configured_tensor_count += 1

    if not ranges_by_storage:
        raise ValueError("KV cache config contains no registerable tensors.")

    ptrs = [region[0] for region in ranges_by_storage.values()]
    lengths = [region[1] - region[0] for region in ranges_by_storage.values()]
    return RegisterRegions(
        ptrs=ptrs,
        lengths=lengths,
        logical_tensor_count=configured_tensor_count,
        logical_total_bytes=sum(lengths),
    )


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


@dataclass
class SizedDict(OrderedDict[_KT, _VT]):
    """Insertion-ordered mapping with a bounded number of entries."""

    def __init__(self, max_size: int = 16000, *args: Any, **kwargs: Any) -> None:
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: _KT, value: _VT) -> None:
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key: _KT) -> _VT:
        try:
            return super().__getitem__(key)
        except KeyError:
            value = cast(_VT, {})
            self[key] = value
            return value


def worker_rank_to_dcp_rank(worker_rank: int, pcp_size: int, dcp_size: int) -> int:
    """Return a Mooncake worker's rank within its vLLM DCP group."""
    worker_rank_in_group = worker_rank % dcp_size
    # todo v0.33.0 and the verified main pin (c410543d) enumerate PCP first.
    # Remove this gate once both pins include vLLM #64534.
    if not vllm_version_is("0.34.0"):
        return worker_rank_in_group

    # vLLM sorts each DCP group by ascending global rank (https://github.com/vllm-project/vllm/pull/64534).
    #     group_ranks = [sorted(x.tolist()) for x in group_ranks]
    # PCP varies fastest in Mooncake worker IDs and TP varies fastest in global rank order.
    tp_rank_in_group, pcp_rank_in_group = divmod(worker_rank_in_group, pcp_size)
    tp_size_in_group = dcp_size // pcp_size
    return pcp_rank_in_group * tp_size_in_group + tp_rank_in_group


def group_concurrent_contiguous(
    src: list[int],
    dst: list[int],
    src_block_stride: int = 1,
    dst_block_stride: int = 1,
    block_len: int = 1,
) -> tuple[list[list[int]], list[list[int]]]:
    """Group block ids that are contiguous in both id space and memory."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    src_byte_contiguous = np.diff(src_indices) * src_block_stride == block_len
    dst_byte_contiguous = np.diff(dst_indices) * dst_block_stride == block_len
    brk = np.where(~(src_byte_contiguous & dst_byte_contiguous))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def split_if_not_byte_contiguous(
    src_groups: list[list[int]],
    dst_groups: list[list[int]],
    src_block_stride: int,
    dst_block_stride: int,
    block_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if src_block_stride == block_len and dst_block_stride == block_len:
        return src_groups, dst_groups

    src = [bid for group in src_groups for bid in group]
    dst = [bid for group in dst_groups for bid in group]
    return group_concurrent_contiguous(
        src,
        dst,
        src_block_stride=src_block_stride,
        dst_block_stride=dst_block_stride,
        block_len=block_len,
    )


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def _supports_socket_monitor(sock: Any) -> bool:
    """Return True if sock is a real/non-mock socket supporting ZMQ monitors."""
    if isinstance(sock, unittest.mock.NonCallableMock):
        return False
    return callable(getattr(sock, "get_monitor_socket", None))


@contextlib.contextmanager
def lost_peer_monitor(
    sock: Any,
    events: int = PULL_LOST_PEER_EVENTS,
) -> Iterator[Any | None]:
    """Attach a ZMQ socket monitor for disconnect and refused-connect events."""
    if not _supports_socket_monitor(sock):
        yield None
        return

    monitor = sock.get_monitor_socket(events)
    try:
        yield monitor
    finally:
        try:
            disable_monitor = getattr(sock, "disable_monitor", None)
            if callable(disable_monitor):
                with contextlib.suppress(zmq.ZMQError):  # type: ignore[attr-defined]
                    disable_monitor()
        finally:
            close_zmq_socket(monitor)


def _is_monitor_stopped_event(frames: Any) -> bool:
    """Return True when a 6-byte ZMQ monitor event frame is EVENT_MONITOR_STOPPED."""
    if not frames or not isinstance(frames[0], (bytes, bytearray)) or len(frames[0]) != _ZMQ_MONITOR_EVENT_STRUCT_SIZE:
        return False
    event_id = struct.unpack("=H", frames[0][:2])[0]
    stopped_event = int(getattr(zmq, "EVENT_MONITOR_STOPPED", 1024))
    return event_id == stopped_event and (event_id & PULL_LOST_PEER_EVENTS) == 0


def _format_monitor_event(frames: Any) -> str:
    """Format a ZMQ monitor frame into a human-readable event label."""
    if not frames or not isinstance(frames[0], (bytes, bytearray)) or len(frames[0]) < 2:
        return "unknown_event"
    event_id = struct.unpack("=H", frames[0][:2])[0]
    if event_id == getattr(zmq, "EVENT_DISCONNECTED", None):
        return "EVENT_DISCONNECTED"
    if event_id == getattr(zmq, "EVENT_CONNECT_RETRIED", None):
        return "EVENT_CONNECT_RETRIED"
    return f"event={event_id}"


def _validate_timeout_ms(timeout_ms: int | None) -> int | None:
    """Validate an explicit timeout_ms argument while rejecting booleans."""
    if timeout_ms is None:
        return None
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
        raise ValueError(f"timeout_ms must be an int or None, got {timeout_ms!r} ({type(timeout_ms).__name__})")
    return timeout_ms if timeout_ms >= 0 else None


def _resolve_recv_timeout_ms(sock: Any, timeout_ms: int | None) -> int | None:
    """Resolve an effective non-negative receive timeout in milliseconds."""
    if timeout_ms is not None:
        return _validate_timeout_ms(timeout_ms)
    getsockopt = getattr(sock, "getsockopt", None)
    if not callable(getsockopt) or not hasattr(zmq, "RCVTIMEO"):
        return None
    try:
        raw_timeout = getsockopt(zmq.RCVTIMEO)  # type: ignore[attr-defined]
    except Exception:
        return None
    if isinstance(raw_timeout, int) and not isinstance(raw_timeout, bool) and raw_timeout >= 0:
        return raw_timeout
    return None


def recv_unless_lost(
    sock: Any,
    monitor: Any,
    path: str,
    *,
    timeout_ms: int | None = None,
) -> bytes:
    """Return the peer's next reply, or raise MooncakePeerLostError once lost."""
    if monitor is None:
        return sock.recv()

    effective_timeout_ms = _resolve_recv_timeout_ms(sock, timeout_ms)
    deadline = None if effective_timeout_ms is None else time.monotonic() + effective_timeout_ms / 1000.0
    poller = zmq.Poller()  # type: ignore[attr-defined]
    poller.register(sock, zmq.POLLIN)  # type: ignore[attr-defined]
    poller.register(monitor, zmq.POLLIN)  # type: ignore[attr-defined]
    first_poll = True
    try:
        while True:
            if deadline is None:
                poll_timeout_ms: int | None = None
            elif first_poll and effective_timeout_ms == 0:
                poll_timeout_ms = 0
            else:
                remaining_ms = math.ceil((deadline - time.monotonic()) * 1000.0)
                if remaining_ms <= 0:
                    raise zmq.Again()  # type: ignore[attr-defined]
                poll_timeout_ms = remaining_ms
            first_poll = False

            events = dict(poller.poll(poll_timeout_ms))
            if not events:
                raise zmq.Again()  # type: ignore[attr-defined]
            # Check sock before monitor so a completed reply arriving right
            # before the peer disconnects is still returned cleanly.
            if sock in events:
                try:
                    return sock.recv(flags=zmq.NOBLOCK)  # type: ignore[attr-defined]
                except zmq.Again:  # type: ignore[attr-defined]
                    pass
            if monitor in events:
                try:
                    frames = monitor.recv_multipart(flags=zmq.NOBLOCK)  # type: ignore[attr-defined]
                except zmq.Again:  # type: ignore[attr-defined]
                    continue
                if not _is_monitor_stopped_event(frames):
                    raise MooncakePeerLostError(
                        f"Lost connection to Mooncake peer {path} ({_format_monitor_event(frames)})"
                    )
    finally:
        with contextlib.suppress(KeyError, zmq.ZMQError):  # type: ignore[attr-defined]
            poller.unregister(monitor)
        with contextlib.suppress(KeyError, zmq.ZMQError):  # type: ignore[attr-defined]
            poller.unregister(sock)


def _recv_with_retries(
    socket: Any,
    path: str,
    max_retries: int,
    *,
    monitor: Any | None,
    timeout_ms: int | None,
) -> bytes:
    """Receive ZMQ data with retries on transient ZMQError, failing fast on lost peer."""
    retries_left = max_retries
    while True:
        try:
            if monitor is not None:
                return recv_unless_lost(socket, monitor, path, timeout_ms=timeout_ms)
            return socket.recv()
        except MooncakePeerLostError as e:
            logger.error("Receive aborted on lost peer. source=%s, error=%s. ", path, e)
            raise
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed after all retries. source=%s, error=%s. ", path, e)
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}") from e


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    path: str,
    max_retries: int = 3,
    *,
    monitor: Any | None = None,
    timeout_ms: int | None = None,
    detect_peer_lost: bool = False,
) -> bytes:
    """Receive a ZMQ reply, optionally failing fast via socket monitor when opted in."""
    _validate_timeout_ms(timeout_ms)
    if monitor is not None:
        return _recv_with_retries(socket, path, max_retries, monitor=monitor, timeout_ms=timeout_ms)
    if detect_peer_lost and _supports_socket_monitor(socket):
        with lost_peer_monitor(socket) as local_monitor:
            return _recv_with_retries(socket, path, max_retries, monitor=local_monitor, timeout_ms=timeout_ms)
    return _recv_with_retries(socket, path, max_retries, monitor=None, timeout_ms=timeout_ms)


class MooncakeRemotePeerMixin:
    """Shared ZMQ client socket pool and remote metadata invalidation helpers."""

    remote_sockets: dict[str, deque[Any]]
    remote_sockets_lock: threading.Lock
    remote_metadata_lock: threading.Lock
    _remote_metadata_map_attrs: tuple[str, ...] = ()

    @staticmethod
    def _close_socket(sock: Any, *, poller: Any | None = None) -> None:
        if sock is None:
            return
        if poller is not None and not isinstance(sock, unittest.mock.NonCallableMock):
            with contextlib.suppress(KeyError, zmq.ZMQError):  # type: ignore[attr-defined]
                poller.unregister(sock)
        if isinstance(sock, unittest.mock.NonCallableMock):
            sock.close()
        else:
            close_zmq_socket(sock)

    def _get_remote_sockets_lock(self) -> threading.Lock:
        lock = getattr(self, "remote_sockets_lock", None)
        if lock is None:
            lock = threading.Lock()
            self.remote_sockets_lock = lock
        return lock

    def _get_remote_sockets_map(self) -> dict[str, deque[Any]]:
        sockets: dict[str, deque[Any]] | None = getattr(self, "remote_sockets", None)
        if sockets is None:
            sockets = defaultdict(deque)
            self.remote_sockets = sockets
        return sockets

    def _get_zmq_context(self, context_factory: Any = None) -> Any:
        ctx = getattr(self, "_zmq_context", None)
        if ctx is None:
            factory = zmq.Context if context_factory is None else context_factory  # type: ignore[attr-defined]
            ctx = factory()
            self._zmq_context = ctx
        return ctx

    def _borrow_remote_socket(
        self,
        remote_host: str,
        remote_handshake_port: int,
        *,
        timeout_ms: int,
        context_factory: Any = None,
        client_socket_fn: Any = None,
        make_socket_fn: Any = make_zmq_socket,
        heartbeat_ivl_ms: int | None | _DefaultOption = _DefaultOption.UNSET,
        heartbeat_timeout_ms: int | None | _DefaultOption = _DefaultOption.UNSET,
        register_poller: Any | None = None,
    ) -> Any:
        """Borrow a pooled REQ socket or open one with pre-connect options."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self._get_remote_sockets_lock():
            sockets = self._get_remote_sockets_map()
            if sockets[remote_path]:
                return sockets[remote_path].popleft()

            ctx = self._get_zmq_context(context_factory=context_factory)
            socket_builder = make_mooncake_client_socket if client_socket_fn is None else client_socket_fn
            sock = socket_builder(
                ctx,
                remote_path,
                zmq.REQ,  # type: ignore[attr-defined]
                linger=0,
                heartbeat_ivl_ms=heartbeat_ivl_ms,
                heartbeat_timeout_ms=heartbeat_timeout_ms,
                make_socket_fn=make_socket_fn,
            )
            sock.setsockopt(zmq.SNDTIMEO, timeout_ms)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)  # type: ignore[attr-defined]
            if register_poller is not None:
                register_poller.register(sock, zmq.POLLIN)  # type: ignore[attr-defined]
            return sock

    def _return_remote_socket(
        self,
        sock: Any,
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return a healthy client socket to the endpoint pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self._get_remote_sockets_lock():
            self._get_remote_sockets_map()[remote_path].append(sock)

    def _discard_remote_sockets(
        self,
        remote_host: str,
        remote_handshake_port: int,
        *,
        poller: Any | None = None,
    ) -> None:
        """Close and drop pooled REQ sockets for a lost remote peer."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self._get_remote_sockets_lock():
            sockets = self._get_remote_sockets_map().pop(remote_path, None)
        if not sockets:
            return
        target_poller = getattr(self, "remote_poller", None) if poller is None else poller
        for sock in sockets:
            self._close_socket(sock, poller=target_poller)

    def _get_remote_metadata_lock(self) -> threading.Lock:
        lock = getattr(self, "remote_metadata_lock", None)
        if lock is None:
            lock = threading.Lock()
            self.remote_metadata_lock = lock
        return lock

    def _get_remote_endpoint_engine_ids(self) -> defaultdict[tuple[str, int], set[str]]:
        endpoint_map = getattr(self, "_remote_endpoint_engine_ids", None)
        if endpoint_map is None:
            endpoint_map = defaultdict(set)
            self._remote_endpoint_engine_ids = endpoint_map
        return endpoint_map

    def _record_remote_endpoint_engine_id(
        self,
        remote_host: str,
        remote_handshake_port: int,
        engine_id: str,
    ) -> None:
        self._get_remote_endpoint_engine_ids()[(remote_host, remote_handshake_port)].add(engine_id)

    def _get_remote_metadata_maps(self) -> tuple[dict[str, dict[int, Any]], ...]:
        return tuple(
            cache_map
            for attr in self._remote_metadata_map_attrs
            if isinstance((cache_map := getattr(self, attr, None)), dict)
        )

    def _get_local_engine_id_for_invalidation(self) -> str | None:
        local_id = getattr(self, "local_engine_id", None)
        if isinstance(local_id, str) and local_id.strip():
            return local_id
        engine_id = getattr(self, "engine_id", None)
        return engine_id if isinstance(engine_id, str) else None

    def _collect_invalidation_engine_ids(
        self,
        metadata_maps: tuple[dict[str, dict[int, Any]], ...],
        remote_engine_id: str | None,
        remote_host: str | None,
        remote_handshake_port: int | None,
    ) -> set[str]:
        local_engine_id = self._get_local_engine_id_for_invalidation()
        endpoint_map = self._get_remote_endpoint_engine_ids()
        engine_ids: set[str] = set()
        if remote_engine_id is not None and remote_engine_id != local_engine_id:
            engine_ids.add(remote_engine_id)
        if remote_host is not None and remote_handshake_port is not None:
            for eid in endpoint_map.pop((remote_host, remote_handshake_port), ()):
                if eid != local_engine_id:
                    engine_ids.add(eid)
        if not engine_ids and remote_engine_id is None and remote_handshake_port is not None:
            other_host_eids: set[str] = (
                {
                    eid
                    for (host, port), eids in endpoint_map.items()
                    if port == remote_handshake_port and host != remote_host
                    for eid in eids
                }
                if remote_host is not None
                else set()
            )
            for cache_map in metadata_maps:
                for engine_id, ports in list(cache_map.items()):
                    if (
                        engine_id != local_engine_id
                        and engine_id not in other_host_eids
                        and remote_handshake_port in ports
                    ):
                        engine_ids.add(engine_id)
        return engine_ids

    def _invalidate_remote_metadata(
        self,
        remote_engine_id: str | None = None,
        *,
        remote_host: str | None = None,
        remote_handshake_port: int | None = None,
    ) -> None:
        """Evict cached remote metadata for a lost or restarted producer."""
        metadata_maps = self._get_remote_metadata_maps()
        with self._get_remote_metadata_lock():
            endpoint_map = self._get_remote_endpoint_engine_ids()
            engine_ids = self._collect_invalidation_engine_ids(
                metadata_maps, remote_engine_id, remote_host, remote_handshake_port
            )
            for engine_id in engine_ids:
                for cache_map in metadata_maps:
                    cache_map.pop(engine_id, None)
                for endpoint, eids in list(endpoint_map.items()):
                    eids.discard(engine_id)
                    if not eids:
                        endpoint_map.pop(endpoint, None)

    def _handle_remote_zmq_error(
        self,
        sock: Any,
        remote_host: str,
        remote_handshake_port: int,
        *,
        peer_lost: bool,
        poller: Any | None = None,
    ) -> Any:
        """Close the failed socket, flush its endpoint pool, and evict metadata."""
        target_poller = getattr(self, "remote_poller", None) if poller is None else poller
        if peer_lost:
            if sock is not None:
                self._close_socket(sock, poller=target_poller)
                sock = None
            self._discard_remote_sockets(remote_host, remote_handshake_port, poller=target_poller)
        elif isinstance(sock, zmq.Socket):  # type: ignore[attr-defined]
            self._close_socket(sock, poller=target_poller)
            sock = None
            self._discard_remote_sockets(remote_host, remote_handshake_port, poller=target_poller)
        elif sock is None:
            self._discard_remote_sockets(remote_host, remote_handshake_port, poller=target_poller)
        self._invalidate_remote_metadata(
            remote_host=remote_host,
            remote_handshake_port=remote_handshake_port,
        )
        return sock


class MooncakeRecvFailureTrackingMixin(MooncakeRemotePeerMixin):
    """Shared request failure, lost-peer isolation, and block eviction tracking."""

    failed_recv_requests_lock: threading.Lock
    failed_recv_requests: set[str]
    peer_lost_requests: set[tuple[str, str]]
    invalid_block_ids: set[int]
    use_request_level_recv_errors: bool

    def _ensure_failure_tracking(self) -> None:
        if not hasattr(self, "failed_recv_requests_lock"):
            self.failed_recv_requests_lock = threading.Lock()
        if not hasattr(self, "failed_recv_requests"):
            self.failed_recv_requests = set()
        if not hasattr(self, "peer_lost_requests"):
            self.peer_lost_requests = set()
        if not hasattr(self, "invalid_block_ids"):
            self.invalid_block_ids = set()
        if not hasattr(self, "use_request_level_recv_errors"):
            kv_groups = getattr(getattr(self, "kv_cache_config", None), "kv_cache_groups", ())
            self.use_request_level_recv_errors = bool(getattr(self, "use_hybrid", False)) or len(kv_groups) > 1

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        """Get and clear block ids that failed to load."""
        self._ensure_failure_tracking()
        with self.failed_recv_requests_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def _is_failed_recv_request(self, request_id: str) -> bool:
        self._ensure_failure_tracking()
        with self.failed_recv_requests_lock:
            return request_id in self.failed_recv_requests

    def get_and_clear_failed_recv_requests(self, finished_req_ids: set[str]) -> set[str]:
        """Retain failures until every local pull for the request has finished."""
        self._ensure_failure_tracking()
        with self.failed_recv_requests_lock:
            failed_req_ids = self.failed_recv_requests & finished_req_ids
            self.failed_recv_requests.difference_update(failed_req_ids)
            for key in [item for item in self.peer_lost_requests if item[0] in finished_req_ids]:
                self.peer_lost_requests.discard(key)
        return failed_req_ids if self.use_request_level_recv_errors else set()

    def _mark_failed_recv_request(self, request_id: str, local_block_ids: Any) -> None:
        self._ensure_failure_tracking()
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.add(request_id)
            if not self.use_request_level_recv_errors and local_block_ids:
                self.invalid_block_ids.update(local_block_ids[0])

    def _mark_peer_lost_request(self, request_id: str, remote_host: str) -> None:
        self._ensure_failure_tracking()
        with self.failed_recv_requests_lock:
            self.peer_lost_requests.add((request_id, remote_host))

    def _is_peer_lost_request(self, request_id: str, remote_host: str) -> bool:
        self._ensure_failure_tracking()
        with self.failed_recv_requests_lock:
            return (request_id, remote_host) in self.peer_lost_requests

    def _clear_peer_lost_request(self, request_id: str) -> None:
        self._ensure_failure_tracking()
        with self.failed_recv_requests_lock:
            for key in [item for item in self.peer_lost_requests if item[0] == request_id]:
                self.peer_lost_requests.discard(key)


__all__ = [
    "DEFAULT_PULL_ZMQ_TIMEOUT_MS",
    "MooncakePeerLostError",
    "MooncakeRecvFailureTrackingMixin",
    "MooncakeRemotePeerMixin",
    "PULL_HEARTBEAT_IVL_MS",
    "PULL_HEARTBEAT_TIMEOUT_MS",
    "PULL_LOST_PEER_EVENTS",
    "SizedDict",
    "as_kv_cache_tensors",
    "close_zmq_socket",
    "collect_configured_register_regions",
    "ensure_zmq_recv",
    "ensure_zmq_send",
    "group_concurrent_contiguous",
    "lost_peer_monitor",
    "make_mooncake_client_socket",
    "recv_unless_lost",
    "split_if_not_byte_contiguous",
    "string_to_int64_hash",
    "worker_rank_to_dcp_rank",
    "zmq_ctx",
]
