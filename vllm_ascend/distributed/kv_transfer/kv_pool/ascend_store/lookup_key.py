from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    StoreLookupResult,
    TailKeyBoundary,
    block_hash_to_bytes,
)
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.34.0"):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (  # type: ignore[import-not-found]
        LOOKUP_MSG,
        RESET_MSG,
        RESP_ERR,
        RESP_OK,
    )
else:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol import (  # type: ignore[import-not-found]
        LOOKUP_MSG,
        RESET_MSG,
        RESP_ERR,
        RESP_OK,
    )

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker

DEFAULT_LOOKUP_BATCH_SIZE: int = 512
DEFAULT_LOOKUP_TIMEOUT_MS: int = 1000
DEFAULT_RESET_TIMEOUT_MS: int = 30_000
DEFAULT_LOOKUP_ASYNC: bool = False
DEFAULT_LOOKUP_MAX_INFLIGHT: int = 64
DEFAULT_ASYNC_LOOKUP_TTL_S: float = 5.0
_ASYNC_DRAIN_GRACE_S: float = 0.5
_MAX_ZMQ_TIMEOUT_MS: int = (1 << 31) - 1
# Request token counts and timeout budgets are unsigned 32-bit values.
_LOOKUP_INT_BYTES_LEN: int = 4
_LOOKUP_BUDGET_FRAME_PREFIX: bytes = b"budget:"
_LOOKUP_BUDGET_FRAME_LEN: int = len(_LOOKUP_BUDGET_FRAME_PREFIX) + _LOOKUP_INT_BYTES_LEN

LookupSignature = tuple[int, int, int, bytes, bytes]


def _encode_lookup_budget_frame(timeout_ms: int) -> bytes:
    """Encode an explicit client lookup timeout budget frame (in ms)."""
    clamped_ms = max(1, min(_MAX_ZMQ_TIMEOUT_MS, int(timeout_ms)))
    return _LOOKUP_BUDGET_FRAME_PREFIX + clamped_ms.to_bytes(_LOOKUP_INT_BYTES_LEN, byteorder="big")


def _split_hash_and_budget_frames(hash_and_optional_frames: Sequence[Any]) -> tuple[list[Any], int | None]:
    """Separate trailing client budget frame from serialized block-hash frames."""
    frames = list(hash_and_optional_frames)
    if len(frames) >= 2:
        tail = bytes(frames[-1])
        if len(tail) == _LOOKUP_BUDGET_FRAME_LEN and tail.startswith(_LOOKUP_BUDGET_FRAME_PREFIX):
            budget_ms = int.from_bytes(tail[len(_LOOKUP_BUDGET_FRAME_PREFIX) :], byteorder="big")
            return frames[:-1], (budget_ms if budget_ms > 0 else None)
    return frames, None


@dataclass(frozen=True)
class LookupConfig:
    """Validated AscendStore lookup bounding and async configuration."""

    lookup_batch_size: int = DEFAULT_LOOKUP_BATCH_SIZE
    lookup_timeout_ms: int = DEFAULT_LOOKUP_TIMEOUT_MS
    lookup_async: bool = DEFAULT_LOOKUP_ASYNC
    lookup_max_inflight: int = DEFAULT_LOOKUP_MAX_INFLIGHT


def _parse_opt(
    extra_config: Mapping[str, Any], key: str, default: Any, typ: type, lo: int = 0, hi: int | None = None
) -> Any:
    val = extra_config.get(key)
    if val is None:
        return default
    if (typ is int and isinstance(val, bool)) or not isinstance(val, typ):
        raise ValueError(f"Invalid {key}={val!r} in kv_connector_extra_config: expected {typ.__name__}.")
    if typ is int and isinstance(val, int) and (val < lo or (hi is not None and val > hi)):
        raise ValueError(f"Invalid {key}={val!r} in kv_connector_extra_config: expected [{lo}, {hi or 'inf'}].")
    return val


def parse_lookup_config(extra_config: Mapping[str, Any] | None) -> LookupConfig:
    """Parse and validate AscendStore lookup knobs from ``kv_connector_extra_config``."""
    if not isinstance(extra_config, Mapping):
        return LookupConfig()
    return LookupConfig(
        lookup_batch_size=_parse_opt(extra_config, "lookup_batch_size", DEFAULT_LOOKUP_BATCH_SIZE, int, 0),
        lookup_timeout_ms=_parse_opt(
            extra_config, "lookup_timeout_ms", DEFAULT_LOOKUP_TIMEOUT_MS, int, 1, _MAX_ZMQ_TIMEOUT_MS
        ),
        lookup_async=_parse_opt(extra_config, "lookup_async", DEFAULT_LOOKUP_ASYNC, bool),
        lookup_max_inflight=_parse_opt(extra_config, "lookup_max_inflight", DEFAULT_LOOKUP_MAX_INFLIGHT, int, 1),
    )


def _extra_config(vllm_config: VllmConfig) -> Mapping[str, Any] | None:
    return getattr(getattr(vllm_config, "kv_transfer_config", None), "kv_connector_extra_config", None)


def make_lookup_signature(
    token_len: int, hbm_hit_tokens: int, block_hashes: Sequence[BlockHash | str]
) -> LookupSignature:
    """Build an O(1) prefix signature for async lookup deduplication."""
    if not block_hashes:
        return (token_len, hbm_hit_tokens, 0, b"", b"")
    first, last = block_hash_to_bytes(block_hashes[0]), block_hash_to_bytes(block_hashes[-1])
    return (token_len, hbm_hit_tokens, len(block_hashes), first, last)


@dataclass
class _PendingLookup:
    signature: LookupSignature
    future: Future[StoreLookupResult]
    deadline: float
    completed_at: float | None = None
    result: StoreLookupResult | None = None


class AsyncLookupManager:
    """Single-worker background executor for non-blocking KV pool lookups."""

    def __init__(
        self,
        timeout_ms: int = DEFAULT_LOOKUP_TIMEOUT_MS,
        max_inflight: int = DEFAULT_LOOKUP_MAX_INFLIGHT,
        result_ttl_s: float = DEFAULT_ASYNC_LOOKUP_TTL_S,
    ) -> None:
        self._timeout_s, self._max_inflight, self._result_ttl_s = timeout_ms / 1000.0, max_inflight, result_ttl_s
        self._lock = threading.Lock()
        self._entries: dict[str, _PendingLookup] = {}
        self._inflight_futures: set[Future[StoreLookupResult]] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AscendStoreLookup")
        self._closed = False

    def _on_future_done(self, future: Future[StoreLookupResult]) -> None:
        with self._lock:
            self._inflight_futures.discard(future)

    def _sweep_expired_locked(self, now: float) -> None:
        expired = [
            k
            for k, e in self._entries.items()
            if e.completed_at is not None and now - e.completed_at > self._result_ttl_s
        ]
        for req_id in expired:
            self._entries.pop(req_id, None)

    @staticmethod
    def _run_with_deadline(fn: Callable[[float], StoreLookupResult], deadline: float) -> StoreLookupResult:
        return StoreLookupResult(0) if time.monotonic() >= deadline else fn(deadline)

    def _complete_state(self, req_id: str, state: _PendingLookup) -> StoreLookupResult:
        try:
            res = state.future.result(timeout=0)
            hit = res if res.hit_length > 0 else StoreLookupResult(0)
        except Exception:
            logger.exception("KV pool async lookup failed req=%s; failing open with 0", req_id)
            hit = StoreLookupResult(0)
        with self._lock:
            if self._entries.get(req_id) is state:
                state.result = hit
                state.completed_at = time.monotonic()
        return hit

    def _submit_new_lookup(
        self,
        req_id: str,
        signature: LookupSignature,
        submit_factory: Callable[[], Callable[[float], StoreLookupResult]],
        now: float,
    ) -> StoreLookupResult | None:
        fn, deadline = submit_factory(), now + self._timeout_s
        future: Future[StoreLookupResult] | None = None
        new_state: _PendingLookup | None = None
        with self._lock:
            if self._closed:
                return StoreLookupResult(0)
            if len(self._inflight_futures) < self._max_inflight:
                try:
                    future = self._executor.submit(self._run_with_deadline, fn, deadline)
                except RuntimeError:
                    return StoreLookupResult(0)
                self._inflight_futures.add(future)
                new_state = _PendingLookup(signature=signature, future=future, deadline=deadline)
                self._entries[req_id] = new_state
        if future is None or new_state is None:
            logger.warning("KV pool async lookup inflight limit reached req=%s max=%d", req_id, self._max_inflight)
            return StoreLookupResult(0)
        # Register outside self._lock because Future.add_done_callback runs inline if already completed.
        future.add_done_callback(self._on_future_done)
        return self._complete_state(req_id, new_state) if future.done() else None

    def poll_or_submit(
        self,
        req_id: str,
        signature: LookupSignature,
        submit_factory: Callable[[], Callable[[float], StoreLookupResult]],
    ) -> StoreLookupResult | None:
        """Return completed lookup result, zero result on fail-open timeout/cap, or None if pending."""
        now = time.monotonic()
        done_state: _PendingLookup | None = None
        stale_future = timed_out_future = None

        with self._lock:
            if self._closed:
                return StoreLookupResult(0)
            self._sweep_expired_locked(now)
            state = self._entries.get(req_id)
            if state is not None and state.signature != signature:
                self._entries.pop(req_id, None)
                stale_future, state = state.future, None
            if state is not None:
                if state.result is not None:
                    if state.completed_at is not None and (now - state.completed_at) <= self._result_ttl_s:
                        return state.result
                    self._entries.pop(req_id, None)
                elif state.future.done():
                    done_state = state
                elif now >= state.deadline:
                    timed_out_future = state.future
                    state.result, state.completed_at = StoreLookupResult(0), now
                else:
                    return None

        # Cancel outside self._lock because Future.cancel() invokes done callbacks synchronously.
        if stale_future is not None:
            stale_future.cancel()
        if done_state is not None:
            return self._complete_state(req_id, done_state)
        if timed_out_future is not None:
            cancelled = timed_out_future.cancel()
            logger.warning("KV pool async lookup timed out req=%s cancelled=%s; failing open with 0", req_id, cancelled)
            return StoreLookupResult(0)
        return self._submit_new_lookup(req_id, signature, submit_factory, now)

    def discard(self, req_id: str) -> None:
        with self._lock:
            state = self._entries.pop(req_id, None)
        if state is not None:
            state.future.cancel()

    def _drain_and_wait(self, timeout_s: float | None = None, *, close: bool = False) -> bool:
        with self._lock:
            if close:
                self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
            pending = list(self._inflight_futures)
        for entry in entries:
            entry.future.cancel()
        if not pending:
            return True
        budget = (self._timeout_s + _ASYNC_DRAIN_GRACE_S) if timeout_s is None else max(0.0, timeout_s)
        _, not_done = wait(pending, timeout=budget)
        return not not_done

    def wait_and_clear(self, timeout_s: float | None = None) -> bool:
        """Cancel queued lookups and wait for any running worker future to settle."""
        return self._drain_and_wait(timeout_s)

    def shutdown(self, wait_timeout_s: float | None = None) -> bool:
        drained = self._drain_and_wait(wait_timeout_s, close=True)
        self._executor.shutdown(wait=drained, cancel_futures=True)
        return drained


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    # NOTE: This isolates lookup IPC paths only. MooncakeConnector (V1) worker
    # handshake ports and MooncakeConnectorV2 scheduler control ports may still
    # overlap ports reserved by other workers or components. DP indices do not
    # guarantee disjoint TCP port ranges.
    # Dense engines reset their compute DP rank, but retain their serving index.
    dp_index = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    elif "mooncake_rpc_port" in extra_config:
        rpc_port = extra_config["mooncake_rpc_port"]
        logger.warning(
            "It is recommended to use the lookup_rpc_port, as the mooncake_rpc_port will be removed in the future."
        )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_dp_rank{dp_index}"


class LookupKeyClient:
    def __init__(self, vllm_config: VllmConfig):
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(tuple[int, tuple[TailKeyBoundary, ...]])
        self._lookup_timeout_ms = parse_lookup_config(_extra_config(vllm_config)).lookup_timeout_ms
        self._current_rcvtimeo_ms = self._lookup_timeout_ms
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self.socket = make_zmq_socket(self.ctx, get_zmq_rpc_path_lookup(vllm_config), zmq.REQ, bind=False)  # type: ignore[attr-defined]
        self._lock = threading.Lock()
        self.socket.setsockopt(zmq.RCVTIMEO, self._lookup_timeout_ms)  # type: ignore[attr-defined]
        self._closed = False

    def _rpc_locked(self, frames: list[bytes] | bytes, timeout_ms: int, err_msg: str) -> Any:
        if timeout_ms != self._current_rcvtimeo_ms:
            self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)  # type: ignore[attr-defined]
            self._current_rcvtimeo_ms = timeout_ms
        try:
            if isinstance(frames, bytes):
                self.socket.send(frames)
                return self.socket.recv()
            self.socket.send_multipart(frames, copy=False)
            return self.socket.recv_multipart()
        except zmq.ZMQError:  # type: ignore[attr-defined]
            logger.exception(err_msg)
            self._close_locked()
            return None

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
        *,
        deadline: float | None = None,
    ) -> StoreLookupResult:
        """Return the recoverable prefix length and per-group tail-key boundaries."""
        kv_cache_group_ids = kv_cache_group_ids or [0]
        with self._lock:
            if self._closed:
                return StoreLookupResult(0)
            target_ms = self._lookup_timeout_ms
            if deadline is not None:
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    logger.warning("KV pool lookup skipped: deadline expired before ZMQ RPC")
                    return StoreLookupResult(0)
                target_ms = max(1, min(target_ms, remaining_ms))
            frames = [
                LOOKUP_MSG,
                token_len.to_bytes(_LOOKUP_INT_BYTES_LEN, byteorder="big"),
                *self.encoder.encode(kv_cache_group_ids),
                hbm_hit_tokens.to_bytes(_LOOKUP_INT_BYTES_LEN, byteorder="big"),
                *self.encoder.encode([h.hex() for h in block_hashes]),
                _encode_lookup_budget_frame(target_ms),
            ]
            resp = self._rpc_locked(frames, target_ms, "KV pool lookup failed; reporting 0 hit tokens")
            if resp is None:
                return StoreLookupResult(0)

        hit_length, boundaries = self.decoder.decode(resp)
        return StoreLookupResult(hit_length, boundaries)

    def reset(self) -> bool:
        """Ask rank 0 to drain queued puts, then ``remove_all(force=True)``."""
        with self._lock:
            if self._closed:
                return False
            # Prior deadline-bounded lookups may have lowered RCVTIMEO to 1ms.
            resp = self._rpc_locked(
                RESET_MSG, max(self._lookup_timeout_ms, DEFAULT_RESET_TIMEOUT_MS), "KV pool reset RPC failed"
            )
            return resp is not None and bytes(resp) == RESP_OK

    def _close_locked(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.socket.close(linger=0)
        finally:
            # Each client owns a private context. Close it too so repeated
            # lookup failures do not leak ZeroMQ IO threads.
            self.ctx.term()

    def close(self):
        with self._lock:
            self._close_locked()

    @property
    def closed(self) -> bool:
        return self._closed


class LookupKeyServer:
    def __init__(self, pool_worker: BaseKVPoolWorker, vllm_config: VllmConfig):
        self.decoder, self.encoder = MsgpackDecoder(), MsgpackEncoder()
        self._lookup_timeout_s = parse_lookup_config(_extra_config(vllm_config)).lookup_timeout_ms / 1000.0
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(self.ctx, path, zmq.REP, bind=True)  # type: ignore[attr-defined]

        self.pool_worker = pool_worker
        self.running = True
        self._closed = False

        def process_request():
            poller = zmq.Poller()  # type: ignore[attr-defined]
            poller.register(self.socket, zmq.POLLIN)  # type: ignore[attr-defined]
            try:
                while self.running:
                    if not poller.poll(timeout=100):
                        continue

                    # A failed receive leaves a REP socket in an unknown
                    # protocol state. Do not try to send a synthetic reply.
                    try:
                        all_frames = self.socket.recv_multipart(copy=False)
                    except zmq.ZMQError:  # type: ignore[attr-defined]
                        if self.running:
                            logger.exception("Failed to receive KV pool lookup request")
                        break

                    try:
                        msg_type = bytes(all_frames[0])
                        if msg_type == LOOKUP_MSG:
                            token_len = int.from_bytes(all_frames[1], byteorder="big")
                            kv_group_ids = self.decoder.decode([all_frames[2]])
                            hbm_hit_tokens = int.from_bytes(all_frames[3], byteorder="big")
                            hash_frames, client_budget_ms = _split_hash_and_budget_frames(all_frames[4:])
                            hashes_str = self.decoder.decode(hash_frames)
                            timeout_s = self._lookup_timeout_s
                            if client_budget_ms is not None:
                                timeout_s = min(timeout_s, client_budget_ms / 1000.0)
                            result = self.pool_worker.lookup_scheduler(
                                token_len,
                                hashes_str,
                                kv_group_ids,
                                use_layerwise=False,
                                hbm_hit_tokens=hbm_hit_tokens,
                                deadline=time.monotonic() + timeout_s,
                            )
                            logger.debug("KV pool lookup len=%d hit=%d", token_len, result.hit_length)
                            response = self.encoder.encode((result.hit_length, result.tail_key_boundaries))
                        elif msg_type == RESET_MSG:
                            self.pool_worker.reset_store()
                            logger.info("AscendStore reset via remove_all succeeded.")
                            response = [RESP_OK]
                        else:
                            logger.warning("LookupKeyServer received unknown msg_type: %r", msg_type)
                            response = [RESP_ERR]
                    except Exception as e:
                        logger.error("Error handling lookup request: %s", e)
                        response = [RESP_ERR]

                    try:
                        self.socket.send_multipart(response)
                    except zmq.ZMQError:  # type: ignore[attr-defined]
                        if self.running:
                            logger.exception("Failed to send KV pool lookup response")
                        break
            finally:
                self.socket.close(linger=0)
                self.ctx.term()

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.running = False
        self.thread.join()
