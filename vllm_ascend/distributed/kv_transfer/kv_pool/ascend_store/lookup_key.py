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

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import block_hash_to_bytes
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (  # type: ignore[import-not-found]
        LOOKUP_MSG,
        RESET_MSG,
        RESP_ERR,
        RESP_OK,
    )
else:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol import (
        LOOKUP_MSG,
        RESET_MSG,
        RESP_ERR,
        RESP_OK,
    )

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker

DEFAULT_LOOKUP_BATCH_SIZE: int = 512
DEFAULT_LOOKUP_TIMEOUT_MS: int = 1000
DEFAULT_LOOKUP_ASYNC: bool = False
DEFAULT_LOOKUP_MAX_INFLIGHT: int = 64
DEFAULT_ASYNC_LOOKUP_TTL_S: float = 5.0
_MAX_ZMQ_TIMEOUT_MS: int = (1 << 31) - 1

LookupSignature = tuple[int, int, int, bytes, bytes]


@dataclass(frozen=True)
class LookupConfig:
    """Validated AscendStore lookup bounding and async configuration."""

    lookup_batch_size: int = DEFAULT_LOOKUP_BATCH_SIZE
    lookup_timeout_ms: int = DEFAULT_LOOKUP_TIMEOUT_MS
    lookup_async: bool = DEFAULT_LOOKUP_ASYNC
    lookup_max_inflight: int = DEFAULT_LOOKUP_MAX_INFLIGHT


def _parse_int_option(
    extra_config: Mapping[str, Any],
    key: str,
    default: int,
    *,
    min_value: int,
    max_value: int | None = None,
) -> int:
    if key not in extra_config or extra_config[key] is None:
        return default
    value = extra_config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid {key} {value!r} in kv_connector_extra_config: expected an integer.")
    if value < min_value or (max_value is not None and value > max_value):
        upper = str(max_value) if max_value is not None else "inf"
        raise ValueError(
            f"Invalid {key} {value!r} in kv_connector_extra_config: expected integer in [{min_value}, {upper}]."
        )
    return value


def parse_lookup_config(extra_config: Mapping[str, Any] | None) -> LookupConfig:
    """Parse and validate AscendStore lookup knobs from ``kv_connector_extra_config``."""
    if not isinstance(extra_config, Mapping):
        return LookupConfig()
    lookup_batch_size = _parse_int_option(extra_config, "lookup_batch_size", DEFAULT_LOOKUP_BATCH_SIZE, min_value=0)
    lookup_timeout_ms = _parse_int_option(
        extra_config,
        "lookup_timeout_ms",
        DEFAULT_LOOKUP_TIMEOUT_MS,
        min_value=1,
        max_value=_MAX_ZMQ_TIMEOUT_MS,
    )
    lookup_async = DEFAULT_LOOKUP_ASYNC
    if "lookup_async" in extra_config and extra_config["lookup_async"] is not None:
        raw_async = extra_config["lookup_async"]
        if not isinstance(raw_async, bool):
            raise ValueError(f"Invalid lookup_async {raw_async!r} in kv_connector_extra_config: expected a boolean.")
        lookup_async = raw_async
    lookup_max_inflight = _parse_int_option(
        extra_config, "lookup_max_inflight", DEFAULT_LOOKUP_MAX_INFLIGHT, min_value=1
    )
    return LookupConfig(
        lookup_batch_size=lookup_batch_size,
        lookup_timeout_ms=lookup_timeout_ms,
        lookup_async=lookup_async,
        lookup_max_inflight=lookup_max_inflight,
    )


def make_lookup_signature(
    token_len: int,
    hbm_hit_tokens: int,
    block_hashes: Sequence[BlockHash | str],
) -> LookupSignature:
    """Build an O(1) prefix signature for async lookup deduplication."""
    if not block_hashes:
        return (token_len, hbm_hit_tokens, 0, b"", b"")
    first_bytes = block_hash_to_bytes(block_hashes[0])
    last_bytes = block_hash_to_bytes(block_hashes[-1])
    return (token_len, hbm_hit_tokens, len(block_hashes), first_bytes, last_bytes)


@dataclass
class _PendingLookup:
    signature: LookupSignature
    future: Future[int]
    deadline: float
    completed_at: float | None = None
    result: int | None = None
    timed_out: bool = False


class AsyncLookupManager:
    """Single-worker background executor for non-blocking KV pool lookups."""

    def __init__(
        self,
        timeout_ms: int = DEFAULT_LOOKUP_TIMEOUT_MS,
        max_inflight: int = DEFAULT_LOOKUP_MAX_INFLIGHT,
        result_ttl_s: float = DEFAULT_ASYNC_LOOKUP_TTL_S,
    ) -> None:
        self._timeout_s = timeout_ms / 1000.0
        self._max_inflight = max_inflight
        self._result_ttl_s = result_ttl_s
        self._lock = threading.Lock()
        self._entries: dict[str, _PendingLookup] = {}
        self._inflight_futures: set[Future[int]] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AscendStoreLookup")
        self._closed = False

    def _on_future_done(self, future: Future[int]) -> None:
        with self._lock:
            self._inflight_futures.discard(future)

    def _sweep_expired_locked(self, now: float) -> None:
        expired = [
            req_id
            for req_id, entry in self._entries.items()
            if entry.completed_at is not None
            and (entry.future.done() or entry.timed_out)
            and (now - entry.completed_at) > self._result_ttl_s
        ]
        for req_id in expired:
            self._entries.pop(req_id, None)

    @staticmethod
    def _run_with_deadline(fn: Callable[[float], int], deadline: float) -> int:
        if time.monotonic() >= deadline:
            return 0
        return int(fn(deadline))

    @staticmethod
    def _harvest_future(req_id: str, future: Future[int]) -> int:
        try:
            return max(0, int(future.result(timeout=0)))
        except Exception:
            logger.exception("KV pool async lookup failed req=%s; failing open with 0", req_id)
            return 0

    def _submit_new_lookup(
        self,
        req_id: str,
        signature: LookupSignature,
        submit_factory: Callable[[], Callable[[float], int]],
        now: float,
    ) -> int | None:
        fn = submit_factory()
        deadline = now + self._timeout_s
        future: Future[int] | None = None
        new_state: _PendingLookup | None = None
        closed = False
        submit_failed = False
        with self._lock:
            if self._closed:
                closed = True
            elif len(self._inflight_futures) < self._max_inflight:
                try:
                    future = self._executor.submit(self._run_with_deadline, fn, deadline)
                except RuntimeError:
                    submit_failed = True
                else:
                    self._inflight_futures.add(future)
                    new_state = _PendingLookup(signature=signature, future=future, deadline=deadline)
                    self._entries[req_id] = new_state

        if closed or submit_failed:
            return 0
        if future is None or new_state is None:
            logger.warning(
                "KV pool async lookup inflight limit reached req=%s max_inflight=%d; failing open with 0",
                req_id,
                self._max_inflight,
            )
            return 0

        # Register outside self._lock because Future.add_done_callback runs inline if already completed.
        future.add_done_callback(self._on_future_done)
        if future.done():
            hit = self._harvest_future(req_id, future)
            with self._lock:
                if self._entries.get(req_id) is new_state:
                    new_state.result = hit
                    new_state.completed_at = time.monotonic()
            return hit
        return None

    def poll_or_submit(
        self,
        req_id: str,
        signature: LookupSignature,
        submit_factory: Callable[[], Callable[[float], int]],
    ) -> int | None:
        """Return completed hit tokens, 0 on fail-open timeout/cap, or None if pending."""
        now = time.monotonic()
        done_state: _PendingLookup | None = None
        stale_future: Future[int] | None = None
        timed_out_future: Future[int] | None = None
        inflight_limited = False
        timeout_ms = int(self._timeout_s * 1000)

        with self._lock:
            if self._closed:
                return 0
            self._sweep_expired_locked(now)
            state = self._entries.get(req_id)
            if state is not None and state.signature != signature:
                self._entries.pop(req_id, None)
                stale_future = state.future
                state = None
            if state is not None:
                if state.result is not None:
                    if state.completed_at is not None and (now - state.completed_at) <= self._result_ttl_s:
                        return state.result
                    self._entries.pop(req_id, None)
                    state = None
                elif state.future.done():
                    done_state = state
                elif now >= state.deadline:
                    timed_out_future = state.future
                    state.timed_out = True
                    state.result = 0
                    state.completed_at = now
                else:
                    return None
            if (
                state is None
                and stale_future is None
                and len(self._inflight_futures) >= self._max_inflight
            ):
                inflight_limited = True

        # Cancel outside self._lock because Future.cancel() invokes done callbacks synchronously.
        if stale_future is not None:
            stale_future.cancel()

        if inflight_limited:
            logger.warning(
                "KV pool async lookup inflight limit reached req=%s max_inflight=%d; failing open with 0",
                req_id,
                self._max_inflight,
            )
            return 0

        if done_state is not None:
            if done_state.timed_out:
                return 0
            hit = self._harvest_future(req_id, done_state.future)
            with self._lock:
                if self._entries.get(req_id) is done_state:
                    done_state.result = hit
                    done_state.completed_at = time.monotonic()
            return hit

        if timed_out_future is not None:
            cancelled = timed_out_future.cancel()
            logger.warning(
                "KV pool async lookup timed out req=%s timeout_ms=%d cancelled=%s; failing open with 0",
                req_id,
                timeout_ms,
                cancelled,
            )
            return 0

        return self._submit_new_lookup(req_id, signature, submit_factory, now)

    def discard(self, req_id: str) -> None:
        with self._lock:
            state = self._entries.pop(req_id, None)
        if state is not None:
            state.future.cancel()

    def clear(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            entry.future.cancel()

    def wait_and_clear(self, timeout_s: float | None = None) -> bool:
        """Cancel queued lookups and wait briefly for any running worker future."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            pending = list(self._inflight_futures)
        for entry in entries:
            entry.future.cancel()
        if not pending:
            return True
        budget = min(self._timeout_s + 0.5, 2.0) if timeout_s is None else max(0.0, timeout_s)
        _, not_done = wait(pending, timeout=budget)
        return not not_done

    def shutdown(self, wait_timeout_s: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
            pending = list(self._inflight_futures)
        for entry in entries:
            entry.future.cancel()
        if pending:
            budget = min(self._timeout_s + 0.5, 2.0) if wait_timeout_s is None else max(0.0, wait_timeout_s)
            wait(pending, timeout=budget)
        self._executor.shutdown(wait=False, cancel_futures=True)


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
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
        lookup_config = parse_lookup_config(extra_config)
        self._lookup_timeout_ms = lookup_config.lookup_timeout_ms
        self._current_rcvtimeo_ms = self._lookup_timeout_ms
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        self.socket.setsockopt(zmq.RCVTIMEO, self._lookup_timeout_ms)  # type: ignore[attr-defined]
        self._closed = False

    def _sync_receive_timeout(self, deadline: float | None) -> bool:
        if deadline is None:
            target_ms = self._lookup_timeout_ms
        else:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                logger.warning("KV pool lookup skipped: deadline expired before ZMQ RPC")
                return False
            target_ms = max(1, min(self._lookup_timeout_ms, remaining_ms))
        if target_ms != self._current_rcvtimeo_ms:
            self.socket.setsockopt(zmq.RCVTIMEO, target_ms)  # type: ignore[attr-defined]
            self._current_rcvtimeo_ms = target_ms
        return True

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
        *,
        deadline: float | None = None,
    ) -> int:
        if self._closed:
            return 0
        if not self._sync_receive_timeout(deadline):
            return 0

        kv_cache_group_ids = kv_cache_group_ids or [0]
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        kv_group_frames = self.encoder.encode(kv_cache_group_ids)
        all_frames = [
            LOOKUP_MSG,
            token_len.to_bytes(4, byteorder="big"),
            *kv_group_frames,
            hbm_hit_tokens.to_bytes(4, byteorder="big"),
            *hash_frames,
        ]
        try:
            self.socket.send_multipart(all_frames, copy=False)
            resp = self.socket.recv()
        except zmq.ZMQError:  # type: ignore[attr-defined]
            logger.exception("KV pool lookup failed; reporting 0 hit tokens")
            self.close()
            return 0

        resp_bytes = bytes(resp)
        if resp_bytes == RESP_ERR or len(resp_bytes) != 4:
            logger.warning("KV pool lookup received invalid/error response (%d bytes); reporting 0", len(resp_bytes))
            return 0
        return int.from_bytes(resp_bytes, "big")

    def reset(self) -> bool:
        """Ask rank 0 to drain queued puts, then ``remove_all(force=True)``."""
        if self._closed:
            return False
        self.socket.send(RESET_MSG)
        resp = self.socket.recv()
        return bytes(resp) == RESP_OK

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.socket.close(linger=0)
        finally:
            # Each client owns a private context. Close it too so repeated
            # lookup failures do not leak ZeroMQ IO threads.
            self.ctx.term()

    @property
    def closed(self) -> bool:
        return self._closed


class LookupKeyServer:
    def __init__(
        self,
        pool_worker: BaseKVPoolWorker,
        vllm_config: VllmConfig,
    ):
        self.decoder = MsgpackDecoder()
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
        self._lookup_timeout_s = parse_lookup_config(extra_config).lookup_timeout_ms / 1000.0
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

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
                            hashes_str = self.decoder.decode(all_frames[4:])
                            deadline = time.monotonic() + self._lookup_timeout_s
                            result = self.pool_worker.lookup_scheduler(
                                token_len,
                                hashes_str,
                                kv_group_ids,
                                use_layerwise=False,
                                hbm_hit_tokens=hbm_hit_tokens,
                                deadline=deadline,
                            )
                            logger.debug(
                                "KV pool lookup response token_len=%d groups=%s hit_tokens=%d",
                                token_len,
                                kv_group_ids,
                                result,
                            )
                            response = result.to_bytes(4, "big")
                        elif msg_type == RESET_MSG:
                            self.pool_worker.reset_store()
                            logger.info("AscendStore reset via remove_all succeeded.")
                            response = RESP_OK
                        else:
                            logger.warning("LookupKeyServer received unknown msg_type: %r", msg_type)
                            response = RESP_ERR
                    except Exception as e:
                        logger.error("Error handling lookup request: %s", e)
                        response = RESP_ERR

                    try:
                        self.socket.send(response)
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
