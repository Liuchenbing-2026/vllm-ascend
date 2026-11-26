# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2-local native binding records, deferred snapshots and output checks."""

from __future__ import annotations

import functools
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import contextmanager
from threading import Lock
from typing import Any

import torch

from vllm_ascend.core.scheduler_output import resolve_max_concurrent_batches
from vllm_ascend.kv_trace.kv_trace_config import KVTraceConfig
from vllm_ascend.kv_trace.kv_trace_connector import record_output, record_plan
from vllm_ascend.kv_trace.kv_trace_plan import AccessPlan, PlanSnapshotter, audit_access_plan
from vllm_ascend.kv_trace.kv_trace_state import KVCacheState

_MAX_REPORT_KEYS = 1024


class KVTraceError(RuntimeError):
    """Stop returning model results after a visible address-plan violation."""


class _PendingAudit:
    """One snapshot, consumed once by the output reader or next forward."""

    def __init__(
        self,
        complete: Any,
        slot_id: int | None = None,
        release_slot: Callable[[int], None] | None = None,
    ):
        self._complete = complete
        self.slot_id = slot_id
        self._release_slot = release_slot
        self._lock = Lock()
        self._error: BaseException | None = None

    @property
    def is_done(self) -> bool:
        with self._lock:
            return self._complete is None and self._error is None

    def finish(self) -> None:
        release_cb: Callable[[int], None] | None = None
        release_slot_id: int | None = None
        with self._lock:
            if self._complete is not None:
                try:
                    self._complete()
                except BaseException as exc:
                    self._error = exc
                finally:
                    self._complete = None
                    if self.slot_id is not None and self._release_slot is not None:
                        release_cb = self._release_slot
                        release_slot_id = self.slot_id
                        self._release_slot = None
            error = self._error
        if release_cb is not None and release_slot_id is not None:
            release_cb(release_slot_id)
        if error is not None:
            raise error


class KVTraceRuntime:
    def __init__(self, vllm_config: Any):
        from vllm.logger import logger

        self.logger = logger
        self.config = vllm_config
        self.state = KVCacheState(self.report)
        self.layer_groups: dict[str, int] = {}
        self._metadata: dict[int, Any] = {}
        self._phase: str | None = None
        self._full_graph = False
        self._reported: OrderedDict[tuple, None] = OrderedDict()
        self._scheduled: dict[str, tuple[int | None, int]] | None = None
        self._failure: str | None = None
        self.action = KVTraceConfig.from_vllm_config(vllm_config).action
        self._configure_runner: Callable[[], None] | None = None
        self._max_pending_audits = max(1, resolve_max_concurrent_batches(vllm_config))
        self._snapshotter = PlanSnapshotter(
            self._max_pending_audits,
            on_shrink=lambda _target: self._finish_pending(),
        )
        self._defer_audit = False
        self._pending_lock = Lock()
        self._pending_audits: deque[_PendingAudit] = deque()
        self._unattached_pending: deque[_PendingAudit] = deque()

    @property
    def _pending(self) -> _PendingAudit | None:
        with self._pending_lock:
            return self._pending_audits[-1] if self._pending_audits else None

    @_pending.setter
    def _pending(self, value: _PendingAudit | None) -> None:
        with self._pending_lock:
            cleared = list(self._pending_audits)
            self._pending_audits.clear()
            self._unattached_pending.clear()
            if value is not None:
                self._pending_audits.append(value)
                self._unattached_pending.append(value)
        for item in cleared:
            self._snapshotter.release_slot(item.slot_id)

    def _check_failure(self) -> None:
        if self.action == "raise" and self._failure is not None:
            raise KVTraceError(f"KV Trace stopped model output: {self._failure}; restart the worker before reuse")

    def report(self, event: str, **fields: Any) -> None:
        failed = event == "kv_trace.violation"
        if failed:
            with self.state._lock:
                if self._failure is None:
                    self._failure = str(fields.get("reason", event))
                key = (
                    event,
                    *(
                        fields.get(name)
                        for name in (
                            "operation",
                            "reason",
                            "group_id",
                            "block_id",
                            "request_id",
                        )
                    ),
                    tuple(fields.get("request_ids", ())),
                )
                if key in self._reported:
                    self._reported.move_to_end(key)
                    return
                self._reported[key] = None
                if len(self._reported) > _MAX_REPORT_KEYS:
                    self._reported.popitem(last=False)
        payload = " ".join(f"{key}={value}" for key, value in fields.items())
        log = self.logger.error if failed else self.logger.info
        log("[KV_TRACE] %s %s", event, payload)

    def gap(self, reason: str, **fields: Any) -> None:
        self.logger.debug("[KV_TRACE] check skipped: %s %s", reason, fields)

    def configure(self, runner: Any) -> None:
        config = self.config
        parallel = config.parallel_config
        transfer = config.kv_transfer_config
        unsupported = any(
            (
                getattr(config.model_config, "is_encoder_decoder", False),
                config.speculative_config is not None,
                getattr(parallel, "prefill_context_parallel_size", 1) != 1,
                getattr(parallel, "decode_context_parallel_size", 1) != 1,
                getattr(parallel, "enable_dbo", False),
            )
        )
        sizes = {}
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            sizes[group_id] = spec.block_size
            for name in group.layer_names:
                self.layer_groups[name] = group_id
                layer_spec = getattr(spec, "kv_cache_specs", {}).get(name, spec)
                append = (
                    hasattr(layer_spec, "head_size")
                    and getattr(layer_spec, "sliding_window", None) is None
                    and getattr(layer_spec, "compress_ratio", 1) == 1
                    and type(layer_spec).__name__ != "CircularBufferSpec"
                )
                if not append:
                    raise ValueError(f"KV Trace: unadapted cache validity rules: {type(layer_spec).__name__}")
        if unsupported:
            raise ValueError(
                "KV Trace: unadapted execution mode or block coordinates (CP, DBO, speculation, encoder/decoder)"
            )
        self.state.configure(sizes, runner.kv_cache_config.num_blocks)
        self._defer_audit = (
            bool(getattr(getattr(config, "scheduler_config", None), "async_scheduling", False))
            and transfer is None
            # Pooling bypasses sample_tokens; non-reply workers may discard outputs.
            and getattr(runner, "is_pooling_model", None) is False
            and getattr(parallel, "tensor_parallel_size", None) == 1
            and getattr(parallel, "pipeline_parallel_size", None) == 1
            and hasattr(runner, "sample_tokens")
        )
        new_max_pending = max(1, resolve_max_concurrent_batches(config)) if self._defer_audit else 1
        if new_max_pending < self._max_pending_audits:
            self._finish_pending()
        self._max_pending_audits = new_max_pending
        self._snapshotter.set_max_in_flight(self._max_pending_audits)
        if self._configure_runner is not None:
            self._configure_runner()
        self.report("kv_trace.enabled", failure_action=self.action)

    def collect_metadata(self, metadata: Any) -> None:
        self._metadata = {}
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            self.gap("microbatch_metadata_unadapted")
            return
        for name, item in metadata.items():
            group = self.layer_groups.get(name)
            if group is not None and item is not None:
                self._metadata.setdefault(group, item)
        for group in self.state.group_sizes.keys() - self._metadata.keys():
            self.gap("metadata_write_mapping_missing", group_id=group)

    def observe_metadata(self, metadata: Any) -> None:
        try:
            self.collect_metadata(metadata)
        except Exception as exc:
            self._metadata = {}
            self.gap(f"metadata_collection_failed:{type(exc).__name__}")

    def _prune_completed_pending_locked(self) -> None:
        if not self._pending_audits:
            return
        self._pending_audits = deque(item for item in self._pending_audits if not item.is_done)
        if self._unattached_pending:
            self._unattached_pending = deque(item for item in self._unattached_pending if not item.is_done)

    def _prune_completed_pending(self) -> None:
        with self._pending_lock:
            self._prune_completed_pending_locked()

    def _busy_slot_ids_locked(self) -> set[int]:
        return {item.slot_id for item in self._pending_audits if item.slot_id is not None}

    def _run_finish_audits(self, to_finish: list[_PendingAudit]) -> None:
        if not to_finish:
            return
        first_error: BaseException | None = None
        for pending in to_finish:
            try:
                pending.finish()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self._snapshotter.release_slot(pending.slot_id)
        if first_error is not None:
            raise first_error
        self._check_failure()

    def _pop_oldest_pending_locked(self) -> _PendingAudit:
        oldest = self._pending_audits.popleft()
        if oldest in self._unattached_pending:
            self._unattached_pending.remove(oldest)
        return oldest

    def _finish_oldest_if_full(self) -> None:
        while True:
            with self._pending_lock:
                self._prune_completed_pending_locked()
                if not self._pending_audits:
                    break
                has_free = self._snapshotter.has_free_slot(excluded=self._busy_slot_ids_locked())
                if len(self._pending_audits) < self._max_pending_audits and has_free:
                    break
                oldest = self._pop_oldest_pending_locked()
            self._run_finish_audits([oldest])
        self._check_failure()

    def _finish_pending(self) -> None:
        with self._pending_lock:
            self._unattached_pending.clear()
            to_finish = list(self._pending_audits)
            self._pending_audits.clear()
        self._run_finish_audits(to_finish)

    def _acquire_snapshot_slot(self) -> int:
        while True:
            self._finish_oldest_if_full()
            with self._pending_lock:
                busy = self._busy_slot_ids_locked()
            slot_id = self._snapshotter.try_acquire_slot(excluded=busy)
            if slot_id is not None:
                return slot_id
            with self._pending_lock:
                if not self._pending_audits:
                    break
                oldest = self._pop_oldest_pending_locked()
            self._run_finish_audits([oldest])
        return self._snapshotter.acquire_slot()

    def _current_plan(self, runner: Any) -> AccessPlan:
        assert self._phase is not None
        return AccessPlan(
            phase=self._phase,
            request_ids=list(runner.request_ids) if self._phase == "forward" else [],
            scheduled=None if self._scheduled is None else dict(self._scheduled),
            metadata=dict(self._metadata),
            full_graph=self._full_graph,
            bindings=dict(self.state.requests),
            allow_null_block=getattr(runner.runner.block_tables, "redirect_writes_to_null_block", False),
        )

    def _snapshot_audit(self, runner: Any) -> None:
        """Retain only this step's evidence, never a copy of the runtime."""
        slot_id = self._acquire_snapshot_slot()
        try:
            plan, ready = self._snapshotter.freeze(self._current_plan(runner), slot_id=slot_id)
        except BaseException:
            self._snapshotter.release_slot(slot_id)
            raise

        def complete():
            with torch.inference_mode():
                try:
                    if ready is not None:
                        ready.synchronize()
                    audit_access_plan(self.state, plan, self.gap)
                except Exception as exc:
                    self.gap(f"access_audit_failed:{type(exc).__name__}")
                self._check_failure()

        pending = _PendingAudit(complete, slot_id=slot_id, release_slot=self._snapshotter.release_slot)
        with self._pending_lock:
            self._pending_audits.append(pending)
            self._unattached_pending.append(pending)

    def _guard_output(self, output: Any) -> Any:
        self._check_failure()
        with self._pending_lock:
            pending = self._unattached_pending.popleft() if self._unattached_pending else None
        get_output = getattr(output, "get_output", None)

        def complete(result):
            if pending is not None:
                pending.finish()
                self._prune_completed_pending()
            self._check_failure()
            record_output(result, self.report)
            return result

        if get_output is None:
            return complete(output)

        @functools.wraps(get_output)
        def checked(*args, **kwargs):
            return complete(get_output(*args, **kwargs))

        try:
            output.get_output = checked
        except (AttributeError, TypeError):
            if pending is not None:
                pending.finish()
                self._prune_completed_pending()
            self._check_failure()
        return output

    def wrap_sample(self, sample: Any):
        @functools.wraps(sample)
        def observed(*args, **kwargs):
            return self._guard_output(sample(*args, **kwargs))

        return observed

    @contextmanager
    def forward_scope(
        self, runner: Any, *, phase: str, metadata: Any = None, full_graph: bool = False, required: bool = True
    ):
        """Audit final per-group metadata at the native return boundary."""
        self._check_failure()
        self._phase, self._full_graph = phase, full_graph
        self.observe_metadata(metadata)
        try:
            yield
            if required or self._metadata:
                try:
                    if phase == "forward":
                        self._snapshot_audit(runner)
                        if not self._defer_audit:
                            self._finish_pending()
                    else:
                        audit_access_plan(self.state, self._current_plan(runner), self.gap)
                except Exception as exc:
                    self.gap(f"access_audit_failed:{type(exc).__name__}")
            self._check_failure()
        finally:
            self._phase = None

    @contextmanager
    def execute_scope(self, output: Any, *, bind: bool = True):
        """Consume the original SchedulerOutput locally, without scheduler hooks."""
        self._check_failure()
        self._scheduled = self.state.bind(output) if bind else None
        if bind:
            record_plan(output.kv_connector_metadata, self.report)
        try:
            self._check_failure()
            yield
            self._check_failure()
        finally:
            self._scheduled = None

    def zero_blocks(self, block_ids: list[int]) -> None:
        self.report("kv_trace.boundary", boundary="model_runner.update_requests.return", zero_block_ids=block_ids)

    def copy_blocks(self, copies: Any) -> None:
        self.report(
            "kv_trace.boundary",
            boundary="model_runner.update_requests.return",
            block_copies=[copy._asdict() for copy in copies],
        )


def get_trace(vllm_config: Any) -> KVTraceRuntime | None:
    if not KVTraceConfig.from_vllm_config(vllm_config).enabled:
        return None
    guard = getattr(vllm_config, "_ascend_kv_trace", None)
    if guard is None:
        guard = KVTraceRuntime(vllm_config)
        vllm_config._ascend_kv_trace = guard
    return guard
