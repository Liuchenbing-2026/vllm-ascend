# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-step address evidence, independent of subsequent binding updates."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, cast

import torch

from vllm_ascend.core.scheduler_output import require_non_bool_int
from vllm_ascend.kv_trace.kv_trace_state import BlockTable, KVCacheState


@dataclass(slots=True)
class AccessPlan:
    phase: str
    request_ids: list[str]
    scheduled: dict[str, tuple[int | None, int]] | None
    metadata: dict[int, Any]
    full_graph: bool
    bindings: dict[str, BlockTable]
    allow_null_block: bool = False


@dataclass(slots=True)
class ExpectedAccess:
    slots: list[int]
    tables: list[list[int]]
    lengths: list[int]


@dataclass(slots=True)
class MetadataSnapshot:
    slot_mapping: Any
    block_tables: Any
    seq_lens: Any


def as_list(value: Any) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return value.detach().cpu().tolist()


def metadata_field(metadata: Any, *names: str) -> Any:
    return next((value for name in names if (value := getattr(metadata, name, None)) is not None), None)


def tensor_identity(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(tensor_identity(item) for item in value)
    if isinstance(value, (int, float, str)):
        return value
    return value.data_ptr(), tuple(value.shape), tuple(value.stride())


@dataclass
class _SnapshotSlot:
    """Reusable host buffer pool and copy event for one in-flight snapshot."""

    host_buffers: list[Any] = field(default_factory=list)
    copy_event: Any = None


class PlanSnapshotter:
    """Reusable D2H buffers sized for configured in-flight async batches."""

    def __init__(self, max_in_flight: int = 1, on_shrink: Callable[[int], None] | None = None):
        count = max(1, require_non_bool_int(max_in_flight, "max_in_flight"))
        self._slots: list[_SnapshotSlot] = [_SnapshotSlot() for _ in range(count)]
        self._free_slots: deque[int] = deque(range(count))
        self._in_use: set[int] = set()
        self._next_slot = 0
        self._on_shrink = on_shrink
        self._lock = Lock()

    def has_free_slot(self, excluded: set[int] | None = None) -> bool:
        """Return whether an unused snapshot slot not in ``excluded`` is available."""
        blocked = excluded or ()
        with self._lock:
            return any(0 <= slot_id < len(self._slots) and slot_id not in blocked for slot_id in self._free_slots)

    def try_acquire_slot(self, excluded: set[int] | None = None) -> int | None:
        """Reserve an unused snapshot slot not in ``excluded``, or return None."""
        blocked = excluded or ()
        with self._lock:
            skipped: list[int] = []
            chosen: int | None = None
            while self._free_slots:
                candidate = self._free_slots.popleft()
                if not (0 <= candidate < len(self._slots)):
                    continue
                if candidate in blocked:
                    skipped.append(candidate)
                    continue
                chosen = candidate
                self._in_use.add(candidate)
                break
            if skipped:
                self._free_slots.extendleft(reversed(skipped))
            return chosen

    def _next_unmanaged_slot_locked(self) -> int:
        """Select a round-robin slot without claiming from ``_free_slots``."""
        total = len(self._slots)
        for offset in range(total):
            candidate = (self._next_slot + offset) % total
            if candidate not in self._in_use:
                self._next_slot = (candidate + 1) % total
                return candidate
        candidate = self._next_slot % total
        self._next_slot = (candidate + 1) % total
        return candidate

    def acquire_slot(self) -> int:
        """Reserve a free snapshot slot, falling back to round-robin if none are free."""
        with self._lock:
            while self._free_slots:
                candidate = self._free_slots.popleft()
                if 0 <= candidate < len(self._slots):
                    self._in_use.add(candidate)
                    return candidate
            candidate = self._next_slot % len(self._slots)
            self._next_slot = (candidate + 1) % len(self._slots)
            self._in_use.add(candidate)
            return candidate

    def release_slot(self, slot_id: int | None) -> None:
        """Return a finished snapshot slot to the free pool if still in range."""
        if slot_id is None:
            return
        slot_id = require_non_bool_int(slot_id, "slot_id")
        with self._lock:
            was_active = slot_id in self._in_use
            self._in_use.discard(slot_id)
            if was_active and 0 <= slot_id < len(self._slots) and slot_id not in self._free_slots:
                self._free_slots.append(slot_id)

    def set_max_in_flight(self, max_in_flight: int) -> None:
        """Ensure at least ``max_in_flight`` independent snapshot buffer slots exist."""
        target = max(1, require_non_bool_int(max_in_flight, "max_in_flight"))
        with self._lock:
            shrinking = target < len(self._slots)
        if shrinking and self._on_shrink is not None:
            # Drain in-flight audits outside _lock before truncating slots they reference.
            self._on_shrink(target)
        with self._lock:
            while len(self._slots) < target:
                new_id = len(self._slots)
                self._slots.append(_SnapshotSlot())
                self._free_slots.append(new_id)
            if target < len(self._slots):
                self._slots = self._slots[:target]
                self._free_slots = deque(slot_id for slot_id in self._free_slots if slot_id < target)
                self._in_use = {slot_id for slot_id in self._in_use if slot_id < target}
                self._next_slot %= target

    def freeze(self, plan: AccessPlan, *, slot_id: int | None = None) -> tuple[AccessPlan, Any]:
        """Copy final values now; return the event that makes them host-readable."""
        if slot_id is not None:
            slot_id = require_non_bool_int(slot_id, "slot_id")
        with self._lock:
            chosen_slot = self._next_unmanaged_slot_locked() if slot_id is None else slot_id
            slot = self._slots[chosen_slot]
        copies: dict[Any, Any] = {}
        event = None
        device_copies = 0

        def freeze(value):
            nonlocal event, device_copies
            if isinstance(value, torch.Tensor):
                key = value.device, value.dtype, tensor_identity(value)
                if key not in copies:
                    if value.device.type == "cpu":
                        host = value.clone()
                    else:
                        if slot.copy_event is None:
                            slot.copy_event = cast(Any, torch).npu.Event()
                        event = slot.copy_event
                        index = device_copies
                        device_copies += 1
                        if index >= len(slot.host_buffers):
                            slot.host_buffers.append(None)
                        host = slot.host_buffers[index]
                        if host is None or host.shape != value.shape or host.dtype != value.dtype:
                            host = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
                            slot.host_buffers[index] = host
                        host.copy_(value, non_blocking=True)
                    copies[key] = host
                return copies[key]
            if isinstance(value, (list, tuple)):
                return [freeze(item) for item in value]
            return value

        try:
            metadata = {}
            for group, item in plan.metadata.items():
                slots = metadata_field(item, "slot_mapping")
                count = getattr(item, "num_actual_tokens", None)
                metadata[group] = MetadataSnapshot(
                    slot_mapping=freeze(slots if slots is None or count is None else slots[:count]),
                    block_tables=freeze(metadata_field(item, "block_tables", "block_table", "block_table_tensor")),
                    seq_lens=freeze(metadata_field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")),
                )
            if event is not None:
                event.record(cast(Any, torch).npu.current_stream())
        except Exception:
            if event is not None:
                # Even a partially failed snapshot must drain before buffer reuse.
                event.record(cast(Any, torch).npu.current_stream())
                event.synchronize()
            raise

        return replace(plan, metadata=metadata), event


def expected_access(
    state: KVCacheState, plan: AccessPlan, group_id: int, gap: Callable[..., None]
) -> ExpectedAccess | None:
    """Derive destinations from the scheduler, never from the audited mapping.

    Per-token lists exist only for this forward and are shared by all layers
    in a cache group. No persistent token-ID history or KV tensor is copied.
    """
    if plan.phase != "forward" or plan.scheduled is None:
        return None
    size = state.group_sizes[group_id]
    slots: list[int] = []
    tables: list[list[int]] = []
    lengths: list[int] = []
    for request_id in plan.request_ids:
        computed, count = plan.scheduled.get(request_id, (None, 0))
        if computed is None or computed < 0 or count <= 0:
            gap("scheduler_write_range_missing", request_id=request_id)
            return None
        groups = plan.bindings.get(request_id, ())
        refs = groups[group_id] if group_id < len(groups) else ()
        needed = (computed + count + size - 1) // size
        if len(refs) < needed:
            gap("scheduler_read_blocks_missing", request_id=request_id, group_id=group_id)
            return None
        tables.append(list(refs[:needed]))
        lengths.append(computed + count)
        for logical in range(computed // size, (computed + count - 1) // size + 1):
            block_id = refs[logical]
            start = max(computed, logical * size) - logical * size
            end = min(computed + count, (logical + 1) * size) - logical * size
            slots.extend(range(block_id * size + start, block_id * size + end))
    return ExpectedAccess(slots, tables, lengths)


def audit_access_plan(state: KVCacheState, plan: AccessPlan, gap: Callable[..., None]) -> None:
    """Compare one frozen input plan with its own native scheduler bindings."""
    if not plan.metadata:
        if plan.phase != "warmup":
            gap("execution_access_plan_missing")
        return
    for group, item in plan.metadata.items():
        slots = metadata_field(item, "slot_mapping")
        if slots is None:
            gap("metadata_write_mapping_missing", group_id=group)
            continue
        slots = as_list(slots)
        slots = [slot for row in slots for slot in (row if isinstance(row, list) else [row])]
        size = state.group_sizes[group]

        def fail(operation, reason, *, group_id=group, **fields):
            state.emit(
                "kv_trace.violation",
                operation=operation,
                reason=reason,
                group_id=group_id,
                phase=plan.phase,
                evidence="execution_plan",
                **fields,
            )

        for slot in slots:
            if slot < -1 or slot >= state.num_blocks * size:
                fail("write", "write_outside_cache", slot=slot)
                break
            if 0 <= slot < size and not plan.allow_null_block:
                fail("write", "write_null_block", slot=slot)
                break
        expected = expected_access(state, plan, group, gap)
        if plan.phase == "dummy":
            if any(slot != -1 and not (plan.allow_null_block and 0 <= slot < size) for slot in slots):
                fail("write", "dummy_write_target")
        elif expected is not None:
            count = len(expected.slots)
            if slots[:count] != expected.slots or any(slot != -1 for slot in slots[count:]):
                fail("write", "write_target_mismatch", expected_slots=expected.slots, actual_slots=slots)
            for block in {block for row in expected.tables for block in row}:
                if block < 0 or block >= state.num_blocks or (block == 0 and not plan.allow_null_block):
                    fail("write", "invalid_scheduled_block", block_id=block)
                    break
        state.emit(
            "kv_trace.write",
            group_id=group,
            request_ids=plan.request_ids,
            phase=plan.phase,
            evidence="execution_plan",
            full_graph=plan.full_graph,
            slots=slots,
        )
        if plan.phase != "forward":
            continue
        table = metadata_field(item, "block_tables", "block_table", "block_table_tensor")
        lengths = metadata_field(item, "seq_lens_gpu", "seq_lens", "seq_lens_list", "seq_lens_cpu")
        if table is None or lengths is None:
            gap("metadata_read_range_missing", group_id=group)
            continue
        rows = len(plan.request_ids)
        lengths = as_list(lengths)[:rows]
        columns = max(((length + size - 1) // size for length in lengths), default=0)
        table = (
            [row[:columns] for row in table[:rows]]
            if isinstance(table, (list, tuple))
            else as_list(table[:rows, :columns])
        )
        if expected is not None:
            if lengths != expected.lengths:
                fail("read", "read_length_mismatch", expected_lengths=expected.lengths, actual_lengths=lengths)
            if len(table) != len(expected.tables) or any(
                row[: len(want)] != want for row, want in zip(table, expected.tables)
            ):
                fail("read", "read_target_mismatch", expected_tables=expected.tables, actual_tables=table)
        state.emit(
            "kv_trace.read",
            group_id=group,
            request_ids=plan.request_ids,
            evidence="execution_plan",
            full_graph=plan.full_graph,
            block_table=table,
            seq_lens=lengths,
        )
