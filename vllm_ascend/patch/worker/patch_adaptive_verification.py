# SPDX-License-Identifier: Apache-2.0
"""Ascend compatibility patches for MRV2 adaptive speculative verification."""

from __future__ import annotations

import functools
import importlib
import inspect
import sys
import types
from contextlib import nullcontext
from typing import Any

from vllm.v1.worker.gpu.spec_decode import adaptive_verification

from vllm_ascend.core.scheduler_output import (
    require_non_bool_int,
    resolve_max_concurrent_batches,
)

_MIN_CONFIDENCE_RING_SLOTS = 2
_RING_PATCH_FLAG = "_vllm_ascend_confidence_ring_patched"


def _collect_code_names(code: types.CodeType) -> set[str]:
    """Collect bytecode ``co_names`` across a code object and nested code constants."""
    names: set[str] = set()
    stack: list[types.CodeType] = [code]
    seen: set[types.CodeType] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        names.update(cur.co_names)
        for const in cur.co_consts:
            if isinstance(const, types.CodeType) and const not in seen:
                stack.append(const)
    return names


def _uses_generic_accelerator_events(fn: Any) -> bool:
    """Detect whether upstream AdaptiveVerification uses torch.Event / accelerator."""
    if fn is None:
        return False
    try:
        target = inspect.unwrap(fn)
    except Exception:
        target = fn
    code = getattr(target, "__code__", None)
    if not isinstance(code, types.CodeType):
        return False
    names = _collect_code_names(code)
    return "accelerator" in names or ("Event" in names and "cuda" not in names)


def _make_copy_event(device: Any, *, use_generic_events: bool) -> Any:
    torch_mod = adaptive_verification.torch
    if use_generic_events and hasattr(torch_mod, "Event"):
        return torch_mod.Event(blocking=True, device=device)
    return torch_mod.cuda.Event(blocking=True)


def _get_current_stream(device: Any, *, use_generic_events: bool) -> Any:
    torch_mod = adaptive_verification.torch
    if use_generic_events and hasattr(torch_mod, "accelerator"):
        return torch_mod.accelerator.current_stream(device)
    return torch_mod.cuda.current_stream(device)


def _resize_confidence_ring(manager: Any, max_concurrent_batches: int, *, use_generic_events: bool = False) -> None:
    """Ensure confidence/event ring slots and per-slot reset lists match depth."""
    num_slots = max(_MIN_CONFIDENCE_RING_SLOTS, require_non_bool_int(max_concurrent_batches, "max_concurrent_batches"))
    stale_confidences = getattr(manager, "_stale_confidences", None)
    copy_events = getattr(manager, "_copy_events", None)
    if not isinstance(stale_confidences, list) or not isinstance(copy_events, list):
        return

    if len(stale_confidences) < num_slots:
        req_states = manager.req_states
        device = req_states.device
        max_num_reqs = req_states.max_num_reqs
        num_spec_steps = manager.num_speculative_steps
        for _ in range(num_slots - len(stale_confidences)):
            slot = adaptive_verification.CpuGpuBuffer(
                max_num_reqs,
                num_spec_steps,
                dtype=adaptive_verification.torch.float32,
                device=device,
            )
            slot.np.fill(1.0)
            stale_confidences.append(slot)
            copy_events.append(_make_copy_event(device, use_generic_events=use_generic_events))

    total_slots = len(stale_confidences)
    pending = getattr(manager, "_pending_resets", None)
    if (
        not isinstance(pending, list)
        or len(pending) != total_slots
        or any(not isinstance(item, list) for item in pending)
    ):
        manager._pending_resets = [[] for _ in range(total_slots)]


def _patched_add_request(self: Any, req_idx: int) -> None:
    """Reset ``req_idx`` in the active slot now and defer resets for other slots."""
    self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
    pending = getattr(self, "_pending_resets", None)
    num_slots = len(self._stale_confidences)
    if isinstance(pending, list) and len(pending) == num_slots and all(isinstance(r, list) for r in pending):
        for i, resets in enumerate(pending):
            if i != self._stale_idx:
                resets.append(req_idx)
    else:
        self._pending_resets = [[req_idx] if i != self._stale_idx else [] for i in range(num_slots)]
    self._confidence_probs[req_idx].fill_(1.0)


def _make_record_confidences_fn(use_generic_events: bool) -> Any:
    """Create a ring-sized ``record_confidences`` bound to the upstream event API."""

    def _patched_record_confidences(self: Any, confidence_probs: Any, input_batch: Any) -> None:
        num_reqs = input_batch.num_reqs
        num_slots = len(self._stale_confidences)
        ready_idx = (self._stale_idx + 1) % num_slots
        gpu_sync_ctx = getattr(adaptive_verification, "gpu_sync_allowed", nullcontext)
        with gpu_sync_ctx():
            self._copy_events[ready_idx].synchronize()
        pending = getattr(self, "_pending_resets", None)
        if isinstance(pending, list) and len(pending) == num_slots and all(isinstance(r, list) for r in pending):
            resets = pending[ready_idx]
            if resets:
                self._stale_confidences[ready_idx].np[resets] = 1.0
                resets.clear()
        elif isinstance(pending, list) and pending:
            self._stale_confidences[ready_idx].np[pending] = 1.0
            pending.clear()
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = _get_current_stream(self.req_states.device, use_generic_events=use_generic_events)
        self._copy_stream.wait_stream(current_stream)
        with adaptive_verification.stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    return _patched_record_confidences


def _apply_confidence_ring_patch() -> None:
    """Backport configurable confidence/event ring and per-slot row resets."""
    manager_cls = getattr(adaptive_verification, "AdaptiveVerificationManager", None)
    if manager_cls is None:
        return

    orig_init = manager_cls.__init__
    try:
        if "max_concurrent_batches" in inspect.signature(orig_init).parameters:
            return
    except (ValueError, TypeError):
        return

    orig_factory = getattr(adaptive_verification, "maybe_create_adaptive_verification_manager", None)
    factory_sig: inspect.Signature | None = None
    if callable(orig_factory) and not getattr(orig_factory, _RING_PATCH_FLAG, False):
        try:
            factory_sig = inspect.signature(orig_factory)
        except (ValueError, TypeError):
            return

    use_generic_events = _uses_generic_accelerator_events(orig_init) or _uses_generic_accelerator_events(
        getattr(manager_cls, "record_confidences", None)
    )

    if not getattr(manager_cls, _RING_PATCH_FLAG, False):

        @functools.wraps(orig_init)
        def _patched_init(
            self: Any,
            *args: Any,
            max_concurrent_batches: int = _MIN_CONFIDENCE_RING_SLOTS,
            **kwargs: Any,
        ) -> None:
            orig_init(self, *args, **kwargs)
            _resize_confidence_ring(self, max_concurrent_batches, use_generic_events=use_generic_events)

        manager_cls.__init__ = _patched_init
        manager_cls.add_request = _patched_add_request
        manager_cls.record_confidences = _make_record_confidences_fn(use_generic_events)
        setattr(manager_cls, _RING_PATCH_FLAG, True)

    if callable(orig_factory) and factory_sig is not None:
        resolved_factory_sig = factory_sig

        @functools.wraps(orig_factory)
        def _patched_factory(*args: Any, **kwargs: Any) -> Any:
            manager = orig_factory(*args, **kwargs)
            if manager is None or not isinstance(getattr(manager, "_stale_confidences", None), list):
                return manager
            vllm_config = kwargs.get("vllm_config")
            if vllm_config is None:
                try:
                    vllm_config = resolved_factory_sig.bind_partial(*args, **kwargs).arguments.get("vllm_config")
                except TypeError:
                    vllm_config = None
            _resize_confidence_ring(
                manager,
                resolve_max_concurrent_batches(vllm_config),
                use_generic_events=use_generic_events,
            )
            return manager

        setattr(_patched_factory, _RING_PATCH_FLAG, True)
        adaptive_verification.maybe_create_adaptive_verification_manager = _patched_factory
        gpu_model_runner: Any = sys.modules.get("vllm.v1.worker.gpu.model_runner")
        if gpu_model_runner is None:
            try:
                gpu_model_runner = importlib.import_module("vllm.v1.worker.gpu.model_runner")
            except ImportError:
                gpu_model_runner = None
        if gpu_model_runner is not None and hasattr(gpu_model_runner, "maybe_create_adaptive_verification_manager"):
            gpu_model_runner.maybe_create_adaptive_verification_manager = _patched_factory


_apply_confidence_ring_patch()
