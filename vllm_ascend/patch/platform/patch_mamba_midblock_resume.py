# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Patch Mamba align-mode mid-block resume, mid-block hash masking, and sparse decode retention."""

from __future__ import annotations

import functools
import inspect
import sys
import types
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import torch
import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator_mod
import vllm.v1.core.kv_cache_manager as kv_cache_manager_mod
import vllm.v1.core.single_type_kv_cache_manager as st_mgr
import vllm.v1.kv_cache_interface as kv_cache_interface_mod
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MambaSpec

_MIDBLOCK_RULE_ATTR = "_ascend_midblock_checkpoint_rule"
_MIDBLOCK_MANAGER_ATTR = "_ascend_midblock_manager_patched"
_DECODE_RETENTION_ATTR = "_ascend_decode_retention_patched"
_ACCEPTS_REQUEST_ATTR = "_ascend_accepts_request"

_CHECKPOINT_VALID_MODULES = (
    "vllm.v1.kv_cache_interface",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.core.single_type_kv_cache_manager",
    "vllm.model_executor.layers.mamba.checkpoint",
)

_EXPECTED_CHECKPOINT_PARAMS = (
    "query_start",
    "query_end",
    "checkpoint_position",
    "hash_block_size",
    "mamba_block_size",
    "checkpoint_alignment",
)

_EXPECTED_NEEDS_CHECKPOINT_PARAMS = (
    "self",
    "request_id",
    "query_start",
    "query_end",
    "checkpoint_position",
    "prefill_end",
)


def _safe_param_names(fn: Any) -> tuple[str, ...] | None:
    """Return ordered parameter names for ``fn``, or ``None`` if uninspectable."""
    try:
        return tuple(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def _code_references_name(fn: Any, name: str) -> bool | None:
    """Inspect in-memory Python bytecode (without disk source lookup) for ``name``."""
    try:
        unwrapped = inspect.unwrap(fn)
    except Exception:
        return None
    root_code = getattr(unwrapped, "__code__", None)
    if not isinstance(root_code, types.CodeType):
        return None

    stack = [root_code]
    while stack:
        code = stack.pop()
        if name in code.co_names or name in code.co_varnames or name in code.co_freevars:
            return True
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
    return False


def is_mamba_prefill_checkpoint_valid(
    query_start: int,
    query_end: int,
    checkpoint_position: int,
    hash_block_size: int,
    mamba_block_size: int,
    checkpoint_alignment: int | None,
) -> bool:
    """Allow mid-block prefill queries to export a checkpoint into their first private block."""
    if not checkpoint_alignment:
        return False
    assert checkpoint_alignment > 0

    checkpoint_col = cdiv(query_end, mamba_block_size) - 2
    return (
        query_start % hash_block_size == 0
        # A mid-block query may export into its first column (private copy).
        and checkpoint_col >= query_start // mamba_block_size
        and query_start + hash_block_size <= checkpoint_position
        and query_start < checkpoint_position < query_end
        and (checkpoint_position - query_start) % checkpoint_alignment == 0
    )


setattr(is_mamba_prefill_checkpoint_valid, _MIDBLOCK_RULE_ATTR, True)


def apply_mamba_midblock_checkpoint_rule() -> None:
    """Rebind ``is_mamba_prefill_checkpoint_valid`` across interface and consumer modules."""
    current = getattr(kv_cache_interface_mod, "is_mamba_prefill_checkpoint_valid", None)
    if current is None:
        return
    if not getattr(current, _MIDBLOCK_RULE_ATTR, False):
        if _safe_param_names(current) != _EXPECTED_CHECKPOINT_PARAMS:
            return
        try:
            already_allows_midblock = bool(
                current(
                    query_start=128,
                    query_end=450,
                    checkpoint_position=384,
                    hash_block_size=128,
                    mamba_block_size=256,
                    checkpoint_alignment=64,
                )
            )
        except Exception:
            return
        if already_allows_midblock:
            return
        kv_cache_interface_mod.is_mamba_prefill_checkpoint_valid = is_mamba_prefill_checkpoint_valid

    target_fn = kv_cache_interface_mod.is_mamba_prefill_checkpoint_valid
    for mod_name in _CHECKPOINT_VALID_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "is_mamba_prefill_checkpoint_valid", None) is current:
            mod.is_mamba_prefill_checkpoint_valid = target_fn  # type: ignore[attr-defined]


def _resolve_base_mamba_manager_cls() -> type[Any]:
    mamba_cls = st_mgr.MambaManager
    return next((cls for cls in mamba_cls.__mro__ if cls.__name__ == "MambaManager"), mamba_cls)


def _needs_internal_checkpoint_has_midblock_support(fn: Any) -> bool:
    """Check via bytecode + behavioral probe whether ``fn`` already handles mid-block resume."""
    if _safe_param_names(fn) != _EXPECTED_NEEDS_CHECKPOINT_PARAMS:
        return True
    bytecode_match = _code_references_name(fn, "reuses_resumed_block")
    if bytecode_match is None or bytecode_match:
        return True

    mamba_fields = {f.name for f in fields(MambaSpec)}
    spec_kwargs: dict[str, Any] = {
        "block_size": 256,
        "shapes": ((1,),),
        "dtypes": (torch.float32,),
        "num_prefill_checkpoint_blocks": 1,
        "prefill_checkpoint_alignment": 64,
    }
    if "mamba_cache_mode" in mamba_fields:
        spec_kwargs["mamba_cache_mode"] = "align"
    try:
        probe_spec = MambaSpec(**spec_kwargs)
        probe = SimpleNamespace(
            kv_cache_spec=probe_spec,
            block_size=256,
            has_prefill_checkpoint_blocks=True,
            block_pool=SimpleNamespace(hash_block_size=128),
            req_to_blocks={"_probe": [SimpleNamespace(is_null=False)]},
            _allocated_block_reqs=set(),
            num_speculative_blocks=0,
        )
        return bool(fn(probe, "_probe", 128, 450, 384, 450))
    except Exception:
        return True


def _patch_mamba_midblock_needs_checkpoint(base_mamba_cls: type[Any]) -> None:
    orig_needs = getattr(base_mamba_cls, "_needs_internal_checkpoint", None)
    if orig_needs is None or getattr(orig_needs, _MIDBLOCK_MANAGER_ATTR, False):
        return
    if _needs_internal_checkpoint_has_midblock_support(orig_needs):
        return

    @functools.wraps(orig_needs)
    def _needs_internal_checkpoint(
        self: Any,
        request_id: str,
        query_start: int,
        query_end: int,
        checkpoint_position: int,
        prefill_end: int,
    ) -> bool:
        if orig_needs(self, request_id, query_start, query_end, checkpoint_position, prefill_end):
            return True
        if getattr(self, "mamba_cache_mode", None) != "align":
            return False
        checkpoint_idx = cdiv(query_end, self.block_size) - 2
        reuses_resumed_block = query_start % self.block_size != 0 and checkpoint_idx == query_start // self.block_size
        if not reuses_resumed_block or checkpoint_idx < 0:
            return False
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or checkpoint_idx >= len(blocks):
            return False
        # Temporarily trim `blocks` so the upstream predicate evaluates every guard
        # except the pre-existing-resumed-block rejection.
        self.req_to_blocks[request_id] = blocks[:checkpoint_idx]
        try:
            return bool(orig_needs(self, request_id, query_start, query_end, checkpoint_position, prefill_end))
        finally:
            self.req_to_blocks[request_id] = blocks

    setattr(_needs_internal_checkpoint, _MIDBLOCK_MANAGER_ATTR, True)
    base_mamba_cls._needs_internal_checkpoint = _needs_internal_checkpoint


def _record_mid_block_states(self: Any, request: Any, query_start: int, query_end: int) -> None:
    num_prompt_tokens = getattr(request, "num_prompt_tokens", query_end)
    num_tokens = getattr(request, "num_tokens", query_end)
    prefill_end = max(num_prompt_tokens, num_tokens - 1)
    if query_start + 1 >= prefill_end:
        checkpoints = getattr(self, "_checkpoints", None)
        if isinstance(checkpoints, dict):
            checkpoints.pop(request.request_id, None)
        return
    if not hasattr(self, "_mid_block_states"):
        self._mid_block_states = {}
    states = self._mid_block_states.setdefault(request.request_id, {})
    start_idx = query_start // self.block_size
    if query_start % self.block_size and start_idx < cdiv(query_end, self.block_size) - 1:
        states[start_idx] = query_start
    checkpoints = getattr(self, "_checkpoints", None)
    checkpoint = checkpoints.get(request.request_id) if isinstance(checkpoints, dict) else None
    if checkpoint is not None:
        checkpoint_position, checkpoint_idx = checkpoint
        if checkpoint_position != (checkpoint_idx + 1) * self.block_size:
            states[checkpoint_idx] = checkpoint_position
        else:
            states.pop(checkpoint_idx, None)


def _mamba_refine_block_mask(
    self: Any,
    request: Any,
    start_block: int,
    end_block: int,
    block_mask: list[bool] | None,
) -> list[bool] | None:
    states = getattr(self, "_mid_block_states", {}).get(request.request_id)
    if not states:
        return block_mask
    mask = [True] * (end_block - start_block) if block_mask is None else list(block_mask)
    for block_idx in states:
        if start_block <= block_idx < end_block:
            mask[block_idx - start_block] = False
    return mask


def _patch_mamba_midblock_state_tracking(base_mamba_cls: type[Any]) -> None:
    single_type_cls: type[Any] = st_mgr.SingleTypeKVCacheManager
    if hasattr(single_type_cls, "_refine_block_mask") and hasattr(base_mamba_cls, "_record_mid_block_states"):
        return

    def _base_refine_block_mask(
        self: Any,
        request: Any,
        start_block: int,
        end_block: int,
        block_mask: list[bool] | None,
    ) -> list[bool] | None:
        return block_mask

    if not hasattr(single_type_cls, "_refine_block_mask"):
        single_type_cls._refine_block_mask = _base_refine_block_mask

    orig_init = base_mamba_cls.__init__
    if not getattr(orig_init, _MIDBLOCK_MANAGER_ATTR, False):

        @functools.wraps(orig_init)
        def _init_midblock(self: Any, kv_cache_spec: Any, block_pool: Any, **kwargs: Any) -> None:
            orig_init(self, kv_cache_spec, block_pool, **kwargs)
            if getattr(self, "mamba_cache_mode", None) == "align":
                if not hasattr(self, "_req_chunk_ranges"):
                    req_chunk_ranges: dict[str, tuple[int, int]] = {}
                    self._req_chunk_ranges = req_chunk_ranges
                if not hasattr(self, "_mid_block_states"):
                    mid_block_states: dict[str, dict[int, int]] = {}
                    self._mid_block_states = mid_block_states

        setattr(_init_midblock, _MIDBLOCK_MANAGER_ATTR, True)
        base_mamba_cls.__init__ = _init_midblock

    _wrap_mamba_alloc_and_cache_methods(base_mamba_cls)


def _wrap_mamba_alloc_and_cache_methods(base_mamba_cls: type[Any]) -> None:
    orig_alloc = getattr(base_mamba_cls, "get_num_blocks_to_allocate", None)
    if orig_alloc is not None and not getattr(orig_alloc, _MIDBLOCK_MANAGER_ATTR, False):

        @functools.wraps(orig_alloc)
        def _get_num_blocks_to_allocate(
            self: Any,
            request_id: str,
            num_tokens: int,
            new_computed_blocks: Any,
            total_computed_tokens: int,
            num_local_computed_tokens: int | None = None,
            num_tokens_main_model: int | None = None,
            apply_admission_cap: bool = False,
            prefill_end: int = 0,
        ) -> int:
            num_new_blocks = orig_alloc(
                self,
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
                prefill_end=prefill_end,
            )
            if (
                getattr(self, "mamba_cache_mode", None) == "align"
                and not apply_admission_cap
                and num_new_blocks <= self.block_pool.num_gpu_blocks
            ):
                ranges = getattr(self, "_req_chunk_ranges", None)
                if ranges is None:
                    self._req_chunk_ranges = {}
                    ranges = self._req_chunk_ranges
                end_tokens = num_tokens_main_model if num_tokens_main_model is not None else num_tokens
                ranges[request_id] = (total_computed_tokens, end_tokens)
            return num_new_blocks

        setattr(_get_num_blocks_to_allocate, _MIDBLOCK_MANAGER_ATTR, True)
        base_mamba_cls.get_num_blocks_to_allocate = _get_num_blocks_to_allocate

    orig_pop = base_mamba_cls.pop_blocks_for_free
    if not getattr(orig_pop, _MIDBLOCK_MANAGER_ATTR, False):

        @functools.wraps(orig_pop)
        def _pop_blocks_for_free(self: Any, request_id: str) -> Any:
            if getattr(self, "mamba_cache_mode", None) == "align":
                for attr in ("_req_chunk_ranges", "_mid_block_states"):
                    mapping = getattr(self, attr, None)
                    if isinstance(mapping, dict):
                        mapping.pop(request_id, None)
            return orig_pop(self, request_id)

        setattr(_pop_blocks_for_free, _MIDBLOCK_MANAGER_ATTR, True)
        base_mamba_cls.pop_blocks_for_free = _pop_blocks_for_free

    base_mamba_cls._record_mid_block_states = _record_mid_block_states
    base_mamba_cls._refine_block_mask = _mamba_refine_block_mask

    orig_cache_blocks = base_mamba_cls.cache_blocks
    if not getattr(orig_cache_blocks, _MIDBLOCK_MANAGER_ATTR, False):

        @functools.wraps(orig_cache_blocks)
        def _cache_blocks(self: Any, request: Any, num_tokens: int, *args: Any, **kwargs: Any) -> None:
            if getattr(self, "mamba_cache_mode", None) != "align":
                orig_cache_blocks(self, request, num_tokens, *args, **kwargs)
                return

            ranges = getattr(self, "_req_chunk_ranges", None)
            chunk_range = ranges.pop(request.request_id, None) if isinstance(ranges, dict) else None
            if chunk_range is not None:
                self._record_mid_block_states(request, chunk_range[0], chunk_range[1])

            states = getattr(self, "_mid_block_states", {}).get(request.request_id)
            if not states:
                orig_cache_blocks(self, request, num_tokens, *args, **kwargs)
                return

            base_mask_fn = type(self).reachable_block_mask
            prev_instance_fn = self.__dict__.get("reachable_block_mask")

            def _refined_reachable_block_mask(*mask_args: Any, **mask_kwargs: Any) -> list[bool] | None:
                start_block = mask_kwargs["start_block"] if "start_block" in mask_kwargs else mask_args[0]
                end_block = mask_kwargs["end_block"] if "end_block" in mask_kwargs else mask_args[1]
                mask = base_mask_fn(*mask_args, **mask_kwargs)
                return self._refine_block_mask(request, start_block, end_block, mask)

            self.__dict__["reachable_block_mask"] = _refined_reachable_block_mask
            try:
                orig_cache_blocks(self, request, num_tokens, *args, **kwargs)
            finally:
                if prev_instance_fn is None:
                    self.__dict__.pop("reachable_block_mask", None)
                else:
                    self.__dict__["reachable_block_mask"] = prev_instance_fn

        setattr(_cache_blocks, _MIDBLOCK_MANAGER_ATTR, True)
        base_mamba_cls.cache_blocks = _cache_blocks


def _decode_resume_block_idx(
    self: Any,
    request_id: str,
    processed_tokens: int,
    num_prompt_tokens: int,
) -> int | None:
    idx = processed_tokens // self.block_size - 1
    if idx < num_prompt_tokens // self.block_size:
        return None
    checkpoints = getattr(self, "_checkpoints", None)
    checkpoint = checkpoints.get(request_id) if isinstance(checkpoints, dict) else None
    if checkpoint is not None and checkpoint[1] == idx and checkpoint[0] != (idx + 1) * self.block_size:
        return None
    mid_states = getattr(self, "_mid_block_states", None)
    req_mid_states = mid_states.get(request_id) if isinstance(mid_states, dict) else None
    if isinstance(req_mid_states, dict) and idx in req_mid_states:
        return None
    return idx


def _cache_blocks_on_free(self: Any, request: Any) -> None:
    if getattr(self, "mamba_cache_mode", None) != "align" or not getattr(self, "keep_decode_resume_state", False):
        return
    blocks = self.req_to_blocks.get(request.request_id, [])
    num_in_flight = getattr(request, "num_in_flight_tokens", 0)
    num_computed = getattr(request, "num_computed_tokens", 0)
    num_prompt = getattr(request, "num_prompt_tokens", 0)
    idx = self._decode_resume_block_idx(
        request.request_id,
        max(0, num_computed - num_in_flight),
        num_prompt,
    )
    num_hashed_tokens = len(getattr(request, "block_hashes", ())) * self.block_pool.hash_block_size
    if (
        idx is None
        or idx >= len(blocks)
        or blocks[idx].is_null
        or blocks[idx].block_hash is not None
        or num_hashed_tokens < (idx + 1) * self.block_size
    ):
        return
    self.block_pool.cache_full_blocks(
        request=request,
        blocks=blocks,
        num_cached_blocks=idx,
        num_full_blocks=idx + 1,
        block_size=self.block_size,
        kv_cache_group_id=self.kv_cache_group_id,
    )


def _has_upstream_decode_retention(base_mamba_cls: type[Any]) -> bool:
    single_type_cls = st_mgr.SingleTypeKVCacheManager
    if hasattr(single_type_cls, "cache_blocks_on_free") and hasattr(base_mamba_cls, "_decode_resume_block_idx"):
        return True
    coord_free_params = _safe_param_names(getattr(kv_cache_coordinator_mod.KVCacheCoordinator, "free", None))
    return coord_free_params is not None and "request" in coord_free_params


def _patch_mamba_decode_retention(base_mamba_cls: type[Any]) -> None:
    single_type_cls: type[Any] = st_mgr.SingleTypeKVCacheManager
    if _has_upstream_decode_retention(base_mamba_cls):
        return

    if not hasattr(single_type_cls, "cache_blocks_on_free"):

        def _noop_cache_blocks_on_free(self: Any, request: Any) -> None:
            return None

        single_type_cls.cache_blocks_on_free = _noop_cache_blocks_on_free

    orig_mamba_init = base_mamba_cls.__init__
    if not getattr(orig_mamba_init, _DECODE_RETENTION_ATTR, False):
        accepts_retention = "retention_interval" in (_safe_param_names(single_type_cls.__init__) or ())

        @functools.wraps(orig_mamba_init)
        def _mamba_init(self: Any, kv_cache_spec: Any, block_pool: Any, **kwargs: Any) -> None:
            if accepts_retention:
                orig_mamba_init(self, kv_cache_spec, block_pool, **kwargs)
            else:
                retention_interval = kwargs.pop("retention_interval", None)
                orig_mamba_init(self, kv_cache_spec, block_pool, **kwargs)
                if retention_interval is not None and getattr(self, "retention_interval", None) is None:
                    self.retention_interval = retention_interval
            if getattr(self, "mamba_cache_mode", None) == "align":
                self.keep_decode_resume_state = bool(
                    getattr(self, "enable_caching", False) and getattr(self, "retention_interval", None) is not None
                )
            else:
                self.keep_decode_resume_state = False

        setattr(_mamba_init, _DECODE_RETENTION_ATTR, True)
        base_mamba_cls.__init__ = _mamba_init

    base_mamba_cls._decode_resume_block_idx = _decode_resume_block_idx
    base_mamba_cls.cache_blocks_on_free = _cache_blocks_on_free

    orig_remove = base_mamba_cls.remove_skipped_blocks
    if not getattr(orig_remove, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_remove)
        def _remove_skipped_blocks(
            self: Any,
            request_id: str,
            processed_computed_tokens: int,
            num_prompt_tokens: int | None = None,
        ) -> None:
            if (
                getattr(self, "mamba_cache_mode", None) == "align"
                and getattr(self, "keep_decode_resume_state", False)
                and num_prompt_tokens is not None
            ):
                resume_idx = self._decode_resume_block_idx(request_id, processed_computed_tokens, num_prompt_tokens)
                if resume_idx is not None:
                    # Hold the latest decode-crossed boundary state until the next
                    # boundary is crossed so `cache_blocks_on_free` can persist it.
                    processed_computed_tokens = resume_idx * self.block_size + 1
            orig_remove(self, request_id, processed_computed_tokens, num_prompt_tokens)

        setattr(_remove_skipped_blocks, _DECODE_RETENTION_ATTR, True)
        base_mamba_cls.remove_skipped_blocks = _remove_skipped_blocks


def _coordinator_has_decode_retention(coordinator: Any, base_mamba_cls: type[Any]) -> bool:
    cached = getattr(coordinator, "_ascend_has_decode_retention_mamba", None)
    if cached is not None:
        return bool(cached)
    has_retention = any(
        isinstance(manager, base_mamba_cls)
        and getattr(manager, "mamba_cache_mode", None) == "align"
        and getattr(manager, "keep_decode_resume_state", False)
        for manager in getattr(coordinator, "single_type_managers", ())
    )
    coordinator._ascend_has_decode_retention_mamba = has_retention
    return has_retention


def _patch_coordinator_and_kv_manager_free(base_mamba_cls: type[Any]) -> None:
    coordinator_cls: type[Any] = kv_cache_coordinator_mod.KVCacheCoordinator
    kv_manager_cls: type[Any] = kv_cache_manager_mod.KVCacheManager

    orig_coord_init = coordinator_cls.__init__
    if not getattr(orig_coord_init, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_coord_init)
        def _coord_init(self: Any, kv_cache_config: Any, *args: Any, **kwargs: Any) -> None:
            orig_coord_init(self, kv_cache_config, *args, **kwargs)
            retention_interval = getattr(kv_cache_config, "prefix_cache_retention_interval", None)
            has_decode_retention_mamba = False
            for manager in getattr(self, "single_type_managers", ()):
                if not isinstance(manager, base_mamba_cls) or getattr(manager, "mamba_cache_mode", None) != "align":
                    continue
                if retention_interval is not None and getattr(manager, "retention_interval", None) is None:
                    manager.retention_interval = retention_interval
                keep_decode = bool(
                    getattr(manager, "enable_caching", False)
                    and getattr(manager, "retention_interval", None) is not None
                )
                manager.keep_decode_resume_state = keep_decode
                if keep_decode:
                    has_decode_retention_mamba = True
            self._ascend_has_decode_retention_mamba = has_decode_retention_mamba

        setattr(_coord_init, _DECODE_RETENTION_ATTR, True)
        coordinator_cls.__init__ = _coord_init

    orig_coord_free = coordinator_cls.free
    if not getattr(orig_coord_free, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_coord_free)
        def _coord_free(self: Any, request: Any) -> None:
            if not _coordinator_has_decode_retention(self, base_mamba_cls) or not hasattr(request, "request_id"):
                orig_coord_free(self, request.request_id if hasattr(request, "request_id") else request)
                return
            for manager in self.single_type_managers:
                if isinstance(manager, base_mamba_cls) and getattr(manager, "keep_decode_resume_state", False):
                    manager.cache_blocks_on_free(request)
            orig_coord_free(self, request.request_id)

        setattr(_coord_free, _DECODE_RETENTION_ATTR, True)
        setattr(_coord_free, _ACCEPTS_REQUEST_ATTR, True)
        coordinator_cls.free = _coord_free

    orig_coord_pop = coordinator_cls.pop_blocks_for_free
    if not getattr(orig_coord_pop, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_coord_pop)
        def _coord_pop_blocks_for_free(self: Any, request: Any) -> Any:
            if not _coordinator_has_decode_retention(self, base_mamba_cls) or not hasattr(request, "request_id"):
                return orig_coord_pop(self, request.request_id if hasattr(request, "request_id") else request)
            for manager in self.single_type_managers:
                if isinstance(manager, base_mamba_cls) and getattr(manager, "keep_decode_resume_state", False):
                    manager.cache_blocks_on_free(request)
            return orig_coord_pop(self, request.request_id)

        setattr(_coord_pop_blocks_for_free, _DECODE_RETENTION_ATTR, True)
        setattr(_coord_pop_blocks_for_free, _ACCEPTS_REQUEST_ATTR, True)
        coordinator_cls.pop_blocks_for_free = _coord_pop_blocks_for_free

    _wrap_kv_manager_free_paths(kv_manager_cls, base_mamba_cls)


def _wrap_kv_manager_free_paths(kv_manager_cls: type[Any], base_mamba_cls: type[Any]) -> None:
    orig_mgr_free = kv_manager_cls.free
    if not getattr(orig_mgr_free, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_mgr_free)
        def _mgr_free(self: Any, request: Any) -> None:
            coord = getattr(self, "coordinator", None)
            if coord is not None and _coordinator_has_decode_retention(coord, base_mamba_cls):
                coord_free = getattr(coord, "free", None)
                if coord_free is not None and getattr(coord_free, _ACCEPTS_REQUEST_ATTR, False):
                    coord_free(request)
                    return
            orig_mgr_free(self, request)

        setattr(_mgr_free, _DECODE_RETENTION_ATTR, True)
        kv_manager_cls.free = _mgr_free

    orig_mgr_pop = kv_manager_cls.pop_blocks_for_free
    if not getattr(orig_mgr_pop, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_mgr_pop)
        def _mgr_pop_blocks_for_free(self: Any, request: Any) -> Any:
            coord = getattr(self, "coordinator", None)
            if coord is not None and _coordinator_has_decode_retention(coord, base_mamba_cls):
                coord_pop = getattr(coord, "pop_blocks_for_free", None)
                if coord_pop is not None and getattr(coord_pop, _ACCEPTS_REQUEST_ATTR, False):
                    return coord_pop(request)
            return orig_mgr_pop(self, request)

        setattr(_mgr_pop_blocks_for_free, _DECODE_RETENTION_ATTR, True)
        kv_manager_cls.pop_blocks_for_free = _mgr_pop_blocks_for_free


def apply_patch() -> None:
    """Apply mid-block checkpoint resume, hash masking, and decode boundary retention patches."""
    apply_mamba_midblock_checkpoint_rule()
    base_mamba_cls = _resolve_base_mamba_manager_cls()
    _patch_mamba_midblock_needs_checkpoint(base_mamba_cls)
    _patch_mamba_midblock_state_tracking(base_mamba_cls)
    if not _has_upstream_decode_retention(base_mamba_cls):
        _patch_mamba_decode_retention(base_mamba_cls)
        _patch_coordinator_and_kv_manager_free(base_mamba_cls)


apply_patch()
