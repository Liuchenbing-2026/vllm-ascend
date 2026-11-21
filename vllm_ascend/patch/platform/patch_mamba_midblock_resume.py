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
from typing import Any

import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator_mod
import vllm.v1.core.kv_cache_manager as kv_cache_manager_mod
import vllm.v1.core.single_type_kv_cache_manager as st_mgr
import vllm.v1.kv_cache_interface as kv_cache_interface_mod
from vllm.utils.math_utils import cdiv

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
        if tuple(inspect.signature(current).parameters) != _EXPECTED_CHECKPOINT_PARAMS:
            return
        kv_cache_interface_mod.is_mamba_prefill_checkpoint_valid = is_mamba_prefill_checkpoint_valid

    target_fn = kv_cache_interface_mod.is_mamba_prefill_checkpoint_valid
    for mod_name in _CHECKPOINT_VALID_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "is_mamba_prefill_checkpoint_valid"):
            mod.is_mamba_prefill_checkpoint_valid = target_fn  # type: ignore[attr-defined]


def _resolve_base_mamba_manager_cls() -> type[Any]:
    mamba_cls = st_mgr.MambaManager
    return next((cls for cls in mamba_cls.__mro__ if cls.__name__ == "MambaManager"), mamba_cls)


def _needs_internal_checkpoint_has_midblock_support(fn: Any) -> bool:
    try:
        return "reuses_resumed_block" in inspect.getsource(fn)
    except (OSError, TypeError):
        return False


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


def _patch_mamba_midblock_state_tracking(base_mamba_cls: type[Any]) -> None:
    single_type_cls = st_mgr.SingleTypeKVCacheManager
    if hasattr(single_type_cls, "_refine_block_mask") and hasattr(base_mamba_cls, "_record_mid_block_states"):
        return

    if not hasattr(single_type_cls, "_refine_block_mask"):

        def _refine_block_mask(
            self: Any,
            request: Any,
            start_block: int,
            end_block: int,
            block_mask: list[bool] | None,
        ) -> list[bool] | None:
            return block_mask

        single_type_cls._refine_block_mask = _refine_block_mask

    orig_init = base_mamba_cls.__init__
    if not getattr(orig_init, _MIDBLOCK_MANAGER_ATTR, False):

        @functools.wraps(orig_init)
        def _init_midblock(self: Any, kv_cache_spec: Any, block_pool: Any, **kwargs: Any) -> None:
            orig_init(self, kv_cache_spec, block_pool, **kwargs)
            if self.mamba_cache_mode == "align":
                if not hasattr(self, "_req_chunk_ranges"):
                    self._req_chunk_ranges: dict[str, tuple[int, int]] = {}
                if not hasattr(self, "_mid_block_states"):
                    self._mid_block_states: dict[str, dict[int, int]] = {}

        setattr(_init_midblock, _MIDBLOCK_MANAGER_ATTR, True)
        base_mamba_cls.__init__ = _init_midblock

    orig_alloc = base_mamba_cls.get_num_blocks_to_allocate
    if not getattr(orig_alloc, _MIDBLOCK_MANAGER_ATTR, False):

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
            for attr in ("_req_chunk_ranges", "_mid_block_states"):
                mapping = getattr(self, attr, None)
                if isinstance(mapping, dict):
                    mapping.pop(request_id, None)
            return orig_pop(self, request_id)

        setattr(_pop_blocks_for_free, _MIDBLOCK_MANAGER_ATTR, True)
        base_mamba_cls.pop_blocks_for_free = _pop_blocks_for_free

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

    base_mamba_cls._record_mid_block_states = _record_mid_block_states
    base_mamba_cls._refine_block_mask = _mamba_refine_block_mask

    orig_cache_blocks = base_mamba_cls.cache_blocks
    if not getattr(orig_cache_blocks, _MIDBLOCK_MANAGER_ATTR, False):

        @functools.wraps(orig_cache_blocks)
        def _cache_blocks(self: Any, request: Any, num_tokens: int, *args: Any, **kwargs: Any) -> None:
            if getattr(self, "mamba_cache_mode", None) == "align":
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


def _patch_single_type_retention_init(single_type_cls: type[Any]) -> None:
    orig_st_init = single_type_cls.__init__
    if getattr(orig_st_init, _DECODE_RETENTION_ATTR, False):
        return
    accepts_retention = "retention_interval" in inspect.signature(orig_st_init).parameters

    @functools.wraps(orig_st_init)
    def _st_init(self: Any, *args: Any, retention_interval: int | None = None, **kwargs: Any) -> None:
        if accepts_retention:
            orig_st_init(self, *args, retention_interval=retention_interval, **kwargs)
        else:
            orig_st_init(self, *args, **kwargs)
        if not hasattr(self, "retention_interval") or self.retention_interval is None:
            self.retention_interval = retention_interval

    setattr(_st_init, _DECODE_RETENTION_ATTR, True)
    single_type_cls.__init__ = _st_init


def _patch_mamba_decode_retention(base_mamba_cls: type[Any]) -> None:
    single_type_cls = st_mgr.SingleTypeKVCacheManager
    if hasattr(single_type_cls, "cache_blocks_on_free") and hasattr(base_mamba_cls, "_decode_resume_block_idx"):
        return

    _patch_single_type_retention_init(single_type_cls)
    if not hasattr(single_type_cls, "cache_blocks_on_free"):

        def _noop_cache_blocks_on_free(self: Any, request: Any) -> None:
            return None

        single_type_cls.cache_blocks_on_free = _noop_cache_blocks_on_free

    orig_mamba_init = base_mamba_cls.__init__
    if not getattr(orig_mamba_init, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_mamba_init)
        def _mamba_init(self: Any, kv_cache_spec: Any, block_pool: Any, **kwargs: Any) -> None:
            orig_mamba_init(self, kv_cache_spec, block_pool, **kwargs)
            self.keep_decode_resume_state = bool(
                getattr(self, "enable_caching", False) and getattr(self, "retention_interval", None) is not None
            )

        setattr(_mamba_init, _DECODE_RETENTION_ATTR, True)
        base_mamba_cls.__init__ = _mamba_init

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
        return idx

    def _cache_blocks_on_free(self: Any, request: Any) -> None:
        if not getattr(self, "keep_decode_resume_state", False):
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
            if getattr(self, "keep_decode_resume_state", False) and num_prompt_tokens is not None:
                resume_idx = self._decode_resume_block_idx(request_id, processed_computed_tokens, num_prompt_tokens)
                if resume_idx is not None:
                    # Hold the latest decode-crossed boundary state until the next
                    # boundary is crossed so `cache_blocks_on_free` can persist it.
                    processed_computed_tokens = resume_idx * self.block_size + 1
            orig_remove(self, request_id, processed_computed_tokens, num_prompt_tokens)

        setattr(_remove_skipped_blocks, _DECODE_RETENTION_ATTR, True)
        base_mamba_cls.remove_skipped_blocks = _remove_skipped_blocks


def _patch_coordinator_and_kv_manager_free() -> None:
    coordinator_cls = kv_cache_coordinator_mod.KVCacheCoordinator
    kv_manager_cls = kv_cache_manager_mod.KVCacheManager

    orig_coord_init = coordinator_cls.__init__
    if not getattr(orig_coord_init, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_coord_init)
        def _coord_init(self: Any, kv_cache_config: Any, *args: Any, **kwargs: Any) -> None:
            orig_coord_init(self, kv_cache_config, *args, **kwargs)
            retention_interval = getattr(kv_cache_config, "prefix_cache_retention_interval", None)
            for manager in getattr(self, "single_type_managers", ()):
                if getattr(manager, "retention_interval", None) is None:
                    manager.retention_interval = retention_interval
                if hasattr(manager, "keep_decode_resume_state"):
                    manager.keep_decode_resume_state = bool(
                        getattr(manager, "enable_caching", False) and manager.retention_interval is not None
                    )

        setattr(_coord_init, _DECODE_RETENTION_ATTR, True)
        coordinator_cls.__init__ = _coord_init

    orig_coord_free = coordinator_cls.free
    if not getattr(orig_coord_free, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_coord_free)
        def _coord_free(self: Any, request: Any) -> None:
            if hasattr(request, "request_id"):
                for manager in self.single_type_managers:
                    cache_on_free = getattr(manager, "cache_blocks_on_free", None)
                    if cache_on_free is not None:
                        cache_on_free(request)
                orig_coord_free(self, request.request_id)
            else:
                orig_coord_free(self, request)

        setattr(_coord_free, _DECODE_RETENTION_ATTR, True)
        setattr(_coord_free, _ACCEPTS_REQUEST_ATTR, True)
        coordinator_cls.free = _coord_free

    orig_coord_pop = coordinator_cls.pop_blocks_for_free
    if not getattr(orig_coord_pop, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_coord_pop)
        def _coord_pop_blocks_for_free(self: Any, request: Any) -> Any:
            if hasattr(request, "request_id"):
                for manager in self.single_type_managers:
                    cache_on_free = getattr(manager, "cache_blocks_on_free", None)
                    if cache_on_free is not None:
                        cache_on_free(request)
                return orig_coord_pop(self, request.request_id)
            return orig_coord_pop(self, request)

        setattr(_coord_pop_blocks_for_free, _DECODE_RETENTION_ATTR, True)
        setattr(_coord_pop_blocks_for_free, _ACCEPTS_REQUEST_ATTR, True)
        coordinator_cls.pop_blocks_for_free = _coord_pop_blocks_for_free

    orig_mgr_free = kv_manager_cls.free
    if not getattr(orig_mgr_free, _DECODE_RETENTION_ATTR, False):

        @functools.wraps(orig_mgr_free)
        def _mgr_free(self: Any, request: Any) -> None:
            coord_free = getattr(self.coordinator, "free", None)
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
            coord_pop = getattr(self.coordinator, "pop_blocks_for_free", None)
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
    if not (
        hasattr(st_mgr.SingleTypeKVCacheManager, "cache_blocks_on_free")
        and hasattr(base_mamba_cls, "_decode_resume_block_idx")
    ):
        _patch_mamba_decode_retention(base_mamba_cls)
        _patch_coordinator_and_kv_manager_free()


apply_patch()
