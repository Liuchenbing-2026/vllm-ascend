# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Patch upstream KDA checkpoint helpers and KimiGatedDeltaNetAttention for Ascend Triton KDA."""

from __future__ import annotations

import contextvars
import copy
import functools
import importlib
import inspect
import types
from collections.abc import Callable
from dataclasses import is_dataclass, replace
from typing import Any

import torch

from vllm_ascend.ops.kda_checkpoint import (
    KDA_PREFILL_CHECKPOINT_ALIGNMENT,
    build_kda_kv_cache_spec,
    export_kda_prefill_checkpoint,
    get_triton_kda_checkpoint_buffer,
    initialize_kda_checkpoints,
)
from vllm_ascend.ops.triton.kda.kda import chunk_kda_with_fused_gate as ascend_chunk_kda_with_fused_gate

_KDA_CKPT_PATCH_ATTR = "_ascend_kda_checkpoint_patched"
_KDA_CKPT_HOOK_VAR_ATTR = "_ascend_kda_checkpoint_hook_var"


def _safe_param_names(fn: Any) -> frozenset[str] | None:
    """Return explicit parameter names for ``fn``, or ``None`` if uninspectable."""
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def _code_references_name(fn: Any, name: str) -> bool | None:
    """Return whether ``fn`` bytecode (including nested code objects) references ``name``."""
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


def _layer_has_native_kda_checkpoint(layer_cls: Any) -> bool:
    """Detect native upstream prefill checkpoint support from class attributes and bytecode."""
    cls_dict = getattr(layer_cls, "__dict__", {})
    for method_name in ("get_kv_cache_checkpoint_alignment", "get_prefill_checkpoint_alignment"):
        method = cls_dict.get(method_name)
        if method is not None and not getattr(method, _KDA_CKPT_PATCH_ATTR, False):
            return True
    orig_forward = cls_dict.get("_forward", getattr(layer_cls, "_forward", None))
    if orig_forward is not None and not getattr(orig_forward, _KDA_CKPT_PATCH_ATTR, False):
        if _code_references_name(orig_forward, "checkpoint_state") is True:
            return True
    return False


def _non_bool_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _clone_with_updates(obj: Any, updates: dict[str, Any]) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **updates)
    cloned = copy.copy(obj)
    for field_name, value in updates.items():
        setattr(cloned, field_name, value)
    return cloned


def _select_checkpoint_raw_qkv(raw_qkv_ns: Any, attn_metadata: Any, q: Any) -> tuple[Any, bool]:
    num_decodes = _non_bool_int(getattr(attn_metadata, "num_decodes", 0))
    decode_tokens = _non_bool_int(getattr(attn_metadata, "num_decode_tokens", 0)) if num_decodes > 0 else 0
    if decode_tokens <= 0:
        return raw_qkv_ns, False
    q_tokens = q.shape[1] if isinstance(q, torch.Tensor) and q.ndim >= 2 else None
    if q_tokens is not None and isinstance(raw_qkv_ns, torch.Tensor) and q_tokens == raw_qkv_ns.shape[0]:
        return raw_qkv_ns, False
    return raw_qkv_ns[decode_tokens:], True


def _scatter_masked_tensor(value: torch.Tensor, full_mask: torch.Tensor, pad_val: int) -> torch.Tensor:
    """Scatter compact checkpoint rows onto ``full_mask`` True positions without D2H sync."""
    if value.shape[0] == 0:
        return value.new_full((full_mask.shape[0], *value.shape[1:]), pad_val)
    # Clamp indices to [0, base_len - 1] so False rows safely gather before torch.where overwrites.
    src_indices = (full_mask.to(dtype=torch.int64).cumsum(dim=0) - 1).clamp(min=0, max=value.shape[0] - 1)
    gathered = value.index_select(0, src_indices)
    mask_view = full_mask.view(full_mask.shape[0], *([1] * (value.ndim - 1)))
    return torch.where(mask_view, gathered, torch.full_like(gathered, pad_val))


def _resolve_cpu_cu_seqlens(
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    expected_boundaries: int,
) -> torch.Tensor | None:
    for candidate in (cu_seqlens_cpu, cu_seqlens):
        if (
            isinstance(candidate, torch.Tensor)
            and candidate.device.type == "cpu"
            and candidate.ndim == 1
            and candidate.shape[0] == expected_boundaries
        ):
            return candidate
    return None


def _scatter_checkpoint_with_mask(
    checkpoint: Any,
    base_len: int,
    full_mask: torch.Tensor,
    pad_values: dict[str, int],
    *,
    full_mask_list: list[bool] | None,
) -> Any:
    updates: dict[str, Any] = {}
    cached_dev: Any = None
    cached_mask: torch.Tensor | None = None
    for field_name, pad_val in pad_values.items():
        value = getattr(checkpoint, field_name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == base_len:
            target_dev = value.device
            if cached_mask is None or target_dev != cached_dev:
                cached_dev = target_dev
                if full_mask.device == target_dev:
                    cached_mask = full_mask
                else:
                    cached_mask = full_mask.to(device=target_dev, non_blocking=True)
            updates[field_name] = _scatter_masked_tensor(value, cached_mask, pad_val)

    for field_name in ("offsets", "query_lens"):
        value = getattr(checkpoint, field_name, None)
        if full_mask_list is not None:
            if isinstance(value, (tuple, list)) and len(value) == base_len:
                it = iter(value)
                scattered = [next(it) if keep else 0 for keep in full_mask_list]
                updates[field_name] = tuple(scattered) if isinstance(value, tuple) else scattered
        elif value is not None:
            # Clear stale host lists on device-only fallback without triggering D2H .tolist().
            updates[field_name] = None
    return _clone_with_updates(checkpoint, updates) if updates else checkpoint


def _align_checkpoint_rows(
    checkpoint: Any,
    num_seqs: int,
    num_decodes: int,
    peeled_decodes: bool,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> Any:
    """Align compacted checkpoint rows with ``num_seqs`` rows (handling decodes & zero-len pads)."""
    if checkpoint is None or num_seqs <= 0:
        return checkpoint
    ckpt_offsets = getattr(checkpoint, "checkpoint_offsets", None)
    if not isinstance(ckpt_offsets, torch.Tensor) or ckpt_offsets.ndim < 1:
        return checkpoint

    base_len = ckpt_offsets.shape[0]
    if peeled_decodes and num_decodes > 0 and base_len == num_decodes + num_seqs:
        updates: dict[str, Any] = {}
        for field_name in ("checkpoint_offsets", "state_indices", "request_rows", "block_cols"):
            value = getattr(checkpoint, field_name, None)
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == base_len:
                updates[field_name] = value[num_decodes:]
        for field_name in ("offsets", "query_lens"):
            value = getattr(checkpoint, field_name, None)
            if isinstance(value, (tuple, list)) and len(value) == base_len:
                updates[field_name] = value[num_decodes:]
        return _clone_with_updates(checkpoint, updates) if updates else checkpoint

    if base_len >= num_seqs:
        return checkpoint

    prefix_rows = 0 if peeled_decodes else max(num_decodes, 0)
    if prefix_rows > num_seqs:
        return checkpoint

    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    pad_values = {
        "checkpoint_offsets": 0,
        "state_indices": NULL_BLOCK_ID,
        "request_rows": 0,
        "block_cols": -1,
    }
    if prefix_rows > 0 and prefix_rows + base_len == num_seqs:
        updates = {}
        for field_name, pad_val in pad_values.items():
            value = getattr(checkpoint, field_name, None)
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == base_len:
                prefix = torch.full(
                    (prefix_rows, *value.shape[1:]),
                    pad_val,
                    dtype=value.dtype,
                    device=value.device,
                )
                updates[field_name] = torch.cat((prefix, value), dim=0)
        for field_name in ("offsets", "query_lens"):
            value = getattr(checkpoint, field_name, None)
            if isinstance(value, (tuple, list)) and len(value) == base_len:
                pad_seq = [0] * prefix_rows
                updates[field_name] = tuple(pad_seq) + value if isinstance(value, tuple) else pad_seq + value
        return _clone_with_updates(checkpoint, updates) if updates else checkpoint

    cu_cpu = _resolve_cpu_cu_seqlens(cu_seqlens, cu_seqlens_cpu, num_seqs + 1)
    if cu_cpu is not None:
        live_prefill_cpu = (cu_cpu[prefix_rows + 1 : num_seqs + 1] - cu_cpu[prefix_rows:num_seqs]) > 0
        if int(live_prefill_cpu.sum().item()) != base_len:
            return checkpoint
        full_mask_cpu = torch.zeros((num_seqs,), dtype=torch.bool)
        full_mask_cpu[prefix_rows:] = live_prefill_cpu
        return _scatter_checkpoint_with_mask(
            checkpoint,
            base_len,
            full_mask_cpu,
            pad_values,
            full_mask_list=full_mask_cpu.tolist(),
        )

    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1 or cu_seqlens.shape[0] != num_seqs + 1:
        return checkpoint

    live_prefill_dev = (cu_seqlens[prefix_rows + 1 : num_seqs + 1] - cu_seqlens[prefix_rows:num_seqs]) > 0
    if prefix_rows > 0:
        full_mask_dev = torch.cat(
            (cu_seqlens.new_zeros((prefix_rows,), dtype=torch.bool), live_prefill_dev),
            dim=0,
        )
    else:
        full_mask_dev = live_prefill_dev
    return _scatter_checkpoint_with_mask(
        checkpoint,
        base_len,
        full_mask_dev,
        pad_values,
        full_mask_list=None,
    )


def _make_kda_dispatch_wrapper(
    hook_var: contextvars.ContextVar[Callable[..., Any] | None] | None = None,
) -> tuple[Callable[..., Any], contextvars.ContextVar[Callable[..., Any] | None]]:
    active_hook: contextvars.ContextVar[Callable[..., Any] | None] = (
        hook_var if hook_var is not None else contextvars.ContextVar("ascend_kda_checkpoint_hook", default=None)
    )

    @functools.wraps(ascend_chunk_kda_with_fused_gate)
    def _dispatch_chunk_kda_with_fused_gate(*args: Any, **kwargs: Any) -> Any:
        hook = active_hook.get()
        if hook is not None:
            return hook(ascend_chunk_kda_with_fused_gate, *args, **kwargs)
        return ascend_chunk_kda_with_fused_gate(*args, **kwargs)

    setattr(_dispatch_chunk_kda_with_fused_gate, _KDA_CKPT_PATCH_ATTR, True)
    setattr(_dispatch_chunk_kda_with_fused_gate, _KDA_CKPT_HOOK_VAR_ATTR, active_hook)
    return _dispatch_chunk_kda_with_fused_gate, active_hook


def _install_kda_dispatch_binding(
    target_mod: Any,
    dispatch_fn: Callable[..., Any] | None,
    hook_var: contextvars.ContextVar[Callable[..., Any] | None] | None,
) -> tuple[Callable[..., Any] | None, contextvars.ContextVar[Callable[..., Any] | None] | None]:
    current = getattr(target_mod, "chunk_kda_with_fused_gate", None)
    if current is None:
        return dispatch_fn, hook_var
    if getattr(current, _KDA_CKPT_PATCH_ATTR, False):
        existing_var = getattr(current, _KDA_CKPT_HOOK_VAR_ATTR, hook_var)
        return current, existing_var

    param_names = _safe_param_names(current)
    # Strict no-op post-upstream-pin (explicit checkpoint_state param) and fail-safe when uninspectable.
    if param_names is None or "checkpoint_state" in param_names:
        return dispatch_fn, hook_var

    if dispatch_fn is None:
        dispatch_fn, hook_var = _make_kda_dispatch_wrapper(hook_var)
    elif hook_var is None:
        hook_var = getattr(dispatch_fn, _KDA_CKPT_HOOK_VAR_ATTR, None)

    target_mod.chunk_kda_with_fused_gate = dispatch_fn  # type: ignore[attr-defined]
    return dispatch_fn, hook_var


def _patch_nvidia_triton_kda_binding() -> tuple[
    Callable[..., Any] | None, contextvars.ContextVar[Callable[..., Any] | None] | None
]:
    try:
        nvidia_kda_mod = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
    except ImportError:
        return None, None
    return _install_kda_dispatch_binding(nvidia_kda_mod, None, None)


def _patch_kimi_gdn_linear_attn(
    dispatch_fn: Callable[..., Any] | None,
    hook_var: contextvars.ContextVar[Callable[..., Any] | None] | None,
) -> None:
    try:
        kimi_gdn_mod = importlib.import_module("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
    except ImportError:
        return

    layer_cls = getattr(kimi_gdn_mod, "KimiGatedDeltaNetAttention", None)
    if layer_cls is None or _layer_has_native_kda_checkpoint(layer_cls):
        return

    dispatch_fn, hook_var = _install_kda_dispatch_binding(kimi_gdn_mod, dispatch_fn, hook_var)

    orig_init = layer_cls.__init__
    if not getattr(orig_init, _KDA_CKPT_PATCH_ATTR, False):

        @functools.wraps(orig_init)
        def _init_with_checkpoints(self: Any, *args: Any, **kwargs: Any) -> None:
            orig_init(self, *args, **kwargs)
            vllm_config = kwargs.get("vllm_config")
            if vllm_config is None and len(args) >= 2:
                vllm_config = args[1]
            if vllm_config is None:
                vllm_config = getattr(self, "vllm_config", None)
            if getattr(self, "_triton_checkpoint_spec", None) is None:
                initialize_kda_checkpoints(self, vllm_config)
            else:
                self._ascend_internal_checkpoint_enabled = True

        setattr(_init_with_checkpoints, _KDA_CKPT_PATCH_ATTR, True)
        layer_cls.__init__ = _init_with_checkpoints

    if not hasattr(layer_cls, "get_prefill_checkpoint_alignment"):

        def _get_prefill_checkpoint_alignment(self: Any) -> int | None:
            if getattr(self, "_ascend_internal_checkpoint_enabled", False):
                return KDA_PREFILL_CHECKPOINT_ALIGNMENT
            return None

        setattr(_get_prefill_checkpoint_alignment, _KDA_CKPT_PATCH_ATTR, True)
        layer_cls.get_prefill_checkpoint_alignment = _get_prefill_checkpoint_alignment

    if not hasattr(layer_cls, "get_kv_cache_checkpoint_alignment"):

        def _get_kv_cache_checkpoint_alignment(self: Any, vllm_config: Any) -> int | None:
            if getattr(self, "_ascend_internal_checkpoint_enabled", False):
                return KDA_PREFILL_CHECKPOINT_ALIGNMENT
            return None

        setattr(_get_kv_cache_checkpoint_alignment, _KDA_CKPT_PATCH_ATTR, True)
        layer_cls.get_kv_cache_checkpoint_alignment = _get_kv_cache_checkpoint_alignment

    orig_get_kv_spec = getattr(layer_cls, "get_kv_cache_spec", None)
    if orig_get_kv_spec is not None and not getattr(orig_get_kv_spec, _KDA_CKPT_PATCH_ATTR, False):

        @functools.wraps(orig_get_kv_spec)
        def _get_kv_cache_spec(self: Any, vllm_config: Any) -> Any:
            return build_kda_kv_cache_spec(self, vllm_config, orig_get_kv_spec(self, vllm_config))

        setattr(_get_kv_cache_spec, _KDA_CKPT_PATCH_ATTR, True)
        layer_cls.get_kv_cache_spec = _get_kv_cache_spec

    orig_forward = getattr(layer_cls, "_forward", None)
    if orig_forward is None or hook_var is None or getattr(orig_forward, _KDA_CKPT_PATCH_ATTR, False):
        return
    # Fail-safe: skip wrapping _forward if uninspectable/compiled (None) or already handles checkpoints (True).
    if _code_references_name(orig_forward, "checkpoint_state") is not False:
        return

    _wrap_kimi_gdn_forward(layer_cls, orig_forward, hook_var)


def _extract_eager_break_decorator(fn: Any) -> tuple[Any, Callable[[Any], Any] | None]:
    """Unwrap an outer ``@eager_break_during_capture`` decorator so it can be re-applied outermost."""
    inner = getattr(fn, "__wrapped__", None)
    code = getattr(fn, "__code__", None)
    if inner is None or not isinstance(code, types.CodeType):
        return fn, None
    if "BreakableCUDAGraphCapture" not in code.co_names or "add_eager" not in code.co_names:
        return fn, None

    fn_globals = getattr(fn, "__globals__", None)
    decorator = fn_globals.get("eager_break_during_capture") if isinstance(fn_globals, dict) else None
    if not callable(decorator):
        try:
            bc_mod = importlib.import_module("vllm.compilation.breakable_cudagraph")
            decorator = getattr(bc_mod, "eager_break_during_capture", None)
        except ImportError:
            decorator = None
    if not callable(decorator):
        return fn, None
    return inner, decorator


def _wrap_kimi_gdn_forward(
    layer_cls: Any,
    orig_forward: Any,
    hook_var: contextvars.ContextVar[Callable[..., Any] | None],
) -> None:
    """Wrap ``KimiGatedDeltaNetAttention._forward`` inside any outer ``@eager_break_during_capture``."""
    import vllm.forward_context as forward_context_mod
    import vllm.model_executor.layers.mamba.mamba_utils as mamba_utils_mod

    inner_forward, eager_break_decorator = _extract_eager_break_decorator(orig_forward)

    @functools.wraps(inner_forward)
    def _forward_with_checkpoint(
        self: Any,
        mixed_qkv: Any,
        g1: Any,
        g2: Any,
        beta: Any,
        core_attn_out: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not getattr(self, "_ascend_internal_checkpoint_enabled", False):
            return inner_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)

        attn_metadata_raw = forward_context_mod.get_forward_context().attn_metadata
        attn_metadata = attn_metadata_raw.get(self.prefix) if isinstance(attn_metadata_raw, dict) else None
        checkpoint = getattr(attn_metadata, "checkpoint", None)
        if attn_metadata is None or checkpoint is None or _non_bool_int(getattr(attn_metadata, "num_prefills", 0)) <= 0:
            return inner_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)

        num_actual_tokens = _non_bool_int(getattr(attn_metadata, "num_actual_tokens", mixed_qkv.shape[0]))
        trimmed_qkv = mixed_qkv[:num_actual_tokens]
        non_spec_token_indx = getattr(attn_metadata, "non_spec_token_indx", None)
        if getattr(attn_metadata, "spec_sequence_masks", None) is not None and non_spec_token_indx is not None:
            raw_qkv_ns = trimmed_qkv.index_select(0, non_spec_token_indx)
        else:
            raw_qkv_ns = trimmed_qkv

        num_decodes = _non_bool_int(getattr(attn_metadata, "num_decodes", 0))
        dim_first = mamba_utils_mod.is_conv_state_dim_first()

        def _chunk_kda_and_export(base_chunk_fn: Callable[..., Any], *call_args: Any, **call_kwargs: Any) -> Any:
            cu_seqlens = call_kwargs.get("cu_seqlens")
            initial_state = call_kwargs.get("initial_state")
            q_arg = call_kwargs.get("q") if "q" in call_kwargs else (call_args[0] if call_args else None)
            num_seqs = (
                initial_state.shape[0]
                if initial_state is not None
                else (cu_seqlens.numel() - 1 if cu_seqlens is not None else 0)
            )
            raw_qkv_for_export, peeled_decodes = _select_checkpoint_raw_qkv(raw_qkv_ns, attn_metadata, q_arg)
            active_checkpoint = _align_checkpoint_rows(
                checkpoint,
                num_seqs,
                num_decodes,
                peeled_decodes,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=getattr(attn_metadata, "non_spec_query_start_loc_cpu", None),
            )
            checkpoint_state, checkpoint_offsets = get_triton_kda_checkpoint_buffer(
                active_checkpoint,
                self._triton_checkpoint_spec,
                num_seqs,
                reference_tensor=self.kv_cache[1],
            )
            call_kwargs["checkpoint_state"] = checkpoint_state
            call_kwargs["checkpoint_offsets"] = checkpoint_offsets
            output, last_state = base_chunk_fn(*call_args, **call_kwargs)
            export_kda_prefill_checkpoint(
                self._checkpoint_exporter,
                active_checkpoint,
                raw_qkv=raw_qkv_for_export,
                conv_state=self.kv_cache[0],
                conv_state_is_dim_first=dim_first,
                recurrent_checkpoint=checkpoint_state,
                recurrent_state=self.kv_cache[1],
                cu_seqlens=cu_seqlens,
            )
            return output, last_state

        token = hook_var.set(_chunk_kda_and_export)
        try:
            return inner_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)
        finally:
            hook_var.reset(token)

    setattr(_forward_with_checkpoint, _KDA_CKPT_PATCH_ATTR, True)
    # Keep @eager_break_during_capture outermost so capture skips host/graph-break work
    # and BreakableCUDAGraphCapture records _forward_with_checkpoint for eager replay.
    wrapped_forward = (
        eager_break_decorator(_forward_with_checkpoint)
        if eager_break_decorator is not None
        else _forward_with_checkpoint
    )
    setattr(wrapped_forward, _KDA_CKPT_PATCH_ATTR, True)
    layer_cls._forward = wrapped_forward


def apply_patch() -> None:
    """Apply KDA checkpoint patches for Triton bindings and legacy attention layers."""
    dispatch_fn, hook_var = _patch_nvidia_triton_kda_binding()
    _patch_kimi_gdn_linear_attn(dispatch_fn, hook_var)


apply_patch()
