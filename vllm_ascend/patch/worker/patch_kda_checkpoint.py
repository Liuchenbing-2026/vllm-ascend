# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Patch upstream KDA checkpoint helpers and KimiGatedDeltaNetAttention for Ascend Triton KDA."""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Any

from vllm_ascend.ops.kda_checkpoint import (
    KDA_PREFILL_CHECKPOINT_ALIGNMENT,
    build_kda_kv_cache_spec,
    export_kda_prefill_checkpoint,
    get_triton_kda_checkpoint_buffer,
    initialize_kda_checkpoints,
    kda_prefill_checkpoint_alignment,
    reserve_triton_kda_checkpoint_buffer,
)
from vllm_ascend.ops.triton.kda.kda import chunk_kda_with_fused_gate as ascend_chunk_kda_with_fused_gate

_KDA_CKPT_PATCH_ATTR = "_ascend_kda_checkpoint_patched"


def _patch_upstream_kda_checkpoint_module() -> None:
    try:
        kda_ckpt_mod = importlib.import_module("vllm.model_executor.layers.mamba.kda_checkpoint")
    except ImportError:
        return

    orig_align = getattr(kda_ckpt_mod, "kda_prefill_checkpoint_alignment", None)
    if orig_align is not None and not getattr(orig_align, _KDA_CKPT_PATCH_ATTR, False):
        if orig_align("triton") is None:
            patched_align = functools.wraps(orig_align)(kda_prefill_checkpoint_alignment)
            setattr(patched_align, _KDA_CKPT_PATCH_ATTR, True)
            kda_ckpt_mod.kda_prefill_checkpoint_alignment = patched_align  # type: ignore[attr-defined]

    if not hasattr(kda_ckpt_mod, "reserve_triton_kda_checkpoint_buffer"):
        kda_ckpt_mod.reserve_triton_kda_checkpoint_buffer = reserve_triton_kda_checkpoint_buffer  # type: ignore[attr-defined]
    if not hasattr(kda_ckpt_mod, "get_triton_kda_checkpoint_buffer"):
        kda_ckpt_mod.get_triton_kda_checkpoint_buffer = get_triton_kda_checkpoint_buffer  # type: ignore[attr-defined]


def _patch_nvidia_triton_kda_binding() -> Any | None:
    try:
        nvidia_kda_mod = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
    except ImportError:
        return None

    current = getattr(nvidia_kda_mod, "chunk_kda_with_fused_gate", None)
    if not getattr(current, _KDA_CKPT_PATCH_ATTR, False):

        @functools.wraps(ascend_chunk_kda_with_fused_gate)
        def _ascend_chunk_kda_with_fused_gate(*args: Any, **kwargs: Any) -> Any:
            return ascend_chunk_kda_with_fused_gate(*args, **kwargs)

        setattr(_ascend_chunk_kda_with_fused_gate, _KDA_CKPT_PATCH_ATTR, True)
        nvidia_kda_mod.chunk_kda_with_fused_gate = _ascend_chunk_kda_with_fused_gate  # type: ignore[attr-defined]
    return nvidia_kda_mod


def _forward_already_supports_checkpoint(fn: Any) -> bool:
    try:
        return "checkpoint_state" in inspect.getsource(fn)
    except (OSError, TypeError):
        return False


def _patch_kimi_gdn_linear_attn(nvidia_kda_mod: Any | None) -> None:
    try:
        kimi_gdn_mod = importlib.import_module("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
    except ImportError:
        return

    layer_cls = getattr(kimi_gdn_mod, "KimiGatedDeltaNetAttention", None)
    if layer_cls is None:
        return

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

        layer_cls.get_prefill_checkpoint_alignment = _get_prefill_checkpoint_alignment

    if not hasattr(layer_cls, "get_kv_cache_checkpoint_alignment"):

        def _get_kv_cache_checkpoint_alignment(self: Any, vllm_config: Any) -> int | None:
            if getattr(self, "_ascend_internal_checkpoint_enabled", False):
                return KDA_PREFILL_CHECKPOINT_ALIGNMENT
            return None

        layer_cls.get_kv_cache_checkpoint_alignment = _get_kv_cache_checkpoint_alignment

    orig_get_kv_spec = getattr(layer_cls, "get_kv_cache_spec", None)
    if orig_get_kv_spec is not None and not getattr(orig_get_kv_spec, _KDA_CKPT_PATCH_ATTR, False):

        @functools.wraps(orig_get_kv_spec)
        def _get_kv_cache_spec(self: Any, vllm_config: Any) -> Any:
            return build_kda_kv_cache_spec(self, vllm_config, orig_get_kv_spec(self, vllm_config))

        setattr(_get_kv_cache_spec, _KDA_CKPT_PATCH_ATTR, True)
        layer_cls.get_kv_cache_spec = _get_kv_cache_spec

    orig_forward = getattr(layer_cls, "_forward", None)
    if (
        orig_forward is None
        or nvidia_kda_mod is None
        or getattr(orig_forward, _KDA_CKPT_PATCH_ATTR, False)
        or _forward_already_supports_checkpoint(orig_forward)
    ):
        return

    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    @functools.wraps(orig_forward)
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
            return orig_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)

        attn_metadata_raw = get_forward_context().attn_metadata
        attn_metadata = attn_metadata_raw.get(self.prefix) if isinstance(attn_metadata_raw, dict) else None
        checkpoint = getattr(attn_metadata, "checkpoint", None)
        if checkpoint is None or getattr(attn_metadata, "num_prefills", 0) <= 0:
            return orig_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)

        trimmed_qkv = mixed_qkv[: attn_metadata.num_actual_tokens]
        if (
            getattr(attn_metadata, "spec_sequence_masks", None) is not None
            and getattr(attn_metadata, "non_spec_token_indx", None) is not None
        ):
            raw_qkv_ns = trimmed_qkv.index_select(0, attn_metadata.non_spec_token_indx)
        else:
            raw_qkv_ns = trimmed_qkv

        base_chunk_fn = nvidia_kda_mod.chunk_kda_with_fused_gate
        dim_first = is_conv_state_dim_first()

        def _chunk_kda_and_export(*call_args: Any, **call_kwargs: Any) -> Any:
            cu_seqlens = call_kwargs.get("cu_seqlens")
            initial_state = call_kwargs.get("initial_state")
            num_seqs = (
                initial_state.shape[0]
                if initial_state is not None
                else (cu_seqlens.numel() - 1 if cu_seqlens is not None else 0)
            )
            checkpoint_state, checkpoint_offsets = get_triton_kda_checkpoint_buffer(
                checkpoint,
                self._triton_checkpoint_spec,
                num_seqs,
                reference_tensor=self.kv_cache[1],
            )
            call_kwargs["checkpoint_state"] = checkpoint_state
            call_kwargs["checkpoint_offsets"] = checkpoint_offsets
            output, last_state = base_chunk_fn(*call_args, **call_kwargs)
            export_kda_prefill_checkpoint(
                self._checkpoint_exporter,
                checkpoint,
                raw_qkv=raw_qkv_ns,
                conv_state=self.kv_cache[0],
                conv_state_is_dim_first=dim_first,
                recurrent_checkpoint=checkpoint_state,
                recurrent_state=self.kv_cache[1],
                cu_seqlens=cu_seqlens,
            )
            return output, last_state

        nvidia_kda_mod.chunk_kda_with_fused_gate = _chunk_kda_and_export  # type: ignore[attr-defined]
        try:
            return orig_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)
        finally:
            nvidia_kda_mod.chunk_kda_with_fused_gate = base_chunk_fn  # type: ignore[attr-defined]

    setattr(_forward_with_checkpoint, _KDA_CKPT_PATCH_ATTR, True)
    layer_cls._forward = _forward_with_checkpoint


def apply_patch() -> None:
    """Apply KDA checkpoint patches for upstream helpers and attention layers."""
    _patch_upstream_kda_checkpoint_module()
    nvidia_kda_mod = _patch_nvidia_triton_kda_binding()
    _patch_kimi_gdn_linear_attn(nvidia_kda_mod)


apply_patch()
