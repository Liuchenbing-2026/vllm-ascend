# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Shared Ascend KDA prefill checkpoint capability checks, buffer allocation, and export helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ops.triton.kda.gate import DEFAULT_KDA_LOWER_BOUND
from vllm_ascend.ops.triton.kda.kda import chunk_kda_with_fused_gate
from vllm_ascend.ops.triton.kda.utils import FLA_CHUNK_SIZE

KDA_PREFILL_CHECKPOINT_ALIGNMENT = FLA_CHUNK_SIZE
KDA_FLASH_CHECKPOINT_ALIGNMENT = 16
KDA_MAX_HEAD_DIM = 256
MIN_KDA_CONV_SIZE = 2

_KDA_ALIGNMENT_BY_BACKEND = {
    "flashkda": KDA_FLASH_CHECKPOINT_ALIGNMENT,
    "triton": KDA_PREFILL_CHECKPOINT_ALIGNMENT,
}

TritonKDACheckpointSpec = tuple[tuple[int, ...], torch.dtype]


def kda_prefill_checkpoint_alignment(backend: str = "triton") -> int | None:
    """Return the token alignment for KDA internal prefill checkpoints."""
    return _KDA_ALIGNMENT_BY_BACKEND.get(backend)


def _supports_checkpoint_boundary_policy(vllm_config: Any) -> bool:
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        return False
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return True
    if getattr(vllm_config, "num_hash_lookahead_tokens", 0):
        return True
    use_eagle_block_drop = getattr(speculative_config, "use_eagle_block_drop", None)
    if callable(use_eagle_block_drop):
        return not use_eagle_block_drop()
    return True


def supports_kda_prefill_checkpoint(
    vllm_config: Any,
    *,
    head_dim: int,
    conv_size: int,
    local_num_heads: int,
) -> bool:
    """Return whether KDA internal prefill checkpoints can run safely for the layer/config."""
    if vllm_config is None:
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if cache_config is None or parallel_config is None or scheduler_config is None:
        return False
    if getattr(cache_config, "mamba_cache_mode", None) != "align":
        return False
    if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
        return False
    if getattr(parallel_config, "use_ubatching", False):
        return False
    if not _supports_checkpoint_boundary_policy(vllm_config):
        logger.warning_once(
            "KDA internal prefill checkpoints use existing prefill scheduling with "
            "KV transfer or legacy EAGLE block dropping without lookahead hashes."
        )
        return False
    if not (0 < head_dim <= KDA_MAX_HEAD_DIM) or conv_size < MIN_KDA_CONV_SIZE or local_num_heads <= 0:
        return False
    if getattr(scheduler_config, "max_num_seqs", 0) <= 0:
        return False

    from vllm.v1.kv_cache_interface import MambaSpec

    return "num_prefill_checkpoint_blocks" in MambaSpec.__dataclass_fields__


def reserve_triton_kda_checkpoint_buffer(
    vllm_config: Any,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> TritonKDACheckpointSpec:
    """Reserve the `[max_num_seqs, H, V, K]` workspace spec for Triton KDA checkpoints."""
    max_seqs = vllm_config.scheduler_config.max_num_seqs
    spec: TritonKDACheckpointSpec = ((max_seqs, num_heads, head_dim, head_dim), dtype)
    from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized

    if is_workspace_manager_initialized():
        current_workspace_manager().get_simultaneous(spec)
    return spec


def get_triton_kda_checkpoint_buffer(
    checkpoint: Any | None,
    spec: TritonKDACheckpointSpec | None,
    num_seqs: int,
    *,
    reference_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return `(checkpoint_state, checkpoint_offsets)` for a KDA prefill call or `(None, None)`."""
    if checkpoint is None:
        return None, None
    assert spec is not None
    shape, dtype = spec
    from vllm.v1.worker.workspace import current_workspace_manager, is_workspace_manager_initialized

    if is_workspace_manager_initialized():
        (checkpoint_state,) = current_workspace_manager().get_simultaneous(spec)
    elif reference_tensor is not None:
        checkpoint_state = reference_tensor.new_empty((num_seqs, *shape[1:]), dtype=dtype)
    else:
        raise RuntimeError("KDA prefill checkpoint workspace requires initialized workspace manager.")
    return checkpoint_state[:num_seqs], checkpoint.checkpoint_offsets


def initialize_kda_checkpoints(layer: Any, vllm_config: Any | None = None) -> bool:
    """Initialize KDA internal prefill checkpoint attributes safely on an attention layer."""
    layer._ascend_internal_checkpoint_enabled = False
    layer._checkpoint_exporter = None
    layer._triton_checkpoint_spec = None

    cfg = vllm_config if vllm_config is not None else getattr(layer, "vllm_config", None)
    head_dim = getattr(layer, "head_dim", 0)
    conv_size = getattr(layer, "conv_size", 0)
    local_num_heads = getattr(layer, "local_num_heads", 0)
    if not supports_kda_prefill_checkpoint(
        cfg,
        head_dim=head_dim,
        conv_size=conv_size,
        local_num_heads=local_num_heads,
    ):
        return False

    get_state_dtype = getattr(layer, "get_state_dtype", None)
    if not callable(get_state_dtype):
        return False
    state_dtypes = get_state_dtype()
    if len(state_dtypes) < 2:
        return False

    from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointExporter

    layer._ascend_internal_checkpoint_enabled = True
    layer._checkpoint_exporter = MambaPrefillCheckpointExporter(state_len=conv_size - 1)
    layer._triton_checkpoint_spec = reserve_triton_kda_checkpoint_buffer(
        cfg,
        local_num_heads,
        head_dim,
        state_dtypes[1],
    )
    return True


def build_kda_kv_cache_spec(layer: Any, vllm_config: Any, base_spec: Any) -> Any:
    """Upgrade a KDA `MambaSpec` with checkpoint block capacity when enabled on the layer."""
    if not getattr(layer, "_ascend_internal_checkpoint_enabled", False):
        return base_spec
    from vllm.v1.kv_cache_interface import MambaSpec

    if (
        isinstance(base_spec, MambaSpec)
        and base_spec.mamba_cache_mode == "align"
        and getattr(base_spec, "num_prefill_checkpoint_blocks", 0) == 0
    ):
        return replace(
            base_spec,
            num_prefill_checkpoint_blocks=1,
            prefill_checkpoint_alignment=KDA_PREFILL_CHECKPOINT_ALIGNMENT,
        )
    return base_spec


def export_kda_prefill_checkpoint(
    exporter: Any,
    checkpoint: Any,
    *,
    raw_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_is_dim_first: bool,
    recurrent_checkpoint: torch.Tensor | None,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    """Normalize conv-state layout to `[block, dim, state_len]` and export KDA checkpoints."""
    normalized_conv_state = conv_state if conv_state_is_dim_first else conv_state.transpose(-1, -2)
    exporter.export(
        checkpoint,
        raw_qkv=raw_qkv,
        conv_state=normalized_conv_state,
        recurrent_checkpoint=recurrent_checkpoint,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
    )


def run_kda_checkpoint_prefill(
    *,
    exporter: Any,
    checkpoint_spec: TritonKDACheckpointSpec | None,
    checkpoint: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    raw_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_is_dim_first: bool,
    recurrent_state: torch.Tensor,
    beta: torch.Tensor | None = None,
    raw_beta: torch.Tensor | None = None,
    safe_gate: bool | None = None,
    lower_bound: float | None = DEFAULT_KDA_LOWER_BOUND,
    prebuilt_meta: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Triton KDA prefill with checkpoint capture and export conv/recurrent states."""
    checkpoint_state, checkpoint_offsets = get_triton_kda_checkpoint_buffer(
        checkpoint,
        checkpoint_spec,
        initial_state.shape[0],
        reference_tensor=recurrent_state,
    )
    output, final_state = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_gate,
        beta=beta,
        raw_beta=raw_beta,
        A_log=a_log,
        g_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        prebuilt_meta=prebuilt_meta,
        checkpoint_state=checkpoint_state,
        checkpoint_offsets=checkpoint_offsets,
    )
    if checkpoint is not None:
        export_kda_prefill_checkpoint(
            exporter,
            checkpoint,
            raw_qkv=raw_qkv,
            conv_state=conv_state,
            conv_state_is_dim_first=conv_state_is_dim_first,
            recurrent_checkpoint=checkpoint_state,
            recurrent_state=recurrent_state,
            cu_seqlens=cu_seqlens,
        )
    return output, final_state
