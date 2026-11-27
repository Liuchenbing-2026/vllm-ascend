# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Startup configuration and default-deny qualification for DSv4.1 MonoKernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig

DEFAULT_MAX_MONO_ROWS = 48
_VALID_DECODE_BACKENDS = ("auto", "stock", "latency_specialized")
_VALID_HC_MULT = 4
_HIDDEN_ALIGNMENT = 64


@dataclass(frozen=True)
class AscendDSV41MonoConfig:
    """Validated per-model MonoKernel runtime settings."""

    max_rows: int = DEFAULT_MAX_MONO_ROWS
    backend_name: str = "ascend_triton_mono"

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        hf_config: Any,
    ) -> AscendDSV41MonoConfig | None:
        """Build MonoKernel config when opted in, or fail fast listing all blockers."""
        if not resolve_mono_request(vllm_config):
            return None
        reasons = collect_unsupported_reasons(vllm_config, hf_config)
        if reasons:
            joined = "; ".join(reasons)
            raise ValueError(f"DeepSeek V4.1 --monokernel is unsupported for this deployment: {joined}")
        additional = getattr(vllm_config, "additional_config", None) or {}
        max_rows = additional.get("monokernel_max_rows", DEFAULT_MAX_MONO_ROWS)
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
            raise ValueError(f"monokernel_max_rows must be a positive integer, got {max_rows!r}")
        return cls(max_rows=max_rows)


def resolve_mono_request(vllm_config: VllmConfig) -> bool:
    """Resolve ``--monokernel`` and ``additional_config.decode_backend`` consistently."""
    kernel_config = getattr(vllm_config, "kernel_config", None)
    monokernel = bool(getattr(kernel_config, "monokernel", False))
    additional = getattr(vllm_config, "additional_config", None) or {}
    decode_backend = additional.get("decode_backend", "auto")
    if decode_backend not in _VALID_DECODE_BACKENDS:
        raise ValueError(f"Unsupported decode_backend={decode_backend!r}; expected one of {_VALID_DECODE_BACKENDS}.")
    if monokernel and decode_backend == "stock":
        raise ValueError("Conflicting options: --monokernel cannot be combined with decode_backend='stock'.")
    return monokernel or decode_backend == "latency_specialized"


def collect_unsupported_reasons(vllm_config: VllmConfig, hf_config: Any) -> list[str]:
    """Collect every startup incompatibility in one pass."""
    reasons: list[str] = []
    if getattr(vllm_config, "lora_config", None) is not None:
        reasons.append("LoRA is not supported with DeepSeek V4.1 --monokernel")

    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is not None:
        if getattr(parallel, "use_ubatching", False):
            reasons.append("micro-batching (use_ubatching) is not supported")
        if getattr(parallel, "prefill_context_parallel_size", 1) > 1:
            reasons.append("prefill_context_parallel_size > 1 is not supported")
        if getattr(parallel, "use_sequence_parallel_moe", False):
            reasons.append("use_sequence_parallel_moe is not supported")

    model_config = getattr(vllm_config, "model_config", None)
    dtype = getattr(model_config, "dtype", None)
    if dtype is not None and dtype != torch.bfloat16:
        reasons.append(f"requires bfloat16 activations, got {dtype}")

    hc_mult = int(getattr(hf_config, "hc_mult", 0) or 0)
    if hc_mult != _VALID_HC_MULT:
        reasons.append(f"requires hc_mult={_VALID_HC_MULT}, got {hc_mult}")

    hidden_size = int(getattr(hf_config, "hidden_size", 0) or 0)
    if hidden_size <= 0 or hidden_size % _HIDDEN_ALIGNMENT != 0:
        reasons.append(f"hidden_size must be a positive multiple of {_HIDDEN_ALIGNMENT}, got {hidden_size}")

    return reasons
