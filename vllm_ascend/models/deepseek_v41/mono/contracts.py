# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Typed contracts for per-layer DeepSeek V4.1 MonoKernel execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class MonoStepPlan:
    """Deterministic per-step decision shared across all TP ranks."""

    active: bool
    num_tokens: int = 0
    reason: str = ""


@dataclass(frozen=True)
class MonoLayerBinding:
    """Zero-copy layer descriptor and cross-layer seam-fusion boundary flag."""

    layer_idx: int
    layer: Any
    can_defer_post: bool


@dataclass(frozen=True)
class PendingPostState:
    """Deferred FFN output + mixing coefficients folded into the next layer's seam."""

    output: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


class DSV41MonoLayerBackend(Protocol):
    """Pluggable execution backend for one fused DSv4.1 decode layer.

    External AscendC/PTO/BlockRT providers can implement this protocol and
    supply an instance at model init without touching ``model.py`` or adding
    module-level mutable caches.
    """

    name: str

    def run_seam_post_collapse_norm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_module: Any,
        norm_4d_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse hc_post + hc_collapse + RMSNorm + 4D inv_rms computation."""
        ...

    def run_hc_mixes(
        self,
        residual_flat_f32: torch.Tensor,
        inv_rms_4d: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        sinkhorn_iters: int,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (pre, post, comb) mHC coefficients from flattened 4D residual."""
        ...
