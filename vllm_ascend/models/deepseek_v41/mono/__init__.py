# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Per-layer MonoKernel decode execution for DeepSeek V4.1 on Ascend."""

from .ascendc_backend import AscendCFullLayerMonoBackend, compute_mono_workspace_floats, is_full_layer_kernel_eligible
from .config import AscendDSV41MonoConfig, collect_unsupported_reasons, resolve_mono_request
from .contracts import DSV41MonoLayerBackend, MonoLayerBinding, MonoStepPlan, PendingPostState
from .runner import DSV41AscendMonoRunner, maybe_build_mono_runner

__all__ = [
    "AscendCFullLayerMonoBackend",
    "AscendDSV41MonoConfig",
    "DSV41AscendMonoRunner",
    "DSV41MonoLayerBackend",
    "MonoLayerBinding",
    "MonoStepPlan",
    "PendingPostState",
    "collect_unsupported_reasons",
    "compute_mono_workspace_floats",
    "is_full_layer_kernel_eligible",
    "maybe_build_mono_runner",
    "resolve_mono_request",
]
