# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the KDA causal-conv1d rebind in ``worker/patch_triton.py``.

PR #14620 deleted the Ascend Triton ``causal_conv1d_update_npu`` kernel and the
monkey-patch that installed it. A later feature commit re-added a rebind that
still imported the deleted symbol, so every worker start-up logged a warning
that claimed a host-syncing PyTorch fallback had been selected. That warning was
not a real fallback signal (the intended NPU wrapper is installed earlier in the
same module) and was cited as the root cause of an unrelated crash report.
"""

import importlib
import logging

import vllm.model_executor.layers.mamba.ops.causal_conv1d as upstream_cc1d
from vllm.logger import logger as vllm_logger

import vllm_ascend.patch.worker.patch_triton as patch_triton
from vllm_ascend.ops.triton.mamba import causal_conv1d as npu_cc1d

REMOVED_SYMBOL = "causal_conv1d_update_npu"
MISLEADING_WARNING = "NPU Triton causal_conv1d_update is unavailable"


def test_removed_triton_symbol_is_still_absent():
    # Guards against another stray reintroduction of the deleted kernel.
    assert not hasattr(npu_cc1d, REMOVED_SYMBOL)


def test_kda_conv1d_binds_the_ascend_wrappers():
    # The active binding is the speculative-decode-aware NPU wrapper installed by
    # the KDA bind block, not the upstream CUDA Triton kernel.
    assert upstream_cc1d.causal_conv1d_update is patch_triton._npu_causal_conv1d_update
    assert upstream_cc1d.causal_conv1d_fn is patch_triton._npu_causal_conv1d_fn


def test_reimport_does_not_log_the_misleading_fallback_warning():
    # The vllm logger does not necessarily propagate to the root logger, so
    # capture on the module logger itself instead of relying on ``caplog``.
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Capture()
    vllm_logger.addHandler(handler)
    try:
        importlib.reload(patch_triton)
    finally:
        vllm_logger.removeHandler(handler)

    assert not any(MISLEADING_WARNING in message for message in messages)
