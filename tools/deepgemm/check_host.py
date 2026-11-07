# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run adapter/delivery tests with import-boundary stubs; never execute NPU kernels."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def main():
    root = Path(__file__).resolve().parents[2]
    for name, path in [
        ("vllm", None),
        ("vllm.config", None),
        ("vllm.distributed", None),
        ("vllm.logger", None),
        ("torch_npu", None),
        ("vllm_ascend", root / "vllm_ascend"),
        ("vllm_ascend.ops", root / "vllm_ascend/ops"),
        ("vllm_ascend.ops.fused_moe", root / "vllm_ascend/ops/fused_moe"),
        ("vllm_ascend.ascend_config", None),
    ]:
        module = ModuleType(name)
        module.__path__ = [str(path)] if path else []
        sys.modules[name] = module
    for name, attrs in {
        "vllm.config": ["get_current_vllm_config"],
        "vllm.distributed": ["get_ep_group", "get_tensor_model_parallel_world_size"],
        "vllm.logger": ["logger"],
        "vllm_ascend.ascend_config": ["get_ascend_config"],
        "torch_npu": ["npu_dynamic_mx_quant"],
    }.items():
        for attr in attrs:
            setattr(sys.modules[name], attr, MagicMock())
    raise SystemExit(
        pytest.main(
            [
                "--noconftest",
                "-q",
                str(root / "tests/ut/ops/fused_moe/test_deepgemm_v41.py"),
                str(root / "tests/ut/ops/fused_moe/test_deepgemm_delivery.py"),
            ]
        )
    )


if __name__ == "__main__":
    main()
