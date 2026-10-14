#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression test for the device-id selection in
``examples/offline_data_parallel.py``.

The example builds ``device_ids`` from ``torch.npu.device_count()``. Those
values must be **integers** (indices into the ``ASCEND_RT_VISIBLE_DEVICES``
visible set), not strings: vLLM's ``EngineArgs._resolve_device_ids`` resolves
string entries as physical device IDs (the CUDA-UUID path) but integer entries
as indices into the visible set. With the string form, a node whose visible
devices are not 0-based (e.g. ``ASCEND_RT_VISIBLE_DEVICES=4,5,6,7``) resolves
the ids to physical ``0..N-1``, which are not visible, and worker
``init_device()`` raises
``Physical device 1 ... is not visible in ASCEND_RT_VISIBLE_DEVICES=4,5,6,7``.

Runs on CPU: the example's id-construction expression is evaluated against a
stand-in ``torch`` object, and vLLM's resolution branches are reproduced from
``vllm/engine/arg_utils.py`` (v0.30.0). No NPU is required.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "examples" / "offline_data_parallel.py"


def _extract_dp_device_ids_rhs() -> ast.expr:
    """Return the AST of the right-hand side of ``_dp_device_ids = ...``."""
    tree = ast.parse(_EXAMPLE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_dp_device_ids":
                return node.value
    raise AssertionError("`_dp_device_ids` assignment not found in the example")


def _eval_device_ids(device_count: int):
    """Evaluate the example's ``_dp_device_ids`` expression for a given count."""

    class _Npu:
        @staticmethod
        def device_count() -> int:
            return device_count

    class _Torch:
        npu = _Npu()

    expr = ast.Expression(body=_extract_dp_device_ids_rhs())
    ast.fix_missing_locations(expr)
    code = compile(expr, str(_EXAMPLE), "eval")
    return eval(code, {"torch": _Torch()})  # noqa: S307 - trusted, repo-local source


def _to_physical(device_id: str) -> int:
    # Base Platform.device_control_id_to_physical_device_id() is int(device_id).
    return int(device_id)


def _resolve_device_ids(device_ids, cvd: str):
    """Reproduce the device-id branches of EngineArgs._resolve_device_ids."""
    if all(isinstance(i, str) for i in device_ids):
        return [_to_physical(i) for i in device_ids]
    assert not any(isinstance(i, str) for i in device_ids), "mixed str/int ids"
    if cvd:
        cvd_ids = [_to_physical(x) for x in cvd.split(",")]
        for i in device_ids:
            assert i < len(cvd_ids), f"index {i} out of range for {cvd}"
        return [cvd_ids[i] for i in device_ids]
    return list(device_ids)


def test_device_ids_are_integer_visible_indices():
    device_ids = _eval_device_ids(4)
    assert device_ids == [0, 1, 2, 3]
    assert all(isinstance(i, int) and not isinstance(i, bool) for i in device_ids), (
        "device_ids must be integers; string entries are resolved as physical "
        "device IDs and break non-zero-based ASCEND_RT_VISIBLE_DEVICES"
    )


@pytest.mark.parametrize(
    "cvd,expected",
    [
        # Non-zero-based visible set: the exact case from the bug report.
        ("4,5,6,7", [4, 5, 6, 7]),
        # ASCEND_RT_VISIBLE_DEVICES unset -> ids are already physical.
        ("", [0, 1, 2, 3]),
    ],
)
def test_ids_resolve_to_visible_physical_devices(cvd, expected):
    device_ids = _eval_device_ids(4)
    assert _resolve_device_ids(device_ids, cvd) == expected


def test_string_form_reproduces_the_reported_mismatch():
    # Documents the defect: the old string form always resolves to physical
    # 0..N-1, so with only 4,5,6,7 visible it maps to non-visible devices.
    old_style = [str(i) for i in range(4)]
    assert _resolve_device_ids(old_style, "4,5,6,7") == [0, 1, 2, 3]
