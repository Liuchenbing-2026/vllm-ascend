# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl


def _attn(wrapped: set[str]):
    names = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
    return SimpleNamespace(**{n: f"{n}-module" for n in names}), wrapped


@pytest.mark.parametrize(
    ("wrapped", "expected"),
    [
        (set(), False),
        ({"wq_a"}, True),
        ({"wq_b"}, True),
        ({"wkv"}, True),
        ({"wo_a"}, False),
        ({"wo_b"}, False),
        ({"wq_a", "wkv"}, True),
    ],
)
def test_prolog_lora_enabled_follows_q_kv_projections(wrapped, expected) -> None:
    attn, names = _attn(wrapped)
    with patch(
        "vllm_ascend.attention.dsa_v41._is_lora_wrapped",
        side_effect=lambda mod: mod in {f"{n}-module" for n in names},
    ):
        assert AscendDSAV41Impl._prolog_lora_enabled(attn) is expected


def test_prolog_lora_enabled_tolerates_missing_projection() -> None:
    attn = SimpleNamespace(wq_a=None, wq_b=None, wkv=None)
    with patch("vllm_ascend.attention.dsa_v41._is_lora_wrapped", return_value=False):
        assert AscendDSAV41Impl._prolog_lora_enabled(attn) is False
