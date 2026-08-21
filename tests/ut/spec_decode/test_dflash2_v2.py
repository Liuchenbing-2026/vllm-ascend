# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import torch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dflash2 import speculator as dflash2_module
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
)


def test_dflash2_combines_upstream_selector_with_ascend_runtime() -> None:
    assert AscendDFlash2Speculator.__mro__[:4] == (
        AscendDFlash2Speculator,
        DFlash2Speculator,
        AscendDFlashSpeculator,
        DFlashSpeculator,
    )
    assert (
        AscendDFlash2Speculator._generate_draft
        is DFlash2Speculator._generate_draft
    )
    assert AscendDFlash2Speculator.propose is AscendDFlashSpeculator.propose


def test_dflash2_dispatches_to_ascend_combined_speculator(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        dflash2_module,
        "AscendDFlash2Speculator",
        lambda *_args, **_kwargs: sentinel,
    )
    speculative_config = SimpleNamespace(
        use_dspark=lambda: False,
        use_dflash=lambda: True,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        _is_dflash2_draft=lambda: True,
    )

    actual = init_speculator(vllm_config, torch.device("cpu"))

    assert actual is sentinel
