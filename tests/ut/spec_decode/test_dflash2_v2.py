# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
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


def test_dflash_uses_the_runtime_attention_metadata_signature(monkeypatch) -> None:
    captured = {}

    def build_metadata(**kwargs):
        captured.update(kwargs)
        return "metadata"

    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", nullcontext)
    speculator = SimpleNamespace(
        num_query_per_req=8,
        input_batch=SimpleNamespace(num_reqs=2),
        _group_causal=False,
        _build_draft_attn_metadata=build_metadata,
    )
    seq_lens = object()

    actual = AscendDFlashSpeculator.build_draft_attn_metadatas(
        speculator,
        num_reqs_padded=4,
        seq_lens_cpu_upper_bound=seq_lens,
    )

    assert actual == ["metadata"]
    assert captured["seq_lens_cpu_upper_bound"] is seq_lens
    assert captured["step"] == 8
    assert captured["num_reqs"] == 2
    assert captured["num_reqs_padded"] == 4
    assert captured["num_tokens_padded"] == 32
