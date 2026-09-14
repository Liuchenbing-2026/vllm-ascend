# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash_module
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dflash2 import is_dflash2_draft
from vllm_ascend.worker.v2.spec_decode.dflash2 import speculator as dflash2_module
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
    DFlash2Speculator,
)


def _speculative_config(method: str = "dflash", architectures=("DFlash2DraftModel",)):
    return SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(architectures=list(architectures)),
        use_dspark=lambda: False,
        use_dflash=lambda: method == "dflash",
    )


def test_dflash2_combines_selector_with_ascend_runtime() -> None:
    assert AscendDFlash2Speculator.__mro__[:4] == (
        AscendDFlash2Speculator,
        DFlash2Speculator,
        AscendDFlashSpeculator,
        DFlashSpeculator,
    )
    assert AscendDFlash2Speculator._generate_draft is DFlash2Speculator._generate_draft
    assert AscendDFlash2Speculator.propose is AscendDFlashSpeculator.propose


def test_dflash2_speculator_is_owned_by_vllm_ascend() -> None:
    """vLLM 0.26 has no v1/worker/gpu/spec_decode/dflash2 package to import from."""
    assert DFlash2Speculator.__module__ == dflash2_module.__name__


@pytest.mark.parametrize(
    ("method", "architectures", "expected"),
    [
        ("dflash", ["DFlash2DraftModel"], True),
        ("dflash", ["DFlashDraftModel"], False),
        ("eagle", ["DFlash2DraftModel"], False),
        ("dflash", None, False),
    ],
)
def test_is_dflash2_draft_reads_the_draft_architecture(method, architectures, expected) -> None:
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(architectures=architectures),
    )

    assert is_dflash2_draft(SimpleNamespace(speculative_config=speculative_config)) is expected


def test_is_dflash2_draft_without_a_draft_model() -> None:
    assert not is_dflash2_draft(SimpleNamespace(speculative_config=None))
    speculative_config = SimpleNamespace(method="dflash", draft_model_config=None)
    assert not is_dflash2_draft(SimpleNamespace(speculative_config=speculative_config))


def test_dflash2_dispatches_to_ascend_combined_speculator(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        dflash2_module,
        "AscendDFlash2Speculator",
        lambda *_args, **_kwargs: sentinel,
    )
    # A real vLLM 0.26 VllmConfig has no _is_dflash2_draft; dispatch must not need it.
    vllm_config = SimpleNamespace(speculative_config=_speculative_config())

    actual = init_speculator(vllm_config, torch.device("cpu"))

    assert actual is sentinel


def test_dflash1_dispatches_to_ascend_dflash_speculator(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        dflash_module,
        "AscendDFlashSpeculator",
        lambda *_args, **_kwargs: sentinel,
    )
    vllm_config = SimpleNamespace(speculative_config=_speculative_config(architectures=("DFlashDraftModel",)))

    actual = init_speculator(vllm_config, torch.device("cpu"))

    assert actual is sentinel


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would."""

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3}))
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch) -> None:
    """Greedy caches no proposal distribution; verification reads `draft_logits is None`."""
    _stub_base(monkeypatch, None)

    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_proposal_logits_start_at_negative_infinity(monkeypatch) -> None:
    """vLLM 0.26 allocates zeros; the cache kernel writes only K columns per step."""
    _stub_base(monkeypatch, torch.zeros((2, 4, 17), dtype=torch.float32))

    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits.dtype is torch.float32
    assert torch.isneginf(speculator.draft_logits).all()


def test_selector_asks_for_fp32_proposal_logits() -> None:
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


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
