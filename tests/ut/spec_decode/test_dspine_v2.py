# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Exercise MRv2 DSpine dispatch and its Ascend runtime boundary on CPU."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    CandidateSampler,
    DFlash2Speculator,
)
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dspine.speculator import (
    AscendDSpineSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import (
    LmheadTPDraftSamplingMixin,
)


def _make_speculative_config(*, enforce_eager=None):
    return SimpleNamespace(
        method="dspine",
        enforce_eager=enforce_eager,
        draft_tensor_parallel_size=1,
        use_dspark=lambda: False,
        use_dflash=lambda: False,
        speculator_config=SimpleNamespace(),
        draft_model_config=SimpleNamespace(architectures=["DSpineDraftModel"]),
    )


def _make_config(*, enforce_eager=None):
    return SimpleNamespace(
        speculative_config=_make_speculative_config(enforce_eager=enforce_eager),
        parallel_config=SimpleNamespace(tensor_parallel_size=8, pipeline_parallel_size=4),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[16, 32],
            max_cudagraph_capture_size=32,
        ),
    )


def test_mrv2_dspine_dispatch_uses_speculator_class():
    config = _make_config()
    device = torch.device("cpu")

    with patch("vllm_ascend.worker.v2.spec_decode.dspine.speculator.AscendDSpineSpeculator") as speculator_cls:
        result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)
    draft_config = speculator_cls.call_args.args[0]
    assert draft_config.parallel_config.tensor_parallel_size == 8
    assert draft_config.parallel_config.pipeline_parallel_size == 4
    assert draft_config.speculative_config.draft_tensor_parallel_size == 1


def test_mrv2_dspine_factory_constructs_ascend_speculator(monkeypatch):
    constructed = []

    def recording_init(self, vllm_config, device):
        constructed.append((self, vllm_config, device))

    monkeypatch.setattr(AscendDSpineSpeculator, "__init__", recording_init)
    config = _make_config()
    device = torch.device("cpu")

    speculator = init_speculator(config, device)

    assert type(speculator) is AscendDSpineSpeculator
    assert constructed == [(speculator, config, device)]


@pytest.mark.parametrize(
    ("enforce_eager", "expected_mode"),
    [
        (False, CompilationMode.VLLM_COMPILE),
        (None, CompilationMode.VLLM_COMPILE),
        (True, CompilationMode.NONE),
    ],
)
def test_mrv2_dspine_preserves_target_graph_config(enforce_eager, expected_mode):
    config = _make_config(enforce_eager=enforce_eager)
    target_compilation = config.compilation_config
    device = torch.device("cpu")

    with patch("vllm_ascend.worker.v2.spec_decode.dspine.speculator.AscendDSpineSpeculator") as speculator_cls:
        init_speculator(config, device)

    draft_config = speculator_cls.call_args.args[0]
    if enforce_eager:
        assert draft_config is not config
        assert draft_config.compilation_config is not target_compilation
    else:
        assert draft_config is config
        assert draft_config.compilation_config is target_compilation

    draft_compilation = draft_config.compilation_config
    assert draft_compilation.mode == expected_mode
    assert draft_compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert draft_compilation.cudagraph_capture_sizes == [16, 32]
    assert draft_compilation.max_cudagraph_capture_size == 32
    assert config.compilation_config is target_compilation
    assert target_compilation.mode == CompilationMode.VLLM_COMPILE


def test_mrv2_dspine_runs_core_and_ascend_initialization_once(monkeypatch):
    calls = []
    original_dflash2_init = DFlash2Speculator.__init__

    def core_init(self, vllm_config, device):
        calls.append(("core", vllm_config, device))
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=5,
                dflash_config={
                    "selected_prefix_last_write_refine": {
                        "kind": "exact_sender",
                        "head_topk": 8,
                    }
                },
            )
        )
        self.max_num_reqs = 2
        self.num_speculative_steps = 3

    def recording_dflash2_init(self, vllm_config, device):
        calls.append(("dflash2", self))
        original_dflash2_init(self, vllm_config, device)

    class RecordingCandidateSampler:
        def __init__(self, max_num_reqs, num_steps, top_k, device):
            calls.append(("candidate_sampler", max_num_reqs, num_steps, top_k, device))

    monkeypatch.setattr(DFlashSpeculator, "__init__", core_init)
    monkeypatch.setattr(DFlash2Speculator, "__init__", recording_dflash2_init)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator.CandidateSampler",
        RecordingCandidateSampler,
    )
    monkeypatch.setattr(
        AscendDSpineSpeculator,
        "_lmhead_tp_validate_draft_sampling",
        lambda self: calls.append(("lmhead", self)),
    )

    config = SimpleNamespace(
        model_config=SimpleNamespace(use_fp64_gumbel=False),
        speculative_config=SimpleNamespace(),
    )
    device = torch.device("cpu")
    result = AscendDSpineSpeculator(config, device)

    assert [call[0] for call in calls] == [
        "dflash2",
        "core",
        "lmhead",
        "candidate_sampler",
    ]
    assert calls[0][1] is result
    assert calls[1] == ("core", config, device)
    assert sum(call[0] == "core" for call in calls) == 1
    assert sum(call[0] == "dflash2" for call in calls) == 1
    assert calls[2][1] is result
    assert calls[3] == ("candidate_sampler", 2, 3, 8, device)
    assert result.top_k == 8


def test_mrv2_dspine_fp64_gumbel_fails_before_core_allocation(monkeypatch):
    calls = []

    def core_init(self, vllm_config, device):
        calls.append((vllm_config, device))

    monkeypatch.setattr(DFlashSpeculator, "__init__", core_init)
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_fp64_gumbel=True),
        speculative_config=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="does not support use_fp64_gumbel=True"):
        AscendDSpineSpeculator(config, torch.device("cpu"))

    assert calls == []


def test_mrv2_dspine_lmhead_tp_fails_closed(monkeypatch):
    from vllm_ascend.worker.v2.spec_decode import lmhead_tp_utils

    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_enable", lambda: True)
    speculator = AscendDSpineSpeculator.__new__(AscendDSpineSpeculator)

    with pytest.raises(NotImplementedError, match="AscendDSpineSpeculator"):
        speculator._lmhead_tp_validate_draft_sampling()


def test_mrv2_dspine_reuses_core_proposal_and_ascend_runtime_hooks():
    init_chain = (
        AscendDSpineSpeculator,
        DSpineSpeculator,
        DFlash2Speculator,
        AscendDFlashSpeculator,
        LmheadTPDraftSamplingMixin,
        DFlashSpeculator,
    )

    assert AscendDSpineSpeculator.__mro__[: len(init_chain)] == init_chain
    assert AscendDSpineSpeculator._generate_draft is DSpineSpeculator._generate_draft
    assert AscendDSpineSpeculator._score_candidates is DSpineSpeculator._score_candidates
    assert AscendDSpineSpeculator.load_draft_model is DSpineSpeculator.load_draft_model
    assert AscendDSpineSpeculator.propose is AscendDFlashSpeculator.propose
    assert AscendDSpineSpeculator.set_attn is AscendDFlashSpeculator.set_attn
    assert AscendDSpineSpeculator.build_draft_attn_metadatas is AscendDFlashSpeculator.build_draft_attn_metadatas
    assert AscendDSpineSpeculator.init_cudagraph_manager is AscendDFlashSpeculator.init_cudagraph_manager
    assert AscendDSpineSpeculator._lmhead_tp_sample_draft_supported is False


def test_mrv2_dspine_dispatch_accepts_method_without_dflash_alias():
    config = _make_config()
    config.speculative_config.method = "dspine"
    config.speculative_config.use_dflash = lambda: False
    device = torch.device("cpu")

    with patch("vllm_ascend.worker.v2.spec_decode.dspine.speculator.AscendDSpineSpeculator") as speculator_cls:
        result = init_speculator(config, device)

    assert result is speculator_cls.return_value
    speculator_cls.assert_called_once_with(config, device)


def test_mrv2_dspine_candidate_sampler_uses_patched_selector_walk(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as upstream

    import vllm_ascend.patch.worker.patch_dflash_speculator  # noqa: F401
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
        _selector_walk_kernel_ascend,
    )

    assert upstream._selector_walk_kernel is _selector_walk_kernel_ascend

    launches = []

    class RecordingKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(upstream, "_selector_walk_kernel", RecordingKernel())
    # The CPU-only triton stub used by this test environment does not expose
    # the CUDA runtime helper, but the sampler only needs its value here.
    monkeypatch.setattr(upstream.triton, "next_power_of_2", lambda _top_k: 4, raising=False)
    sampler = CandidateSampler(max_num_reqs=1, num_steps=2, top_k=4, device=torch.device("cpu"))
    sampler.sample(
        candidate_ids=torch.zeros((1, 2, 4), dtype=torch.int64),
        scores=torch.zeros((1, 2, 4, 4), dtype=torch.float32),
        num_reqs=1,
        sample_pos=torch.zeros(2, dtype=torch.int64),
        idx_mapping=torch.zeros(1, dtype=torch.int64),
        temperature=torch.ones(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        draft_tokens=torch.zeros(2, dtype=torch.int64),
        draft_logits=None,
        use_fp64=False,
    )

    assert len(launches) == 1
    grid, _, kwargs = launches[0]
    assert grid == (1,)
    assert kwargs["num_steps"] == 2
    assert kwargs["top_k"] == 4
    assert kwargs["BLOCK_K"] == 4
    assert kwargs["SAMPLE_PROBABILISTIC"] is False
    assert kwargs["USE_FP64"] is False
