# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise MRV2 dispatch and the DSpine/Ascend loader boundary on CPU.

The repository conftest mocks device APIs; these tests do not validate kernels.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CompilationMode
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dspine.speculator import AscendDSpineSpeculator


def _config(*, eager=True, sampling="greedy"):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            enforce_eager=eager,
            draft_sample_method=sampling,
            draft_model_config=SimpleNamespace(architectures=["DSpineDraftModel"]),
            use_dspark=lambda: False,
            use_dflash=lambda: True,
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )


def test_mrv2_dispatch_preserves_target_compilation_and_validates_draft(monkeypatch):
    config = _config()
    calls = []
    monkeypatch.setattr(DSpineSpeculator, "__init__", lambda self, cfg, device: calls.append((cfg, device)))
    validate = Mock()
    monkeypatch.setattr(AscendDSpineSpeculator, "_lmhead_tp_validate_draft_sampling", validate)
    device = torch.device("cpu")
    result = init_speculator(config, device)
    assert isinstance(result, AscendDSpineSpeculator)
    assert calls[0][0].compilation_config.mode == CompilationMode.NONE
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert calls[0][1] == device
    validate.assert_called_once_with()


@pytest.mark.parametrize(
    "eager,sampling,message",
    [(False, "greedy", "enforce_eager"), (True, "probabilistic", "greedy")],
)
def test_unsupported_draft_mode_rejected_before_allocation(monkeypatch, eager, sampling, message):
    allocate = Mock()
    monkeypatch.setattr(DSpineSpeculator, "__init__", allocate)
    with pytest.raises(ValueError, match=message):
        init_speculator(_config(eager=eager, sampling=sampling), torch.device("cpu"))
    allocate.assert_not_called()


def test_shared_core_loader_calls_ascend_loader_then_materializes_bridge(monkeypatch):
    """The MRO must retain NPU loading hooks before the shared bridge setup."""
    target = object()
    layer_names = {"target.attn"}
    weight = torch.randn(16, 8)
    materialize = Mock()
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=weight), materialize_inference_buffers=materialize)
    load = Mock(return_value=model)
    monkeypatch.setattr(AscendDFlashSpeculator, "load_draft_model", load)
    speculator = AscendDSpineSpeculator.__new__(AscendDSpineSpeculator)
    assert speculator.load_draft_model(target, layer_names) is model
    load.assert_called_once_with(target, layer_names)
    materialize.assert_called_once_with(weight)


@pytest.mark.parametrize(
    "name",
    ["tensor_parallel_size", "pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"],
)
def test_sharded_target_rejected_before_bridge_allocation(monkeypatch, name):
    config = _config()
    setattr(config.parallel_config, name, 2)
    allocate = Mock()
    monkeypatch.setattr(DSpineSpeculator, "__init__", allocate)
    with pytest.raises(ValueError, match=f"{name}=1"):
        init_speculator(config, torch.device("cpu"))
    allocate.assert_not_called()


def test_dspine_keeps_ascend_attention_and_proposal_hooks():
    # Reusing the GPU methods here would omit Ascend metadata/stream handling.
    assert AscendDSpineSpeculator.propose is AscendDFlashSpeculator.propose
    assert AscendDSpineSpeculator.set_attn is AscendDFlashSpeculator.set_attn
    assert AscendDSpineSpeculator.build_draft_attn_metadatas is AscendDFlashSpeculator.build_draft_attn_metadatas
    assert AscendDSpineSpeculator._generate_draft is DSpineSpeculator._generate_draft


def test_mrv2_worker_modules_import_with_paired_core(monkeypatch):
    import importlib.util

    import vllm.v1.core.kv_cache_utils as planner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker, get_kv_cache_groups

    assert callable(NPUModelRunner.load_model)
    assert callable(NPUWorker.init_device)
    if importlib.util.find_spec("vllm.v1.core.kv_cache_config_builder") is None:
        groups = [object()]
        config, specs = object(), {}
        plan = Mock(return_value=groups)
        monkeypatch.setattr(planner, "get_kv_cache_groups", plan)
        assert get_kv_cache_groups(config, specs) is groups
        plan.assert_called_once_with(config, specs)
