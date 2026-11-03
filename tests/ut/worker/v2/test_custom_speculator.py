# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for the upstream registered-speculator extension point."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CompilationMode, SpeculativeConfig, VllmConfig

from vllm_ascend.worker.v2.spec_decode import init_speculator

pytest.importorskip(
    "vllm.v1.worker.gpu.spec_decode.speculator_config",
    reason="Registered MRV2 speculators require the upstream speculator-config API",
)

from vllm.v1.worker.gpu.spec_decode import speculator_config  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.ngram.config import NgramGPUSpeculatorConfig  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator  # noqa: E402


class CPUCustomSpeculator(BaseSpeculator):
    """Record constructor inputs without loading weights or using a device."""

    def __init__(self, vllm_config, device, *, req_states):
        self.vllm_config = vllm_config
        self.device = device
        self.req_states = req_states

    def init_cudagraph_manager(self, cudagraph_mode):
        raise NotImplementedError

    def capture(self):
        raise NotImplementedError

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class CPUCustomSpeculatorWithoutStates(CPUCustomSpeculator):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device, req_states=None)


class CPUCustomConfig(speculator_config.SpeculatorConfig):
    speculator_class = f"{__name__}.CPUCustomSpeculator"
    supported_options = frozenset({"speculator_options"})


class CPUCustomNgramConfig(NgramGPUSpeculatorConfig):
    speculator_class = f"{__name__}.CPUCustomSpeculator"


class CPUCustomWithoutStatesConfig(CPUCustomConfig):
    speculator_class = f"{__name__}.CPUCustomSpeculatorWithoutStates"


class InheritedNgramConfig(NgramGPUSpeculatorConfig):
    pass


class InvalidCustomConfig(CPUCustomConfig):
    speculator_class = "builtins.object"


@pytest.fixture
def register(monkeypatch):
    # Isolate registration while exercising the real upstream registry.
    monkeypatch.setattr(speculator_config, "_CUSTOM_CONFIGS", {})
    return speculator_config.register_speculative_method


@pytest.mark.parametrize("config_cls", [CPUCustomConfig, CPUCustomNgramConfig])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_custom_class_takes_precedence(register, config_cls, lazy, enforce_eager):
    register("cpu_custom", f"{__name__}.{config_cls.__name__}" if lazy else config_cls)
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(
        method="cpu_custom", num_speculative_tokens=3, enforce_eager=enforce_eager
    )
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    target_compilation = config.compilation_config
    req_states = object()

    result = init_speculator(config, torch.device("cpu"), req_states)

    assert type(result) is CPUCustomSpeculator
    assert result.device == torch.device("cpu")
    assert result.req_states is req_states
    assert result.vllm_config.speculative_config is config.speculative_config
    assert config.speculative_config.custom_method == "cpu_custom"
    assert config.speculative_config.method == config_cls.base_method
    assert config.compilation_config is target_compilation
    assert target_compilation.mode == CompilationMode.VLLM_COMPILE
    if enforce_eager:
        assert result.vllm_config is not config
        assert result.vllm_config.compilation_config is not target_compilation
        assert result.vllm_config.compilation_config.mode == CompilationMode.NONE
    else:
        assert result.vllm_config is config


def test_custom_options_and_constructor_without_states(register):
    register("cpu_options", CPUCustomWithoutStatesConfig)
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(
        method="cpu_options", num_speculative_tokens=3, speculator_options={"seed": 7}
    )

    result = init_speculator(config, torch.device("cpu"), object())

    assert type(result) is CPUCustomSpeculatorWithoutStates
    assert result.req_states is None
    assert result.vllm_config.speculative_config.speculator_options == {"seed": 7}


def test_invalid_custom_class_uses_upstream_validation(register):
    register("cpu_invalid", InvalidCustomConfig)
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(method="cpu_invalid", num_speculative_tokens=3)

    with pytest.raises(TypeError, match="must be a subclass of"):
        init_speculator(config, torch.device("cpu"), object())


def test_inherited_implementation_preserves_native_dispatch(register, monkeypatch):
    register("inherited_ngram", InheritedNgramConfig)
    config = VllmConfig()
    config.speculative_config = SpeculativeConfig(method="inherited_ngram", num_speculative_tokens=3)
    req_states = object()
    module = ModuleType("vllm.v1.worker.gpu.spec_decode.ngram.speculator")
    native_cls = MagicMock()
    module.NgramGPUSpeculator = native_cls
    monkeypatch.setitem(sys.modules, module.__name__, module)

    result = init_speculator(config, torch.device("cpu"), req_states)

    assert result is native_cls.return_value
    native_cls.assert_called_once_with(config, torch.device("cpu"), req_states)
    assert config.speculative_config.custom_method == "inherited_ngram"
