# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest


@pytest.fixture
def make_model_state(tmp_path, monkeypatch):
    """Initialize state through the real MRO before a focused test overrides inputs."""
    import torch
    from transformers import LlamaConfig
    from vllm.config import DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.model_executor.models import registry

    from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
    from vllm_ascend.worker.v2.model_states import default

    # Preserve real architecture inspection under the existing CPU hardware mocks.
    def inspect_in_process(inspect_model):
        return inspect_model()

    monkeypatch.setattr(registry, "_run_in_subprocess", inspect_in_process)
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True, dtype="float32", max_model_len=2048
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_model_len=2048, max_num_seqs=4, max_num_batched_tokens=2048, is_encoder_decoder=False
        ),
    )

    def make(state_cls, **overrides):
        from unittest.mock import patch

        ascend_config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig.from_additional_config(config, {}))
        with patch.object(default, "get_ascend_config", return_value=ascend_config):
            state = state_cls(config, torch.nn.Module(), None, torch.device("cpu"))
        assert state.mm_prefix_ranges is None
        if "max_model_len" in overrides:
            state.model_config.max_model_len = overrides.pop("max_model_len")
        for name, value in overrides.items():
            setattr(state, name, value)
        return state

    return make


@pytest.fixture(autouse=True)
def _stub_ascend_config(monkeypatch):
    # Bare-runner harnesses reach fine-grained TP gates (sample_tokens, the
    # dummy joins) that read the global ascend config; stub these gates off so the
    # gates no-op instead of raising on the uninitialized config read.
    from vllm_ascend import ascend_config as _ascend_config_module

    monkeypatch.setattr(
        _ascend_config_module,
        "_ASCEND_CONFIG",
        SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(lmhead_tensor_parallel_size=0, embedding_tensor_parallel_size=0),
            ascend_compilation_config=object(),
            eplb_config=object(),
            ascend_warmup_config=SimpleNamespace(enable_early_kernel_warmup=False, enable_early_nz_warmup=False),
        ),
    )
