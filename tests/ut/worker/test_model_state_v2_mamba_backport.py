from unittest.mock import MagicMock, patch

import torch
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from vllm_ascend.worker.v2.model_states import init_asecnd_model_state
from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
    AscendMambaHybridModelState,
)


def test_mamba_model_state_inherits_upstream_state_management():
    assert issubclass(AscendMambaHybridModelState, MambaHybridModelState)
    assert (
        AscendMambaHybridModelState.preprocess_state
        is MambaHybridModelState.preprocess_state
    )
    assert (
        AscendMambaHybridModelState.postprocess_state
        is MambaHybridModelState.postprocess_state
    )


@patch(
    "vllm_ascend.worker.v2.model_states.mamba_hybrid."
    "AscendMambaHybridModelState"
)
def test_hybrid_model_selects_mamba_model_state(mock_mamba_state):
    vllm_config = MagicMock()
    vllm_config.model_config.is_hybrid = True
    model = torch.nn.Module()
    encoder_cache = MagicMock()
    device = torch.device("cpu")

    state = init_asecnd_model_state(
        vllm_config,
        model,
        encoder_cache,
        device,
    )

    assert state is mock_mamba_state.return_value
    mock_mamba_state.assert_called_once_with(
        vllm_config,
        model,
        encoder_cache,
        device,
    )


def test_v2_attn_utils_rebinds_ascend_kv_cache_binder():
    import vllm.v1.worker.gpu.attn_utils as gpu_attn_utils

    from vllm_ascend.patch.worker.patch_qwen3_next_mtp import bind_kv_cache

    # gpu.attn_utils imports the binder by value, so patching only
    # vllm.v1.worker.utils does not update the reference used by init_kv_cache.
    assert gpu_attn_utils.bind_kv_cache is bind_kv_cache
