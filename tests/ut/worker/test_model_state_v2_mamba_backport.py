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
