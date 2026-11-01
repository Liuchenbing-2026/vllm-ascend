# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.ops.gdn import (
    _MAX_RECURRENT_GDN_TOKENS,
    AscendGatedDeltaNetAttention,
    _check_recurrent_gdn_window,
)
from vllm_ascend.ops.gdn_attn_builder import (
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)

from .test_gdn_mixed_batch import _make_layer, _make_mixed_metadata


@pytest.mark.parametrize("window", [1, 8])
def test_spec_window_within_operator_cap(window: int) -> None:
    state_indices = torch.zeros(2, window, dtype=torch.int64)
    _check_recurrent_gdn_window(state_indices, num_seqs=2)


@pytest.mark.parametrize("window", [9, 16])
def test_spec_window_wider_than_operator_cap_raises(window: int) -> None:
    state_indices = torch.zeros(2, window, dtype=torch.int64)
    with pytest.raises(ValueError, match=f"window of {window}") as excinfo:
        _check_recurrent_gdn_window(state_indices, num_seqs=2)
    message = str(excinfo.value)
    assert str(_MAX_RECURRENT_GDN_TOKENS) in message
    assert "num_speculative_tokens" in message


def test_flattened_state_indices_skip_the_window_check() -> None:
    # One-dimensional tables carry no per-sequence window to validate.
    _check_recurrent_gdn_window(torch.zeros(32, dtype=torch.int64), num_seqs=2)


def test_rebuilt_operator_cap_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_RGDR_MAX_MTP", "16")
    _check_recurrent_gdn_window(torch.zeros(2, 9, dtype=torch.int64), num_seqs=2)
    with pytest.raises(ValueError, match="window of 17"):
        _check_recurrent_gdn_window(torch.zeros(2, 17, dtype=torch.int64), num_seqs=2)


@pytest.mark.parametrize("value", ["0", "-8", "wide"])
def test_invalid_operator_cap_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("VLLM_ASCEND_RGDR_MAX_MTP", value)
    with pytest.raises(ValueError, match="VLLM_ASCEND_RGDR_MAX_MTP"):
        _check_recurrent_gdn_window(torch.zeros(2, 8, dtype=torch.int64), num_seqs=2)


def test_forward_core_rejects_nine_token_window_before_the_operator() -> None:
    """A DSpark-style window of 8 drafts + 1 bonus token must fail fast."""
    layer = _make_layer()
    metadata = _make_mixed_metadata()
    num_seqs, window = 2, 9
    num_tokens = num_seqs * window
    metadata.num_prefills = 0
    metadata.num_prefill_tokens = 0
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.num_actual_tokens = num_tokens
    metadata.num_spec_decodes = num_seqs
    metadata.num_spec_decode_tokens = num_tokens
    metadata.spec_sequence_masks = torch.ones(num_seqs, dtype=torch.bool)
    metadata.spec_state_indices_tensor = torch.arange(num_tokens, dtype=torch.int64).view(num_seqs, window)
    metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
            query_start_loc=torch.arange(0, num_tokens + 1, window, dtype=torch.int32),
            cache_indices=metadata.spec_state_indices_tensor[:, 0],
            num_accepted_tokens=torch.ones(num_seqs, dtype=torch.int32),
        ),
        actual_seq_lengths=torch.tensor([0] * (num_seqs + 1), dtype=torch.int32),
    )
    packed = torch.zeros(num_tokens, 6)
    output = torch.empty(num_tokens, 1, 2)
    context = ForwardContext(
        no_compile_layers={layer.prefix: layer},
        attn_metadata={layer.prefix: metadata},
        slot_mapping={},
    )

    with (
        override_forward_context(context),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch(
            "vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating",
            return_value=(torch.zeros(1, num_tokens, 1), torch.ones(1, num_tokens, 1)),
        ),
        patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: x),
        patch("vllm_ascend.ops.gdn.causal_conv1d_update", side_effect=lambda x, *a, **k: x.clone()),
        patch(
            "vllm_ascend.ops.gdn.recurrent_gated_delta_rule",
            side_effect=AssertionError("the operator must not run with an unsupported window"),
        ),
        patch("vllm_ascend.ops.gdn.maybe_save_kv_layer_to_connector"),
        pytest.raises(ValueError, match=f"window of {window}"),
    ):
        AscendGatedDeltaNetAttention._forward_core(
            layer, packed, torch.zeros(num_tokens, 1), torch.zeros(num_tokens, 1), output
        )
