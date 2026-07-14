#!/usr/bin/env python3

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ascend_forward_context import _cann_megamoe_supported_by_config
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    _append_cann_megamoe_dummy_tokens,
    _normalize_cann_megamoe_activation,
)
from vllm_ascend.utils import (
    get_cann_megamoe_dummy_token_capacity,
    resolve_cann_megamoe_max_recv_tokens,
)


def check_dummy_routing() -> None:
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)

    for ep_rank, expected_dummy_mask in ((0, 1), (1, 0)):
        output = _append_cann_megamoe_dummy_tokens(
            hidden_states,
            topk_ids,
            topk_weights,
            None,
            num_experts=8,
            ep_rank_id=ep_rank,
        )
        output_hidden, output_ids, output_weights, output_mask, original_tokens = output
        assert original_tokens == 2
        assert output_hidden.shape == (6, 4)
        assert output_ids.shape == (6, 2)
        assert output_weights.shape == (6, 2)
        assert output_mask.tolist() == [1, 1] + [expected_dummy_mask] * 4
        assert torch.count_nonzero(output_hidden[2:]).item() == 0
        assert torch.count_nonzero(output_weights[2:]).item() == 0
        assert set(output_ids[2:].reshape(-1).tolist()) == set(range(8))


def check_capacity_helpers() -> None:
    assert get_cann_megamoe_dummy_token_capacity(256, 8) == 32
    config = SimpleNamespace(mega_moe_max_recv_tokens=0)
    with patch("vllm_ascend.utils.get_ascend_config", return_value=config):
        assert resolve_cann_megamoe_max_recv_tokens(96, 32, 8, 8) == 24576

    config.mega_moe_max_recv_tokens = 8192
    with patch("vllm_ascend.utils.get_ascend_config", return_value=config):
        assert resolve_cann_megamoe_max_recv_tokens(96, 32, 8, 8) == 8192


def check_a2_selection_guard() -> None:
    hf_text_config = SimpleNamespace(
        hidden_size=6144,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
    )
    model_config = SimpleNamespace(
        hf_text_config=hf_text_config,
        get_num_experts=lambda: 256,
    )
    vllm_config = SimpleNamespace(model_config=model_config, quant_config=None)
    ascend_config = SimpleNamespace(
        enable_fused_mc2=2,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    ep_group = SimpleNamespace(world_size=32)
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
    ):
        assert _cann_megamoe_supported_by_config(vllm_config, "w8a8", False)
        assert not _cann_megamoe_supported_by_config(vllm_config, "w8a8", True)


def main() -> None:
    check_dummy_routing()
    check_capacity_helpers()
    check_a2_selection_guard()
    assert _normalize_cann_megamoe_activation("silu") == "swiglu"
    print("A2 MegaMoe helper checks: PASS", flush=True)


if __name__ == "__main__":
    main()
