# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.router import fused_topk_router as module


def make_router(*, top_k: int, num_experts: int, scoring_func: str = "softmax", **kwargs):
    return module.AscendFusedTopKRouter(
        top_k=top_k,
        global_num_experts=num_experts,
        scoring_func=scoring_func,
        **kwargs,
    )


def fused_call(router, hidden_states, router_logits):
    """Run ``_compute_routing`` and report whether the fused kernel was used."""
    result = (
        torch.ones(hidden_states.shape[0], router.top_k),
        torch.zeros(hidden_states.shape[0], router.top_k, dtype=torch.int32),
        None,
    )
    with (
        patch.object(module.torch_npu, "npu_moe_gating_top_k", return_value=result, create=True) as fused,
        patch.object(
            module.AscendGroupedTopKRouter, "_compute_routing", return_value=(None, None), create=True
        ) as fallback,
    ):
        router._compute_routing(hidden_states, router_logits, None)
    return fused, fallback


@pytest.mark.parametrize(
    ("num_experts", "num_expert_group", "topk_group", "top_k"),
    [
        # Experts per group is the fused kernel bound, not the hidden size.
        (48, 3, 2, 4),
        # ``top_k`` above ``experts_per_group / topk_group`` is still valid as
        # long as the selected groups hold enough candidates.
        (64, 8, 4, 8),
    ],
)
def test_supported_when_expert_dim_satisfies_group_bounds(num_experts, num_expert_group, topk_group, top_k):
    router = make_router(
        top_k=top_k,
        num_experts=num_experts,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )
    # ``is_fused_supported`` only consumes the expert dim of the logits, so the
    # hidden size of the tokens cannot influence the capacity checks anymore.
    router_logits = torch.zeros(3, num_experts)
    assert router.is_fused_supported(router_logits)


def test_unsupported_when_group_holds_two_experts():
    router = make_router(top_k=4, num_experts=16, num_expert_group=8, topk_group=2)
    # hidden size 2048 divides 8 and holds far more than two "experts" per
    # group, so a hidden-size based check would wrongly enable the fused path.
    assert not router.is_fused_supported(torch.zeros(3, 16))


def test_unsupported_when_top_k_exceeds_selected_group_capacity():
    router = make_router(top_k=24, num_experts=64, num_expert_group=8, topk_group=2)
    assert not router.is_fused_supported(torch.zeros(3, 64))


def test_compute_routing_falls_back_for_few_experts_per_group():
    router = make_router(top_k=4, num_experts=16, num_expert_group=8, topk_group=2)
    fused, fallback = fused_call(router, torch.zeros(3, 2048), torch.zeros(3, 16))
    fused.assert_not_called()
    fallback.assert_called_once()


def test_compute_routing_fuses_when_only_expert_dim_matches():
    router = make_router(top_k=4, num_experts=48, num_expert_group=3, topk_group=2)
    fused, fallback = fused_call(router, torch.zeros(3, 4096), torch.zeros(3, 48))
    fused.assert_called_once()
    fallback.assert_not_called()
