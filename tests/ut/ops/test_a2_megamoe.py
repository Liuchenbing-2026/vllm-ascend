import pytest
import torch

from vllm_ascend.ops.fused_moe.comm_utils import (
    append_cann_megamoe_dummy_tokens,
    get_cann_megamoe_buffer_params,
)


def test_dummy_routes_cover_all_experts_across_ep_ranks():
    routed_experts = []
    for ep_rank_id in range(4):
        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        active_mask = torch.tensor([1, 0], dtype=torch.int8)

        hidden_states, topk_ids, topk_weights, active_mask, original_num_tokens = append_cann_megamoe_dummy_tokens(
            hidden_states,
            topk_ids,
            topk_weights,
            active_mask,
            num_experts=8,
            ep_rank_id=ep_rank_id,
            ep_world_size=4,
        )

        assert original_num_tokens == 2
        assert torch.equal(hidden_states[-1], torch.ones(4, dtype=torch.bfloat16))
        assert torch.equal(topk_weights[-1], torch.full((2,), 0.5))
        assert active_mask.tolist() == [1, 0, 1]
        routed_experts.extend(topk_ids[-1].tolist())

    assert sorted(routed_experts) == list(range(8))


def test_receive_bound_includes_a2_dummy_capacity():
    assert get_cann_megamoe_buffer_params(512, 32, 256, 8) == (544, 8, 32, 139264)


def test_dummy_capacity_rejects_operator_token_overflow():
    with pytest.raises(ValueError, match="num_max_tokens_per_rank"):
        get_cann_megamoe_buffer_params(4096, 32, 256, 8)
