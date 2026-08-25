from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_comm_method import _append_cann_megamoe_dummy_tokens
from vllm_ascend.quantization.methods import w4a8 as w4a8_method
from vllm_ascend.utils import get_cann_megamoe_buffer_params


def test_dummy_routes_cover_all_experts_across_ep_ranks():
    routed_experts = []
    for ep_rank_id in range(4):
        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        active_mask = torch.tensor([1, 0], dtype=torch.int8)

        hidden_states, topk_ids, topk_weights, active_mask, original_num_tokens = (
            _append_cann_megamoe_dummy_tokens(
                hidden_states,
                topk_ids,
                topk_weights,
                active_mask,
                num_experts=8,
                ep_rank_id=ep_rank_id,
                ep_world_size=4,
            )
        )

        assert original_num_tokens == 2
        assert torch.equal(hidden_states[-1], torch.ones(4, dtype=torch.bfloat16))
        assert torch.equal(topk_weights[-1], torch.full((2,), 0.5))
        assert active_mask.tolist() == [1, 0, 1]
        routed_experts.extend(topk_ids[-1].tolist())

    assert sorted(routed_experts) == list(range(8))


def test_receive_bound_uses_documented_worst_case():
    assert get_cann_megamoe_buffer_params(480, 32, 256, 8) == (512, 8, 32, 131072)


@pytest.mark.parametrize("quant_name", ["w8a8_dynamic", "w4a8_dynamic"])
def test_a2_mode_2_selects_supported_megamoe_quantization(monkeypatch, quant_name):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=2,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            hidden_size=4096,
            moe_intermediate_size=2048,
            num_experts_per_tok=8,
            quantize=quant_name,
        ),
        get_hidden_size=lambda: 4096,
        get_num_experts=lambda: 256,
    )
    config = SimpleNamespace(
        model_config=model_config,
        quant_config=SimpleNamespace(quant_description={"moe": quant_name}),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=32,
            pipeline_parallel_size=1,
        ),
    )

    assert afc.select_moe_comm_method(512, config) == MoECommType.FUSED_MC2


def test_w4a8_builds_compact_per_expert_megamoe_weights(monkeypatch):
    formatted = []

    def fake_format_cast(weight, npu_format):
        formatted.append((tuple(weight.shape), npu_format, weight.storage_offset()))
        return weight

    monkeypatch.setattr(w4a8_method.torch_npu, "npu_format_cast", fake_format_cast)
    monkeypatch.setattr(w4a8_method, "maybe_trans_nz", lambda weight: weight)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(data=torch.zeros((2, 8, 16), dtype=torch.int8)),
        w2_weight=SimpleNamespace(data=torch.zeros((2, 16, 8), dtype=torch.int8)),
        w13_weight_scale=SimpleNamespace(data=torch.zeros((2, 16), dtype=torch.int64)),
        w2_weight_scale=SimpleNamespace(data=torch.zeros((2, 8), dtype=torch.int64)),
        w13_scale_bias=SimpleNamespace(data=torch.zeros((2, 16), dtype=torch.float32)),
        w2_scale_bias=SimpleNamespace(data=torch.zeros((2, 8), dtype=torch.float32)),
    )
    method = object.__new__(w4a8_method.AscendW4A8DynamicFusedMoEMethod)
    method.pack_to_int32 = lambda weight: weight.view(torch.int32)

    method._build_cann_megamoe_weights(layer)

    assert formatted == [
        ((8, 16), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0),
        ((8, 16), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0),
        ((16, 8), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0),
        ((16, 8), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0),
    ]
    assert len(layer.cann_mega_moe_w13_weight_list) == 2
    assert len(layer.cann_mega_moe_w2_weight_list) == 2
