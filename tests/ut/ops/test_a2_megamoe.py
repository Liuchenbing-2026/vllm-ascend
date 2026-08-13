from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method as comm_method
from vllm_ascend.ops.fused_moe.comm_utils import (
    _CANN_ACL_INT4,
    _CANN_MEGA_MOE_QUANT_MODE_INT8,
    _CANN_TORCH_INT8,
    _get_cann_mega_moe_quant_settings,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import _append_cann_megamoe_dummy_tokens
from vllm_ascend.quantization.methods import w4a8 as w4a8_method
from vllm_ascend.quantization.methods.base import QuantType
from vllm_ascend.utils import get_cann_megamoe_buffer_params


def test_dummy_routes_cover_all_experts_across_ep_ranks():
    routed_experts = []
    for ep_rank_id in range(4):
        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        active_mask = torch.tensor([1, 0], dtype=torch.int8)

        hidden_states, topk_ids, topk_weights, active_mask, original_num_tokens = _append_cann_megamoe_dummy_tokens(
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


def test_receive_bound_uses_documented_worst_case():
    assert get_cann_megamoe_buffer_params(480, 32, 256, 8) == (512, 8, 32, 131072)


@pytest.mark.parametrize(
    ("quant_name", "expected"),
    [
        ("w8a8_dynamic", MoECommType.FUSED_MC2),
        ("w4a8_dynamic", MoECommType.FUSED_MC2),
    ],
)
def test_a2_mode_2_selects_supported_megamoe_quantization(monkeypatch, quant_name, expected):
    monkeypatch.setattr(afc, "_MEGA_MOE_SUPPORTED", True)
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
            moe_intermediate_size=1536,
            num_experts_per_tok=8,
            quantize=quant_name,
        ),
        get_hidden_size=lambda: 4096,
        get_num_experts=lambda: 256,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config=SimpleNamespace(quant_description={"moe": quant_name}),
        lora_config=None,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=32,
            pipeline_parallel_size=1,
        ),
    )

    assert afc.select_moe_comm_method(512, vllm_config) == expected


def test_w4a8_uses_int4_megamoe_weights():
    dispatch_mode, dispatch_dtype, weight_type = _get_cann_mega_moe_quant_settings(QuantType.W4A8)

    assert dispatch_mode == _CANN_MEGA_MOE_QUANT_MODE_INT8
    assert dispatch_dtype == _CANN_TORCH_INT8
    assert weight_type == _CANN_ACL_INT4


def test_w4a8_prepares_megamoe_weights_for_a2_mode_2(monkeypatch):
    ascend_config = SimpleNamespace(enable_fused_mc2=2)
    monkeypatch.setattr(w4a8_method, "_MEGA_MOE_SUPPORTED", True)
    monkeypatch.setattr(w4a8_method, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(w4a8_method, "_is_a2_megamoe_enabled", lambda config: config is ascend_config)

    assert w4a8_method.AscendW4A8DynamicFusedMoEMethod._cann_megamoe_enabled()


@pytest.mark.parametrize(("configured_max_recv", "expected_max_recv"), [(None, 24), ("8", 8)])
def test_symm_buffer_uses_receive_bound_override(monkeypatch, configured_max_recv, expected_max_recv):
    if configured_max_recv is not None:
        monkeypatch.setenv("VLLM_ASCEND_MEGAMOE_MAX_RECV_TOKENS", configured_max_recv)
    dispatcher = object.__new__(comm_method.TokenDispatcherWithMC2)
    dispatcher.global_bs = 0
    dispatcher.max_num_tokens_per_rank = 2
    dispatcher.ep_world_size = 8
    dispatcher.ep_rank_id = 0

    impl = object.__new__(comm_method.FusedMC2CommImpl)
    impl.token_dispatcher = dispatcher
    impl.moe_config = SimpleNamespace(
        experts_per_token=8,
        num_experts=8,
        hidden_dim=1024,
        intermediate_size_per_partition=512,
    )
    call = {}

    def get_buffer(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return object()

    impl.get_symm_buffer_for_mega_moe = get_buffer
    monkeypatch.setattr(comm_method, "get_mc2_group", lambda: SimpleNamespace(device_group="group"))
    monkeypatch.setattr(
        comm_method.comm_utils,
        "_get_cann_mega_moe_quant_settings",
        lambda _: (2, torch.int8, _CANN_ACL_INT4),
    )

    impl._init_mega_moe_symm_buffer(SimpleNamespace(quant=SimpleNamespace(quant_type=QuantType.W4A8)))

    assert call["args"] == ("group", 8, 3, 8)
    assert call["kwargs"]["max_recv_token_num"] == expected_max_recv


def test_w4a8_compacts_each_megamoe_expert_before_nz(monkeypatch):
    formatted_weights = []

    def fake_format_cast(weight, npu_format):
        formatted_weights.append(
            (tuple(weight.shape), npu_format, weight.storage_offset(), weight.untyped_storage().nbytes())
        )
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
    method.new_quant_version = True
    method.quant_method = ""
    method._maybe_build_cann_mega_moe_lists(layer)

    assert formatted_weights == [
        ((8, 16), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0, 128),
        ((8, 16), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0, 128),
        ((16, 8), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0, 128),
        ((16, 8), w4a8_method.ACL_FORMAT_FRACTAL_NZ, 0, 128),
    ]
    assert len(layer.cann_mega_moe_w13_weight_list) == 2
    assert len(layer.cann_mega_moe_w2_weight_list) == 2
    assert layer.cann_mega_moe_w13_weight_list[0].dtype == torch.int8
    assert layer.cann_mega_moe_w2_weight_list[0].dtype == torch.int8
    assert (
        layer.cann_mega_moe_w13_weight_list[0].untyped_storage().data_ptr()
        != layer.cann_mega_moe_w13_weight_list[1].untyped_storage().data_ptr()
    )
    assert hasattr(layer, "w13_weight")
    assert hasattr(layer, "w2_weight")
    assert layer.w13_weight.data.dtype == torch.int32
    assert layer.w2_weight.data.dtype == torch.int32
