#!/usr/bin/env python3

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ascend_forward_context import (
    MoECommType,
    _get_a2_megamoe_step_fallback_reason,
    _cann_megamoe_supported_by_config,
    _resolve_moe_comm_type,
    _select_a2_moe_comm_method,
    _should_force_eager_megamoe_runtime,
    _should_skip_compiled_megamoe_profile,
    _should_skip_compiled_megamoe_runtime,
    empty_dp_step_requires_dummy_forward,
)
from vllm_ascend.envs import env_variables
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    FusedMC2CommImpl,
    _append_cann_megamoe_dummy_tokens,
    _get_cann_megamoe_layer_index,
    _normalize_cann_megamoe_activation,
    _parse_cann_megamoe_fallback_layer_indices,
)
from vllm_ascend.utils import (
    AscendDeviceType,
    get_cann_megamoe_dummy_token_capacity,
    resolve_cann_megamoe_max_recv_tokens,
)


def check_dummy_routing() -> None:
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)

    all_dummy_masks = []
    for ep_rank, expected_dummy_mask in (
        (0, [1, 0, 1, 0]),
        (1, [0, 1, 0, 1]),
    ):
        output = _append_cann_megamoe_dummy_tokens(
            hidden_states,
            topk_ids,
            topk_weights,
            None,
            num_experts=8,
            ep_rank_id=ep_rank,
            ep_world_size=2,
        )
        output_hidden, output_ids, output_weights, output_mask, original_tokens = output
        assert original_tokens == 2
        assert output_hidden.shape == (6, 4)
        assert output_ids.shape == (6, 2)
        assert output_weights.shape == (6, 2)
        assert output_mask.tolist() == [1, 1] + expected_dummy_mask
        all_dummy_masks.append(output_mask[2:])
        assert torch.count_nonzero(output_hidden[2:]).item() == 0
        assert torch.count_nonzero(output_weights[2:]).item() == 0
        assert set(output_ids[2:].reshape(-1).tolist()) == set(range(8))
    assert torch.stack(all_dummy_masks).sum(dim=0).tolist() == [1, 1, 1, 1]


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


def check_a2_min_tokens_selection() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_num_experts=lambda: 256),
        parallel_config=SimpleNamespace(world_size_across_dp=32, pipeline_parallel_size=1),
    )
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "vllm_ascend.ascend_forward_context._cann_megamoe_supported_by_config",
            return_value=True,
        ),
    ):
        assert _select_a2_moe_comm_method(1, vllm_config, "w8a8", 4096, False) == MoECommType.FUSED_MC2
        assert _select_a2_moe_comm_method(4096, vllm_config, "w8a8", 4096, False) == MoECommType.FUSED_MC2

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_MEGAMOE_MIN_TOKENS": "512"}),
        patch(
            "vllm_ascend.ascend_forward_context._cann_megamoe_supported_by_config",
            return_value=True,
        ),
    ):
        assert _select_a2_moe_comm_method(511, vllm_config, "w8a8", 4096, False) == MoECommType.MC2
        assert _select_a2_moe_comm_method(512, vllm_config, "w8a8", 4096, False) == MoECommType.FUSED_MC2


def check_dp_policy_defaults() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert not env_variables["VLLM_ASCEND_MEGAMOE_REQUIRE_UNIFORM_DP_TOKENS"]()
        assert env_variables["VLLM_ASCEND_MEGAMOE_REQUIRE_UNIFORM_DP_GRAPH"]()
        assert not env_variables["VLLM_ASCEND_MEGAMOE_FORCE_EAGER_DECODE"]()
        assert env_variables["VLLM_ASCEND_MEGAMOE_REQUIRE_NONZERO_DP_TOKENS"]()


def check_force_eager_megamoe_runtime() -> None:
    config = SimpleNamespace(enable_fused_mc2=2)
    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_MEGAMOE_FORCE_EAGER_DECODE": "1"},
            clear=True,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
    ):
        assert _should_force_eager_megamoe_runtime(config)
        assert not _should_force_eager_megamoe_runtime(
            config,
            is_graph_capturing=True,
        )

    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "vllm_ascend.ascend_forward_context.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
    ):
        assert not _should_force_eager_megamoe_runtime(config)

    config.enable_fused_mc2 = 0
    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_MEGAMOE_FORCE_EAGER_DECODE": "1"},
            clear=True,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
    ):
        assert not _should_force_eager_megamoe_runtime(config)


def check_selective_compiled_megamoe_runtime() -> None:
    config = SimpleNamespace(enable_fused_mc2=2)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "vllm_ascend.ascend_forward_context.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
    ):
        assert not _should_skip_compiled_megamoe_runtime(
            config,
            decode_graph_safe=True,
        )
        assert _should_skip_compiled_megamoe_runtime(
            config,
            decode_graph_safe=False,
        )

    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_MEGAMOE_FORCE_EAGER_DECODE": "1"},
            clear=True,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
    ):
        assert _should_skip_compiled_megamoe_runtime(
            config,
            decode_graph_safe=True,
        )
        assert not _should_skip_compiled_megamoe_runtime(
            config,
            decode_graph_safe=False,
            is_graph_capturing=True,
        )

    config.enable_fused_mc2 = 0
    with patch(
        "vllm_ascend.ascend_forward_context.get_ascend_device_type",
        return_value=AscendDeviceType.A2,
    ):
        assert not _should_skip_compiled_megamoe_runtime(
            config,
            decode_graph_safe=False,
        )


def check_eager_megamoe_profile() -> None:
    config = SimpleNamespace(enable_fused_mc2=2)
    with patch(
        "vllm_ascend.ascend_forward_context.get_ascend_device_type",
        return_value=AscendDeviceType.A2,
    ):
        assert _should_skip_compiled_megamoe_profile(config, is_profile=True)
        assert not _should_skip_compiled_megamoe_profile(config, is_profile=False)

    config.enable_fused_mc2 = 0
    with patch(
        "vllm_ascend.ascend_forward_context.get_ascend_device_type",
        return_value=AscendDeviceType.A2,
    ):
        assert not _should_skip_compiled_megamoe_profile(config, is_profile=True)


def check_empty_dp_dummy_forward() -> None:
    assert empty_dp_step_requires_dummy_forward("external_launcher", 4)
    assert not empty_dp_step_requires_dummy_forward("external_launcher", 1)
    assert not empty_dp_step_requires_dummy_forward("mp", 4)


def check_step_moe_comm_type_override() -> None:
    config = SimpleNamespace()
    with patch(
        "vllm_ascend.ascend_forward_context.select_moe_comm_method",
        side_effect=AssertionError("override must bypass dynamic selection"),
    ):
        assert _resolve_moe_comm_type(1, config, False, MoECommType.MC2) == MoECommType.MC2

    with patch(
        "vllm_ascend.ascend_forward_context.select_moe_comm_method",
        return_value=MoECommType.FUSED_MC2,
    ):
        assert _resolve_moe_comm_type(1, config, False, None) == MoECommType.FUSED_MC2


def check_step_fallback_reason() -> None:
    config = SimpleNamespace(enable_fused_mc2=2)
    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_MEGAMOE_REQUIRE_UNIFORM_DP_TOKENS": "1",
                "VLLM_ASCEND_MEGAMOE_REQUIRE_NONZERO_DP_TOKENS": "1",
            },
            clear=True,
        ),
        patch(
            "vllm_ascend.ascend_forward_context.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
    ):
        assert _get_a2_megamoe_step_fallback_reason(config, True, True) is None
        assert _get_a2_megamoe_step_fallback_reason(config, False, True) == "non-uniform-dp-tokens"
        assert _get_a2_megamoe_step_fallback_reason(config, True, False) == "idle-dp-rank"
        assert (
            _get_a2_megamoe_step_fallback_reason(config, False, False)
            == "non-uniform-dp-tokens,idle-dp-rank"
        )

    config.enable_fused_mc2 = 0
    with patch(
        "vllm_ascend.ascend_forward_context.get_ascend_device_type",
        return_value=AscendDeviceType.A2,
    ):
        assert _get_a2_megamoe_step_fallback_reason(config, False, False) is None


def check_cross_rank_contract() -> None:
    impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    impl._cann_megamoe_call_index = 0
    impl._cann_megamoe_last_contract_signature = None
    impl._cann_megamoe_contract_check = True
    impl._cann_megamoe_trace_every_call = False
    impl.moe_config = SimpleNamespace(num_experts=8)
    impl.token_dispatcher = SimpleNamespace(ep_rank_id=0, ep_world_size=2)

    hidden_states = torch.ones((6, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3]],
        dtype=torch.int32,
    )
    topk_weights = torch.full((6, 2), 0.5, dtype=torch.float32)
    active_mask = torch.ones(6, dtype=torch.int8)

    def gather_matching(output, input_tensor, group=None):
        del group
        output.copy_(input_tensor.repeat(2))

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch("torch.distributed.all_gather_into_tensor", side_effect=gather_matching),
    ):
        call_index, _ = impl._check_cann_megamoe_contract(
            hidden_states,
            topk_ids,
            topk_weights,
            active_mask,
        )
        assert call_index == 0

    def gather_mismatched(output, input_tensor, group=None):
        del group
        remote = input_tensor.clone()
        remote[2] += 1
        output.copy_(torch.cat((input_tensor, remote)))

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch("torch.distributed.all_gather_into_tensor", side_effect=gather_mismatched),
    ):
        try:
            impl._check_cann_megamoe_contract(
                hidden_states,
                topk_ids,
                topk_weights,
                active_mask,
            )
        except RuntimeError as exc:
            assert "identical call order and num_tokens" in str(exc)
        else:
            raise AssertionError("Expected a cross-rank MegaMoe contract mismatch.")


def check_route_fallback_guard() -> None:
    impl = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    impl._cann_megamoe_max_tokens_per_expert = 0
    impl._cann_megamoe_require_uniform_dp_tokens = False
    impl._cann_megamoe_require_nonzero_dp_tokens = False
    impl._cann_megamoe_fallback_count = 0
    impl._cann_megamoe_uniform_dp_fallback_count = 0
    impl._cann_megamoe_idle_dp_fallback_count = 0
    impl._cann_megamoe_expert_threshold_fallback_count = 0
    impl._cann_megamoe_layer_fallback_count = 0
    impl._cann_megamoe_fallback_layer_indices = set()
    impl.moe_config = SimpleNamespace(num_experts=8)
    impl.token_dispatcher = SimpleNamespace(ep_rank_id=0, ep_world_size=4)
    impl.prepare_finalize = SimpleNamespace(tp_size=2)

    fused_input = SimpleNamespace(routing=SimpleNamespace(mc2_mask=torch.ones(2, dtype=torch.int8)))
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    def set_max_count(max_count):
        def all_reduce(counts, op=None, group=None):
            del op, group
            counts.zero_()
            counts[0] = max_count

        return all_reduce

    def set_active_counts(counts):
        def all_gather(output, input_tensor, group=None):
            del input_tensor, group
            output.copy_(torch.tensor(counts, dtype=output.dtype))

        return all_gather

    should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(fused_input, topk_ids)
    assert not should_fallback
    assert max_count == 0
    assert (active_min, active_max) == (0, 0)

    impl._cann_megamoe_max_tokens_per_expert = 1792
    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("decode-sized route must not synchronize"),
        ),
    ):
        should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(
            fused_input, topk_ids
        )
        assert not should_fallback
        assert max_count == 0
        assert (active_min, active_max) == (0, 0)

    impl._cann_megamoe_require_uniform_dp_tokens = True
    fused_input = SimpleNamespace(routing=SimpleNamespace(mc2_mask=torch.ones(256, dtype=torch.int8)))
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).repeat(128, 1)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch("torch.distributed.all_gather_into_tensor", side_effect=set_active_counts([256, 0, 256, 0])),
        patch("torch.distributed.all_reduce", side_effect=set_max_count(1792)),
    ):
        should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(
            fused_input, topk_ids
        )
        assert not should_fallback
        assert max_count == 1792
        assert (active_min, active_max) == (256, 256)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch("torch.distributed.all_gather_into_tensor", side_effect=set_active_counts([256, 0, 256, 0])),
        patch("torch.distributed.all_reduce", side_effect=set_max_count(1793)),
    ):
        should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(
            fused_input, topk_ids
        )
        assert should_fallback
        assert max_count == 1793
        assert (active_min, active_max) == (256, 256)
        assert impl._cann_megamoe_fallback_count == 1

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch("torch.distributed.all_gather_into_tensor", side_effect=set_active_counts([128, 0, 256, 0])),
        patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("mixed DP fallback must short-circuit route synchronization"),
        ),
    ):
        should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(
            fused_input, topk_ids
        )
        assert should_fallback
        assert max_count == 0
        assert (active_min, active_max) == (128, 256)
        assert impl._cann_megamoe_fallback_count == 2
        assert impl._cann_megamoe_uniform_dp_fallback_count == 1
        assert impl._cann_megamoe_expert_threshold_fallback_count == 1

    impl._cann_megamoe_require_uniform_dp_tokens = False
    impl._cann_megamoe_require_nonzero_dp_tokens = True
    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch("torch.distributed.all_gather_into_tensor", side_effect=set_active_counts([128, 0, 256, 0])),
        patch("torch.distributed.all_reduce", side_effect=set_max_count(128)),
    ):
        should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(
            fused_input, topk_ids
        )
        assert not should_fallback
        assert max_count == 128
        assert (active_min, active_max) == (128, 256)
        assert impl._cann_megamoe_fallback_count == 2

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group",
            return_value=SimpleNamespace(device_group=None),
        ),
        patch("torch.distributed.all_gather_into_tensor", side_effect=set_active_counts([0, 0, 256, 0])),
        patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("idle DP fallback must short-circuit route synchronization"),
        ),
    ):
        should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(
            fused_input, topk_ids
        )
        assert should_fallback
        assert max_count == 0
        assert (active_min, active_max) == (0, 256)
        assert impl._cann_megamoe_fallback_count == 3
        assert impl._cann_megamoe_idle_dp_fallback_count == 1

    impl._cann_megamoe_require_uniform_dp_tokens = False
    impl._cann_megamoe_require_nonzero_dp_tokens = False
    impl._cann_megamoe_max_tokens_per_expert = 0
    impl._cann_megamoe_fallback_layer_indices = {17}
    with patch(
        "vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX",
        SimpleNamespace(moe_layer_index=17),
    ):
        should_fallback, max_count, active_min, active_max = impl._cann_megamoe_should_fallback(
            fused_input, topk_ids
        )
    assert should_fallback
    assert max_count == 0
    assert (active_min, active_max) == (0, 0)
    assert impl._cann_megamoe_layer_fallback_count == 1


def check_fallback_layer_parser() -> None:
    assert _parse_cann_megamoe_fallback_layer_indices("") == set()
    assert _parse_cann_megamoe_fallback_layer_indices("17, 3,17") == {3, 17}
    assert _get_cann_megamoe_layer_index() == -1
    try:
        _parse_cann_megamoe_fallback_layer_indices("-1")
    except ValueError as exc:
        assert "non-negative integers" in str(exc)
    else:
        raise AssertionError("Expected negative fallback layer index to fail.")


def main() -> None:
    check_dummy_routing()
    check_capacity_helpers()
    check_a2_selection_guard()
    check_a2_min_tokens_selection()
    check_dp_policy_defaults()
    check_force_eager_megamoe_runtime()
    check_selective_compiled_megamoe_runtime()
    check_eager_megamoe_profile()
    check_empty_dp_dummy_forward()
    check_step_moe_comm_type_override()
    check_step_fallback_reason()
    check_cross_rank_contract()
    check_route_fallback_guard()
    check_fallback_layer_parser()
    assert _normalize_cann_megamoe_activation("silu") == "swiglu"
    print("A2 MegaMoe helper checks: PASS", flush=True)


if __name__ == "__main__":
    main()
