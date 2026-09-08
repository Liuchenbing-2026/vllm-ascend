from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method as comm_module
from vllm_ascend.ops.fused_moe.moe_comm_method import _append_cann_megamoe_dummy_tokens
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import _warn_if_megamoe_shape_is_unfavourable, get_cann_megamoe_buffer_params


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


@pytest.mark.parametrize("ep_rank_id", [0, 1, 7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dummy_cache_reuses_constants_across_batch_sizes(monkeypatch, ep_rank_id, dtype):
    cache = {}

    def inputs(rows):
        return (
            torch.zeros((rows, 16), dtype=dtype),
            torch.zeros((rows, 8), dtype=torch.int32),
            torch.full((rows, 8), 0.125),
            torch.zeros(rows, dtype=torch.int8),
        )

    first_inputs, next_inputs = inputs(2), inputs(5)
    first = _append_cann_megamoe_dummy_tokens(*first_inputs, 256, ep_rank_id, 8, dummy_cache=cache)
    assert len(cache) == 1
    expected_routes = torch.arange(ep_rank_id * 32, (ep_rank_id + 1) * 32).reshape(4, 8)
    assert torch.equal(first[1][-4:], expected_routes)
    expected = tuple(x.clone() for x in first[:4])
    for tensor in first[:4]:
        tensor.zero_()

    def unexpected_factory(*args, **kwargs):
        raise AssertionError("Warm dummy padding must reuse constant tensors")

    for name in ("arange", "ones", "full"):
        monkeypatch.setattr(torch, name, unexpected_factory)
    second = _append_cann_megamoe_dummy_tokens(*next_inputs, 256, ep_rank_id, 8, dummy_cache=cache)
    assert len(cache) == 1 and second[4] == 5
    for actual, original, padded in zip(second[:4], next_inputs, expected):
        assert torch.equal(actual[:5], original)
        assert torch.equal(actual[-4:], padded[-4:])


def test_dummy_cache_separates_rank_dtype_and_mask_contracts():
    cache = {}
    for rank, dtype, mask_dtype in (
        (0, torch.bfloat16, torch.int8),
        (1, torch.bfloat16, torch.int8),
        (1, torch.float16, torch.bool),
    ):
        x = torch.zeros((3, 16), dtype=dtype)
        ids = torch.zeros((3, 8), dtype=torch.int64)
        weights = torch.full((3, 8), 0.125)
        mask = torch.tensor([1, 0, 1], dtype=mask_dtype)
        result = _append_cann_megamoe_dummy_tokens(x, ids, weights, mask, 256, rank, 8, dummy_cache=cache)
        assert result[0].dtype == dtype and result[3].dtype == mask_dtype
        assert result[3].tolist() == [1, 0, 1, 1, 1, 1, 1]
        assert result[1][-4:].flatten().tolist() == list(range(rank * 32, (rank + 1) * 32))
    assert len(cache) == 3
    result = _append_cann_megamoe_dummy_tokens(x, ids, weights, None, 256, rank, 8, dummy_cache=cache)
    assert result[3].dtype == torch.int8 and result[3].tolist() == [1] * 7


def _make_a2_config(
    *,
    quantize: str | None,
    use_v2_model_runner: bool = False,
    hidden_size: int = 4096,
    moe_intermediate_size: int = 1536,
):
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=8,
            quantize=quantize,
        ),
        get_hidden_size=lambda: hidden_size,
        get_num_experts=lambda: 256,
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        lora_config=None,
        use_v2_model_runner=use_v2_model_runner,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )


@pytest.mark.parametrize("quantize", ["w8a8_dynamic", "w4a8_dynamic"])
def test_a2_mode_1_selects_megamoe_for_supported_v1_config(monkeypatch, quantize):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )

    assert afc.select_moe_comm_method(512, _make_a2_config(quantize=quantize)) == MoECommType.FUSED_MC2


@pytest.mark.parametrize(
    ("is_draft_model", "use_v2_model_runner"),
    [(True, False), (False, True)],
)
def test_a2_megamoe_falls_back_for_unvalidated_runner_paths(monkeypatch, is_draft_model, use_v2_model_runner):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )

    result = afc.select_moe_comm_method(
        512,
        _make_a2_config(quantize="w4a8_dynamic", use_v2_model_runner=use_v2_model_runner),
        is_draft_model=is_draft_model,
    )
    assert result == MoECommType.ALLGATHER


def _patch_a2_megamoe_env(monkeypatch):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_min_tokens=512,
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )


@pytest.mark.parametrize(
    ("moe_intermediate_size", "expected"),
    [
        (512, MoECommType.FUSED_MC2),  # Qwen3.5/3.6-35B-A3B, documented lower bound
        (3072, MoECommType.FUSED_MC2),  # documented upper bound
        (256, MoECommType.ALLGATHER),  # below the documented range
        (3584, MoECommType.ALLGATHER),  # above the documented range
        (768, MoECommType.ALLGATHER),  # not a multiple of 512
    ],
)
def test_a2_megamoe_intermediate_hidden_range(monkeypatch, moe_intermediate_size, expected):
    _patch_a2_megamoe_env(monkeypatch)
    config = _make_a2_config(
        quantize="w8a8_dynamic",
        hidden_size=2048,
        moe_intermediate_size=moe_intermediate_size,
    )
    assert afc.select_moe_comm_method(512, config) == expected


@pytest.mark.parametrize("ep_world_size", [4, 8])
@pytest.mark.parametrize("num_tokens", [32, 512, 4096, 4097])
def test_a2_nonquantized_bf16_uses_megamoe_only_in_prefill_window(monkeypatch, ep_world_size, num_tokens):
    _patch_a2_megamoe_env(monkeypatch)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    config = _make_a2_config(quantize=None, hidden_size=2048, moe_intermediate_size=512)
    config.model_config.dtype = torch.bfloat16
    config.parallel_config.world_size_across_dp = ep_world_size
    expected = MoECommType.FUSED_MC2 if 512 <= num_tokens <= 4096 else MoECommType.ALLGATHER
    assert afc.select_moe_comm_method(num_tokens, config) == expected


@pytest.mark.parametrize(
    ("dtype", "quantize", "quant_config", "quantization"),
    [
        (torch.float16, None, None, None),
        (torch.float32, None, None, None),
        (None, None, None, None),
        (torch.bfloat16, "fp8", None, None),
        (torch.bfloat16, None, SimpleNamespace(quant_description={}), None),
        (torch.bfloat16, None, None, "fp8"),
    ],
)
def test_a2_bf16_gate_does_not_admit_other_dtypes_or_unknown_quantization(
    monkeypatch, dtype, quantize, quant_config, quantization
):
    _patch_a2_megamoe_env(monkeypatch)
    config = _make_a2_config(quantize=quantize, hidden_size=2048, moe_intermediate_size=512)
    config.model_config.dtype = dtype
    config.model_config.quantization = quantization
    config.quant_config = quant_config
    assert afc.select_moe_comm_method(512, config) == MoECommType.ALLGATHER


def test_a2_nonquantized_bf16_off_stays_on_allgather(monkeypatch):
    _patch_a2_megamoe_env(monkeypatch)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=0, mega_moe_min_tokens=512),
    )
    config = _make_a2_config(quantize=None, hidden_size=2048, moe_intermediate_size=512)
    config.model_config.dtype = torch.bfloat16
    assert afc.select_moe_comm_method(512, config) == MoECommType.ALLGATHER


def test_a2_bf16_operator_call_keeps_quantization_disabled(monkeypatch):
    monkeypatch.setattr(comm_module, "_is_a2_megamoe_enabled", lambda _: True)
    monkeypatch.setattr(comm_module, "get_ascend_config", lambda: SimpleNamespace(mega_moe_replicated_dispatch=False))
    impl = object.__new__(comm_module.FusedMC2CommImpl)
    impl.token_dispatcher = object.__new__(comm_module.TokenDispatcherWithMC2)
    impl.token_dispatcher.global_bs = 0
    impl.token_dispatcher.ep_rank_id = 0
    impl.token_dispatcher.ep_world_size = 4
    impl.moe_config = SimpleNamespace(num_experts=8)
    impl.swiglu_limit = 0
    impl._cann_megamoe_dummy_cache = {}
    impl.mega_moe_symm_buffer = SimpleNamespace()
    weight = torch.ones((4, 4), dtype=torch.bfloat16)
    x = torch.ones((2, 4), dtype=torch.bfloat16)
    inp = SimpleNamespace(
        hidden_states=x,
        topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        topk_weights=torch.full((2, 2), 0.5),
        quant=SimpleNamespace(quant_type=QuantType.NONE),
        routing=SimpleNamespace(mc2_mask=None),
        weights=SimpleNamespace(
            w1=[weight], w2=[weight], w1_scale=None, w2_scale=None, w1_scale_bias=None, w2_scale_bias=None
        ),
    )
    calls = []

    def fake_mega_moe(hidden, ids, probs, w1, w2, sym, **kwargs):
        calls.append(kwargs)
        assert hidden.dtype == w1[0].dtype == w2[0].dtype == torch.bfloat16
        assert sym.dispatch_quant_mode == 0 and sym.dispatch_quant_out_dtype is None
        assert all(kwargs[k] is None for k in ("l1_weights_sf", "l2_weights_sf", "weight1_type", "weight2_type"))
        return hidden.clone(), torch.zeros(2, dtype=torch.int32)

    impl.mega_moe = fake_mega_moe
    output, _ = impl._apply_cann_mega_moe(inp)
    assert len(calls) == 1 and torch.equal(output, x)


@pytest.mark.parametrize(
    ("ep_world_size", "experts_per_rank", "warns"),
    [
        (8, 32, True),  # Qwen3.6-35B-A3B on one A2 node: measured ~3.7x slower than AllGather
        (8, 16, True),  # EP below the threshold upstream requires for the MC2 family
        (16, 32, True),  # too many experts per rank: ~14us fixed cost each, per layer
        (16, 24, False),  # both thresholds satisfied
        (32, 4, False),
    ],
)
def test_warns_on_unfavourable_megamoe_shape(caplog, monkeypatch, ep_world_size, experts_per_rank, warns):
    from vllm_ascend import utils as ascend_utils

    # warning_once is lru_cached upstream; clear it so each parametrisation is
    # independent, but do not assume the attribute exists.
    getattr(ascend_utils.logger.warning_once, "cache_clear", lambda: None)()
    # vllm's parent logger stops propagation before pytest's root handler.
    monkeypatch.setattr(ascend_utils.logger, "handlers", [*ascend_utils.logger.handlers, caplog.handler])
    assert caplog.handler in ascend_utils.logger.handlers
    with caplog.at_level("WARNING"):
        _warn_if_megamoe_shape_is_unfavourable(ep_world_size, experts_per_rank)
    assert ("likely to be SLOWER" in caplog.text) is warns
