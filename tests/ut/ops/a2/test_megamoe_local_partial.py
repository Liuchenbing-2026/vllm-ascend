# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_config, utils
from vllm_ascend.ops.fused_moe import moe_comm_method as comm
from vllm_ascend.ops.fused_moe import prepare_finalize as pf
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("rows", [1, 31, 32, 511, 512, 513, 8191, 8192])
def test_full_input_and_local_output_survive_prepare_finalize(rows, monkeypatch):
    target = (rows + 3) // 4 * 4
    mask = torch.arange(target) < rows - 1
    monkeypatch.setattr(pf, "_EXTRA_CTX", SimpleNamespace(mc2_mask=mask, padded_num_tokens=target))
    x = torch.randn(rows, 4, dtype=torch.bfloat16)
    logits = torch.randn(rows, 8)
    input_ids = torch.arange(rows)
    for rank in range(4):
        prepare = object.__new__(pf.PrepareAndFinalizeWithLocalPartial)
        prepare.tp_size, prepare.tp_rank = 4, rank
        result = prepare.prepare(x, logits)
        assert result.hidden_states is x and result.router_logits is logits
        assert torch.equal(result.mc2_mask, mask[:rows])
        assert result.padded_hidden_states_shape == x.shape
        partial = x * (rank + 1)
        assert prepare.finalize_tp_partial(partial, x.shape) is partial
        assert prepare.pad_and_split_input_ids(input_ids) is input_ids
        with pytest.raises(ValueError, match="full input shape"):
            prepare.finalize_tp_partial(partial[: rows // 4], x.shape)
        with pytest.raises(RuntimeError, match="deferred"):
            prepare.finalize(partial, True)


@pytest.mark.parametrize(
    "replace,quant,dtype",
    [
        (True, QuantType.NONE, torch.bfloat16),
        (False, QuantType.W8A8, torch.bfloat16),
        (False, QuantType.NONE, torch.float16),
    ],
)
def test_rejects_sharded_or_quantized_inputs(replace, quant, dtype):
    prepare = object.__new__(pf.PrepareAndFinalizeWithLocalPartial)
    with pytest.raises(ValueError, match="unsharded BF16"):
        prepare.prepare(torch.ones(8, 4, dtype=dtype), torch.ones(8, 8), replace, quant)


@pytest.mark.parametrize("tokens", [1, 511, 512, 513, 8191, 8192])
def test_group_buffer_query_matches_operator_full_input_capacity(tokens, monkeypatch):
    import vllm.config

    cfg = SimpleNamespace(mega_moe_replicated_dispatch=False, mega_moe_local_partial=True)
    vc = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_experts=lambda: 256,
            hf_text_config=SimpleNamespace(hidden_size=2048, num_experts_per_tok=8),
        ),
        parallel_config=SimpleNamespace(world_size_across_dp=4, pipeline_parallel_size=1, tensor_parallel_size=4),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
    )
    calls = []

    def query(*args, **kwargs):
        calls.append((args, kwargs))
        return 525

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: vc)
    monkeypatch.setattr(utils, "get_ascend_config", lambda: cfg)
    monkeypatch.setattr(utils, "_load_cann_megamoe_ccl_buffer_size", lambda: query)
    assert utils.calculate_cann_megamoe_hccl_buffer_size() == 525
    impl = object.__new__(comm.FusedMC2CommImpl)
    impl.token_dispatcher = object.__new__(comm.TokenDispatcherWithMC2)
    impl.token_dispatcher.global_bs = 0
    impl.token_dispatcher.ep_world_size = 4
    impl.token_dispatcher.max_num_tokens_per_rank = (tokens + 3) // 4
    impl.moe_config = SimpleNamespace(
        experts_per_token=8,
        num_experts=256,
        hidden_dim=2048,
        intermediate_size_per_partition=512,
    )
    impl.get_symm_buffer_for_mega_moe = query
    monkeypatch.setattr(comm, "get_ascend_config", lambda: cfg)
    monkeypatch.setattr(comm, "get_mc2_group", lambda: SimpleNamespace(device_group="test_group"))
    monkeypatch.setattr(comm, "_is_a2_megamoe_enabled", lambda _: True)
    impl._init_mega_moe_symm_buffer()
    group_args, group_kwargs = calls[0]
    op_args, op_kwargs = calls[1]
    expected_rows = (tokens + 3) // 4 * 4 + 32
    assert group_args[1:4] == op_args[1:4] == (256, expected_rows, 8)
    assert group_kwargs["max_recv_token_num"] == op_kwargs["max_recv_token_num"] == expected_rows * 8
    assert group_kwargs["comm_alg"] == op_kwargs["comm_alg"] == "local_partial_tp4"
    assert group_kwargs["dispatch_quant_mode"] == op_kwargs["dispatch_quant_mode"] == 0
    assert group_kwargs["dispatch_quant_out_dtype"] is op_kwargs["dispatch_quant_out_dtype"] is None


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "tp",
        "dp",
        "sp",
        "quant",
        "dtype",
        "shape",
        "capacity",
        "lora",
        "eplb",
        "disabled",
        "replicated",
        "placement",
    ],
)
def test_config_support_boundary(invalid, monkeypatch):
    cfg = SimpleNamespace(
        mega_moe_local_partial=True,
        mega_moe_replicated_dispatch=False,
        enable_fused_mc2=1,
        enable_sp_by_pass=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    pc = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=4,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        expert_placement_strategy="linear",
    )
    hf = SimpleNamespace(hidden_size=2048, moe_intermediate_size=512, num_experts_per_tok=8)
    mc = SimpleNamespace(dtype=torch.bfloat16, hf_text_config=hf, get_num_experts=lambda: 256)
    vc = SimpleNamespace(
        parallel_config=pc,
        model_config=mc,
        quant_config=None,
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        use_v2_model_runner=False,
    )
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A2)
    monkeypatch.setattr(ascend_config, "is_mega_moe_supported", lambda: True)
    if invalid == "tp":
        pc.tensor_parallel_size = 2
    elif invalid == "dp":
        pc.data_parallel_size = 2
    elif invalid == "sp":
        cfg.enable_sp_by_pass = True
    elif invalid == "quant":
        vc.quant_config = object()
    elif invalid == "dtype":
        mc.dtype = torch.float16
    elif invalid == "shape":
        hf.moe_intermediate_size = 1024
    elif invalid == "capacity":
        vc.scheduler_config.max_num_batched_tokens = 8193
    elif invalid == "lora":
        vc.lora_config = object()
    elif invalid == "eplb":
        cfg.eplb_config.dynamic_eplb = True
    elif invalid == "disabled":
        cfg.enable_fused_mc2 = 0
    elif invalid == "replicated":
        cfg.mega_moe_replicated_dispatch = True
    elif invalid == "placement":
        pc.expert_placement_strategy = "round_robin"
    if invalid is None:
        ascend_config.AscendConfig._validate_megamoe_local_partial(cfg, vc)
    else:
        with pytest.raises(ValueError, match="mega_moe_local_partial requires"):
            ascend_config.AscendConfig._validate_megamoe_local_partial(cfg, vc)


def test_local_partial_preserves_shared_add_rounding(monkeypatch):
    # This fails for source-owner zero embedding even with an exact routed sum.
    monkeypatch.setattr(pf, "_EXTRA_CTX", SimpleNamespace(mc2_mask=torch.ones(1, dtype=torch.bool)))
    routed = torch.tensor([1, -1, 0, 0], dtype=torch.bfloat16).reshape(4, 1, 1)
    shared = torch.tensor([1 / 256, 0, 0, 0], dtype=torch.bfloat16).reshape(4, 1, 1)
    partials = []
    for rank in range(4):
        prepare = object.__new__(pf.PrepareAndFinalizeWithLocalPartial)
        result = prepare.prepare(routed[rank], torch.zeros(1, 8))
        partials.append(prepare.finalize_tp_partial(routed[rank], result.padded_hidden_states_shape))
    assert torch.equal((torch.stack(partials) + shared).sum(0), (routed + shared).sum(0))
    old_owner = torch.zeros_like(routed)
    old_owner[0] = routed.sum(0)
    assert not torch.equal((old_owner + shared).sum(0), (routed + shared).sum(0))


def test_only_fused_method_selects_full_input_prepare(monkeypatch):
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(mega_moe_local_partial=True))
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithLocalPartial", lambda _: "local_partial")
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithMC2", lambda _: "ordinary")
    fused, ordinary = object.__new__(comm.FusedMC2CommImpl), object.__new__(comm.MC2CommImpl)
    fused.moe_config = ordinary.moe_config = None
    assert fused._get_prepare_finalize() == "local_partial"
    assert ordinary._get_prepare_finalize() == "ordinary"
