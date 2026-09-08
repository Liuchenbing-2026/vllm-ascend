# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import utils
from vllm_ascend.ops.fused_moe import moe_comm_method as comm
from vllm_ascend.ops.fused_moe import prepare_finalize as pf
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("rows", [1, 511, 512, 2048, 4095, 4096])
def test_replicated_inputs_preserve_each_original_source(rows, monkeypatch):
    target = (rows + 1) // 2 * 2
    mask = torch.arange(target) < rows
    monkeypatch.setattr(pf, "_EXTRA_CTX", SimpleNamespace(padded_num_tokens=target, mc2_mask=mask))
    x = torch.arange(rows * 4).reshape(rows, 4).to(torch.bfloat16)
    logits = torch.arange(rows * 8).reshape(rows, 8).float()
    new = object.__new__(pf.PrepareAndFinalizeWithReplicatedDispatch)
    prepared = new.prepare(x, logits)
    assert prepared.mc2_mask is mask
    assert prepared.padded_hidden_states_shape == (target, 4)
    source_results = []
    for rank in range(2):
        old = object.__new__(pf.PrepareAndFinalizeWithMC2)
        old.tp_size, old.tp_rank = 2, rank
        source = old.prepare(x, logits)
        half = target // 2
        assert torch.equal(prepared.hidden_states[rank * half : (rank + 1) * half], source.hidden_states)
        assert torch.equal(prepared.router_logits[rank * half : (rank + 1) * half], source.router_logits)
        assert torch.equal(prepared.mc2_mask[rank * half : (rank + 1) * half], source.mc2_mask)
        ids = source.router_logits.to(torch.int32).remainder(256)
        probs = torch.full(ids.shape, 0.125)
        source_results.append(
            comm._append_cann_megamoe_dummy_tokens(
                source.hidden_states, ids, probs, source.mc2_mask.to(torch.int8), 256, rank, 2, dummy_cache={}
            )
        )
    ids = prepared.router_logits.to(torch.int32).remainder(256)
    probs = torch.full(ids.shape, 0.125)
    cache = {}
    combined = comm._append_cann_megamoe_replicated_source_tokens(
        prepared.hidden_states, ids, probs, mask.to(torch.int8), 256, 2, cache
    )
    expected = [torch.cat((source_results[0][i], source_results[1][i])) for i in range(4)]
    assert combined[-1] == target
    assert all(torch.equal(combined[i], expected[i]) for i in range(4))
    # Returned tensors may be mutated by the kernel; cached sentinels must survive.
    for tensor in combined[:4]:
        tensor.zero_()
    again = comm._append_cann_megamoe_replicated_source_tokens(
        prepared.hidden_states, ids, probs, mask.to(torch.int8), 256, 2, cache
    )
    assert all(torch.equal(again[i], expected[i]) for i in range(4))
    partials = []
    for rank in range(2):
        raw = torch.zeros_like(again[0]).reshape(2, target // 2 + 16, 4)
        raw[rank, : target // 2] = source_results[rank][0][: target // 2]
        trimmed = raw[:, : target // 2].reshape(target, 4)
        partials.append(new.finalize_tp_partial(trimmed, prepared.padded_hidden_states_shape))
    assert torch.equal(partials[0] + partials[1], x)
    assert torch.equal(
        new.pad_and_split_input_ids(torch.arange(rows)), torch.nn.functional.pad(torch.arange(rows), (0, target - rows))
    )
    with pytest.raises(RuntimeError, match="deferred"):
        new.finalize(x, True)


@pytest.mark.parametrize(
    "replace,quant,dtype",
    [
        (True, QuantType.NONE, torch.bfloat16),
        (False, QuantType.W8A8, torch.bfloat16),
        (False, QuantType.NONE, torch.float16),
    ],
)
def test_replicated_prepare_rejects_incompatible_input(replace, quant, dtype):
    prepare = object.__new__(pf.PrepareAndFinalizeWithReplicatedDispatch)
    with pytest.raises(ValueError, match="unsharded BF16"):
        prepare.prepare(torch.ones(2, 4, dtype=dtype), torch.ones(2, 8), replace, quant)


def test_buffer_bound_preserves_old_receive_capacity():
    assert utils.get_cann_megamoe_buffer_params(2048, 2, 256, 8) == (2080, 128, 32, 33280)
    assert utils.get_cann_megamoe_buffer_params(2048, 2, 256, 8, replicated_dispatch=True) == (4128, 128, 32, 33280)
    for args in [(0, 2, 256, 8), (2049, 2, 256, 8), (2048, 4, 256, 8), (2048, 2, 128, 8), (2048, 2, 256, 4)]:
        with pytest.raises(ValueError):
            utils.get_cann_megamoe_buffer_params(*args, replicated_dispatch=True)


@pytest.mark.parametrize("tokens", [511, 2048, 4095, 4096])
def test_group_query_matches_operator_capacity(tokens, monkeypatch):
    import vllm.config

    config = SimpleNamespace(mega_moe_replicated_dispatch=True)
    vc = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_experts=lambda: 256, hf_text_config=SimpleNamespace(num_experts_per_tok=8, hidden_size=2048)
        ),
        parallel_config=SimpleNamespace(world_size_across_dp=2, pipeline_parallel_size=1, tensor_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
    )
    calls = []

    def query(*args, **kwargs):
        calls.append((args, kwargs))
        return 270

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: vc)
    monkeypatch.setattr(utils, "get_ascend_config", lambda: config)
    monkeypatch.setattr(utils, "_load_cann_megamoe_ccl_buffer_size", lambda: query)
    monkeypatch.setattr(utils, "_warn_if_megamoe_shape_is_unfavourable", lambda *args: None)
    assert utils.calculate_cann_megamoe_hccl_buffer_size() == 270
    impl = object.__new__(comm.FusedMC2CommImpl)
    impl.token_dispatcher = object.__new__(comm.TokenDispatcherWithMC2)
    impl.token_dispatcher.global_bs = 0
    impl.token_dispatcher.ep_world_size = 2
    impl.token_dispatcher.max_num_tokens_per_rank = (tokens + 1) // 2
    impl.moe_config = SimpleNamespace(
        experts_per_token=8, num_experts=256, hidden_dim=2048, intermediate_size_per_partition=512
    )
    impl.get_symm_buffer_for_mega_moe = query
    monkeypatch.setattr(comm, "get_ascend_config", lambda: config)
    monkeypatch.setattr(comm, "get_mc2_group", lambda: SimpleNamespace(device_group="test_group"))
    monkeypatch.setattr(comm, "_is_a2_megamoe_enabled", lambda _: True)
    impl._init_mega_moe_symm_buffer()
    group_args, group_kwargs = calls[0]
    op_args, op_kwargs = calls[1]
    assert group_args[1:4] == op_args[1:4]
    assert group_kwargs["max_recv_token_num"] == op_kwargs["max_recv_token_num"]
    assert group_kwargs["comm_alg"] == op_kwargs["comm_alg"] == "replicated_dispatch"
    assert group_kwargs["dispatch_quant_mode"] == op_kwargs["dispatch_quant_mode"] == 0
    assert group_kwargs["dispatch_quant_out_dtype"] is op_kwargs["dispatch_quant_out_dtype"] is None


@pytest.mark.parametrize(
    "invalid", [None, "tp", "dp", "sp", "quant", "dtype", "hidden", "capacity", "lora", "eplb", "disabled"]
)
def test_config_rejects_incompatible_replication_contract(invalid, monkeypatch):
    from vllm_ascend import ascend_config

    cfg = SimpleNamespace(
        mega_moe_replicated_dispatch=True,
        enable_fused_mc2=1,
        enable_sp_by_pass=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    pc = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    hf = SimpleNamespace(hidden_size=2048, moe_intermediate_size=512, num_experts_per_tok=8)
    mc = SimpleNamespace(dtype=torch.bfloat16, hf_text_config=hf, get_num_experts=lambda: 256)
    vc = SimpleNamespace(
        parallel_config=pc,
        model_config=mc,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        quant_config=None,
        lora_config=None,
        use_v2_model_runner=False,
    )
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A2)
    monkeypatch.setattr(ascend_config, "is_mega_moe_supported", lambda: True)
    if invalid == "tp":
        pc.tensor_parallel_size = 4
    elif invalid == "dp":
        pc.data_parallel_size = 2
    elif invalid == "sp":
        cfg.enable_sp_by_pass = True
    elif invalid == "quant":
        vc.quant_config = object()
    elif invalid == "dtype":
        mc.dtype = torch.float16
    elif invalid == "hidden":
        hf.hidden_size = 4096
    elif invalid == "capacity":
        vc.scheduler_config.max_num_batched_tokens = 4097
    elif invalid == "lora":
        vc.lora_config = object()
    elif invalid == "eplb":
        cfg.eplb_config.dynamic_eplb = True
    elif invalid == "disabled":
        cfg.enable_fused_mc2 = 0
    if invalid is None:
        ascend_config.AscendConfig._validate_megamoe_replicated_dispatch(cfg, vc)
    else:
        with pytest.raises(ValueError, match="mega_moe_replicated_dispatch requires"):
            ascend_config.AscendConfig._validate_megamoe_replicated_dispatch(cfg, vc)


def test_only_fused_mc2_selects_replicated_prepare(monkeypatch):
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(mega_moe_replicated_dispatch=True))
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithReplicatedDispatch", lambda cfg: "replicated")
    monkeypatch.setattr(comm, "PrepareAndFinalizeWithMC2", lambda cfg: "ordinary_mc2")
    fused = object.__new__(comm.FusedMC2CommImpl)
    ordinary = object.__new__(comm.MC2CommImpl)
    fused.moe_config = ordinary.moe_config = None
    assert fused._get_prepare_finalize() == "replicated"
    assert ordinary._get_prepare_finalize() == "ordinary_mc2"
