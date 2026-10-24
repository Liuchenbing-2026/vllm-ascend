# SPDX-License-Identifier: Apache-2.0
"""Compose config resolution with backend selection, including safe fallback."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_config, utils
from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.device import hardware_profile


def make_config(ep, experts=256, quantize=None, local_partial=False, replicated=False):
    config = SimpleNamespace(
        enable_fused_mc2=1,
        mega_moe_local_partial=local_partial,
        mega_moe_replicated_dispatch=replicated,
        mega_moe_min_tokens=512,
        enable_sp_by_pass=False,
        multistream_overlap_shared_expert=True,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=ep,
        world_size_across_dp=ep,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        expert_placement_strategy="linear",
    )
    hf = SimpleNamespace(hidden_size=2048, moe_intermediate_size=512, num_experts_per_tok=8, quantize=quantize)
    model = SimpleNamespace(dtype=torch.bfloat16, hf_text_config=hf, get_num_experts=lambda: experts, quantization=None)
    vllm_config = SimpleNamespace(
        parallel_config=parallel,
        model_config=model,
        quant_config=None,
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        use_v2_model_runner=False,
    )
    return config, vllm_config


def patch_environment(monkeypatch, config, ep):
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A2)
    monkeypatch.setattr(ascend_config, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: utils.AscendDeviceType.A2)
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 4096)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep))
    monkeypatch.setattr(afc, "get_ascend_config", lambda: config)


def resolve(config, vllm_config):
    # Match the startup order: correctness contracts precede policy fallback,
    # which must precede shared-stream decisions and later weight preparation.
    cls = ascend_config.AscendConfig
    cls._validate_megamoe_replicated_dispatch(config, vllm_config)
    cls._validate_megamoe_local_partial(config, vllm_config)
    cls._resolve_a2_bf16_megamoe_shape(config, vllm_config)
    cls._resolve_fused_mc2_shared_overlap(config)


@pytest.mark.parametrize(
    "ep,local_partial,replicated,experts",
    [
        (2, False, True, 256),
        (4, False, False, 256),
        (8, False, False, 256),
        (16, False, False, 512),
    ],
)
def test_protected_shape_resolves_to_actual_allgather(monkeypatch, ep, local_partial, replicated, experts):
    config, vc = make_config(ep, experts, local_partial=local_partial, replicated=replicated)
    patch_environment(monkeypatch, config, ep)
    resolve(config, vc)
    assert config.enable_fused_mc2 == 0
    assert not config.mega_moe_local_partial
    assert not config.mega_moe_replicated_dispatch
    assert config.multistream_overlap_shared_expert
    for tokens in (32, 511, 512, 4096, 4097):
        assert afc.select_moe_comm_method(tokens, vc) is afc.MoECommType.ALLGATHER
    # Effective config can pass the same resolution again without depending
    # on the now-cleared original request flags.
    resolve(config, vc)
    assert config.enable_fused_mc2 == 0


def test_validated_ep4_local_partial_keeps_fused_backend(monkeypatch):
    config, vc = make_config(4, local_partial=True)
    patch_environment(monkeypatch, config, 4)
    resolve(config, vc)
    assert config.enable_fused_mc2 == 1
    assert config.mega_moe_local_partial
    assert config.multistream_overlap_shared_expert
    assert afc.select_moe_comm_method(512, vc) is afc.MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(4096, vc) is afc.MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(4097, vc) is afc.MoECommType.ALLGATHER
    assert afc.select_moe_comm_method(32, vc) is afc.MoECommType.ALLGATHER


@pytest.mark.parametrize("ep", [16, 32])
def test_eligible_shape_keeps_existing_dispatch_and_capacity_policy(monkeypatch, ep):
    config, vc = make_config(ep)
    patch_environment(monkeypatch, config, ep)
    resolve(config, vc)
    assert config.enable_fused_mc2 == 1
    assert afc.select_moe_comm_method(512, vc) is afc.MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(4096, vc) is afc.MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(4097, vc) is afc.MoECommType.ALLGATHER
    assert afc.select_moe_comm_method(32, vc) is afc.MoECommType.MC2


@pytest.mark.parametrize("quantize", ["w8a8_dynamic", "w4a8_dynamic"])
def test_quantized_path_does_not_inherit_bf16_protection(monkeypatch, quantize):
    config, vc = make_config(8, quantize=quantize)
    patch_environment(monkeypatch, config, 8)
    resolve(config, vc)
    assert config.enable_fused_mc2 == 1
    assert afc.select_moe_comm_method(512, vc) is afc.MoECommType.FUSED_MC2


def test_ep16_does_not_make_local_partial_contract_valid(monkeypatch):
    config, vc = make_config(16, local_partial=True)
    patch_environment(monkeypatch, config, 16)
    with pytest.raises(ValueError, match="TP4/EP4"):
        resolve(config, vc)


def test_selector_defends_against_unresolved_protected_config(monkeypatch):
    config, vc = make_config(4, local_partial=False)
    patch_environment(monkeypatch, config, 4)
    # Manually constructed ordinary EP4 configs still cannot bypass the slow-shape guard.
    assert afc.select_moe_comm_method(4096, vc) is afc.MoECommType.ALLGATHER


@pytest.mark.parametrize(
    "field,bad_value",
    [("hidden_size", 4096), ("moe_intermediate_size", 1024), ("num_experts_per_tok", 4), ("tensor_parallel_size", 8)],
)
def test_selector_rejects_unvalidated_local_partial_contract(monkeypatch, field, bad_value):
    config, vc = make_config(4, local_partial=True)
    target = vc.parallel_config if field == "tensor_parallel_size" else vc.model_config.hf_text_config
    setattr(target, field, bad_value)
    patch_environment(monkeypatch, config, 4)
    assert afc.select_moe_comm_method(4096, vc) is afc.MoECommType.ALLGATHER


def test_other_hardware_is_not_normalized_by_a2_policy(monkeypatch):
    config, vc = make_config(8)
    patch_environment(monkeypatch, config, 8)
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: utils.AscendDeviceType.A3)
    ascend_config.AscendConfig._resolve_a2_bf16_megamoe_shape(config, vc)
    assert config.enable_fused_mc2 == 1


@pytest.mark.parametrize("ep,local_partial", [(4, True), (4, False), (8, False)])
def test_mode_two_resolves_during_full_startup_before_overlap_policy(monkeypatch, ep, local_partial):
    _, vc = make_config(ep, local_partial=local_partial)
    vc.model_config.enforce_eager = True
    vc.kv_transfer_config = None
    vc.compilation_config = SimpleNamespace(pass_config=SimpleNamespace(enable_sp=False))
    original_find_spec = ascend_config.importlib.util.find_spec
    monkeypatch.setattr(
        ascend_config.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace() if name == "cann_ops_transformer" else original_find_spec(name),
    )
    # Construction changes the process-wide capability flag. Restore it after
    # the test, just as the public config fixture does.
    monkeypatch.setattr(ascend_config, "_MEGA_MOE_SUPPORTED", False)
    config = ascend_config.AscendConfig(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        enable_fused_mc2=2,
        mega_moe_local_partial=local_partial,
        multistream_overlap_shared_expert=True,
    )
    assert config.enable_fused_mc2 == 1
    assert ascend_config.is_mega_moe_supported()
    patch_environment(monkeypatch, config, ep)
    monkeypatch.setattr(
        hardware_profile, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True)
    )
    config.derive_and_validate(vc)
    if local_partial:
        assert config.enable_fused_mc2 == 1
        assert config.mega_moe_local_partial
        assert config.multistream_overlap_shared_expert
        assert afc.select_moe_comm_method(4096, vc) is afc.MoECommType.FUSED_MC2
    else:
        assert config.enable_fused_mc2 == 0
        assert not config.mega_moe_local_partial
        assert config.multistream_overlap_shared_expert
        assert afc.select_moe_comm_method(4096, vc) is afc.MoECommType.ALLGATHER
