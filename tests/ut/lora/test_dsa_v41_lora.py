# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import LoRAConfig
from vllm.lora.layers import ReplicatedLinearWithLoRA
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
from vllm_ascend.lora.utils import AscendReplicatedLinearWithLoRA

PROJECTIONS = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")


@pytest.fixture(autouse=True)
def tp_size_one():
    """The CPU tests never build a process group; expose a 1-rank TP world."""
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        local_rank=0,
        is_first_rank=True,
        is_last_rank=True,
    )
    with patch("vllm.distributed.parallel_state._TP", group):
        yield


def _base_layer(input_size: int = 16, output_size: int = 8) -> ReplicatedLinear:
    return ReplicatedLinear(input_size, output_size, bias=False)


def _lora_config(max_loras: int = 1, rank: int = 8, fully_sharded: bool = False) -> LoRAConfig:
    return LoRAConfig(
        max_loras=max_loras,
        max_lora_rank=rank,
        lora_dtype=torch.float32,
        fully_sharded_loras=fully_sharded,
    )


def _can_replace(source_layer, lora_config: LoRAConfig) -> bool:
    # ``_not_fully_sharded_can_replace`` reads the flag from kwargs.
    return AscendReplicatedLinearWithLoRA.can_replace_layer(
        source_layer=source_layer,
        lora_config=lora_config,
        packed_modules_list=[],
        model_config=None,
    )


def test_replicated_layer_is_claimed_only_for_a_single_adapter() -> None:
    layer = _base_layer()
    assert _can_replace(layer, _lora_config())
    assert not _can_replace(layer, _lora_config(max_loras=2))
    assert not _can_replace(layer, _lora_config(fully_sharded=True))


def test_replicated_layer_does_not_claim_other_linear_kinds() -> None:
    # On Ascend ``ColumnParallelLinear`` is replaced by an out-of-tree subclass
    # whose ``__init__`` needs an initialised AscendConfig. ``can_replace_layer``
    # only inspects the type, so allocate the instance without running ``__init__``.
    column = ColumnParallelLinear.__new__(ColumnParallelLinear)
    assert not _can_replace(column, _lora_config())


def test_replicated_wrapper_keeps_b_packed_for_the_ascend_expand_kernel() -> None:
    base = _base_layer()
    wrapper = AscendReplicatedLinearWithLoRA(base)

    assert isinstance(wrapper, ReplicatedLinearWithLoRA)
    assert wrapper.n_slices == 1

    rank = 8
    wrapper.create_lora_weights(1, _lora_config(rank=rank))
    assert wrapper.output_slices == (8,)
    assert wrapper.lora_b_packed.shape == (1, 1, rank, 8)

    lora_a = torch.randn(rank, 16)
    lora_b = torch.randn(8, rank)
    wrapper.set_lora(0, lora_a, lora_b)

    # The expand kernel consumes a contiguous [rank, output] copy of B.
    torch.testing.assert_close(wrapper.lora_b_packed[0, 0], lora_b.T)
    torch.testing.assert_close(wrapper.lora_b_stacked[0][0, 0], lora_b)


def _owner(module_factory) -> SimpleNamespace:
    return SimpleNamespace(**{name: module_factory(name) for name in PROJECTIONS})


def _impl_bound_to(owner: SimpleNamespace) -> AscendDSAImpl:
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    for name in PROJECTIONS:
        setattr(impl, name, getattr(owner, name))
    impl.cv_wq_a = impl.cv_wkv = impl.cv_wq_b = "stale"
    impl.bind_lora_projections(owner)
    return impl


def test_bind_lora_projections_rebinds_captured_modules() -> None:
    impl = _impl_bound_to(_owner(lambda _: _base_layer()))
    assert impl.lora_projections == ()
    assert not impl.lora_enabled
    assert not impl.prolog_lora_enabled
    # No adapter was installed, so the cached CV wrappers must survive untouched.
    assert impl.cv_wq_a == "stale"

    wrapped = _owner(lambda _: AscendReplicatedLinearWithLoRA(_base_layer()))
    wrapped.wo_a.punica_wrapper = SimpleNamespace(_single_lora_mask=torch.ones(2, 1))
    with patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False):
        impl.bind_lora_projections(wrapped)

    # LoRAModelManager swaps the submodules after the impl is constructed.
    for name in PROJECTIONS:
        assert getattr(impl, name) is getattr(wrapped, name)
    assert impl.lora_projections == PROJECTIONS
    assert impl.lora_enabled
    assert impl.prolog_lora_enabled

    # The quantised-prolog caches follow the rebound modules.
    assert impl.cv_wq_a.linear is wrapped.wq_a
    assert impl.cv_wkv.linear is wrapped.wkv
    assert impl.cv_wq_b.linear is wrapped.wq_b

    # A second call with unchanged modules must not rebuild anything.
    cv_wq_a = impl.cv_wq_a
    impl.bind_lora_projections(wrapped)
    assert impl.cv_wq_a is cv_wq_a


def test_prolog_lora_predicate_ignores_output_only_adapters() -> None:
    owner = _owner(lambda _: _base_layer())
    owner.wo_b = AscendReplicatedLinearWithLoRA(_base_layer())
    impl = _impl_bound_to(owner)

    assert impl.lora_projections == ("wo_b",)
    assert impl.lora_enabled
    # An adapter on wo_b does not touch the q/kv prolog shortcuts.
    assert not impl.prolog_lora_enabled


def test_prolog_shortcuts_are_guarded_by_the_lora_predicate() -> None:
    """The W8A8 shortcuts call npu_quant_matmul with the base weights.

    They do not run the module forward, so they would silently drop the
    adapter. Both the shared-hs quant and the wq_b quant branch must be
    guarded; the rest of the path is unchanged, which is why the guard itself
    is the observable contract on a CPU-only host.
    """
    source = inspect.getsource(AscendDSAImpl._mla_prolog_single_stream)
    assert source.count("not self.prolog_lora_enabled") >= 2
    assert "share_hs_quant = not self.prolog_lora_enabled" in source


def test_multistream_overlap_is_disabled_when_the_prolog_has_adapters() -> None:
    """CVLinearWrapper.quantize()/matmul() bypass the wrapped module forward."""
    source = inspect.getsource(AscendDSAImpl._forward_attention)
    assert "multistream_overlap = self.multistream_dsv4_dsa_overlap and not self.prolog_lora_enabled" in source


def _wo_a_impl(tokens: int, groups: int, in_per_group: int, out_per_group: int, rank: int):
    lora_a = torch.randn(rank, in_per_group)
    lora_b = torch.randn(groups * out_per_group, rank)
    mask = torch.ones(tokens, 1)
    mask[1::2] = 0.0
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.n_local_groups = groups
    impl.wo_a = SimpleNamespace(
        lora_a_stacked=(lora_a[None, None],),
        lora_b_stacked=(lora_b[None, None],),
        punica_wrapper=SimpleNamespace(_single_lora_mask=mask),
    )
    grouped_input = torch.randn(tokens, groups, in_per_group)
    wo_a_output = torch.randn(tokens, groups, out_per_group)
    return impl, grouped_input, wo_a_output, lora_a, lora_b, mask


def test_wo_a_lora_delta_matches_the_grouped_reference() -> None:
    tokens, groups, in_per_group, out_per_group, rank = 4, 3, 5, 2, 8
    impl, grouped_input, wo_a_output, lora_a, lora_b, mask = _wo_a_impl(
        tokens, groups, in_per_group, out_per_group, rank
    )

    result = impl._add_wo_a_lora(wo_a_output, grouped_input)
    expected = wo_a_output.clone()
    for group in range(groups):
        group_b = lora_b[group * out_per_group : (group + 1) * out_per_group].T
        delta = (grouped_input[:, group] @ lora_a.T) @ group_b
        expected[:, group] += delta * mask[:, 0:1]

    assert result.shape == wo_a_output.shape
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    # Tokens without an adapter keep the plain projection.
    torch.testing.assert_close(result[1], wo_a_output[1])


def test_wo_a_lora_rejects_fine_grained_o_proj_tp() -> None:
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.wo_a = SimpleNamespace(punica_wrapper=None)
    with (
        patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=True),
        pytest.raises(ValueError, match="o_proj tensor parallelism"),
    ):
        impl._validate_wo_a_lora()


def test_wo_a_lora_rejects_pcp_o_proj_sharding() -> None:
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.wo_a = SimpleNamespace(punica_wrapper=None)
    impl._parallel_o_proj = SimpleNamespace(decode_sharded=True)
    with (
        patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False),
        pytest.raises(ValueError, match="context parallelism"),
    ):
        impl._validate_wo_a_lora()


def test_wo_a_lora_rejects_multi_adapter_punica() -> None:
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.wo_a = SimpleNamespace(punica_wrapper=SimpleNamespace(_single_lora_mask=None))
    with (
        patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False),
        pytest.raises(ValueError, match="single-adapter Punica"),
    ):
        impl._validate_wo_a_lora()
