from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.patch.worker.patch_qwen3_dflash import (
    precompute_and_store_context_kv,
)
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@dataclass
class _FakeCAD:
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool
    num_actual_tokens: int
    num_reqs: int = 1

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class _FakeBuilder:
    def __init__(self, gid: int):
        self.gid = gid
        self.kv_cache_spec = SimpleNamespace(block_size=128)

    def build_for_drafting(self, common_attn_metadata, draft_index):
        return SimpleNamespace(
            gid=self.gid,
            causal=common_attn_metadata.causal,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            attn_mask=object(),
            attn_state=None,
        )


class _FakeGroup:
    def __init__(self, gid: int, layer_names: list[str]):
        self.kv_cache_group_id = gid
        self.layer_names = layer_names
        self._builder = _FakeBuilder(gid)

    def get_metadata_builder(self):
        return self._builder


def _make_mixed_proposer() -> tuple[AscendDflashProposer, list[str], str]:
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    swa_layers = [f"model.layers.{i}.self_attn.attn" for i in range(5)]
    full_layer = "model.layers.5.self_attn.attn"
    all_layers = [*swa_layers, full_layer]
    proposer.model = SimpleNamespace(
        sliding_attention_layer_names=set(swa_layers)
    )
    proposer._draft_attn_layer_names = set(all_layers)
    proposer._draft_kv_cache_group_ids = [1, 2]
    proposer._draft_layer_to_kv_cache_gid = {
        **{name: 1 for name in swa_layers},
        full_layer: 2,
    }
    proposer.kv_cache_gid = 1
    proposer.draft_attn_groups = [
        _FakeGroup(1, swa_layers),
        _FakeGroup(2, [full_layer]),
    ]
    proposer._slot_mapping_buffers_by_gid = {
        1: (torch.arange(8, dtype=torch.int32), torch.arange(9, dtype=torch.int32)),
        2: (
            torch.arange(8, dtype=torch.int32) + 100,
            torch.arange(9, dtype=torch.int32) + 100,
        ),
    }
    proposer._per_group_block_tables = {
        1: torch.tensor([[11, 12]], dtype=torch.int32),
        2: torch.tensor([[21, 22]], dtype=torch.int32),
    }
    proposer._per_group_input_slot_mappings = {}
    proposer.runner = None
    return proposer, swa_layers, full_layer


def test_mixed_dflash_builds_per_layer_causal_metadata_and_per_group_slots():
    proposer, swa_layers, full_layer = _make_mixed_proposer()
    cad = _FakeCAD(
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.arange(9, dtype=torch.int32),
        causal=False,
        num_actual_tokens=9,
    )

    per_group, per_layer = proposer.build_per_group_and_layer_attn_metadata(cad)

    assert len(per_group) == 2
    assert set(per_layer) == set([*swa_layers, full_layer])
    for layer_name in swa_layers:
        metadata = per_layer[layer_name]
        assert metadata.causal is True
        assert metadata.gid == 1
        assert metadata.block_table is proposer._per_group_block_tables[1]
        assert metadata.slot_mapping.data_ptr() == (
            proposer._slot_mapping_buffers_by_gid[1][1].data_ptr()
        )
    full_metadata = per_layer[full_layer]
    assert full_metadata.causal is False
    assert full_metadata.attn_mask is None
    assert full_metadata.gid == 2
    assert full_metadata.block_table is proposer._per_group_block_tables[2]
    assert full_metadata.slot_mapping.data_ptr() == (
        proposer._slot_mapping_buffers_by_gid[2][1].data_ptr()
    )


def test_context_precompute_routes_slot_mapping_by_layer_name():
    source = inspect.getsource(precompute_and_store_context_kv)
    assert "isinstance(context_slot_mapping, Mapping)" in source
    assert "context_slot_mapping[attn.layer_name]" in source
    assert "layer_slot_mapping" in source


def test_context_precompute_runs_inside_ascend_forward_context():
    source = inspect.getsource(AscendSpecDecodeBaseProposer._propose)
    context_offset = source.index("with set_ascend_forward_context(")
    precompute_offset = source.index("self.precompute_context_kv()")
    assert precompute_offset > context_offset


def test_runner_collects_every_dflash_kv_group_and_flattens_block_sizes():
    build_source = inspect.getsource(NPUModelRunner._build_attention_metadata)
    init_source = inspect.getsource(NPUModelRunner.initialize_kv_cache)
    assert "clear_per_group_attn_metadata" in build_source
    assert "self.drafter.set_per_group_attn_metadata" in build_source
    assert "kv_cache_gid == self.drafter.kv_cache_gid" in build_source
    assert "sizes[0] if isinstance(sizes, list)" in init_source


@pytest.mark.parametrize(
    ("max_seq_len", "expected"),
    [(8183, True), (8184, False)],
)
def test_ascend_runner_reserves_bonus_and_masks_near_max_model_len(
    max_seq_len: int,
    expected: bool,
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.num_spec_tokens = 8
    runner.effective_drafter_max_model_len = 8192
    runner.speculative_config = SimpleNamespace(use_dflash=lambda: True)

    assert runner._input_fits_in_drafter(
        SimpleNamespace(max_seq_len=max_seq_len)
    ) is expected


def test_ascend_sample_tokens_wires_runtime_bound_and_clears_stale_drafts():
    source = inspect.getsource(NPUModelRunner.sample_tokens)

    assert "self._input_fits_in_drafter(" in source
    assert "self._draft_token_ids = None" in source
    assert "if input_fits_in_drafter and use_padded_batch:" in source
    assert "zeros_only=True" in source


def test_dflash_proposer_query_window_guard_exact_boundary():
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    proposer.runner = None
    proposer.draft_model_config = SimpleNamespace(max_model_len=None)
    proposer.max_model_len = 8192

    proposer._raise_if_query_window_exceeds_max_model_len(8183, 9)
    with pytest.raises(RuntimeError, match="1 bonus \\+ 8 masks"):
        proposer._raise_if_query_window_exceeds_max_model_len(8184, 9)

    source = inspect.getsource(AscendDflashProposer.set_inputs_first_pass)
    assert source.index("_raise_if_query_window_exceeds_max_model_len") < (
        source.index("copy_and_expand_dflash_inputs_kernel_single_grid")
    )


def _make_platform_config(max_model_len: int):
    hf_config = SimpleNamespace(
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=4096,
    )
    config = MagicMock()
    config.speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=8,
        draft_model_config=SimpleNamespace(hf_config=hf_config),
    )
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.model_config.max_model_len = max_model_len
    config.cache_config.enable_prefix_caching = False
    return config


def test_platform_accepts_qwen36_mixed_swa_at_4k(
    monkeypatch: pytest.MonkeyPatch,
):
    warning = MagicMock()
    monkeypatch.setattr("vllm_ascend.platform.logger.warning", warning)

    NPUPlatform._validate_and_update_dflash_config(
        _make_platform_config(4096)
    )

    warning.assert_not_called()


@pytest.mark.parametrize("max_model_len", [4097, 8192, 9216, 10240])
def test_platform_warns_but_accepts_experimental_long_context(
    max_model_len: int,
    monkeypatch: pytest.MonkeyPatch,
):
    warning = MagicMock()
    monkeypatch.setattr("vllm_ascend.platform.logger.warning", warning)

    NPUPlatform._validate_and_update_dflash_config(
        _make_platform_config(max_model_len)
    )

    assert any(
        "experimental long-context mixed-SWA DFlash" in call.args[0]
        for call in warning.call_args_list
    )
