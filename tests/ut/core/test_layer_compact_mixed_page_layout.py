# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests for mixed page sizes in layer-compact KV cache layouts."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import vllm.v1.core.kv_cache_planning as kv_cache_planning
import vllm.v1.core.single_type_kv_cache_manager as single_type_manager
import vllm.v1.kv_cache_interface as kv_cache_interface
from vllm.v1.kv_cache_layout import KVCacheLayout

import vllm_ascend.core.kv_cache_config_builder as builder_mod
import vllm_ascend.hisparse.binding as binding_mod
import vllm_ascend.patch.platform.patch_kv_cache_layout as patch_kv_cache_layout
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec


@dataclass(frozen=True)
class _FakeSpec:
    page_size_bytes: int
    alignment: int = 1
    dcp_sharded: bool = False
    block_size: int = 16

    def get_block_stride_alignment(self) -> int:
        return self.alignment

    def is_uniform_with_collection(self, specs: dict[str, Any]) -> bool:
        return all(isinstance(spec, _FakeSpec) for spec in specs.values())


def _make_group(
    specs_by_layer: dict[str, Any],
    *,
    legacy: bool,
    layer_names: list[str] | None = None,
    manager_cls: Any = None,
) -> Any:
    projected = list(specs_by_layer) if layer_names is None else layer_names
    if legacy:
        unique = set(specs_by_layer.values())
        spec = (
            next(iter(unique))
            if len(unique) == 1
            else kv_cache_interface.UniformTypeKVCacheSpecs.from_specs(specs_by_layer)
        )
        return kv_cache_interface.KVCacheGroupSpec(layer_names=projected, kv_cache_spec=spec)
    manager_config = None if manager_cls is None else SimpleNamespace(manager_cls=manager_cls)
    return SimpleNamespace(
        layer_names=projected,
        kv_cache_specs={name: specs_by_layer[name] for name in projected},
        manager_config=manager_config,
    )


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "native"])
def test_mixed_page_layout_grouping_stride_and_validation(legacy: bool) -> None:
    mla, idx, swa = _FakeSpec(4096), _FakeSpec(1024), _FakeSpec(4096)
    g_inter = _make_group(
        {"l0.mla": mla, "l0.idx": idx, "l1.mla": mla, "l1.idx": idx},
        legacy=legacy,
    )
    assert patch_kv_cache_layout.group_layers_by_spec(g_inter) == {
        mla: ["l0.mla", "l1.mla"],
        idx: ["l0.idx", "l1.idx"],
    }
    s2 = _make_group({"s0": swa, "s1": swa}, legacy=legacy)
    s3 = _make_group({"s0": swa, "s1": swa, "s2": swa}, legacy=legacy)
    assert patch_kv_cache_layout.overlaid_layers_match([g_inter, s2])
    assert not patch_kv_cache_layout.overlaid_layers_match([g_inter, s3])
    assert not patch_kv_cache_layout.overlaid_layers_match([_make_group({"i": idx, "m": mla}, legacy=legacy), s2])

    aligned_spec = _FakeSpec(3072, alignment=4096)
    assert patch_kv_cache_layout.get_layer_block_stride(aligned_spec) == 4096
    assert patch_kv_cache_layout.overlaid_layers_match(
        [_make_group({"a": aligned_spec}, legacy=False), _make_group({"b": swa}, legacy=False)]
    )
    g_2k = _make_group({"c": _FakeSpec(2048)}, legacy=False)
    g_2k_aligned = _make_group({"d": _FakeSpec(2048, 4096)}, legacy=False)
    assert not patch_kv_cache_layout.overlaid_layers_match([g_2k, g_2k_aligned])

    patch_kv_cache_layout.validate_kv_cache_layout_for_mixed_pages(KVCacheLayout.LBHNC, [g_inter, s2])
    with pytest.raises(ValueError, match=r"LBNHC.*\[1024, 4096\]"):
        patch_kv_cache_layout.validate_kv_cache_layout_for_mixed_pages(KVCacheLayout.LBNHC, [g_inter, s3])
    patch_kv_cache_layout.validate_kv_cache_layout_for_mixed_pages(KVCacheLayout.BLHNC, [g_inter, s3])
    with pytest.raises(ValueError, match="LHBNC"):
        patch_kv_cache_layout.validate_kv_cache_layout_for_mixed_pages(KVCacheLayout.LHBNC, [g_inter])

    empty_proj = _make_group({"r0": _FakeSpec(2048), "r1": _FakeSpec(512)}, legacy=True, layer_names=[])
    assert patch_kv_cache_layout.group_layers_by_spec(empty_proj) == {}
    patch_kv_cache_layout.validate_kv_cache_layout_for_mixed_pages(
        KVCacheLayout.LHBNC,
        [_make_group({"l0": mla}, legacy=True), empty_proj],
    )


def test_apply_kv_cache_layout_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(kv_cache_planning, "validate_kv_cache_layout", sentinel)
    monkeypatch.setattr(kv_cache_planning, "_overlaid_layers_match", lambda _g: True, raising=False)
    assert patch_kv_cache_layout.apply_kv_cache_layout_patch() is False
    assert kv_cache_planning.validate_kv_cache_layout is sentinel

    monkeypatch.delattr(kv_cache_planning, "_overlaid_layers_match", raising=False)
    assert patch_kv_cache_layout.apply_kv_cache_layout_patch() is True
    assert kv_cache_planning.validate_kv_cache_layout is patch_kv_cache_layout.validate_kv_cache_layout_for_mixed_pages
    assert patch_kv_cache_layout.apply_kv_cache_layout_patch() is False


def test_native_group_builder_and_replicated_draft_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_next")),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=None,
        speculative_config=None,
    )
    mamba = kv_cache_interface.MambaSpec(block_size=16, shapes=((32, 16),), dtypes=(torch.bfloat16,))
    attn = kv_cache_interface.AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, dcp_sharded=False
    )
    mla = kv_cache_interface.MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, dcp_sharded=False
    )
    dcp = kv_cache_interface.AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, dcp_sharded=True
    )

    mamba_grp = _make_group({"l0": mamba}, legacy=False, manager_cls=object)
    for bad_cls in (single_type_manager.SlidingWindowManager, single_type_manager.ChunkedLocalAttentionManager, None):
        non_draft = _make_group({"x": attn}, legacy=False, manager_cls=bad_cls)
        assert builder_mod._get_replicated_draft_pools([mamba_grp, non_draft]) is None

    swa_grp = _make_group({"s": attn}, legacy=False, manager_cls=single_type_manager.SlidingWindowManager)
    mla_grp = _make_group({"m": mla}, legacy=False, manager_cls=single_type_manager.FullAttentionManager)
    dcp_grp = _make_group({"d": dcp}, legacy=False, manager_cls=single_type_manager.FullAttentionManager)
    draft_grp = type(
        "_Group",
        (),
        {
            "layer_names": ["dr"],
            "kv_cache_specs": {"dr": attn},
            "manager_cls": single_type_manager.FullAttentionManager,
        },
    )()
    assert builder_mod._get_replicated_draft_pools([mamba_grp, swa_grp, mla_grp, dcp_grp, draft_grp]) == [
        [mamba_grp, swa_grp, mla_grp, dcp_grp],
        [draft_grp],
    ]

    full_spec = kv_cache_interface.FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        dcp_sharded=False,
    )
    mamba_act = _make_group({"l": mamba}, legacy=True)
    mamba_empty = _make_group({"l": mamba}, legacy=True, layer_names=[])
    draft_act = _make_group({"d": full_spec}, legacy=True)
    draft_empty = _make_group({"d": full_spec}, legacy=True, layer_names=[])
    assert builder_mod._get_replicated_draft_pools([mamba_act, draft_empty]) is None
    assert builder_mod._get_replicated_draft_pools([mamba_empty, draft_act]) is None

    monkeypatch.setattr(
        kv_cache_planning.DefaultKVCacheConfigBuilder,
        "_get_pool_bytes_per_block",
        Mock(return_value=5120),
    )
    built_cfg = SimpleNamespace(kv_cache_groups=[mla_grp])
    monkeypatch.setattr(builder_mod, "_upstream_get_kv_cache_config_from_groups", Mock(return_value=built_cfg))
    monkeypatch.setattr(builder_mod, "plan_layerwise_cache_reuse", lambda *_a: None)
    builder = builder_mod.AscendKVCacheConfigBuilder()
    assert builder._get_pool_bytes_per_block(runtime, [mla_grp]) == 5120  # type: ignore[arg-type]
    built = builder_mod.get_ascend_kv_cache_config_from_groups(runtime, [mla_grp], 40960)  # type: ignore[arg-type]
    assert built is built_cfg and built.kv_transfer_config is None


def test_hisparse_binding_page_spec_and_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = AscendSFAIndexerCacheSpec(block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)

    @dataclass(frozen=True)
    class FakePageSpec:
        block_size: int
        page_size: int
        block_stride_alignment: int = 1

        def get_block_stride_alignment(self) -> int:
            return self.block_stride_alignment

    monkeypatch.setattr(kv_cache_interface, "HiSparsePageSpec", FakePageSpec, raising=False)
    raw = FakePageSpec(16, 4096)
    aligned = binding_mod.align_hisparse_indexer_specs(
        {"idx": idx, "page": raw},  # type: ignore[dict-item]
        packed_row_bytes=256,
    )
    assert aligned["idx"].get_block_stride_alignment() == idx.page_size_bytes
    assert aligned["page"].get_block_stride_alignment() == 256
    cfg = SimpleNamespace(
        kv_cache_groups=[_make_group({"idx": idx}, legacy=False), _make_group({"page": raw}, legacy=True)]
    )
    assert binding_mod.get_hisparse_indexer_specs(cfg) == {"idx": idx}  # type: ignore[arg-type]
