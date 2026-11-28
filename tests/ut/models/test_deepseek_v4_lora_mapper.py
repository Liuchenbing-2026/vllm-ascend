# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for DeepSeek-V4 LoRA mapping, Compressor exclusion, and indexer wq_b."""

from types import SimpleNamespace as _NS
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from vllm.config import LoRAConfig
from vllm.lora import lora_model, model_manager
from vllm.lora import utils as vllm_lora_utils
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models import SupportsLoRA, interfaces

from vllm_ascend.lora import punica_npu
from vllm_ascend.lora import utils as asc_lora_utils
from vllm_ascend.models.deepseek_v4 import compressor as dsv4_comp
from vllm_ascend.models.deepseek_v4 import indexer as dsv4_idx
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM as _C
from vllm_ascend.models.deepseek_v4.vl_model import AscendDeepseekV4ForConditionalGeneration as _V
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod

_B, _M, _L, _P = "base_model.model.", "model.layers.", "language_model.", ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
_X, _Y = "ffn.experts.12.", "mlp.experts.12."
_E = (("w1", "gate_proj", "A"), ("w2", "down_proj", "B"), ("w3", "up_proj", "A"))
_CASES = [
    (_C, f"{_B}{_M}7.attn.wq_a.lora_A.weight", f"{_M}7.self_attn.wq_a", True),
    *[(_C, f"{_B}{_M}3.attn.{p}.lora_B.weight", f"{_M}3.self_attn.{p}", False) for p in _P],
    (_C, f"{_M}1.self_attn.wkv.lora_A.weight", f"{_M}1.self_attn.wkv", True),
    *[(_C, f"{_B}{_M}5.{_X}{k}.lora_{ab}.weight", f"{_M}5.{_Y}{v}", ab == "A") for k, v, ab in _E],
    (_C, f"{_B}{_M}5.ffn.shared_experts.w1.lora_A.weight", f"{_M}5.mlp.shared_experts.gate_proj", True),
    (_C, f"{_B}layers.5.attn.wq_a.lora_A.weight", f"{_M}5.self_attn.wq_a", True),
    (_C, "layers.5.ffn.experts.3.w1.lora_B.weight", f"{_M}5.mlp.experts.3.gate_proj", False),
    (_C, "embed_tokens.lora_A.weight", "model.embed_tokens", True),
    (_C, "model.embed_tokens.lora_B.weight", "model.embed_tokens", False),
    (_V, f"{_B}{_M}7.attn.wq_a.lora_A.weight", f"{_L}{_M}7.self_attn.wq_a", True),
    (_V, f"{_B}layers.3.attn.wo_b.lora_B.weight", f"{_L}{_M}3.self_attn.wo_b", False),
    *[(_V, f"{_B}{_M}5.{_X}{k}.lora_{ab}.weight", f"{_L}{_M}5.{_Y}{v}", ab == "A") for k, v, ab in _E[:2]],
    (_V, f"{_B}{_M}5.ffn.shared_experts.w3.lora_B.weight", f"{_L}{_M}5.mlp.shared_experts.up_proj", False),
    (_V, f"{_B}lm_head.lora_A.weight", f"{_L}lm_head", True),
    (_V, f"{_L}{_M}1.self_attn.wkv.lora_A.weight", f"{_L}{_M}1.self_attn.wkv", True),
]


@pytest.mark.parametrize(("cls", "raw_key", "expected_module", "expected_is_a"), _CASES)
def test_lora_weights_mapper(cls: type[_C] | type[_V], raw_key: str, expected_module: str, expected_is_a: bool) -> None:
    mapper = cls.build_lora_weights_mapper().get_rename_mapper()
    assert vllm_lora_utils.parse_fine_tuned_lora_name(raw_key, mapper) == (expected_module, expected_is_a)


def test_vl_wrapper_exposes_lora_contract_with_independent_mapping() -> None:
    assert interfaces.supports_lora(_V) and _V.packed_modules_mapping == {"gate_up_proj": ["gate_proj", "up_proj"]}
    assert _V.packed_modules_mapping is not _C.packed_modules_mapping
    assert all(getattr(_V, a) == getattr(_C, a) for a in ("embedding_modules", "is_3d_moe_weight", "is_non_gated_moe"))
    assert {"vision.", "aligner."}.issubset(_V.lora_skip_prefixes) and not hasattr(_V, "hf_to_vllm_mapper")


def _mod(**children: Any) -> torch.nn.Module:
    m = torch.nn.Module()
    for k, v in children.items():
        setattr(m, k, v)
    return m


def _cpu_linear(in_dim: int, out_dim: int, *, prefix: str = "", **_: object) -> torch.nn.Module:
    layer = cast(Any, object.__new__(maybe_get_oot_by_class(ReplicatedLinear)))
    torch.nn.Module.__init__(layer)
    layer.input_size, layer.output_size, layer.output_partition_sizes = in_dim, out_dim, [out_dim]
    layer.input_size_per_partition, layer.tp_size, layer.tp_rank, layer.prefix = in_dim, 1, 0, prefix
    layer.skip_bias_add, layer.return_bias, layer.bias, layer.quant_method = False, False, None, None
    layer.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim, dtype=torch.bfloat16))
    return cast(torch.nn.Module, layer)


def _build_compressor_tree(is_vl: bool) -> tuple[torch.nn.Module, str]:
    root, comp_cls = ("language_model.model" if is_vl else "model"), dsv4_comp.Compressor
    cfg = cast(Any, _NS(hidden_size=8, qk_rope_head_dim=2, rms_norm_eps=1e-6))
    cc = cast(Any, _NS(block_size=16))
    with (
        patch.object(dsv4_comp, "ReplicatedLinear", side_effect=_cpu_linear),
        patch.object(dsv4_comp, "RMSNorm", side_effect=lambda *_a, **_kw: torch.nn.Identity()),
        patch.object(dsv4_comp, "AscendCompressorStateCache", side_effect=lambda *_a, **_kw: torch.nn.Module()),
        patch.object(dsv4_comp, "get_current_hardware_profile", return_value=_NS(supports=lambda _c: False)),
        patch.dict("vllm_ascend.models.layer.attention.layer.DSV4_BLOCK_SIZES", {16: [(0, 0, 16, 16), (0, 0)]}),
    ):
        c1, c2 = [comp_cls(cast(Any, _NS()), cfg, 4, 4, cache_config=cc, prefix=f"{root}.{s}") for s in ("c", "ic")]
    wkv = _cpu_linear(8, 8, prefix=f"{root}.layers.0.self_attn.wkv")
    attn = _mod(wkv=wkv, compressor=c1, indexer=_mod(compressor=c2))
    inner = _mod(layers=torch.nn.ModuleList([_mod(self_attn=attn)]))
    if is_vl:
        vl_model = _V.__new__(_V)
        torch.nn.Module.__init__(vl_model)
        vl_any = cast(Any, vl_model)
        vl_any.config, vl_any.hf_to_vllm_mapper = None, _V.build_lora_weights_mapper()
        vl_any.vision, vl_any.aligner = _mod(proj=torch.nn.Linear(8, 8)), _mod(w1=torch.nn.Linear(8, 8))
        vl_any.language_model = _mod(model=inner)
        return vl_model, root
    model = _mod(model=inner)
    attrs: dict[str, Any] = {
        "config": None,
        "supports_lora": SupportsLoRA.supports_lora,
        "is_3d_moe_weight": False,
        "is_non_gated_moe": False,
        "embedding_modules": {},
        "packed_modules_mapping": {},
        "lora_skip_prefixes": [],
    }
    model.__dict__.update(attrs)
    return model, root


@pytest.mark.parametrize("is_vl", [False, True])
def test_compressor_projections_are_unwrapped_and_rejected_on_load(is_vl: bool) -> None:
    orig, (model, root) = vllm_lora_utils._all_lora_classes, _build_compressor_tree(is_vl)
    model_any = cast(Any, model)
    try:
        asc_lora_utils.refresh_all_lora_classes()
        layer0 = (model_any.language_model.model if is_vl else model_any.model).layers[0]
        subs = (layer0.self_attn.compressor, layer0.self_attn.indexer.compressor)
        reasons = [getattr(sub, w)._ascend_lora_unsupported_reason for sub in subs for w in ("wkv", "wgate")]
        assert all("Compressor" in r for r in reasons)
        cfg = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16, max_cpu_loras=2)
        v_cfg = cast(Any, _NS(lora_config=cfg, parallel_config=_NS(enable_expert_parallel=False), model_config=None))
        punica = object.__new__(punica_npu.PunicaWrapperNPU)
        PunicaWrapperBase.__init__(punica, 16, 2, "cpu")
        with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica):
            mgr = model_manager.LRUCacheLoRAModelManager(model, 2, 16, 32, cfg, torch.device("cpu"), vllm_config=v_cfg)
        wkv_key = f"{root}.layers.0.self_attn.wkv"
        comp_keys = ("compressor.wkv", "compressor.wgate", "indexer.compressor.wkv", "indexer.compressor.wgate")
        assert isinstance(mgr.modules[wkv_key], asc_lora_utils.AscendReplicatedLinearWithLoRA)
        assert all(f"{root}.layers.0.self_attn.{s}" not in mgr.modules for s in comp_keys)
        if is_vl:
            skip, pfx = lora_model.LoRAModel._should_skip_module, model_any.lora_skip_prefixes
            assert "vision.proj" not in mgr.modules and "aligner.w1" not in mgr.modules
            assert skip("vision.proj.lora_A.weight", pfx) and skip("aligner.w1.lora_A.weight", pfx)
            assert not skip(f"{_M}0.attn.wkv.lora_A.weight", pfx)
        else:
            dummy = mgr.create_dummy_lora(lora_id=1, rank=2, embedding_modules=model_any.embedding_modules)
            assert set(dummy.loras) == {wkv_key} and mgr.add_adapter(dummy) and mgr.activate_adapter(dummy.id)
        wm = _NS(_adapter_manager=mgr, _load_adapter=lambda req: req.loaded_model)
        asc_lora_utils.install_dp_moe_lora_load_guard(wm)
        reg = {1: wm._load_adapter(_NS(loaded_model=_NS(id=1, loras={wkv_key: object()})))}
        mapper = model_any.hf_to_vllm_mapper.get_rename_mapper() if is_vl else None
        fn = vllm_lora_utils.parse_fine_tuned_lora_name
        for idx, s in enumerate(comp_keys, start=2):
            bad = fn(f"{_B}{_M}0.attn.{s}.lora_A.weight", mapper)[0] if mapper else f"{root}.layers.0.self_attn.{s}"
            with pytest.raises(NotImplementedError, match="Compressor"):
                reg[idx] = wm._load_adapter(_NS(loaded_model=_NS(id=idx, loras={wkv_key: object(), bad: object()})))
        assert set(reg) == {1}
    finally:
        vllm_lora_utils._all_lora_classes = orig


def test_dsv4_indexer_wq_b_lora_bypasses_multistream_and_w8a8_direct_matmul() -> None:
    class _StubLoRA(BaseLayerWithLoRA):
        calls: int = 0

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return x + 2.0

    bf = torch.bfloat16
    seen: list[str] = []
    overlap: list[str] = []
    t0, t1 = torch.zeros((2, 1), dtype=torch.int32), torch.ones((2, 1), dtype=torch.int32)
    ckv0 = (torch.zeros((0, 4), dtype=bf), t0[:0, 0])
    wq_b = object.__new__(_StubLoRA)
    indexer = object.__new__(dsv4_idx.DeepseekV4Indexer)
    torch.nn.Module.__init__(wq_b)
    torch.nn.Module.__init__(indexer)
    cast(Any, wq_b).quant_method = _NS(quant_method=object.__new__(AscendW8A8DynamicLinearMethod))
    idx_any = cast(Any, indexer)
    idx_any.wq_b, idx_any.n_heads, idx_any.head_dim, idx_any.rope_head_dim = wq_b, 2, 4, 2
    idx_any.softmax_scale, idx_any.skip_topk = 0.5, False
    idx_any.use_index_cache, idx_any.topk_indices_buffer = False, None
    idx_any.cv_wq_b = _NS(matmul=lambda *_: (_ for _ in ()).throw(AssertionError("no cv_wq_b")))
    idx_any.weights_proj = lambda hs: torch.ones((hs.shape[0], 2), dtype=hs.dtype)
    idx_any.compressor = type("_C", (), {"rotate": False, "__call__": lambda *_, **__: ckv0})()

    def _select_topk(*_: Any) -> torch.Tensor:
        seen.append("select_topk")
        return t0

    def _quant_select(*_: Any) -> torch.Tensor:
        seen.append("quant_select")
        return t1

    idx_any.ops = _NS(
        unpack_dsa_indexer_kv_cache=lambda _kv: ("k", "s", "sc", "f"),
        quantize_query=lambda q: (q, "qs"),
        select_topk=_select_topk,
        quantize_update_cache_and_select_topk=_quant_select,
    )
    cos, sin = torch.ones((2, 2), dtype=bf), torch.zeros((2, 2), dtype=bf)
    cache = _NS(req_metadata=_NS(cos={"layer0": cos}, sin={"layer0": sin}), hadamard=torch.eye(4, dtype=bf))
    meta = dsv4_idx.AscendIndexerMetadata(compressor=cast(Any, _NS(cache=cache)))
    hs, qr, scale = torch.ones((2, 8), dtype=bf), torch.ones((2, 8), dtype=bf), torch.ones((2, 1))

    def _compute_ckv() -> tuple[str, str]:
        overlap.append("compute")
        return ("ckv", "slot")

    def _scatter_ckv(ckv: str, slot: str) -> None:
        overlap.append(f"scatter:{ckv}:{slot}")

    plan = dsv4_idx.IndexerOverlapPlan(
        cast(Any, _compute_ckv),
        cast(Any, _scatter_ckv),
        cast(Any, object()),
    )
    with (
        patch.object(torch.ops, "_C_ascend", _NS(inplace_partial_rotary_mul=lambda *_, **__: None), create=True),
        patch.object(dsv4_idx.torch_npu, "npu_quant_matmul", side_effect=AssertionError("no w8a8")),
        patch.object(dsv4_idx, "rotate_activation", side_effect=lambda q, *_: q),
        patch.object(indexer, "_cv_compute_query_and_update_cache_multistream", side_effect=AssertionError("no ms")),
    ):
        q, *_ = indexer._indexer_qkv_prepare(hs, qr, (), meta, cos, sin, qr_pertoken_scale=scale, write_cache=False)
        out_ro = indexer.forward("layer0", hs, qr, (), meta, plan, qr_pertoken_scale=scale, write_cache=False)
        out_rw = indexer.forward("layer0", hs, qr, (), meta, plan, qr_pertoken_scale=None, write_cache=True)
        assert wq_b.calls == 3 and torch.all(q == 3.0) and seen == ["select_topk", "quant_select"]
        assert overlap == ["compute", "scatter:ckv:slot"] and torch.equal(out_ro, t0) and torch.equal(out_rw, t1)
