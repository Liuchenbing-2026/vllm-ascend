# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CompilationMode, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.model_loader.utils import configure_quant_config

from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm_ascend.models.deepseek_v4 import dspark as dspark_v4_mod
from vllm_ascend.models.deepseek_v4 import mtp as mtp_v4_mod
from vllm_ascend.models.deepseek_v4 import vl_model as vl_v4_mod
from vllm_ascend.models.deepseek_v41 import dspark as dspark_v41_mod
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.configs.fp8_config import AscendDeepseekV4FP8Config, AscendFp8Config
from vllm_ascend.quantization.configs.modelopt_mxfp8_config import AscendModelOptMxFp8Config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod

ALL_QUANTS = ("fp8", "deepseek_v4_fp8", "modelopt_mxfp8", "compressed_tensors")


def _build_quant_config(kind: str, exclusions: list[str], *, whole_layer: bool = False):
    if kind in ("fp8", "deepseek_v4_fp8"):
        cfg = AscendFp8Config() if kind == "fp8" else AscendDeepseekV4FP8Config()
        if kind == "fp8":
            cfg.is_checkpoint_fp8_serialized, cfg.activation_scheme = True, "dynamic"
        cfg.weight_block_size, cfg.ignored_layers = (128, 128), list(exclusions)
        if whole_layer:
            cfg.ignored_layers_match_mode = "substring"
        return cfg
    if kind == "modelopt_mxfp8":
        raw = {"quant_method": "mxfp8", "ignored_layers": list(exclusions), "weight_block_size": [1, 32]}
        return AscendModelOptMxFp8Config.from_config(raw)
    raw = deepcopy(COMPRESSED_TENSORS_W8A8_CONFIG)
    raw["ignore"] = list(exclusions)
    return AscendCompressedTensorsConfig.from_config(raw)


def _make_vllm_config(quant_config, *, mtp_layers: int = 2):
    ns = SimpleNamespace
    hf = ns(model_type="deepseek_v4", hc_mult=2, hc_eps=1e-6, hidden_size=128, head_dim=128)
    hf.o_lora_rank, hf.vocab_size, hf.draft_vocab_size, hf.dspark_block_size = 32, 128, None, 4
    hf.dspark_target_layer_ids, hf.dspark_markov_rank, hf.num_hidden_layers = [59, 60], 4, 61
    hf.n_mtp_layers, hf.num_nextn_predict_layers, hf.rms_norm_eps = mtp_layers, mtp_layers, 1e-6
    hf.n_routed_experts, hf.n_group, hf.quantization_config = 2, 1, {"quant_method": "fp8"}
    hf.hf_text_config = hf
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        cfg = VllmConfig()
    cfg.compilation_config.mode, cfg.quant_config = CompilationMode.NONE, quant_config
    cfg.speculative_config = ns(draft_model_config=ns(hf_config=hf, hf_text_config=hf))  # type: ignore[assignment]
    mc = {"model": "dummy", "dtype": torch.bfloat16, "head_dtype": torch.bfloat16, "hf_config": hf}
    cfg.model_config = ns(**mc, hf_text_config=hf, architectures=[])  # type: ignore[assignment]
    return cfg


def _make_real_linear_decoder_layer(base_cls):
    def _init(self, vllm_config, prefix: str = "", config=None, **kwargs):
        torch.nn.Module.__init__(self)
        del config, kwargs
        self.use_sequence_parallel_moe, qcfg = False, vllm_config.quant_config
        self.mlp, self.self_attn = torch.nn.Module(), torch.nn.Module()
        sh = self.mlp.shared_experts = torch.nn.Module()
        sp, wp = f"{prefix}.mlp.shared_experts.gate_up_proj", f"{prefix}.self_attn.wkv"
        sh.gate_up_proj = MergedColumnParallelLinear(128, [128, 128], bias=False, quant_config=qcfg, prefix=sp)
        self.self_attn.wkv = ColumnParallelLinear(128, 128, bias=False, quant_config=qcfg, prefix=wp)
        self.self_attn.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=None))

    return type(f"_CPU{base_cls.__name__}", (base_cls,), {"__init__": _init})


@contextmanager
def _cpu_real_linear_env(vllm_config):
    tp, _mk = SimpleNamespace(rank_in_group=0, world_size=1, device_group=None), _make_real_linear_decoder_layer
    emb, lin, ds = "vllm_ascend.ops.vocab_parallel_embedding", "vllm_ascend.ops.linear_op", "vllm.model_executor"
    qm, d41 = "vllm_ascend.quantization.methods.w8a8", dspark_v41_mod.__name__
    with ExitStack() as st:
        st.callback(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)
        st.enter_context(torch.device("cpu"))
        st.enter_context(set_current_vllm_config(vllm_config))
        for mod, nm in ((dspark_v4_mod, "DeepseekV4DecoderLayer"), (mtp_v4_mod, "DeepseekV4DecoderLayer")):
            st.enter_context(patch.object(mod, nm, _mk(getattr(mod, nm))))
        d41_cls = dspark_v41_mod.DeepseekV41DSparkDecoderLayer
        st.enter_context(patch.object(dspark_v41_mod, "DeepseekV41DSparkDecoderLayer", _mk(d41_cls)))
        st.enter_context(patch(f"{d41}.DeviceOperator.get_dsv41_packed_cache_ops", return_value=object()))
        for t in (f"{qm}.fp8_block._require_fp8_weight_matmul", f"{qm}.w8a8_mxfp8.require_mx_weight_format"):
            st.enter_context(patch(t, return_value=None))
        for m in (lin, emb, "vllm.distributed.parallel_state"):
            st.enter_context(patch(f"{m}.get_tp_group", return_value=tp))
        st.enter_context(patch(f"{emb}.get_lmhead_tp_group", return_value=tp))
        for f in ("embedding_tp_enable", "lmhead_tp_enable", "enable_pcp_embedding_lmhead_weight_sharding"):
            st.enter_context(patch(f"{emb}.{f}", return_value=False))
        for f in ("mlp_tp_enable", "oproj_tp_enable", "shared_expert_dp_enabled", "shared_expert_tp_enable"):
            st.enter_context(patch(f"{lin}.{f}", return_value=False))
        for p in ("layers.linear", "parameter", "layers.vocab_parallel_embedding"):
            st.enter_context(patch(f"{ds}.{p}.get_tensor_model_parallel_world_size", return_value=1))
            st.enter_context(patch(f"{ds}.{p}.get_tensor_model_parallel_rank", return_value=0))
        yield


def _assert_quant(linear, *, skipped: bool) -> None:
    assert isinstance(linear.quant_method, UnquantizedLinearMethod if skipped else AscendLinearMethod)
    assert linear.weight.dtype in ((torch.bfloat16,) if skipped else (torch.float8_e4m3fn, torch.int8))


def _excl(p: str, sc: str, extra: list[str], *, wkv: str = "mtp.0.attn.wkv", main_e: str = "") -> list[str]:
    m = {"wkv": [wkv], "whole_layer": [p], "h_proj": ["mtp.0.h_proj"], "main_proj": [main_e]}
    packed = [f"{p}.ffn.shared_experts.w1", f"{p}.ffn.shared_experts.w3"]
    return [*(packed if sc == "packed_pair" else m.get(sc, [f"{p}.{sc}"])), *extra]


def test_packed_modules_mapping_contract_and_partial_shard_error():
    exp = {"gate_up_proj": ["gate_proj", "up_proj"]}
    v4_cls, v41_cls = dspark_v4_mod.DSparkDeepseekV4ForCausalLM, dspark_v41_mod.DSparkDeepseekV41ForCausalLM
    vl_cls, mtp_cls = vl_v4_mod.AscendDeepseekV4ForConditionalGeneration, mtp_v4_mod.DeepSeekV4MTP
    for cls in (v4_cls, v41_cls, vl_cls, mtp_cls):
        assert cls.packed_modules_mapping == exp
    v4_cls.packed_modules_mapping["gate_up_proj"].append("extra")
    try:
        assert vl_cls.packed_modules_mapping == exp and mtp_cls.packed_modules_mapping == exp
    finally:
        v4_cls.packed_modules_mapping["gate_up_proj"].pop()
    partial_cfg = _build_quant_config("fp8", ["model.layers.61.ffn.shared_experts.w1"])
    configure_quant_config(partial_cfg, v4_cls)
    with _cpu_real_linear_env(vc := _make_vllm_config(partial_cfg)), pytest.raises(ValueError, match="Detected some"):
        v4_cls(vllm_config=vc, prefix="")


@pytest.mark.parametrize("family", ["v4", "v41"])
@pytest.mark.parametrize("quant_kind", ALL_QUANTS)
@pytest.mark.parametrize("wrapper_prefix", ["model", "language_model.model"])
@pytest.mark.parametrize("scenario", ["packed_pair", "wkv", "whole_layer", "main_proj"])
def test_dspark_real_linears_and_vl_aliases(family, quant_kind, wrapper_prefix, scenario):
    p = f"{wrapper_prefix}.layers.61"
    main_e = "mtp.0.main_proj" if wrapper_prefix == "language_model.model" else f"{p}.main_proj"
    wkv_e = "re:.*wkv.*" if quant_kind == "compressed_tensors" and wrapper_prefix != "model" else "mtp.0.attn.wkv"
    ex = _excl(p, scenario, [], wkv=wkv_e, main_e=main_e)
    qcfg = _build_quant_config(quant_kind, ex, whole_layer=scenario == "whole_layer")
    mod = dspark_v4_mod if family == "v4" else dspark_v41_mod
    c_cls = mod.DSparkDeepseekV4ForCausalLM if family == "v4" else mod.DSparkDeepseekV41ForCausalLM
    m_cls = mod.DeepseekV4DSparkModel if family == "v4" else mod.DeepseekV41DSparkModel
    configure_quant_config(qcfg, c_cls)
    with _cpu_real_linear_env(vc := _make_vllm_config(qcfg)):
        wp = wrapper_prefix
        model = c_cls(vllm_config=vc, prefix="").model if wp == "model" else m_cls(vllm_config=vc, prefix=wp)
    assert wkv_e != "re:.*wkv.*" or scenario != "wkv" or qcfg.ignore == ["re:.*wkv.*"]
    l61, l62 = model.layers["61"], model.layers["62"]
    _assert_quant(l61.mlp.shared_experts.gate_up_proj, skipped=scenario in ("packed_pair", "whole_layer"))
    _assert_quant(l61.self_attn.wkv, skipped=scenario in ("wkv", "whole_layer"))
    _assert_quant(model.context_wkv_proj, skipped=scenario in ("wkv", "whole_layer"))
    _assert_quant(model.main_proj, skipped=scenario in ("whole_layer", "main_proj"))
    _assert_quant(l62.mlp.shared_experts.gate_up_proj, skipped=False)


@pytest.mark.parametrize("quant_kind", ["fp8", "modelopt_mxfp8", "compressed_tensors"])
@pytest.mark.parametrize("outer_prefix", [None, "", "language_model"])
@pytest.mark.parametrize("scenario", ["packed_pair", "wkv", "e_proj", "h_proj", "whole_layer"])
def test_deepseek_v4_mtp_prefix_protection(quant_kind, outer_prefix, scenario):
    wrapper = "language_model.model" if outer_prefix == "language_model" else "model"
    p, h62 = f"{wrapper}.layers.61", f"{wrapper}.layers.62.shared_head.head"
    is_mx = quant_kind == "modelopt_mxfp8"
    he = [h62] if is_mx and scenario == "whole_layer" else ([f"{p}.shared_head.head", h62] if is_mx else [])
    qcfg = _build_quant_config(quant_kind, _excl(p, scenario, he), whole_layer=scenario == "whole_layer")
    configure_quant_config(qcfg, mtp_v4_mod.DeepSeekV4MTP)
    with _cpu_real_linear_env(vc := _make_vllm_config(qcfg)):
        op, pred_cls = outer_prefix, mtp_v4_mod.DeepSeekMultiTokenPredictor
        mtp = pred_cls(vllm_config=vc) if op is None else mtp_v4_mod.DeepSeekV4MTP(vllm_config=vc, prefix=op).model
    s0, s1 = mtp.layers["0"], mtp.layers["1"]
    _assert_quant(s0.e_proj, skipped=scenario in ("e_proj", "whole_layer"))
    _assert_quant(s0.h_proj, skipped=scenario in ("h_proj", "whole_layer"))
    _assert_quant(s0.mtp_block.mlp.shared_experts.gate_up_proj, skipped=scenario in ("packed_pair", "whole_layer"))
    _assert_quant(s0.mtp_block.self_attn.wkv, skipped=scenario in ("wkv", "whole_layer"))
    _assert_quant(s1.e_proj, skipped=False)
    if quant_kind == "modelopt_mxfp8":
        assert isinstance(s0.shared_head.head.quant_method, UnquantizedEmbeddingMethod)


def test_deepseek_v4_mtp_modelslim_protection():
    keys = ("h_proj", "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "head")
    slim_map = dict.fromkeys([f"mtp.0.{k}.weight" for k in keys], "FLOAT")
    slim_map |= {"mtp.0.e_proj.weight": "W8A8_DYNAMIC", "mtp.0.self_attn.wkv.weight": "W8A8_DYNAMIC"}
    slim_cfg = AscendModelSlimConfig(slim_map)
    configure_quant_config(slim_cfg, mtp_v4_mod.DeepSeekV4MTP)
    with _cpu_real_linear_env(vc := _make_vllm_config(slim_cfg, mtp_layers=1)):
        s0 = mtp_v4_mod.DeepSeekV4MTP(vllm_config=vc, prefix="").model.layers["0"]
    _assert_quant(s0.e_proj, skipped=False)
    _assert_quant(s0.h_proj, skipped=True)
    _assert_quant(s0.mtp_block.mlp.shared_experts.gate_up_proj, skipped=True)
