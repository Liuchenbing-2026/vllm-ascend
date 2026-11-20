# mypy: disable-error-code="annotation-unchecked, arg-type, assignment, attr-defined, misc"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for prefill_context_kv_only support across Ascend models and MRV2."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.model_executor.warmup import kernel_warmup
from vllm_ascend.models import glm5_dspark
from vllm_ascend.models.deepseek_v4 import dspark as dsv4_dspark
from vllm_ascend.models.kimi_k3 import kimi_k3_dspark
from vllm_ascend.models.minimax_m3 import minimax_m3_dspark
from vllm_ascend.models.qwen4_exp import cpu_length_sync as qwen4_cpu_sync
from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.patch.worker import patch_qwen3_dflash
from vllm_ascend.worker.device_metadata import TargetDeviceMetadata
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

try:
    from vllm_ascend.models.deepseek_v41 import dspark as dsv41_dspark
except ModuleNotFoundError:
    dsv41_dspark = None

try:
    from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import (
        AscendLiLiCorrSpeculator,
    )
except ModuleNotFoundError:
    AscendLiLiCorrSpeculator = None


def _simulate_free_unused_weights(model: nn.Module, target_model: nn.Module | None = None) -> None:
    """Mirror upstream DFlashSpeculator.free_unused_weights parameter clearing."""
    keep = {id(p) for mod in model.context_kv_modules() for p in mod.parameters()}
    if target_model is not None:
        keep.update(id(p) for p in target_model.parameters())
    for param in model.parameters():
        if id(param) not in keep:
            param.data = param.data.new_empty(0)


def test_patched_dflash_qwen3_context_kv_modules_keeps_knorm_and_projs():
    class DummyAttnLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))
            self.kv_cache = None
            self.impl = SimpleNamespace(do_kv_cache_update=MagicMock())

    class DummySelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_norm = nn.LayerNorm(4)
            self.q_norm = nn.LayerNorm(4)
            self.qkv_proj = nn.Linear(4, 12, bias=False)
            self.o_proj = nn.Linear(4, 4, bias=False)
            self.attn = DummyAttnLayer()
            self.rotary_emb = lambda pos, k, v: (k, v)

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummySelfAttn()
            self.mlp = nn.Linear(4, 4, bias=False)

    model = nn.Module()
    model.hidden_norm = nn.LayerNorm(4)
    model.fc = nn.Linear(8, 4, bias=False)
    model.norm = nn.LayerNorm(4)
    model.layers = nn.ModuleList([DummyLayer()])
    model.use_aux_hidden_state = True
    model._fused_kv_weight = None
    model._fused_kv_bias = None
    model._attn_layers = [layer.self_attn.attn for layer in model.layers]
    model._context_qkv_projs = [layer.self_attn.qkv_proj for layer in model.layers]
    model._context_q_sizes = [4]
    model._num_attn_layers = 1
    model._kv_size = 4
    model._head_dim = 4
    model._num_kv_heads = 1

    model.context_kv_modules = lambda: patch_qwen3_dflash.context_kv_modules(model)
    model._project_context_kv = lambda *args: patch_qwen3_dflash._project_context_kv(model, *args)

    _simulate_free_unused_weights(model)
    assert model.layers[0].self_attn.k_norm.weight.numel() > 0
    assert model.layers[0].self_attn.qkv_proj.weight.numel() > 0
    assert model.fc.weight.numel() > 0
    assert model.layers[0].mlp.weight.numel() == 0
    assert model.layers[0].self_attn.o_proj.weight.numel() == 0
    assert model.norm.weight.numel() == 0

    states = torch.ones(2, 4)
    positions = torch.arange(2)
    patch_qwen3_dflash.precompute_and_store_context_kv(model, states, positions, positions)
    model.layers[0].self_attn.attn.impl.do_kv_cache_update.assert_called_once()


@pytest.mark.parametrize("quantized", [False, True])
def test_minimax_dspark_context_kv_modules_handles_quant_and_unquant(quantized):
    model = minimax_m3_dspark.MiniMaxFlashDSparkModel.__new__(minimax_m3_dspark.MiniMaxFlashDSparkModel)
    nn.Module.__init__(model)
    model.quant_config = object() if quantized else None
    model.use_aux_hidden_state = True
    model.hidden_norm = nn.LayerNorm(4)
    model.fc = nn.Linear(8, 4, bias=False)
    model.norm = nn.LayerNorm(4)

    attn_inner = nn.Module()
    attn_inner.kv_scale = nn.Parameter(torch.ones(1))
    self_attn = nn.Module()
    self_attn.k_norm = nn.LayerNorm(4)
    self_attn.qkv_proj = nn.Linear(4, 12, bias=False)
    self_attn.o_proj = nn.Linear(4, 4, bias=False)
    self_attn.attn = attn_inner
    layer = nn.Module()
    layer.self_attn = self_attn
    layer.mlp = nn.Linear(4, 4, bias=False)
    model.layers = nn.ModuleList([layer])

    built = []
    if not quantized:

        def fake_build():
            built.append(True)
            model._num_attn_layers = 1
            model._fused_kv_weight = torch.ones(8, 4)

        model._build_fused_kv_buffers = fake_build

    _simulate_free_unused_weights(model)
    assert bool(built) is (not quantized)
    assert model.hidden_norm.weight.numel() > 0
    assert model.fc.weight.numel() > 0
    assert self_attn.k_norm.weight.numel() > 0
    assert attn_inner.kv_scale.numel() > 0
    assert (self_attn.qkv_proj.weight.numel() > 0) is quantized
    assert layer.mlp.weight.numel() == 0
    assert self_attn.o_proj.weight.numel() == 0


@pytest.mark.parametrize("prolog_v3", [False, True])
def test_glm5_and_k3_mla_dspark_context_kv_modules(prolog_v3):
    for wrapper_cls, model_cls, is_k3 in (
        (glm5_dspark.Glm5DSparkForCausalLM, glm5_dspark.Glm5DSparkModel, False),
        (kimi_k3_dspark.AscendK3DSparkForCausalLM, kimi_k3_dspark.AscendK3DSparkModel, True),
    ):
        wrapper = wrapper_cls.__new__(wrapper_cls)
        nn.Module.__init__(wrapper)
        model = model_cls.__new__(model_cls)
        nn.Module.__init__(model)
        wrapper.model = model

        model.context_proj = nn.Linear(8, 4, bias=False)
        model.context_norm = nn.LayerNorm(4)
        model.final_norm = nn.LayerNorm(4)
        model.markov_head = nn.Linear(4, 4, bias=False)
        model.embed_tokens = nn.Embedding(8, 4)
        wrapper.lm_head = nn.Linear(4, 8, bias=False)

        impl = SimpleNamespace(
            supports_prolog_v3_writer=lambda: prolog_v3,
            kv_a_layernorm=nn.LayerNorm(4),
            q_a_layernorm=nn.LayerNorm(4),
            q_proj=nn.Linear(4, 4, bias=False),
            kv_b_proj=nn.Linear(4, 4, bias=False),
        )
        attn_leaf = nn.Module()
        attn_leaf.impl = impl
        attn_leaf.kv_scale = nn.Parameter(torch.ones(1))
        # Register impl sub-layers inside self_attn as in Ascend MLA blocks.
        self_attn = nn.Module()
        self_attn.fused_qkv_a_proj = nn.Linear(4, 8, bias=False)
        self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self_attn.kv_a_layernorm = impl.kv_a_layernorm
        self_attn.q_a_layernorm = impl.q_a_layernorm
        self_attn.q_proj = impl.q_proj
        self_attn.kv_b_proj = impl.kv_b_proj
        if is_k3:
            self_attn.impl = impl
            self_attn.mla_attn = nn.Module()
            self_attn.mla_attn.mla_attn = attn_leaf
        else:
            self_attn.attn = attn_leaf

        layer = nn.Module()
        layer.self_attn = self_attn
        layer.mlp = nn.Linear(4, 4, bias=False)
        model.layers = nn.ModuleList([layer])

        assert wrapper.context_kv_modules() == model.context_kv_modules()
        _simulate_free_unused_weights(model)
        assert model.context_proj.weight.numel() > 0
        assert model.context_norm.weight.numel() > 0
        assert self_attn.fused_qkv_a_proj.weight.numel() > 0
        assert impl.kv_a_layernorm.weight.numel() > 0
        assert attn_leaf.kv_scale.numel() > 0
        assert (impl.q_proj.weight.numel() > 0) is prolog_v3
        assert (impl.kv_b_proj.weight.numel() > 0) is prolog_v3
        assert self_attn.o_proj.weight.numel() == 0
        assert layer.mlp.weight.numel() == 0
        assert model.markov_head.weight.numel() == 0
        assert model.final_norm.weight.numel() == 0


@pytest.mark.parametrize("context_kv_only", [False, True])
def test_deepseek_v4_dspark_gates_oproj_buffers_and_frees_unused_weights(monkeypatch, context_kv_only):
    config = SimpleNamespace(
        hc_mult=2,
        hidden_size=4,
        vocab_size=8,
        draft_vocab_size=None,
        dspark_block_size=4,
        dspark_target_layer_ids=[0],
        dspark_markov_rank=2,
        num_hidden_layers=2,
        n_mtp_layers=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-5,
        quantization_config=None,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            prefill_context_kv_only=context_kv_only,
            draft_model_config=SimpleNamespace(hf_config=config),
        ),
        model_config=SimpleNamespace(dtype=torch.float16),
        quant_config=None,
    )

    draft_impls = []

    def make_layer(*args, **kwargs):
        layer = nn.Module()
        layer.use_sequence_parallel_moe = False
        impl = MagicMock(spec=dsv4_dspark.AscendDSAImpl)
        draft_impls.append(impl)
        swa_cache = nn.Module()
        swa_cache.kv_scale = nn.Parameter(torch.ones(1))
        dsa_leaf = nn.Module()
        dsa_leaf.impl = impl
        dsa_leaf.k_scale = nn.Parameter(torch.ones(1))
        layer.self_attn = nn.Module()
        layer.self_attn.kv_norm = nn.LayerNorm(4)
        layer.self_attn.wq_a = nn.Linear(4, 4, bias=False)
        layer.self_attn.wo_a = nn.Linear(4, 4, bias=False)
        layer.self_attn.dsa_attn = SimpleNamespace(swa_cache_layer=swa_cache, dsa_attn=dsa_leaf)
        layer.mlp = nn.Linear(4, 4, bias=False)
        return layer

    monkeypatch.setattr(dsv4_dspark, "DeepseekV4DecoderLayer", make_layer)
    monkeypatch.setattr(dsv4_dspark, "VocabParallelEmbedding", lambda *a, **kw: nn.Embedding(8, 4))
    monkeypatch.setattr(dsv4_dspark, "MergedColumnParallelLinear", lambda *a, **kw: nn.Linear(4, 4, bias=False))
    monkeypatch.setattr(dsv4_dspark, "ColumnParallelLinear", lambda *a, **kw: nn.Linear(4, 4, bias=False))
    monkeypatch.setattr(dsv4_dspark, "RMSNorm", lambda *a, **kw: nn.LayerNorm(4))
    monkeypatch.setattr(dsv4_dspark, "DSparkMarkovHead", lambda *a, **kw: nn.Linear(4, 2, bias=False))
    monkeypatch.setattr(dsv4_dspark, "DSparkConfidenceHead", lambda *a, **kw: nn.Linear(4, 1, bias=False))

    model = dsv4_dspark.DeepseekV4DSparkModel(vllm_config=vllm_config, prefix="model")
    assert len(draft_impls) == 1
    if context_kv_only:
        draft_impls[0].reserve_oproj_buffers.assert_not_called()
    else:
        draft_impls[0].reserve_oproj_buffers.assert_called_once()

    wrapper = dsv4_dspark.DSparkDeepseekV4ForCausalLM.__new__(dsv4_dspark.DSparkDeepseekV4ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    assert wrapper.context_kv_modules() == model.context_kv_modules()

    _simulate_free_unused_weights(model)
    layer = model.layers["2"]
    assert model.main_proj.weight.numel() > 0
    assert model.main_norm.weight.numel() > 0
    assert model.context_wkv_proj.weight.numel() > 0
    assert layer.self_attn.kv_norm.weight.numel() > 0
    assert layer.self_attn.dsa_attn.swa_cache_layer.kv_scale.numel() > 0
    assert layer.self_attn.dsa_attn.dsa_attn.k_scale.numel() > 0
    assert model.embed_tokens.weight.numel() == 0
    assert layer.self_attn.wq_a.weight.numel() == 0
    assert layer.self_attn.wo_a.weight.numel() == 0
    assert layer.mlp.weight.numel() == 0
    assert model.markov_head.weight.numel() == 0
    assert model.confidence_head.weight.numel() == 0
    assert model.hc_head_fn.numel() == 0


def test_deepseek_v41_dspark_gates_oproj_and_context_kv_modules(monkeypatch):
    if dsv41_dspark is None:
        pytest.skip("DeepSeek V4.1 is unavailable on this vLLM release")
    config = SimpleNamespace(
        hc_mult=2,
        hidden_size=4,
        vocab_size=8,
        draft_vocab_size=None,
        dspark_block_size=4,
        dspark_target_layer_ids=[0],
        dspark_markov_rank=2,
        num_hidden_layers=2,
        n_mtp_layers=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-5,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            prefill_context_kv_only=True,
            draft_model_config=SimpleNamespace(hf_text_config=config, hf_config=config),
        ),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        model_config=SimpleNamespace(dtype=torch.float16),
        quant_config=None,
    )
    draft_impls = []

    def make_layer(*args, **kwargs):
        layer = nn.Module()
        impl = MagicMock(spec=dsv41_dspark.AscendDSAImpl)
        draft_impls.append(impl)
        swa_cache = nn.Module()
        swa_cache.kv_scale = nn.Parameter(torch.ones(1))
        dsa_leaf = nn.Module()
        dsa_leaf.impl = impl
        layer.self_attn = nn.Module()
        layer.self_attn.kv_norm = nn.LayerNorm(4)
        layer.self_attn.wq_a = nn.Linear(4, 4, bias=False)
        layer.self_attn.dsa_attn = SimpleNamespace(swa_cache_layer=swa_cache, dsa_attn=dsa_leaf)
        layer.mlp = nn.Linear(4, 4, bias=False)
        return layer

    monkeypatch.setattr(dsv41_dspark, "normalize_deepseek_v41_config", lambda c: c)
    monkeypatch.setattr(dsv41_dspark, "DeepseekV41DSparkDecoderLayer", make_layer)
    monkeypatch.setattr(dsv41_dspark, "VocabParallelEmbedding", lambda *a, **kw: nn.Embedding(8, 4))
    monkeypatch.setattr(dsv41_dspark, "MergedColumnParallelLinear", lambda *a, **kw: nn.Linear(4, 4, bias=False))
    monkeypatch.setattr(dsv41_dspark, "ColumnParallelLinear", lambda *a, **kw: nn.Linear(4, 4, bias=False))
    monkeypatch.setattr(dsv41_dspark, "RMSNorm", lambda *a, **kw: nn.LayerNorm(4))
    monkeypatch.setattr(dsv41_dspark, "DSparkMarkovHead", lambda *a, **kw: nn.Linear(4, 2, bias=False))
    monkeypatch.setattr(dsv41_dspark, "DSparkConfidenceHead", lambda *a, **kw: nn.Linear(4, 1, bias=False))

    model = dsv41_dspark.DeepseekV41DSparkModel(vllm_config=vllm_config, prefix="model")
    draft_impls[0].reserve_oproj_buffers.assert_not_called()

    wrapper = dsv41_dspark.DSparkDeepseekV41ForCausalLM.__new__(dsv41_dspark.DSparkDeepseekV41ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    assert wrapper.context_kv_modules() == model.context_kv_modules()

    _simulate_free_unused_weights(model)
    assert model.main_proj.weight.numel() > 0
    assert model.context_wkv_proj.weight.numel() > 0
    assert model.layers["2"].self_attn.kv_norm.weight.numel() > 0
    assert model.layers["2"].mlp.weight.numel() == 0
    assert model.layers["2"].self_attn.wq_a.weight.numel() == 0


def test_speculators_gate_context_kv_only_and_empty_builders(monkeypatch):
    # AscendDFlashSpeculator forces CUDAGraphMode.NONE when context_kv_only is True.
    dflash = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    dflash.speculative_config = SimpleNamespace(enforce_eager=False)
    dflash.context_kv_only = True
    dflash.update_stream = None
    seen_modes = []

    def fake_super_init(self, mode):
        seen_modes.append(mode)
        self.query_cudagraph_manager = SimpleNamespace()

    with patch.object(AscendDFlashSpeculator.__bases__[1], "init_cudagraph_manager", fake_super_init):
        dflash.init_cudagraph_manager(CUDAGraphMode.FULL_AND_PIECEWISE)
    assert seen_modes == [CUDAGraphMode.NONE]

    # AscendDSparkSpeculator skips LM-head TP wrap, forces NONE graph, and ignores builderless groups.
    dspark = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    dspark.speculative_config = SimpleNamespace(enforce_eager=False)
    dspark.context_kv_only = True
    dspark.update_stream = None
    dummy_model = SimpleNamespace(compute_draft_logits="original")
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.dspark.speculator.lmhead_tp_enable", lambda: True)
    dspark._lmhead_tp_wrap_draft_logits(dummy_model)
    assert dummy_model.compute_draft_logits == "original"

    dspark_modes = []

    def fake_dspark_super_init(self, mode):
        dspark_modes.append(mode)
        self.query_cudagraph_manager = SimpleNamespace()

    with patch.object(AscendDSparkSpeculator.__bases__[1], "init_cudagraph_manager", fake_dspark_super_init):
        dspark.init_cudagraph_manager(CUDAGraphMode.FULL)
    assert dspark_modes == [CUDAGraphMode.NONE]

    # AscendDSparkSpeculator.set_attn skips empty-builder MLA groups safely.
    monkeypatch.setattr(
        AscendDSparkSpeculator,
        "attn_vllm_config",
        property(lambda self: SimpleNamespace(parallel_config=SimpleNamespace())),
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.dflash_speculator.prepare_dflash_inputs",
        object(),
    )
    dspark.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
    dspark.draft_attn_layer_names = {"draft.0"}
    dspark._context_slot_mappings = torch.zeros(1, dtype=torch.int64)
    empty_mla_group = SimpleNamespace(
        metadata_builders=[],
        backend=AscendMLABackend,
        get_metadata_builder=lambda *a: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    dspark.attn_groups = [[empty_mla_group]]
    kv_cfg = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["draft.0"])])
    fake_layer = SimpleNamespace(get_attn_backend=lambda: AscendMLABackend)
    with (
        patch("vllm_ascend.worker.v2.spec_decode.dspark.speculator.set_current_vllm_config"),
        patch.object(AscendDSparkSpeculator.__bases__[1], "set_attn"),
        patch(
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator.get_layers_from_vllm_config",
            return_value={"draft.0": fake_layer},
        ),
        patch("vllm_ascend.worker.v2.spec_decode.dspark.speculator.prepare_dflash_inputs_factory"),
    ):
        dspark.set_attn(None, kv_cfg, None, None, None)
    assert dspark.attn_architecture == "MLA"
    assert dspark.use_flashmla is False

    # AscendDSparkSpeculator.propose skips DCP metadata prep when context_kv_only is True.
    dspark.use_dcp = True
    dspark.attn_architecture = "MLA"
    dspark.pcp_manager = None
    dspark.max_num_reqs = 2
    dspark.max_num_tokens = 8
    dspark.num_query_per_req = 4
    dspark.input_buffers = SimpleNamespace(positions=torch.arange(8))
    batch = SimpleNamespace(num_reqs=1, is_prefilling_np=np.array([True], dtype=np.bool_))
    with (
        patch.object(dspark, "_prepare_draft_dcp_metadata_inputs", side_effect=AssertionError("should skip")),
        patch("vllm_ascend.worker.v2.spec_decode.dspark.speculator.draft_metadata_request_count"),
        patch("vllm_ascend.worker.v2.spec_decode.dspark.speculator.build_attn_metadata_wrapper"),
        patch("vllm_ascend.worker.v2.spec_decode.dspark.speculator.build_attn_metadata_factory"),
        patch.object(AscendDSparkSpeculator.__bases__[1], "propose", return_value=torch.zeros(1, dtype=torch.int32)),
    ):
        dspark.propose(
            batch,
            {},
            {},
            torch.zeros(1, 4),
            None,
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1),
            torch.zeros(1, dtype=torch.int64),
        )

    # AscendLiLiCorrSpeculator allows eager override and special vocab modes under context_kv_only.
    if AscendLiLiCorrSpeculator is not None:
        lili = AscendLiLiCorrSpeculator.__new__(AscendLiLiCorrSpeculator)
        lili.speculative_config = SimpleNamespace(enforce_eager=False)
        lili.context_kv_only = True
        lili.update_stream = None
        lili.target_embeddings = SimpleNamespace(parallel_mode=VocabParallelMode.PCP_X_TP)
        lili_modes = []
        with patch.object(
            AscendDFlashSpeculator.__bases__[1],
            "init_cudagraph_manager",
            lambda self, mode: (lili_modes.append(mode), setattr(self, "query_cudagraph_manager", SimpleNamespace())),
        ):
            lili.init_cudagraph_manager(CUDAGraphMode.FULL)
        assert lili_modes == [CUDAGraphMode.NONE]

        draft_mod = SimpleNamespace(lm_head=SimpleNamespace(parallel_mode=VocabParallelMode.FINE_GRAINED))
        with patch.object(AscendLiLiCorrSpeculator.__bases__[0], "load_draft_model", return_value=draft_mod):
            assert lili.load_draft_model(nn.Module(), set()) is draft_mod


def test_empty_metadata_builders_and_mamba_hybrid_zero_scheduled_spec_tokens():
    # 1. attn_utils._get_parallel_config_for_attn_metadata and build_attn_metadata skip empty groups.
    p_cfg = object()
    empty_group = SimpleNamespace(
        metadata_builders=[],
        backend=AscendMLABackend,
        layer_names=["draft.0"],
        get_metadata_builder=lambda *a: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    valid_group = SimpleNamespace(
        metadata_builders=[object()],
        backend=AscendMLABackend,
        layer_names=["target.0"],
        get_metadata_builder=lambda *a: SimpleNamespace(
            vllm_config=SimpleNamespace(parallel_config=p_cfg),
            flashmla_state=object(),
        ),
    )
    legacy_stub_group = SimpleNamespace(
        backend=AscendMLABackend,
        layer_names=["target.0"],
        get_metadata_builder=lambda *a: SimpleNamespace(
            vllm_config=SimpleNamespace(parallel_config=p_cfg),
            flashmla_state=object(),
        ),
    )
    assert attn_utils._get_parallel_config_for_attn_metadata([[empty_group], [valid_group]]) is p_cfg
    assert attn_utils._get_parallel_config_for_attn_metadata([[empty_group], [legacy_stub_group]]) is p_cfg
    with patch("vllm_ascend.worker.v2.attn_utils.uses_turboquant_groups", return_value=False):
        meta = attn_utils.build_attn_metadata(
            attn_groups=[[empty_group]],
            num_reqs=1,
            num_tokens=1,
            query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            max_query_len=1,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_np=np.array([1], dtype=np.int32),
            max_seq_len=1,
            positions=torch.zeros(1, dtype=torch.int64),
            block_tables=(torch.zeros(1, 1, dtype=torch.int32),),
            slot_mappings=torch.zeros(1, 1, dtype=torch.int32),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace()]),
        )
    assert meta == {}

    # 2. NPUModelRunner._has_target_flashmla_attention ignores empty-builder groups.
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculator = SimpleNamespace(draft_attn_layer_names=set())
    runner.attn_groups = [[empty_group, valid_group]]
    assert runner._has_target_flashmla_attention() is True
    runner.attn_groups = [[empty_group, legacy_stub_group]]
    assert runner._has_target_flashmla_attention() is True

    # 3. _register_dspark_swa_indices_warmup returns 0 cleanly when prefill_context_kv_only=True.
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            jit_warmup_registry=SimpleNamespace(),
            speculative_config=SimpleNamespace(prefill_context_kv_only=True),
        )
    )
    assert kernel_warmup._register_dspark_swa_indices_warmup(worker) == 0

    # 4. TargetDeviceMetadata.build skips empty-builder groups.
    dm = TargetDeviceMetadata.__new__(TargetDeviceMetadata)
    dm.executor = SimpleNamespace(submission_in_flight=False)
    dm._tasks = ()
    dm._failed = False
    with dm.build([[empty_group]], full_graph=False):
        pass

    # 5. cpu_length_sync ignores empty-builder groups gracefully.
    mock_runner = SimpleNamespace(
        ascend_config=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
        model_config=SimpleNamespace(architecture="Qwen4ExpForCausalLM"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        speculator=SimpleNamespace(attn_groups=[[empty_group]]),
        attn_groups=[[empty_group]],
    )
    assert qwen4_cpu_sync.can_skip_qwen_cpu_length_sync(mock_runner) is False

    # 6. AscendMambaHybridModelState uses num_scheduled_spec_tokens == 0 to skip spec classification.
    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=3, num_scheduled_spec_tokens=0)
    # Use a sentinel on num_accepted_tokens_gpu: if entered, indexing would fail.
    state.num_accepted_tokens_gpu = None
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=4,
        is_prefilling_np=np.array([True], dtype=np.bool_),
        prefill_runs_as_decode_np=None,
        num_draft_tokens_per_req=None,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        max_query_len=4,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_np=np.array([0, 4], dtype=np.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
    )
    with patch(
        "vllm_ascend.worker.v2.model_states.mamba_hybrid.build_attn_metadata",
        return_value={"ok": True},
    ) as mock_build:
        res = state._build_attn_metadata_for_batch(
            batch,
            CUDAGraphMode.NONE,
            (torch.zeros(1, 1, dtype=torch.int32),),
            torch.zeros(1, 4, dtype=torch.int32),
            [[empty_group]],
            SimpleNamespace(kv_cache_groups=[]),
            False,
            use_pcp_padding=False,
        )
    assert res == {"ok": True}
    spec_meta = mock_build.call_args.kwargs["model_specific_attn_metadata"]
    assert spec_meta.num_accepted_tokens is None
    assert spec_meta.num_decode_draft_tokens_cpu is None
