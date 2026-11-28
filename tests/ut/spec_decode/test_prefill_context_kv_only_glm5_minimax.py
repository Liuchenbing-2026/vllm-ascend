# mypy: disable-error-code="annotation-unchecked, arg-type, assignment, attr-defined, misc"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused CPU unit tests for GLM-5 and MiniMax-M3 DSpark prefill_context_kv_only."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend import utils
from vllm_ascend.models import glm5_dspark
from vllm_ascend.models.minimax_m3 import minimax_m3_dspark as mm3_dspark


def _lin(o: int = 4, i: int = 4) -> nn.Linear:
    return nn.Linear(i, o, bias=False)


def _mod(**kw: object) -> nn.Module:
    m = nn.Module()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _assert_retained(wrap: nn.Module, kept: tuple[torch.Tensor, ...], freed: tuple[torch.Tensor, ...]) -> None:
    if not utils.vllm_version_is("0.34.0"):
        spec = DFlashSpeculator.__new__(DFlashSpeculator)
        spec.context_kv_only, spec.model = True, wrap
        spec.free_unused_weights(nn.Module())
        assert all(t.numel() > 0 for t in kept) and all(t.numel() == 0 for t in freed)
        return
    kept_ids = {id(p) for mod in wrap.context_kv_modules() for p in mod.parameters()}
    assert all(id(t) in kept_ids for t in kept) and all(id(t) not in kept_ids for t in freed)


@pytest.mark.parametrize("prolog_v3", [False, True])
def test_glm5_dspark_context_kv_modules(prolog_v3: bool):
    def ln() -> nn.LayerNorm:
        return nn.LayerNorm(4)

    g5 = glm5_dspark.Glm5DSparkModel.__new__(glm5_dspark.Glm5DSparkModel)
    nn.Module.__init__(g5)
    g5.context_proj, g5.context_norm, g5.embed_tokens = _lin(), ln(), _lin()
    kv_ln, q_proj, q_ln, kv_b = ln(), _lin(), ln(), _lin()
    g5_impl = NS(
        kv_a_layernorm=kv_ln,
        q_proj=q_proj,
        q_a_layernorm=q_ln,
        kv_b_proj=kv_b,
        supports_prolog_v3_writer=lambda: prolog_v3,
    )
    g5_attn = _mod(scale=_lin())
    g5_attn.impl = g5_impl
    g5_sa = _mod(
        fused_qkv_a_proj=_lin(),
        attn=g5_attn,
        kv_a_layernorm=kv_ln,
        q_proj=q_proj,
        q_a_layernorm=q_ln,
        kv_b_proj=kv_b,
    )
    g5.layers = nn.ModuleList([_mod(self_attn=g5_sa, mlp=_lin())])
    g5_lm = glm5_dspark.Glm5DSparkForCausalLM.__new__(glm5_dspark.Glm5DSparkForCausalLM)
    nn.Module.__init__(g5_lm)
    g5_lm.model = g5
    kept = (g5.context_proj.weight, g5.context_norm.weight, g5_sa.fused_qkv_a_proj.weight, kv_ln.weight)
    freed = (g5.embed_tokens.weight, g5.layers[0].mlp.weight)
    prolog_weights = (q_proj.weight, q_ln.weight, kv_b.weight)
    if prolog_v3:
        kept = (*kept, *prolog_weights)
    else:
        freed = (*freed, *prolog_weights)
    _assert_retained(g5_lm, kept, freed)


@pytest.mark.parametrize(
    ("quantized", "has_fused", "expect_qkv_kept"),
    [
        pytest.param(True, False, True, id="quantized-keeps-qkv"),
        pytest.param(False, True, False, id="unquantized-fused-frees-qkv"),
        pytest.param(False, False, True, id="unquantized-unfused-keeps-qkv"),
    ],
)
def test_minimax_m3_dspark_context_kv_modules(quantized: bool, has_fused: bool, expect_qkv_kept: bool):
    def ln() -> nn.LayerNorm:
        return nn.LayerNorm(4)

    mm = mm3_dspark.MiniMaxFlashDSparkModel.__new__(mm3_dspark.MiniMaxFlashDSparkModel)
    nn.Module.__init__(mm)
    mm.quant_config = object() if quantized else None
    mm.hidden_norm, mm.norm, mm.fc, mm.use_aux_hidden_state = ln(), ln(), _lin(), True
    mm._num_attn_layers = 1
    mm._fused_kv_weight = torch.zeros(8, 4) if has_fused else None
    mm_sa = _mod(k_norm=ln(), qkv_proj=_lin(12), o_proj=_lin(), attn=_lin())
    mm.layers = nn.ModuleList([_mod(self_attn=mm_sa, mlp=_lin())])
    mm_lm = mm3_dspark.MiniMaxFlashDSparkForCausalLM.__new__(mm3_dspark.MiniMaxFlashDSparkForCausalLM)
    nn.Module.__init__(mm_lm)
    mm_lm.model = mm
    kept = (mm.hidden_norm.weight, mm.fc.weight, mm_sa.k_norm.weight, mm_sa.attn.weight)
    freed = (mm.norm.weight, mm_sa.o_proj.weight, mm.layers[0].mlp.weight)
    if expect_qkv_kept:
        kept = (*kept, mm_sa.qkv_proj.weight)
    else:
        freed = (*freed, mm_sa.qkv_proj.weight)
    _assert_retained(mm_lm, kept, freed)
