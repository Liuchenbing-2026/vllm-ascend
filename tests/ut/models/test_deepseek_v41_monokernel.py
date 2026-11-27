# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU unit tests for DeepSeek V4.1 Per-Layer MonoKernel (--monokernel)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from tests.deepseek_v41_utils import hc_mixes_reference, hc_post_reference
from vllm_ascend.models.deepseek_v41.mono import (
    AscendDSV41MonoConfig,
    DSV41AscendMonoRunner,
    maybe_build_mono_runner,
    resolve_mono_request,
)
from vllm_ascend.models.deepseek_v41.mono.seam_kernel import compute_hc_mixes_from_inv_rms


def _make_configs(
    *,
    monokernel: bool = True,
    decode_backend: str = "auto",
    lora: object = None,
    ubatch: bool = False,
    pcp: int = 1,
    sp_moe: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    hc_mult: int = 4,
    hidden_size: int = 5120,
):
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        additional_config={"decode_backend": decode_backend},
        lora_config=lora,
        parallel_config=SimpleNamespace(
            use_ubatching=ubatch,
            prefill_context_parallel_size=pcp,
            use_sequence_parallel_moe=sp_moe,
        ),
        model_config=SimpleNamespace(dtype=dtype),
    )
    hf_config = SimpleNamespace(hc_mult=hc_mult, hidden_size=hidden_size)
    return vllm_config, hf_config


def test_resolve_and_default_off_and_conflict():
    vllm_off, hf_cfg = _make_configs(monokernel=False, decode_backend="auto")
    assert not resolve_mono_request(vllm_off)
    assert maybe_build_mono_runner(vllm_off, hf_cfg, [], 0, 0, (), 0) is None

    vllm_conflict, _ = _make_configs(monokernel=True, decode_backend="stock")
    with pytest.raises(ValueError, match="Conflicting options"):
        resolve_mono_request(vllm_conflict)

    vllm_bad, _ = _make_configs(monokernel=False, decode_backend="invalid_mode")
    with pytest.raises(ValueError, match="Unsupported decode_backend"):
        resolve_mono_request(vllm_bad)


def test_default_deny_collects_all_unsupported_reasons():
    vllm_cfg, hf_cfg = _make_configs(
        monokernel=True,
        lora=object(),
        pcp=4,
        sp_moe=True,
        dtype=torch.float16,
        hc_mult=2,
        hidden_size=100,
    )
    with pytest.raises(ValueError) as exc_info:
        AscendDSV41MonoConfig.from_vllm_config(vllm_cfg, hf_cfg)
    msg = str(exc_info.value)
    for token in ("LoRA", "prefill_context_parallel_size", "use_sequence_parallel_moe", "bfloat16", "hc_mult=4", "64"):
        assert token in msg


@pytest.mark.parametrize(
    ("num_tokens", "cg_mode", "num_prefills", "num_decodes", "num_reqs", "expected"),
    [
        (8, CUDAGraphMode.FULL, 0, 8, 8, True),
        (48, CUDAGraphMode.NONE, 0, 8, 8, True),
        (49, CUDAGraphMode.FULL, 0, 8, 8, False),
        (8, CUDAGraphMode.PIECEWISE, 0, 8, 8, False),
        (8, CUDAGraphMode.FULL, 1, 7, 8, False),
        (8, CUDAGraphMode.FULL, 0, 0, 0, False),
    ],
)
def test_plan_step_deterministic_routing(num_tokens, cg_mode, num_prefills, num_decodes, num_reqs, expected):
    runner = DSV41AscendMonoRunner(AscendDSV41MonoConfig(max_rows=48), {})
    meta = {"l0": SimpleNamespace(num_prefills=num_prefills, num_decodes=num_decodes, num_reqs=num_reqs)}
    ctx = SimpleNamespace(cudagraph_runtime_mode=cg_mode, attn_metadata=meta)
    with (
        patch("vllm_ascend.models.deepseek_v41.mono.runner.is_forward_context_available", return_value=True),
        patch("vllm_ascend.models.deepseek_v41.mono.runner.get_forward_context", return_value=ctx),
    ):
        plan = runner.plan_step(torch.arange(num_tokens))
    assert plan.active is expected


def test_layer_bindings_break_deferral_at_engram_aux_and_replay_boundaries():
    layers = [SimpleNamespace(layer_idx=i, engram=(object() if i == 2 else None)) for i in range(6)]
    runner = DSV41AscendMonoRunner.from_layers(
        config=AscendDSV41MonoConfig(),
        layers=layers,
        start_layer=0,
        end_layer=6,
        aux_hidden_state_layers=(4,),
        decoder_replay_start=5,
    )
    assert runner.bindings[0].can_defer_post is True
    assert runner.bindings[1].can_defer_post is False  # layer 2 has engram
    assert runner.bindings[2].can_defer_post is True
    assert runner.bindings[3].can_defer_post is False  # layer 4 in aux_hidden_state_layers
    assert runner.bindings[4].can_defer_post is False  # layer 5 is decoder_replay_start
    assert runner.bindings[5].can_defer_post is False  # last layer


def test_hc_mixes_from_inv_rms_matches_reference():
    torch.manual_seed(3)
    layer = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=3, norm_eps=1e-6, hc_eps=1e-6)
    res = torch.randn(4, 4, 64, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, 256, dtype=torch.float32) / 64.0
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)

    ref_pre, ref_post, ref_comb = hc_mixes_reference(layer, res, hc_fn, hc_scale, hc_base)
    flat = res.float().flatten(-2)
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + layer.norm_eps)
    pre, post, comb = compute_hc_mixes_from_inv_rms(flat, inv_rms, hc_fn, hc_scale, hc_base, 3, layer.hc_eps)

    torch.testing.assert_close(pre, ref_pre, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(post, ref_post, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(comb, ref_comb, atol=1e-6, rtol=1e-6)


class _RefMonoBackend:
    name = "cpu_ref"

    def run_seam_post_collapse_norm(self, x, residual, post, comb, pre, norm_module, norm_4d_eps):
        res_new = hc_post_reference(x, residual, post, comb)
        collapsed = (pre.unsqueeze(-1) * res_new.float()).sum(-2).to(x.dtype)
        uf = collapsed.float()
        inv_rms = torch.rsqrt(uf.square().mean(-1, keepdim=True) + norm_module.variance_epsilon)
        normed = (uf * inv_rms * norm_module.weight.float()).to(x.dtype)
        flat = res_new.float().flatten(-2)
        return res_new, normed, torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_4d_eps)

    def run_hc_mixes(self, residual_flat_f32, inv_rms_4d, hc_fn, hc_scale, hc_base, sinkhorn_iters, hc_eps):
        return compute_hc_mixes_from_inv_rms(
            residual_flat_f32, inv_rms_4d, hc_fn, hc_scale, hc_base, sinkhorn_iters, hc_eps
        )


class _FakeRMSNorm:
    def __init__(self, d: int):
        self.weight = torch.ones(d, dtype=torch.bfloat16)
        self.variance_epsilon = 1e-6

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        uf = x.float()
        return (uf * torch.rsqrt(uf.square().mean(-1, keepdim=True) + self.variance_epsilon) * self.weight.float()).to(
            x.dtype
        )


class _FakeLayer:
    def __init__(self, idx: int, d: int):
        self.layer_idx, self.engram, self.norm_eps, self.hc_eps, self.hc_mult, self.hc_sinkhorn_iters = (
            idx,
            None,
            1e-6,
            1e-6,
            4,
            3,
        )
        self.hc_attn_fn = torch.randn(24, 4 * d) / (4 * d)
        self.hc_ffn_fn = torch.randn(24, 4 * d) / (4 * d)
        self.hc_attn_scale, self.hc_ffn_scale = torch.ones(3), torch.ones(3)
        self.hc_attn_base, self.hc_ffn_base = torch.zeros(24), torch.zeros(24)
        self.input_layernorm = _FakeRMSNorm(d)
        self.post_attention_layernorm = _FakeRMSNorm(d)
        self.self_attn = lambda pos, x, _: (x * 0.5).to(x.dtype)
        self.mlp = lambda x, input_ids=None, already_sequence_parallel=False: (x * 0.25).to(x.dtype)

    def hc_pre(self, x, fn, sc, b, pre=None):
        u = (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)
        p, po, cm = hc_mixes_reference(self, x, fn, sc, b)
        return u, po, cm, p

    def hc_post(self, x, r, p, c):
        return hc_post_reference(x, r, p, c)


def test_runner_three_layer_deferred_post_matches_stock():
    torch.manual_seed(7)
    layers = [_FakeLayer(i, 64) for i in range(3)]
    runner = DSV41AscendMonoRunner.from_layers(
        AscendDSV41MonoConfig(), layers, 0, 3, (2,), 99, backend=_RefMonoBackend()
    )
    res = torch.randn(4, 4, 64, dtype=torch.bfloat16)
    pre = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
    pos = torch.arange(4)

    ref_res, ref_pre = res.clone(), pre.clone()
    for lyr in layers:
        x_a, ap, ac, apr = lyr.hc_pre(ref_res, lyr.hc_attn_fn, lyr.hc_attn_scale, lyr.hc_attn_base, ref_pre)
        ref_res = lyr.hc_post(lyr.self_attn(pos, lyr.input_layernorm(x_a), None), ref_res, ap, ac)
        x_f, fp, fc, ref_pre = lyr.hc_pre(ref_res, lyr.hc_ffn_fn, lyr.hc_ffn_scale, lyr.hc_ffn_base, apr)
        ref_res = lyr.hc_post(lyr.mlp(lyr.post_attention_layernorm(x_f)), ref_res, fp, fc)

    mono_res, mono_pre, pending = res.clone(), pre.clone(), None
    for lyr in layers:
        if pending is not None and lyr.layer_idx == 2:
            mono_res = runner.flush_pending(layers[lyr.layer_idx - 1], mono_res, pending)
            pending = None
        mono_res, mono_pre, pending = runner.run_layer(
            lyr.layer_idx, pos, mono_res, mono_pre, None, pending, allow_defer_post=lyr.layer_idx != 1
        )
    if pending is not None:
        mono_res = runner.flush_pending(layers[-1], mono_res, pending)

    torch.testing.assert_close(mono_res, ref_res, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(mono_pre, ref_pre, atol=1e-5, rtol=1e-5)
