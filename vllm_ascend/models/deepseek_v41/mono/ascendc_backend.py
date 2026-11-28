# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Single-kernel AscendC per-layer MegaKernel backend for DeepSeek V4.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.forward_context import get_forward_context

_MONO_ALIGN = 64
_MONO_SYNC_INTS = 1024


def _align64(val: int) -> int:
    return ((val + _MONO_ALIGN - 1) // _MONO_ALIGN) * _MONO_ALIGN


def compute_mono_workspace_floats(
    num_tokens: int,
    hidden_size: int,
    q_lora_rank: int,
    n_heads: int,
    head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    n_experts: int,
    top_k: int,
    shared_inter: int,
    routed_inter: int,
) -> int:
    """Compute exact float32 element count required by dsv41_mono_layer workspace."""
    m = max(num_tokens, 1)
    # wsXNorm, wsResMid, wsQA, wsQR, wsKV, wsQ, wsAttnOut, wsOA, wsSharedAct, wsSharedOut, wsRoutedAct, wsRoutedOut
    bf16_chunks = (
        _align64((m * hidden_size + 1) // 2) * 2
        + _align64((m * 4 * hidden_size + 1) // 2)
        + _align64((m * q_lora_rank + 1) // 2) * 2
        + _align64((m * head_dim + 1) // 2)
        + _align64((m * n_heads * head_dim + 1) // 2) * 2
        + _align64((m * o_groups * o_lora_rank + 1) // 2)
        + _align64((m * shared_inter + 1) // 2)
        + _align64((m * top_k * routed_inter + 1) // 2)
        + _align64((m * top_k * hidden_size + 1) // 2)
    )
    # wsSync, wsInvRms, wsMixes, wsAttnPre/Post, wsFfnPre/Post, wsAttnComb, wsFfnComb, logits, topk_ids, topk_w, exp_meta
    exp_meta_ints = 64 + max(m * top_k, 1) * 18
    f32_chunks = (
        _MONO_SYNC_INTS
        + _align64(m)
        + _align64(m * 32)
        + _align64(m * 8) * 4
        + _align64(m * 16) * 2
        + _align64(m * _align64(n_experts))
        + _align64(m * _align64(top_k)) * 2
        + _align64(exp_meta_ints)
    )
    return bf16_chunks + f32_chunks


def _unfold_cache(cache: Any) -> torch.Tensor | None:
    while isinstance(cache, (list, tuple)) and len(cache) == 1:
        cache = cache[0]
    return cache if isinstance(cache, torch.Tensor) else None


def is_full_layer_kernel_eligible(layer: Any) -> bool:
    """Return True when one DSv4.1 decoder layer fits the single-kernel AscendC path."""
    c_ascend = getattr(torch.ops, "_C_ascend", None)
    if c_ascend is None or not hasattr(c_ascend, "dsv41_mono_layer"):
        return False
    if getattr(layer, "use_sequence_parallel", False):
        return False
    if getattr(layer, "prefill_context_parallel_size", 1) != 1:
        return False
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    if attn is None or mlp is None or getattr(attn, "compress_ratio", -1) != 0:
        return False
    if getattr(mlp, "hash", False) or getattr(mlp, "tp_size", 1) != 1 or getattr(mlp, "ep_size", 1) != 1:
        return False
    experts = getattr(mlp, "experts", None)
    w13 = getattr(experts, "w13_weight", None)
    w2 = getattr(experts, "w2_weight", None)
    if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):
        return False
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return False
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        mod = getattr(attn, name, None)
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16:
            return False
    return True


def _resolve_wo_a(attn: Any) -> torch.Tensor:
    w = attn.wo_a.weight
    n_groups = int(attn.n_local_groups)
    o_lora = int(attn.o_lora_rank)
    group_hidden = int(attn.n_local_heads * attn.head_dim // n_groups)
    if w.ndim == 3 and w.shape == (n_groups, group_hidden, o_lora):
        return w.permute(0, 2, 1).contiguous()
    return w.reshape(n_groups, o_lora, group_hidden).contiguous()


def _as_contig(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


@dataclass
class _PreparedLayerWeights:
    swa_key: str
    rot_name: str
    hidden_size: int
    q_lora_rank: int
    n_heads: int
    head_dim: int
    nope_dim: int
    rope_dim: int
    o_groups: int
    o_lora_rank: int
    window_size: int
    n_experts: int
    top_k: int
    shared_inter: int
    routed_inter: int
    sinkhorn_iters: int
    has_e_bias: int
    use_sigmoid: int
    renorm: int
    has_sink: int
    meta_f32: torch.Tensor
    input_ln_w: torch.Tensor
    hc_attn_fn: torch.Tensor
    hc_attn_scale: torch.Tensor
    hc_attn_base: torch.Tensor
    wq_a: torch.Tensor
    q_norm_w: torch.Tensor
    wq_b: torch.Tensor
    wkv: torch.Tensor
    kv_norm_w: torch.Tensor
    attn_sink: torch.Tensor
    wo_a: torch.Tensor
    wo_b: torch.Tensor
    post_attn_ln_w: torch.Tensor
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    gate_w: torch.Tensor
    e_bias: torch.Tensor
    shared_w13: torch.Tensor
    shared_w2: torch.Tensor
    routed_w13: torch.Tensor
    routed_w2: torch.Tensor


class AscendCFullLayerMonoBackend:
    """Dispatches an entire eligible DSv4.1 SWA+MoE layer in one AscendC kernel launch."""

    name = "ascendc_full_layer_mono"

    def __init__(self) -> None:
        self._workspace: torch.Tensor | None = None
        self._layer_cache: dict[int, _PreparedLayerWeights] = {}
        self._meta_i64_cache: dict[tuple[int, int, int, int, str], torch.Tensor] = {}

    def _get_workspace(self, floats_needed: int, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device or self._workspace.numel() < floats_needed:
            self._workspace = torch.empty(floats_needed, dtype=torch.float32, device=device)
        return self._workspace

    def _prepare_layer(self, layer: Any, device: torch.device) -> _PreparedLayerWeights:
        key = id(layer)
        cached = self._layer_cache.get(key)
        if cached is not None:
            return cached
        attn = layer.self_attn
        mlp = layer.mlp
        experts = mlp.experts
        shared = getattr(mlp, "shared_experts", None)
        shared_inter = int(shared.gate_up_proj.weight.shape[0] // 2) if shared is not None else 0
        e_bias = getattr(mlp.gate, "e_score_correction_bias", None)
        attn_sink = getattr(attn, "attn_sink", None)
        empty_bf16 = torch.empty(1, dtype=torch.bfloat16, device=device)
        empty_f32 = torch.empty(1, dtype=torch.float32, device=device)
        routed_w2 = _as_contig(experts.w2_weight)
        scoring = str(getattr(experts, "scoring_func", "softmax")).lower()

        meta_f32 = torch.tensor(
            [
                float(layer.norm_eps),
                float(layer.hc_eps),
                float(attn.softmax_scale),
                float(layer.routed_scaling_factor),
                float(getattr(mlp, "swiglu_limit", 0.0) or 0.0),
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        prepared = _PreparedLayerWeights(
            swa_key=f"{attn.prefix}.swa_cache",
            rot_name=attn.rotary_emb.layername,
            hidden_size=int(layer.hidden_size),
            q_lora_rank=int(attn.q_lora_rank),
            n_heads=int(attn.n_local_heads),
            head_dim=int(attn.head_dim),
            nope_dim=int(attn.nope_head_dim),
            rope_dim=int(attn.rope_head_dim),
            o_groups=int(attn.n_local_groups),
            o_lora_rank=int(attn.o_lora_rank),
            window_size=int(attn.window_size),
            n_experts=int(mlp.n_routed_experts),
            top_k=int(experts.top_k),
            shared_inter=shared_inter,
            routed_inter=int(routed_w2.shape[-1]),
            sinkhorn_iters=int(layer.hc_sinkhorn_iters),
            has_e_bias=1 if e_bias is not None else 0,
            use_sigmoid=1 if scoring == "sigmoid" else 0,
            renorm=1 if getattr(experts, "renormalize", False) else 0,
            has_sink=1 if attn_sink is not None else 0,
            meta_f32=meta_f32,
            input_ln_w=_as_contig(layer.input_layernorm.weight),
            hc_attn_fn=_as_contig(layer.hc_attn_fn),
            hc_attn_scale=_as_contig(layer.hc_attn_scale),
            hc_attn_base=_as_contig(layer.hc_attn_base),
            wq_a=_as_contig(attn.wq_a.weight),
            q_norm_w=_as_contig(attn.q_norm.weight),
            wq_b=_as_contig(attn.wq_b.weight),
            wkv=_as_contig(attn.wkv.weight),
            kv_norm_w=_as_contig(attn.kv_norm.weight),
            attn_sink=_as_contig(attn_sink, torch.float32) if attn_sink is not None else empty_f32,
            wo_a=_resolve_wo_a(attn),
            wo_b=_as_contig(attn.wo_b.weight),
            post_attn_ln_w=_as_contig(layer.post_attention_layernorm.weight),
            hc_ffn_fn=_as_contig(layer.hc_ffn_fn),
            hc_ffn_scale=_as_contig(layer.hc_ffn_scale),
            hc_ffn_base=_as_contig(layer.hc_ffn_base),
            gate_w=_as_contig(mlp.gate.weight),
            e_bias=_as_contig(e_bias, torch.float32) if e_bias is not None else empty_f32,
            shared_w13=_as_contig(shared.gate_up_proj.weight) if shared is not None else empty_bf16,
            shared_w2=_as_contig(shared.down_proj.weight) if shared is not None else empty_bf16,
            routed_w13=_as_contig(experts.w13_weight),
            routed_w2=routed_w2,
        )
        self._layer_cache[key] = prepared
        return prepared

    def _get_meta_i64(
        self,
        layer_id: int,
        pw: _PreparedLayerWeights,
        m_tokens: int,
        block_size: int,
        max_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (layer_id, m_tokens, block_size, max_blocks, str(device))
        cached = self._meta_i64_cache.get(cache_key)
        if cached is not None:
            return cached
        meta_i64 = torch.tensor(
            [
                m_tokens,
                pw.hidden_size,
                pw.q_lora_rank,
                pw.n_heads,
                pw.head_dim,
                pw.nope_dim,
                pw.rope_dim,
                pw.o_groups,
                pw.o_lora_rank,
                pw.window_size,
                block_size,
                max_blocks,
                pw.n_experts,
                pw.top_k,
                pw.shared_inter,
                pw.routed_inter,
                pw.sinkhorn_iters,
                pw.has_e_bias,
                pw.use_sigmoid,
                pw.renorm,
                pw.has_sink,
                0,
                0,
                0,
            ],
            dtype=torch.int64,
            device=device,
        )
        self._meta_i64_cache[cache_key] = meta_i64
        return meta_i64

    def run_full_layer(
        self,
        layer: Any,
        positions: torch.Tensor,
        residual: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one complete DSv4.1 decoder layer via torch.ops._C_ascend.dsv41_mono_layer."""
        ctx = get_forward_context()
        pw = self._prepare_layer(layer, residual.device)
        swa_meta = ctx.attn_metadata[pw.swa_key]
        m_tokens = int(positions.shape[0])

        cos_f32 = _as_contig(swa_meta.cos[pw.rot_name][:m_tokens].reshape(m_tokens, pw.rope_dim), torch.float32)
        sin_f32 = _as_contig(swa_meta.sin[pw.rot_name][:m_tokens].reshape(m_tokens, pw.rope_dim), torch.float32)
        kv_cache = _unfold_cache(layer.self_attn.dsa_attn.swa_cache_layer.kv_cache)
        if kv_cache is None:
            raise RuntimeError("dsv41_mono_layer requires unfolded SWA KV cache tensor")

        block_table = _as_contig(swa_meta.block_table[:m_tokens], torch.int32)
        seq_lens = _as_contig(swa_meta.seq_lens[:m_tokens], torch.int32)
        slot_mapping = _as_contig(swa_meta.slot_mapping[:m_tokens], torch.int64)
        ws = self._get_workspace(
            compute_mono_workspace_floats(
                num_tokens=m_tokens,
                hidden_size=pw.hidden_size,
                q_lora_rank=pw.q_lora_rank,
                n_heads=pw.n_heads,
                head_dim=pw.head_dim,
                o_groups=pw.o_groups,
                o_lora_rank=pw.o_lora_rank,
                n_experts=pw.n_experts,
                top_k=pw.top_k,
                shared_inter=pw.shared_inter,
                routed_inter=pw.routed_inter,
            ),
            residual.device,
        )
        meta_i64 = self._get_meta_i64(
            id(layer),
            pw,
            m_tokens,
            int(kv_cache.shape[1]),
            int(block_table.shape[1]),
            residual.device,
        )

        res_out = torch.empty_like(residual)
        pre_mix_out = torch.empty_like(pre_mix)
        torch.ops._C_ascend.dsv41_mono_layer(
            _as_contig(residual),
            _as_contig(pre_mix),
            _as_contig(positions),
            cos_f32,
            sin_f32,
            slot_mapping,
            block_table,
            seq_lens,
            res_out,
            pre_mix_out,
            pw.input_ln_w,
            pw.hc_attn_fn,
            pw.hc_attn_scale,
            pw.hc_attn_base,
            pw.wq_a,
            pw.q_norm_w,
            pw.wq_b,
            pw.wkv,
            pw.kv_norm_w,
            pw.attn_sink,
            pw.wo_a,
            pw.wo_b,
            pw.post_attn_ln_w,
            pw.hc_ffn_fn,
            pw.hc_ffn_scale,
            pw.hc_ffn_base,
            pw.gate_w,
            pw.e_bias,
            pw.shared_w13,
            pw.shared_w2,
            pw.routed_w13,
            pw.routed_w2,
            kv_cache,
            ws,
            meta_i64,
            pw.meta_f32,
        )
        return res_out, pre_mix_out
