# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Single-kernel AscendC per-layer MegaKernel backend for DeepSeek V4.1."""

from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import get_forward_context

_MONO_ALIGN = 64


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
    bf16_chunks = (
        _align64((m * hidden_size + 1) // 2) * 3
        + _align64((m * 4 * hidden_size + 1) // 2)
        + _align64((m * q_lora_rank + 1) // 2) * 2
        + _align64((m * head_dim + 1) // 2)
        + _align64((m * n_heads * head_dim + 1) // 2) * 2
        + _align64((m * o_groups * o_lora_rank + 1) // 2)
        + _align64((m * 2 * shared_inter + 1) // 2)
        + _align64((m * shared_inter + 1) // 2)
        + _align64((m * top_k * 2 * routed_inter + 1) // 2)
        + _align64((m * top_k * routed_inter + 1) // 2)
    )
    f32_chunks = (
        1024
        + _align64(m)
        + _align64(m * 32)
        + _align64(m * 8) * 4
        + _align64(m * 16) * 2
        + _align64(m * _align64(n_experts))
        + _align64(m * _align64(top_k)) * 2
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


def _resolve_wo_a(attn: Any, cache: dict[int, torch.Tensor], key: int) -> torch.Tensor:
    cached = cache.get(key)
    if cached is not None:
        return cached
    w = attn.wo_a.weight
    n_groups = int(attn.n_local_groups)
    o_lora = int(attn.o_lora_rank)
    group_hidden = int(attn.n_local_heads * attn.head_dim // n_groups)
    if w.ndim == 3 and w.shape == (n_groups, group_hidden, o_lora):
        prepped = w.permute(0, 2, 1).contiguous()
    else:
        prepped = w.reshape(n_groups, o_lora, group_hidden).contiguous()
    cache[key] = prepped
    return prepped


class AscendCFullLayerMonoBackend:
    """Dispatches an entire eligible DSv4.1 SWA+MoE layer in one AscendC kernel launch."""

    name = "ascendc_full_layer_mono"

    def __init__(self) -> None:
        self._workspace: torch.Tensor | None = None
        self._wo_a_cache: dict[int, torch.Tensor] = {}

    def _get_workspace(self, floats_needed: int, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device or self._workspace.numel() < floats_needed:
            self._workspace = torch.empty(floats_needed, dtype=torch.float32, device=device)
        return self._workspace

    def run_full_layer(
        self,
        layer: Any,
        positions: torch.Tensor,
        residual: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one complete DSv4.1 decoder layer via torch.ops._C_ascend.dsv41_mono_layer."""
        ctx = get_forward_context()
        attn = layer.self_attn
        swa_meta = ctx.attn_metadata[f"{attn.prefix}.swa_cache"]
        m_tokens = int(positions.shape[0])
        rot_name = attn.rotary_emb.layername
        cos = swa_meta.cos[rot_name][:m_tokens]
        sin = swa_meta.sin[rot_name][:m_tokens]
        cos_f32 = cos.reshape(m_tokens, attn.rope_head_dim).contiguous().float()
        sin_f32 = sin.reshape(m_tokens, attn.rope_head_dim).contiguous().float()
        kv_cache = _unfold_cache(attn.dsa_attn.swa_cache_layer.kv_cache)
        if kv_cache is None:
            raise RuntimeError("dsv41_mono_layer requires unfolded SWA KV cache tensor")

        mlp = layer.mlp
        experts = mlp.experts
        shared = getattr(mlp, "shared_experts", None)
        shared_inter = int(shared.gate_up_proj.weight.shape[0] // 2) if shared is not None else 0
        shared_w13 = shared.gate_up_proj.weight if shared is not None else residual.new_empty(1, dtype=torch.bfloat16)
        shared_w2 = shared.down_proj.weight if shared is not None else residual.new_empty(1, dtype=torch.bfloat16)
        e_bias = getattr(mlp.gate, "e_score_correction_bias", None)
        attn_sink = getattr(attn, "attn_sink", None)

        routed_w13 = experts.w13_weight
        routed_w2 = experts.w2_weight
        wo_a = _resolve_wo_a(attn, self._wo_a_cache, id(layer))
        ws = self._get_workspace(
            compute_mono_workspace_floats(
                num_tokens=m_tokens,
                hidden_size=layer.hidden_size,
                q_lora_rank=attn.q_lora_rank,
                n_heads=attn.n_local_heads,
                head_dim=attn.head_dim,
                o_groups=attn.n_local_groups,
                o_lora_rank=attn.o_lora_rank,
                n_experts=mlp.n_routed_experts,
                top_k=mlp.experts.top_k,
                shared_inter=shared_inter,
                routed_inter=routed_w2.shape[-1],
            ),
            residual.device,
        )
        block_table = swa_meta.block_table[:m_tokens].contiguous().to(torch.int32)
        seq_lens = swa_meta.seq_lens[:m_tokens].contiguous().to(torch.int32)
        slot_mapping = swa_meta.slot_mapping[:m_tokens].contiguous().to(torch.int64)
        scoring = str(getattr(experts, "scoring_func", "softmax")).lower()

        meta_i64 = torch.tensor(
            [
                m_tokens,
                layer.hidden_size,
                attn.q_lora_rank,
                attn.n_local_heads,
                attn.head_dim,
                attn.nope_head_dim,
                attn.rope_head_dim,
                attn.n_local_groups,
                attn.o_lora_rank,
                attn.window_size,
                kv_cache.shape[1],
                block_table.shape[1],
                mlp.n_routed_experts,
                mlp.experts.top_k,
                shared_inter,
                routed_w2.shape[-1],
                layer.hc_sinkhorn_iters,
                1 if e_bias is not None else 0,
                1 if scoring == "sigmoid" else 0,
                1 if getattr(experts, "renormalize", False) else 0,
                1 if attn_sink is not None else 0,
                0,
                0,
                0,
            ],
            dtype=torch.int64,
            device=residual.device,
        )
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
            device=residual.device,
        )

        res_out = torch.empty_like(residual)
        pre_mix_out = torch.empty_like(pre_mix)
        torch.ops._C_ascend.dsv41_mono_layer(
            residual.contiguous(),
            pre_mix.contiguous(),
            positions.contiguous(),
            cos_f32,
            sin_f32,
            slot_mapping,
            block_table,
            seq_lens,
            res_out,
            pre_mix_out,
            layer.input_layernorm.weight.contiguous(),
            layer.hc_attn_fn.contiguous(),
            layer.hc_attn_scale.contiguous(),
            layer.hc_attn_base.contiguous(),
            attn.wq_a.weight.contiguous(),
            attn.q_norm.weight.contiguous(),
            attn.wq_b.weight.contiguous(),
            attn.wkv.weight.contiguous(),
            attn.kv_norm.weight.contiguous(),
            attn_sink.contiguous() if attn_sink is not None else residual.new_empty(1, dtype=torch.float32),
            wo_a,
            attn.wo_b.weight.contiguous(),
            layer.post_attention_layernorm.weight.contiguous(),
            layer.hc_ffn_fn.contiguous(),
            layer.hc_ffn_scale.contiguous(),
            layer.hc_ffn_base.contiguous(),
            mlp.gate.weight.contiguous(),
            e_bias.contiguous() if e_bias is not None else residual.new_empty(1, dtype=torch.float32),
            shared_w13.contiguous(),
            shared_w2.contiguous(),
            routed_w13.contiguous(),
            routed_w2.contiguous(),
            kv_cache,
            ws,
            meta_i64,
            meta_f32,
        )
        return res_out, pre_mix_out
