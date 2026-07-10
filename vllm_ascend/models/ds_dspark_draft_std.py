# SPDX-License-Identifier: Apache-2.0
"""Standard-PAGED DSpark speculative draft model for vllm-ascend v0.23.

Original implementation. The per-stage draft MATH (MHC hyper-clone, MoE, hc_head,
Markov head, the MLA q/kv projections, q/k RoPE, grouped wo_a/wo_b tail) is the
SAME set already validated bit-identical against DeepSeek's reference
(ds_dspark_math / ds_dspark_draft, offline cos = 1.0) and is reused here.

The genuinely NEW surface vs the eager original is ONLY the attention data path:
  (1) the draft attention now writes context-K and this-block-K into a real vLLM
      PAGED SWA KV-cache (slot_mapping -> dsa_kv_compress_scatter) instead of a
      private register_buffer ring;
  (2) it reads context-K/block-K back with a PURE-TORCH paged gather driven by a
      non-causal dspark_swa_indices slot table (root cause B) and runs the user's
      VALIDATED ds_dspark_math.sparse_attn -- NO fused AscendC kernel, NO .so;
  (3) inheriting DeepseekV4Attention gives each draft layer a registered
      dsa_attn.swa_cache_layer -> the draft owns ONE paged kv-cache group per
      layer (groups != 0), the structural switch vs the eager attention-free path;
  (4) cad.positions drives q/k RoPE (iron-law 1, enforced by the proposer).

Draft weights = rbf16 (dequantized rotated w8a8, dspark_mtp_dequantized_to_bf16):
QuaRot is a NO-OP for rbf16 (no quarot.safetensors); only w8a8 rotates.
K == V (MQA single shared kv head). attn_sink enters the softmax DENOMINATOR
only, unscaled. Ascend RoPE is NOT in-place -> every _apply_dsv4_rope* return
value is caught (iron-law 2).
"""
from __future__ import annotations

import re
import typing
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer, DeepseekV4Attention, _apply_dsv4_rope,
    _apply_dsv4_rope_tail, _grouped_wo_a_projection, _hc_head_torch,
    _linear_output, _make_deepseek_v4_expert_params_mapping,
    _wo_a_weight_for_eager_projection)
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

from vllm_ascend.attention.ds_dspark_meta_std import build_dspark_swa_indices
from vllm_ascend.ops.ds_dspark_attention_std import paged_gather_attend


def _draft_quant_config(vllm_config: VllmConfig):
    # rbf16 draft is bf16 (dspark_mtp_dequantized_to_bf16); no quant on draft linears.
    cfg = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(cfg, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _resolve_per_layer(mapping, prefix):
    """slot_mapping / block_table arrive per-layer (dict keyed by the layer's
    swa_cache_layer.prefix) on the paged path, or as a single tensor. Pick this
    layer's view."""
    if isinstance(mapping, dict):
        return mapping.get(prefix)
    return mapping


# ===========================================================================
# HC (hyper-clone) -- REUSED VERBATIM from the validated ds_dspark_draft.
# ===========================================================================
def _sinkhorn(comb, iters, eps):
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def hc_pre(x_hc, fn, scale, base, rms_eps, hc_eps, hc_mult, iters, post_alpha=2.0):
    lead = x_hc.shape[:-2]
    hc, d = x_hc.shape[-2], x_hc.shape[-1]
    flat = x_hc.reshape(*lead, hc * d).float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    scale = scale.float(); base = base.float()
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = post_alpha * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).view(*lead, hc, hc)
    comb = _sinkhorn(comb, iters, hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * flat.view(*lead, hc, d), dim=-2)
    return y.to(x_hc.dtype), post, comb


def hc_post(xc, residual, post, comb):
    y = post.unsqueeze(-1) * xc.unsqueeze(-2) \
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
    return y.type_as(xc)


# ===========================================================================
# Paged DSpark draft attention (Component B).  Inherits DeepseekV4Attention so
# dsa_attn.swa_cache_layer registers a real paged kv-cache group per draft layer.
# ===========================================================================
def _unwrap_kv(kv_cache):
    while isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 1:
        kv_cache = kv_cache[0]
    return kv_cache


def _try_forward_context():
    try:
        return get_forward_context()
    except Exception:
        return None


class DeepseekV4DSparkPagedAttention(DeepseekV4Attention):
    """DSpark draft attention over a PAGED SWA KV-cache.

    TODO(serve-verify): confirm the base DeepseekV4Attention.__init__ accepts the
    same (vllm_config, config, ...) call the decoder layer makes AND exposes the
    validated projection set (wq_a / q_norm / q_norm_without_weight / wq_b / wkv /
    kv_norm / wo_a / wo_b / attn_sink / rotary_emb) plus dsa_attn.swa_cache_layer.
    If any base attribute name differs, bind it here before first serve.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)          # builds projections + dsa_attn/swa_cache_layer

        def _ctor(name, pos, default=None):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if len(args) > pos else default
        vllm_config = _ctor("vllm_config", 0)
        config = _ctor("config", 1)
        max_pos = _ctor("max_position_embeddings", 2, 0)
        prefix = _ctor("prefix", 5, "")
        if config is None:
            # decoder layer passes config positionally in some builds
            config = getattr(self, "config", None)

        # ---- Fix 1 (math #1+#2): do NOT inherit the base DeepseekV4Attention
        # rotary_emb / scale -- those are built from the TARGET config (wrong
        # compress rope_theta + mscale-carrying scale for the draft). OVERRIDE both
        # to EXACTLY the proven eager draft's construction so q_pe/k_pe RoPE and the
        # score scaling match the validated ds_dspark_math and #11196's proven draft
        # (removes the unverifiable-base-class risk).
        if config is not None and vllm_config is not None:
            rp = config.rope_parameters
            rp["rope_theta"] = config.rope_theta      # plain theta (compress_ratio=1)
            self.rotary_emb = ComplexExpRotaryEmbedding(
                vllm_config=vllm_config, layername=f"{prefix}.attn",
                head_size=self.rope_head_dim, rotary_dim=self.rope_head_dim,
                max_position_embeddings=max_pos, is_neox_style=False,
                scaling_factor=rp["factor"], base=rp["rope_theta"],
                beta_fast=rp["beta_fast"], beta_slow=rp["beta_slow"],
                rope_groups=["default"])
        # full 512 head_dim, NO mscale -> matches sparse_attn's softmax_scale.
        self.scale = self.head_dim ** -0.5

        block = getattr(config, "dspark_block_size", None) if config is not None else None
        self.block_size = int(block) if block else int(getattr(self, "block_size", 5))
        # DSpark forces no DSA compression on the draft -> plain rope theta.
        self.compress_ratio = 1
        if getattr(self, "dsa_attn", None) is not None:
            self.dsa_attn.compress_ratio = 1
        self._draft_index_width = None   # lazily aligned once window/block known
        # NOTE: no private _ctx_k/_ctx_valid ring -- context K/V lives in the
        # paged swa_cache_layer.kv_cache.

    # ---- shared MLA kv projection (REUSE validated math; K==V, single kv head) ----
    def _project_shared_kv(self, hidden_states, positions):
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        # iron-law 2: context/block K gets RoPE at its positions; CATCH the return.
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()

    # ---- paged write: scatter shared_kv into the SWA kv_cache at slot_mapping ----
    def _store_paged_kv(self, shared_kv, slot_mapping, is_query=False):
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        from vllm_ascend.device.device_op import DeviceOperator   # TODO(serve-verify): import path
        swa_layer = self.dsa_attn.swa_cache_layer
        kv_cache = _unwrap_kv(getattr(swa_layer, "kv_cache", None))
        if kv_cache is None:
            return
        slot_mapping = slot_mapping.to(device=shared_kv.device, dtype=torch.int32)
        # Fix 2 (BLOCKING): ONLY the per-block QUERY write may be truncated by the
        # cudagraph/profiling num_actual_tokens padding. The CONTEXT prewrite
        # (is_query=False, from precompute_and_store_context_kv) must scatter EVERY
        # one of its _dflash_num_context tokens -- truncating it to num_actual_tokens
        # (the QUERY count, which differs) drops context K/V, leaves context slots
        # empty, and the non-causal gather then reads garbage -> ~0% accept. Every
        # position in [start_pos, seq_len) is populated here before any read.
        if is_query:
            fctx = _try_forward_context()
            n_act = getattr(fctx, "num_actual_tokens", None)
            if n_act is not None and n_act < slot_mapping.shape[0]:
                shared_kv, slot_mapping = shared_kv[:n_act], slot_mapping[:n_act]
        # This round is draft-eager (no FULL cudagraph): drop invalid (-1) rows.
        valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, -1)
        if not bool(valid.any()):
            return
        if not bool(valid.all()):
            shared_kv, slot_mapping = shared_kv[valid], slot_mapping[valid]
        if slot_mapping.ndim == 1:
            slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, swa_layer.block_size)
        DeviceOperator.dsa_kv_compress_scatter(kv_cache, shared_kv, slot_mapping)
        # PTA reads the raw cache right after scatter -> sync the stream.
        try:
            torch.npu.synchronize(shared_kv.device)   # TODO(serve-verify): sync primitive
        except Exception:
            pass

    # ---- query slot_mapping: pos -> paged slot via block_table (per req) ----
    def _query_slot_mapping(self, positions, slot_mapping, block_table, token_to_req):
        # If the proposer already supplied the paged query slot_mapping, use it.
        if slot_mapping is not None:
            return slot_mapping.to(device=positions.device, dtype=torch.int32)
        if block_table is None:
            return None
        cbs = int(self.dsa_attn.swa_cache_layer.block_size)
        pos = positions.to(torch.long)
        out = torch.full((positions.shape[0],), -1, dtype=torch.int32, device=positions.device)
        if token_to_req is None:
            return out
        req = token_to_req[:positions.numel()].to(positions.device, torch.long)
        valid = (req >= 0) & (req < block_table.shape[0])
        req = req.clamp(0, block_table.shape[0] - 1)
        bn = (pos // cbs).clamp(0, block_table.shape[1] - 1)
        flat_bt = block_table.to(positions.device, torch.long).reshape(-1)
        block_ids = flat_bt.index_select(0, req * block_table.shape[1] + bn)
        out = (block_ids * cbs + pos % cbs).to(torch.int32)
        out.masked_fill_(~valid, -1)
        return out

    def _index_width(self):
        if self._draft_index_width is None:
            from vllm_ascend.attention.ds_dspark_meta_std import aligned_index_width
            self._draft_index_width = aligned_index_width(int(self.window_size), self.block_size)
        return self._draft_index_width

    # ---- swa slot table: prefer proposer metadata, else build inline ----
    def _swa_metadata_or_build(self, positions, slot_mapping, block_table,
                               query_start_loc, seq_lens, token_to_req):
        md = self._lookup_draft_metadata()
        if md is not None:
            return md
        if query_start_loc is None or seq_lens is None or block_table is None:
            return None, None
        return build_dspark_swa_indices(
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            token_to_req_indices=token_to_req,
            block_size=self.block_size,
            window_size=int(self.window_size),
            cache_block_size=int(self.dsa_attn.swa_cache_layer.block_size),
            num_query_tokens=int(positions.numel()),
            index_width=self._index_width(),
            device=positions.device)

    def _lookup_draft_metadata(self):
        # TODO(serve-verify): read forward_context.draft_attn_metadatas keyed by
        # self.dsa_attn.swa_cache_layer.prefix; return (dspark_swa_indices,
        # dspark_swa_lens) from its decode/prefill sub-struct if present.
        fctx = _try_forward_context()
        mds = getattr(fctx, "draft_attn_metadatas", None)
        if not mds:
            return None
        prefix = getattr(self.dsa_attn.swa_cache_layer, "prefix", None)
        per_layer = mds[0] if isinstance(mds, (list, tuple)) else mds
        md = per_layer.get(prefix) if isinstance(per_layer, dict) else None
        sub = getattr(md, "decode", None) or getattr(md, "prefill", None) if md is not None else None
        if sub is not None and getattr(sub, "dspark_swa_indices", None) is not None:
            return sub.dspark_swa_indices, sub.dspark_swa_lens
        return None

    # ---- pure-torch paged gather + validated sparse_attn (NO kernel, NO .so) ----
    def _paged_attend(self, q, positions, q_slots, block_table,
                      query_start_loc, seq_lens, token_to_req):
        swa_layer = self.dsa_attn.swa_cache_layer
        kv_cache = _unwrap_kv(getattr(swa_layer, "kv_cache", None))
        if kv_cache is None or block_table is None:
            return None
        swa_i, swa_l = self._swa_metadata_or_build(
            positions, q_slots, block_table, query_start_loc, seq_lens, token_to_req)
        if swa_i is None:
            return None
        cbs = int(swa_layer.block_size)
        # Reuses ds_dspark_math.sparse_attn verbatim (via ops.paged_gather_attend):
        # per block, gather the shared K==V slot list from the flat paged cache and
        # run non-causal softmax with attn_sink in the denominator only.
        return paged_gather_attend(
            q, kv_cache, swa_i, swa_l, self.block_size, cbs,
            self.attn_sink[:self.n_local_heads], float(self.head_dim ** -0.5))

    # ---- context-KV prewrite (iron-law 2): scatter context K/V into paged cache ----
    def precompute_context_kv(self, main_x, positions, request_slots=None,
                              context_slot_mapping=None):
        if positions is None or positions.numel() == 0:
            return
        shared_kv = self._project_shared_kv(main_x, positions)   # RoPE at context positions, caught
        self._store_paged_kv(shared_kv, context_slot_mapping)

    def reset_request_slots(self, request_slots):
        # Fix 4: guard against stale cross-request K/V from RECYCLED paged blocks
        # leaking through the non-causal gather. The paged SWA blocks are recycled
        # by the runner's block manager, so explicitly zeroing a freed request's OLD
        # paged rows at reassignment is UNSAFE -- by the time a ring slot is reused,
        # the previous occupant's physical blocks may already back a DIFFERENT live
        # request, and zeroing them would corrupt that request. Correctness is
        # therefore guaranteed structurally: every slot the gather reads is
        # (RE)WRITTEN this same round -- precompute_and_store_context_kv scatters
        # ALL context tokens (Fix 2) and the per-block query write scatters ALL
        # query tokens, while dspark_swa_indices references ONLY those written
        # positions (out-of-window / unallocated slots are padded -1 and dropped in
        # paged_gather_attend), so a recycled block's stale rows are never gathered.
        # Recorded only for a follow-up on-device assertion of that invariant.
        self._reset_request_slots = (None if request_slots is None
                                     else request_slots.reshape(-1).to(torch.long))
        return

    # ---- main forward ----
    def forward(self, positions, hidden_states, llama_4_scaling=None,
                request_slots=None, slot_mapping=None, block_table=None,
                dspark_query_start_loc=None, dspark_seq_lens=None,
                dspark_token_to_req_indices=None, **kw):
        del llama_4_scaling, request_slots
        T = hidden_states.shape[0]
        nlh, hd, rd = self.n_local_heads, self.head_dim, self.rope_head_dim
        qr = self.q_norm(_linear_output(self.wq_a, hidden_states))
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        q = _linear_output(self.wq_b, qr).view(T, nlh, hd)
        q = self.q_norm_without_weight(q)
        q_nope, q_pe = q.split([self.nope_head_dim, rd], dim=-1)
        k_nope, k_pe = kv.split([self.nope_head_dim, rd], dim=-1)
        # iron-law 1: positions == cad.positions; iron-law 2: catch RoPE returns.
        q_pe = _apply_dsv4_rope(self.rotary_emb, positions, q_pe)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        shared_kv = torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, hd).contiguous()  # K==V
        q = torch.cat([q_nope, q_pe], dim=-1)

        q_slots = self._query_slot_mapping(positions, slot_mapping, block_table,
                                           dspark_token_to_req_indices)
        self._store_paged_kv(shared_kv, q_slots, is_query=True)  # write this block's K/V first
        attn_out = self._paged_attend(q, positions, q_slots, block_table,
                                      dspark_query_start_loc, dspark_seq_lens,
                                      dspark_token_to_req_indices)
        if attn_out is None:
            # Fix 3: memory-profiling / dummy_run cannot supply a real paged cache
            # (block_table / kv_cache / swa_indices are absent). TOLERATE it -- return
            # zeros of the wo_b output shape [T, hidden] instead of raising, so the
            # proposer's dummy_run can trace this layer during profiling.
            return torch.zeros(T, self.dim, dtype=hidden_states.dtype,
                               device=hidden_states.device)

        attn_out = _apply_dsv4_rope_tail(self.rotary_emb, positions, attn_out, inverse=True)  # catch
        group_dim = nlh * hd // self.n_local_groups
        attn_out = attn_out.reshape(T, self.n_local_groups, group_dim)
        wo_a = _wo_a_weight_for_eager_projection(self.wo_a.weight, self.n_local_groups,
                                                 self.o_lora_rank, group_dim)
        z = _grouped_wo_a_projection(attn_out, wo_a).flatten(1)
        return _linear_output(self.wo_b, z)


# ===========================================================================
# DSpark decoder layer: MHC-pre -> attn -> MHC-post ; MHC-pre -> ffn -> MHC-post.
# Body REUSED from the validated draft; only self_attn call passes paged args.
# ===========================================================================
class DeepseekV4DSparkDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config, prefix):
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        super().__init__(vllm_config=vllm_config, prefix=prefix, config=cfg,
                         is_draft_layer=True, attn_cls=DeepseekV4DSparkPagedAttention,
                         quant_config_override=_draft_quant_config(vllm_config),
                         use_quant_config_override=True)
        self.post_alpha = 2.0

    def _pre(self, h, fn, scale, base):
        return hc_pre(h, fn, scale, base, self.norm_eps, self.hc_eps,
                      self.hc_mult, self.hc_sinkhorn_iters, self.post_alpha)

    def _attn_prefix(self):
        return getattr(self.self_attn.dsa_attn.swa_cache_layer, "prefix", None)

    def forward(self, positions, h, slot_mapping=None, block_table=None,
                dspark_query_start_loc=None, dspark_seq_lens=None,
                dspark_token_to_req_indices=None, **kw):
        prefix = self._attn_prefix()
        sm = _resolve_per_layer(slot_mapping, prefix)
        bt = _resolve_per_layer(block_table, prefix)
        # ---- attn sub-block ----
        r = h
        xc, post, comb = self._pre(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        xc = self.input_layernorm(xc)
        xc = self.self_attn(positions, xc, None, slot_mapping=sm, block_table=bt,
                            dspark_query_start_loc=dspark_query_start_loc,
                            dspark_seq_lens=dspark_seq_lens,
                            dspark_token_to_req_indices=dspark_token_to_req_indices)
        h = hc_post(xc, r, post, comb)
        # ---- ffn sub-block ----
        r = h
        xc, post, comb = self._pre(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        xc = self.post_attention_layernorm(xc)
        xc = self.mlp(xc)
        h = hc_post(xc, r, post, comb)
        return h


class DSparkMarkovHead(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(config.vocab_size, config.dspark_markov_rank,
                                                prefix=f"{prefix}.markov_w1")
        self.markov_w2 = ParallelLMHead(config.vocab_size, config.dspark_markov_rank,
                                        org_num_embeddings=config.vocab_size,
                                        prefix=f"{prefix}.markov_w2")
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed(self, ids):
        return self.markov_w1(ids)

    def bias(self, emb):
        return self.logits_processor(self.markov_w2, emb)


# ===========================================================================
# DSpark draft model (paged).
# ===========================================================================
class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = cfg
        self.hc_mult = cfg.hc_mult
        self.block_size = int(cfg.dspark_block_size)
        self.target_layer_ids = list(cfg.dspark_target_layer_ids)
        self.num_dspark_layers = int(cfg.n_mtp_layers)
        self.start = cfg.num_hidden_layers
        self.last = self.start + self.num_dspark_layers - 1
        # QuaRot: the w8a8 TARGET is QuaRot-rotated, so the target hidden (aux) it
        # emits is in the ROTATED basis while this draft's main_proj is stored
        # UNROTATED. precompute must un-rotate aux (@ Q^T) into the main_proj basis,
        # then re-rotate the projected main_x (@ Q) into the draft cv_wkv basis --
        # UNLESS this is an unrotated-bf16 draft (DSPARK_QUAROT_AUXONLY set), which
        # skips only the re-rotation. rbf16 IS a rotated draft (its serve does NOT
        # set AUXONLY) -> apply BOTH. Ref: deepseek_v4_dspark.py:1349-1362.
        # Without the un-rotation the draft context is garbage (AL ~= 1.0).
        self._dequant_bf16 = bool(getattr(cfg, "dspark_mtp_dequantized_to_bf16", False))
        self._quarot_Q = None
        try:
            import os as _os_q
            _qp = _os_q.path.join(
                str(vllm_config.speculative_config.draft_model_config.model),
                "optional", "quarot.safetensors")
            if _os_q.path.exists(_qp):
                from safetensors.torch import load_file as _qload
                self._quarot_Q = _qload(_qp)["global_rotation"].to(torch.bfloat16)
        except Exception:
            self._quarot_Q = None

        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size,
                                                   quant_config=_draft_quant_config(vllm_config),
                                                   prefix=maybe_prefix(prefix, "embed_tokens"))
        self.layers = nn.ModuleDict({
            str(self.start + i): DeepseekV4DSparkDecoderLayer(
                vllm_config, prefix=maybe_prefix(prefix, f"layers.{self.start + i}"))
            for i in range(self.num_dspark_layers)})

        first = self.layers[str(self.start)]
        self.main_proj = ReplicatedLinear(cfg.hidden_size * len(self.target_layer_ids),
                                          cfg.hidden_size, bias=False, return_bias=False,
                                          quant_config=_draft_quant_config(vllm_config),
                                          prefix=maybe_prefix(prefix, f"layers.{self.start}.main_proj"))
        self.main_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        first.main_proj = self.main_proj
        first.main_norm = self.main_norm

        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(cfg, maybe_prefix(prefix, f"layers.{self.last}.markov_head"))
        hc_dim = self.hc_mult * cfg.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        last = self.layers[str(self.last)]
        last.norm = self.norm
        last.markov_head = self.markov_head
        last.hc_head_fn = self.hc_head_fn
        last.hc_head_base = self.hc_head_base
        last.hc_head_scale = self.hc_head_scale

    # ---- paged kv-cache group hook: one prefix per draft layer's SWA cache ----
    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [self.layers[str(self.start + i)].self_attn.dsa_attn.swa_cache_layer.prefix
                for i in range(self.num_dspark_layers)]

    # ---- QuaRot rotation (ref deepseek_v4_dspark.py:1349-1362) ----
    def _quarot_unrotate(self, states):
        # Un-rotate the (rotated) target hidden into the UNROTATED main_proj basis:
        # per target-layer hidden block (width=hidden_size), block @ Q^T. Always
        # applied when Q is loaded (both rbf16 and w8a8) -- reference step 1.
        if self._quarot_Q is None:
            return states
        _q = self._quarot_Q.to(device=states.device, dtype=states.dtype)
        h = int(self.config.hidden_size)
        n = len(self.target_layer_ids)
        return torch.cat([states[:, k * h:(k + 1) * h] @ _q.t() for k in range(n)], dim=-1)

    def _quarot_rerotate(self, main_x):
        # Re-rotate main_x into the draft cv_wkv (rotated) basis: main_x @ Q. Skipped
        # only for an unrotated-bf16 draft (DSPARK_QUAROT_AUXONLY). rbf16 keeps it.
        if self._quarot_Q is None or __import__("os").environ.get("DSPARK_QUAROT_AUXONLY"):
            return main_x
        _q = self._quarot_Q.to(device=main_x.device, dtype=main_x.dtype)
        return main_x @ _q

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def _combine_aux(self, aux):
        aux = self._quarot_unrotate(aux)
        main_x = self.main_norm(_linear_output(self.main_proj, aux))
        return self._quarot_rerotate(main_x)

    def precompute_and_store_context_kv(self, context_states, context_positions,
                                        context_slot_mapping=None, context_request_slots=None):
        """Paged context-KV prewrite (iron-law 2). context_slot_mapping is a
        per-layer dict {swa_cache_layer.prefix: tensor} (or a single tensor)."""
        main_x = self._combine_aux(context_states)
        for i in range(self.num_dspark_layers):
            layer = self.layers[str(self.start + i)]
            prefix = getattr(layer.self_attn.dsa_attn.swa_cache_layer, "prefix", None)
            per_layer_sm = _resolve_per_layer(context_slot_mapping, prefix)
            layer.self_attn.precompute_context_kv(
                main_x, context_positions, request_slots=context_request_slots,
                context_slot_mapping=per_layer_sm)

    def reset_request_slots(self, slots):
        for i in range(self.num_dspark_layers):
            self.layers[str(self.start + i)].self_attn.reset_request_slots(slots)

    def forward(self, input_ids, positions, inputs_embeds=None, request_slots=None,
                slot_mapping=None, block_table=None, dspark_query_start_loc=None,
                dspark_seq_lens=None, dspark_token_to_req_indices=None, **kw):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        h = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)   # hc expand [T,hc,d]
        for i in range(self.num_dspark_layers):
            h = self.layers[str(self.start + i)](
                positions, h, slot_mapping=slot_mapping, block_table=block_table,
                dspark_query_start_loc=dspark_query_start_loc,
                dspark_seq_lens=dspark_seq_lens,
                dspark_token_to_req_indices=dspark_token_to_req_indices)
        out = _hc_head_torch(h, self.hc_head_fn, self.hc_head_scale,
                             self.hc_head_base, self.config.rms_norm_eps, self.config.hc_eps)
        return out

    def markov_embed(self, ids):
        return self.markov_head.embed(ids)

    def markov_bias(self, emb):
        return self.markov_head.bias(emb)


# ===========================================================================
# ForCausalLM wrapper -- REUSED (embed/lm_head share + rbf16 weight loading).
# ===========================================================================
class DeepseekV4DSparkForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = cfg
        self.model = DeepseekV4DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size,
                                      quant_config=None, prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(cfg.vocab_size)
        # rbf16 draft ckpt ships neither embed nor head -> share the target's.
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        self.draft_id_to_target_id = None      # draft_vocab == target_vocab (identity)

    @property
    def layers(self):
        # Framework + our own group registration access `.layers` on the
        # ForCausalLM; the decoder layers actually live on the inner model.
        return self.model.layers

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None, request_slots=None,
                slot_mapping=None, block_table=None, dspark_query_start_loc=None,
                dspark_seq_lens=None, dspark_token_to_req_indices=None, **kw):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds,
                          request_slots=request_slots, slot_mapping=slot_mapping,
                          block_table=block_table,
                          dspark_query_start_loc=dspark_query_start_loc,
                          dspark_seq_lens=dspark_seq_lens,
                          dspark_token_to_req_indices=dspark_token_to_req_indices)

    def precompute_and_store_context_kv(self, context_states, context_positions,
                                        context_slot_mapping=None, context_request_slots=None):
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping, context_request_slots)

    def reset_request_slots(self, slots):
        self.model.reset_request_slots(slots)

    def markov_embed(self, ids):
        return self.model.markov_embed(ids)

    def markov_bias(self, emb):
        return self.model.markov_bias(emb)

    def compute_logits(self, head_hidden):
        return self.logits_processor(self.lm_head, self.model.norm(head_hidden))

    # ---- weight load -- REUSED VERBATIM from the validated eager draft ----
    def _remap(self, name: str):
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        layer = self.config.num_hidden_layers + int(m.group(1))
        rest = m.group(2)
        if rest.startswith("confidence_head."):
            return None
        n = f"model.layers.{layer}.{rest}"
        for a, b in ((".attn.", ".self_attn."), (".attn_norm.", ".input_layernorm."),
                     (".ffn_norm.", ".post_attention_layernorm."), (".ffn.", ".mlp."),
                     (".w1.", ".gate_proj."), (".w2.", ".down_proj."), (".w3.", ".up_proj."),
                     ("mlp.gate.bias", "mlp.gate.e_score_correction_bias")):
            n = n.replace(a, b)
        return n

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [("mlp.gate_up_proj", "mlp.gate_proj", 0), ("mlp.gate_up_proj", "mlp.up_proj", 1),
                   ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
                   ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1)]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        hpr = self.config.num_attention_heads // tp_size
        h0, h1 = tp_rank * hpr, tp_rank * hpr + hpr
        # BUGFIX: DeepseekV4MoE has NO get_expert_mapping method, so the old
        # hasattr(...) fallback silently yielded [] -> expert weights were never
        # loaded -> uninitialized MoE experts -> NaN on the first forward -> 0%
        # acceptance. Build the mapping at the model level, exactly like #11196.
        expert_mapping = _make_deepseek_v4_expert_params_mapping(
            self.model, num_experts=int(self.config.n_routed_experts))
        esuffix = ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"

        for name, w in weights:
            if name == "embed.weight":
                p = params["model.embed_tokens.weight"]
                getattr(p, "weight_loader", default_weight_loader)(p, w); loaded.add("model.embed_tokens.weight"); continue
            if name == "head.weight":
                p = params["lm_head.weight"]
                getattr(p, "weight_loader", default_weight_loader)(p, w); loaded.add("lm_head.weight"); continue
            n = self._remap(name)
            if n is None:
                continue
            if n.startswith(f"model.layers.{self.model.last}.hc_head_"):
                cn = n.replace(f"model.layers.{self.model.last}.", "model.", 1)
                if cn in params:
                    n = cn
            if n.endswith(".scale"):
                suf = esuffix if re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$", n) else ".weight_scale"
                n = n.removesuffix(".scale") + suf
                if n not in params:
                    continue
            for pn, wn, sid in stacked:
                if ".experts." in n or f".{wn}." not in n:
                    continue
                mp = n.replace(wn, pn)
                if mp in params:
                    params[mp].weight_loader(params[mp], w, sid); loaded.add(mp)
                break
            else:
                if ".experts." in n:
                    for pn, wn, eid, sid in expert_mapping:
                        if wn not in n:
                            continue
                        mp = n.replace(wn, pn)
                        if mp not in params:
                            continue
                        ok = typing.cast(typing.Callable, params[mp].weight_loader)(
                            params[mp], w, mp, shard_id=sid, expert_id=eid, return_success=True)
                        if ok:
                            loaded.add(mp); break
                    continue
                if "attn_sink" in n:
                    if n in params:
                        with torch.no_grad():
                            params[n][: (h1 - h0)].copy_(w[h0:h1])
                        loaded.add(n)
                    continue
                if n in params:
                    getattr(params[n], "weight_loader", default_weight_loader)(params[n], w); loaded.add(n)
        return loaded
