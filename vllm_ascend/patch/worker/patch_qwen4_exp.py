#
# Ascend adaptation for Qwen3.8 Flash-Next (model_type "qwen4_exp").
#
# The upstream model definition is device-neutral apart from a few points in its
# QSA (Qwen Sparse Attention) attention implementation and in the PLE n-gram
# embedding.  Patch exactly those and reuse everything else, so checkpoint
# mapping and model behaviour stay identical to upstream.
#
#  1. QSA construction is gated on CUDA FlashAttention being importable.  The
#     QSA forward never calls FlashAttention -- it runs
#     `qsa_sparse_paged_attention` (Triton) over the paged BF16 K/V cache.  The
#     class derives from FlashAttentionImpl only for the metadata plumbing, and
#     that base __init__ runs fine here, so keep it and drop just the
#     availability assertion.
#
#  2. The KV-cache write inherited from FlashAttentionImpl is CUDA's
#     `reshape_and_cache_flash`.  Replace it with a stride-safe scatter that
#     reproduces that kernel's skip-on-PAD_SLOT_ID behaviour.
#
#  3. Two Triton kernels do not lower on triton-ascend; upstream carries torch
#     equivalents for both, so select those.
#
#  4. The PLE n-gram lookup is hidden behind a custom op so torch.compile does
#     not trace into it (see `_opaque_ngram_embedding`).
#
#  5. The model's own ModelState is re-based on the Ascend hybrid state so
#     attention metadata is built the way this platform's graph replay expects
#     (see `_use_ascend_model_state`).
#
#  6. The Gated DeltaNet output gate.  This model sets
#     `output_gate_type = "sigmoid"` (out = w * rmsnorm(x) * sigmoid(z)); every
#     earlier GDN model used SiLU.  vLLM passes that through RMSNormGated's
#     `activation`, but the Ascend replacement drops the argument and its
#     Triton kernel hard-codes z * sigmoid(z), so all 36 GDN layers gated with
#     SiLU.  Route a sigmoid-gated norm through a torch implementation instead
#     (see `_honour_gated_norm_activation`).
#
#  7. The decode recurrence.  vllm-ascend routes single-token GDN steps to a
#     custom AscendC kernel (npu_recurrent_gated_delta_rule).  In place its
#     output and state update disagree with the gated delta rule (checked on
#     the real tensors, and against the HF reference which the prefill path
#     matches).  The SSM state it is handed is a strided view of the hybrid
#     KV page (see 8), which the kernel addresses as a dense array.  Run the
#     recurrence in torch, indexing the state directly, for the
#     non-speculative decode batch (see `_own_decode_recurrence`).
#
#  8. The conv state.  It lives inside the hybrid KV page too
#     ("[(kv_padding), conv]"), so the layer's conv_state is a strided view
#     (slot stride = page size, not state_len * dim).  The custom CausalConv1d
#     kernel addresses it as dense: in place the prefill call never wrote the
#     state and the decode call wrote another slot, while the same calls on a
#     contiguous tensor are right.  Run the kernel on a contiguous gather of
#     the addressed slots and scatter back (see `_own_conv_state_addressing`).
#
# This file is the source of truth; ours/patches/apply_0024.py installs it into
# vllm_ascend/patch/worker/.
#

import torch
import torch_npu

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _ascend_qsa_init(self, *args, **kwargs) -> None:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    FlashAttentionImpl.__init__(self, *args, **kwargs)
    if self.dcp_world_size != 1:
        raise NotImplementedError("Qwen4Exp QSA does not support decode context parallelism")
    if self.kv_cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
    self.supports_quant_query_input = False


def _ascend_qsa_kv_cache_update(
    self,
    layer,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    del layer, args, kwargs
    # Same view the QSA forward reads back, so writer and reader agree by
    # construction rather than by a separately maintained layout assumption.
    #
    # NOTE: these two are *strided views* of one packed allocation, not the
    # separate contiguous K and V tensors vllm-ascend hands to
    # `npu_scatter_pa_kv_cache` elsewhere. Feeding that fused in-place op a
    # non-contiguous destination hangs the device (vector core timeout) once
    # launches are asynchronous, so scatter with indexing instead: it honours
    # the destination strides, keeps shapes static and does no host
    # synchronisation, and so stays capturable.
    key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
    rows = slot_mapping.shape[0]
    page_size = key_cache.shape[1]
    slots = slot_mapping[:rows].to(torch.int64)

    # slot_mapping is padded up to num_tokens_after_padding with PAD_SLOT_ID
    # (-1), and a dummy / graph-capture run is -1 all the way through. Plain
    # index arithmetic turns -1 into (block -1, offset page_size - 1), i.e. the
    # LAST REAL SLOT of the cache: a silent overwrite of whatever request owns
    # it. `reshape_and_cache_flash` simply skips those rows; an indexed scatter
    # cannot do that without a data-dependent shape, which is illegal under
    # graph capture.
    #
    # Instead point every padded row at a row a real token is already writing
    # ("the donor"), let them all land on that one slot, then rewrite the donor
    # slot with the donor's own value. Shapes stay static, nothing synchronises,
    # and the only slot touched twice ends up with the correct bytes.
    #
    # If no row is valid at all -- dummy and capture runs -- the donor clamps to
    # the last row and the whole batch lands on slot 0. That happens before any
    # request has stored KV, and a request always writes a slot before reading
    # it, so the garbage never survives to be read.
    #
    # Everything here stays index_select-shaped: `t[zero_dim_tensor]` is read as
    # a *scalar* index, which makes torch call .item() on it and synchronise the
    # device -- illegal mid graph capture, and it turns any earlier async kernel
    # fault into a crash at this line.
    valid = slots >= 0
    positions = torch.arange(rows, device=slots.device)
    donor = (
        torch.where(valid, positions, positions.new_full((), rows))
        .min()
        .clamp_max(rows - 1)
        .reshape(1)
    )
    slots = torch.where(valid, slots, torch.index_select(slots, 0, donor))
    slots.clamp_min_(0)
    block_ids = torch.div(slots, page_size, rounding_mode="floor")
    block_offsets = slots - block_ids * page_size
    donor_block = torch.index_select(block_ids, 0, donor)
    donor_offset = torch.index_select(block_offsets, 0, donor)

    key_cache[block_ids, block_offsets] = key[:rows]
    key_cache[donor_block, donor_offset] = torch.index_select(key, 0, donor)
    value_cache[block_ids, block_offsets] = value[:rows]
    value_cache[donor_block, donor_offset] = torch.index_select(value, 0, donor)


def _use_torch_qsa_metadata() -> None:
    """Build the QSA side-cache metadata with torch instead of Triton.

    The Triton metadata kernel scans per-request work counts with
    `tl.cumsum` and then indexes the running totals with `tl.gather`.
    triton-ascend's gather accepts only fp16/fp32/bf16/fp8/int8, so the int32
    offsets it is handed abort compilation:

        ValueError: Expected dtype fp16/fp32/bf16/f8E5M2/f8E4M3FN/int8,
                    but got int32

    `_build_qsa_metadata_torch` is upstream's own equivalent of that kernel --
    it is what gets selected when Triton is unavailable at all -- so switch to
    it rather than reimplementing the scan.  It also bounds-checks the
    compressor-ring request index, which the Triton branch does not.

    This is metadata construction, not a compute kernel: it runs once per
    forward, not per layer.
    """
    from vllm.models.qwen4_exp.common import qsa_cache

    qsa_cache.build_qsa_metadata = qsa_cache._build_qsa_metadata_torch


def _disable_fused_pre_indexer() -> None:
    """Use the unfused QSA pre-indexer path on Ascend.

    The fused kernel's mRoPE branch indexes a Python tuple with the induction
    variable of a `tl.static_range` loop:

        pos_rows = (pos_t, pos_h, pos_w)
        for axis in tl.static_range(3):
            cos += tl.load(base + pos_rows[axis][:, None] * stride, ...)

    triton-ascend cannot lower that -- `unsupported tensor index: constexpr[0]`
    -- so the kernel fails to compile. Upstream already carries a non-fused
    branch for configurations the fused kernel does not cover (compress ->
    GemmaRMSNorm -> RoPE -> store as separate steps); select that instead of
    hand-editing a numerics kernel.

    Revisit if the fused path is needed for throughput: unrolling the
    three-iteration loop is a mechanically equivalent rewrite.
    """
    from vllm.models.qwen4_exp.nvidia import indexer_qsa

    indexer_qsa._supports_fused_pre_indexer = lambda *args, **kwargs: False


# ---------------------------------------------------------------------------
# PLE n-gram embedding as an opaque custom op
# ---------------------------------------------------------------------------
#
# `Qwen4ExpNGramEmbedding.forward` packs the flat token stream back into a
# padded [num_reqs, seq] matrix before hashing the n-grams, so it derives
# `num_reqs` from `query_start_loc.shape[0]` and slices its scratch buffers with
# it.  Under torch.compile that specialises a dimension the runner marks
# dynamic, and compilation aborts:
#
#     ConstraintViolationError: Constraints violated (L['query_start_loc'].size()[0])!
#       - You marked L['query_start_loc'].size()[0] as dynamic but your code
#         specialized it to be a constant (2).
#
# The lookup is a gather, not a fusion opportunity, so there is nothing to gain
# from tracing into it. Hide it behind a custom op -- the same treatment
# upstream already gives the sibling short-conv step in this layer
# (`vllm::qwen4_exp_ple_short_conv`) -- and Dynamo stops at the op boundary.
#
# The op is *not* added to `splitting_ops`: it needs to be opaque, not a graph
# break. Its body is capturable (device ops only, no host synchronisation).

_ORIG_NGRAM_FORWARD = None


def qwen4_exp_ple_ngram_embed(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = _ORIG_NGRAM_FORWARD(
        layer.ple_embedding, input_ids, query_start_loc, ngram_context
    )
    # Dequantise inside the op so `output` has one dtype (the model dtype)
    # regardless of how the embedding table is stored.
    result = layer._dequantize_embeddings(result, output.dtype)
    output[: result.shape[0]].copy_(result)


def qwen4_exp_ple_ngram_embed_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _ascend_ngram_forward(
    self,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
) -> torch.Tensor:
    owner = getattr(self, "_ascend_owner", None)
    if owner is None:
        # Not owned by a registered PLE layer, so there is no forward-context
        # entry to look up; run the original implementation directly.
        return _ORIG_NGRAM_FORWARD(self, input_ids, query_start_loc, ngram_context)
    output = torch.empty(
        (input_ids.numel(), self.embedding_dim),
        dtype=self._ascend_out_dtype,
        device=input_ids.device,
    )
    torch.ops.vllm.qwen4_exp_ple_ngram_embed(
        input_ids, query_start_loc, ngram_context, output, owner
    )
    return output


def _opaque_ngram_embedding() -> None:
    global _ORIG_NGRAM_FORWARD
    from vllm.models.qwen4_exp.nvidia import ple_layer

    if _ORIG_NGRAM_FORWARD is not None:
        return
    _ORIG_NGRAM_FORWARD = ple_layer.Qwen4ExpNGramEmbedding.forward

    direct_register_custom_op(
        op_name="qwen4_exp_ple_ngram_embed",
        op_func=qwen4_exp_ple_ngram_embed,
        mutates_args=["output"],
        fake_impl=qwen4_exp_ple_ngram_embed_fake,
    )
    ple_layer.Qwen4ExpNGramEmbedding.forward = _ascend_ngram_forward

    # The op looks the layer up by name in the forward context, so the
    # embedding has to know which PLE layer owns it. The PLE layer already
    # registers itself under `prefix` in static_forward_context.
    original_init = ple_layer.Qwen4ExpPLELayer.__init__

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.ple_embedding._ascend_owner = self.prefix
        self.ple_embedding._ascend_out_dtype = self.model_config.dtype

    ple_layer.Qwen4ExpPLELayer.__init__ = _init


# ---------------------------------------------------------------------------
# Model state: use the Ascend hybrid state, not the upstream CUDA one
# ---------------------------------------------------------------------------
#
# Qwen4Exp ships its own ModelState (it carries the PLE n-gram context across
# steps) and returns it from `get_model_state_cls`, which takes priority over
# the platform default. It derives from vLLM's `MambaHybridModelState`, so on
# Ascend the model silently bypasses `AscendMambaHybridModelState` -- the only
# difference between them is `prepare_attn`, which builds Ascend's
# `AscendCommonAttentionMetadata` (a subclass of the upstream one, so upstream
# metadata builders still accept it) and keeps the result on the state.
#
# vllm-ascend's full-graph replay reads exactly that:
#
#     set_forward_context(self.model_runner.model_state.attn_metadata, ...)
#     AttributeError: 'Qwen4ExpModelState' object has no attribute 'attn_metadata'
#
# Graft the two together rather than reimplementing either: MRO
# (Qwen4ExpModelState, AscendMambaHybridModelState) keeps Qwen4Exp's input
# preparation and picks up Ascend's prepare_attn, because C3 places
# AscendMambaHybridModelState ahead of the MambaHybridModelState they share.

_ASCEND_MODEL_STATE = None


def _ascend_model_state_cls():
    global _ASCEND_MODEL_STATE
    if _ASCEND_MODEL_STATE is None:
        from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
        from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
            AscendMambaHybridModelState,
        )

        _ASCEND_MODEL_STATE = type(
            "AscendQwen4ExpModelState",
            (Qwen4ExpModelState, AscendMambaHybridModelState),
            {"__doc__": "Qwen4Exp PLE inputs with Ascend attention metadata."},
        )
    return _ASCEND_MODEL_STATE


def _use_ascend_model_state() -> None:
    """Resolve the model state lazily, at model-runner init rather than import.

    vllm_ascend.worker.v2 pulls in the whole V2 runner stack, which is not
    importable this early in the worker patch sequence.
    """
    from vllm.models.qwen4_exp.nvidia import model as _model

    for name in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        cls = getattr(_model, name, None)
        if cls is not None:
            cls.get_model_state_cls = staticmethod(_ascend_model_state_cls)


# ---------------------------------------------------------------------------
# 6. Gated RMSNorm with the activation the model asks for
# ---------------------------------------------------------------------------
_ORIG_GATED_NORM_FORWARD_OOT = None


def _torch_gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    eps: float,
    group_size: int | None,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    """out = w * rmsnorm(x) * act(z)  (norm_before_gate)  or  w * rmsnorm(x * act(z))."""
    shape = x.shape
    width = shape[-1]
    groups = width // (group_size or width)
    gate = z.float()
    gate = torch.sigmoid(gate) if activation == "sigmoid" else torch.nn.functional.silu(gate)
    xf = x.float()
    if not norm_before_gate:
        xf = xf * gate
    grouped = xf.reshape(-1, groups, width // groups)
    normed = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    out = normed.reshape(-1, width) * weight.float()
    if norm_before_gate:
        out = out * gate.reshape(-1, width)
    return out.reshape(shape).to(x.dtype)


def _ascend_gated_norm_forward_oot(self, x: torch.Tensor, z: torch.Tensor | None = None):
    activation = getattr(self, "activation", "swish")
    if z is None or activation in ("swish", "silu"):
        # The Ascend kernel is SiLU-gated; that is what it was written for.
        return _ORIG_GATED_NORM_FORWARD_OOT(self, x, z)
    return _torch_gated_rmsnorm(
        x, self.weight, z, self.eps, self.group_size, self.norm_before_gate, activation
    )


def _honour_gated_norm_activation() -> None:
    global _ORIG_GATED_NORM_FORWARD_OOT
    from vllm_ascend.ops.layernorm import AscendRMSNormGated

    if _ORIG_GATED_NORM_FORWARD_OOT is not None:
        return
    _ORIG_GATED_NORM_FORWARD_OOT = AscendRMSNormGated.forward_oot
    AscendRMSNormGated.forward_oot = _ascend_gated_norm_forward_oot


# ---------------------------------------------------------------------------
# 7. Decode recurrence in torch
# ---------------------------------------------------------------------------
_ORIG_RECURRENT_GDN = None


def _gated_delta_rule_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    *,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Gated delta rule over a decode batch, updating ``state`` in place.

    query/key: [T, Nk, D] already L2-normalised; value: [T, Nv, D];
    g (log decay, fp32) and beta: [T, Nv]; state: [slots, Nv, Dv, Dk];
    actual_seq_lengths: [R + 1] = [0, len_0, ..., len_{R-1}] (the builder's
    convention); ssm_state_indices: [R].

    graph-safe: for the one-row-per-request layout (R == T, the only one a
    full decode graph replays) everything is batched and mask-driven -- no
    host synchronisation, no data-dependent shapes.  Rows the builder padded
    (slot NULL_BLOCK_ID, length 0) or marked invalid read slot 0, write it
    back unchanged and produce zeros.
    """
    T, Nv, Dv = value.shape
    Nk = query.shape[1]
    ratio = Nv // Nk
    q = query.float().repeat_interleave(ratio, dim=1) * scale
    k = key.float().repeat_interleave(ratio, dim=1)
    v = value.float()
    decay = g.float().exp()
    b = beta.float()

    lengths = actual_seq_lengths[1:].to(torch.int64)
    slots = ssm_state_indices.to(torch.int64)
    num_reqs = min(lengths.numel(), slots.numel())
    lengths = lengths[:num_reqs]
    slots = slots[:num_reqs]
    if num_reqs == T:
        valid = (slots >= 0) & (slots < state.shape[0]) & (lengths == 1)
        safe = torch.where(valid, slots, torch.zeros_like(slots))
        S0 = state.index_select(0, safe).float()  # [T, Nv, Dv, Dk]
        S = S0 * decay.view(T, Nv, 1, 1)
        pred = torch.einsum("rhvk,rhk->rhv", S, k)
        u = (v - pred) * b.view(T, Nv, 1)
        S = S + torch.einsum("rhv,rhk->rhvk", u, k)
        out = torch.einsum("rhvk,rhk->rhv", S, q)
        S = torch.where(valid.view(T, 1, 1, 1), S, S0)
        state.index_copy_(0, safe, S.to(state.dtype))
        out = out * valid.view(T, 1, 1).to(out.dtype)
        return out.to(value.dtype)

    # Ragged layout (several tokens for one request): never captured; walk it.
    out = torch.zeros((T, Nv, Dv), dtype=torch.float32, device=value.device)
    starts = torch.cumsum(lengths, 0) - lengths
    for r in range(num_reqs):
        slot = int(slots[r])
        if slot < 0:
            continue
        S = state[slot].float()
        for t in range(int(starts[r]), int(starts[r] + lengths[r])):
            S = S * decay[t].view(Nv, 1, 1)
            u = (v[t] - torch.einsum("hvk,hk->hv", S, k[t])) * b[t].view(Nv, 1)
            S = S + torch.einsum("hv,hk->hvk", u, k[t])
            out[t] = torch.einsum("hvk,hk->hv", S, q[t])
        state[slot] = S.to(state.dtype)
    return out.to(value.dtype)


def _gated_delta_rule_spec(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    *,
    beta: torch.Tensor,
    scale: float,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Speculative-decode step of the gated delta rule (graph-safe).

    query/key [T, Nk, D] (L2-normalised), value [T, Nv, D], g/beta [T, Nv],
    state [slots, Nv, Dv, Dk]; T = B * L with L = num_spec + 1 tokens per
    request; ssm_state_indices [B, L] (or flattened); num_accepted_tokens [B].
    """
    T, Nv, Dv = value.shape
    Nk = query.shape[1]
    B = num_accepted_tokens.numel()
    L = T // B
    idx = ssm_state_indices.reshape(-1)[: B * L].reshape(B, L).to(torch.int64)
    acc = num_accepted_tokens.to(torch.int64).clamp(1, L)
    ratio = Nv // Nk
    q = (query.float().repeat_interleave(ratio, dim=1) * scale).reshape(B, L, Nv, -1)
    k = key.float().repeat_interleave(ratio, dim=1).reshape(B, L, Nv, -1)
    v = value.float().reshape(B, L, Nv, Dv)
    decay = g.float().exp().reshape(B, L, Nv)
    b = beta.float().reshape(B, L, Nv)
    nslots = state.shape[0]

    init_slot = idx.gather(1, (acc - 1).view(B, 1)).squeeze(1)
    valid = (init_slot >= 0) & (init_slot < nslots)
    safe0 = torch.where(valid, init_slot, torch.zeros_like(init_slot))
    S = state.index_select(0, safe0).float()  # [B, Nv, Dv, Dk]
    outs = []
    for t in range(L):
        S = S * decay[:, t].view(B, Nv, 1, 1)
        pred = torch.einsum("bhvk,bhk->bhv", S, k[:, t])
        u = (v[:, t] - pred) * b[:, t].view(B, Nv, 1)
        S = S + torch.einsum("bhv,bhk->bhvk", u, k[:, t])
        outs.append(torch.einsum("bhvk,bhk->bhv", S, q[:, t]))
        slot_t = idx[:, t]
        ok_t = valid & (slot_t >= 0) & (slot_t < nslots)
        safe_t = torch.where(ok_t, slot_t, torch.zeros_like(slot_t))
        keep = state.index_select(0, safe_t).float()
        S_write = torch.where(ok_t.view(B, 1, 1, 1), S, keep)
        state.index_copy_(0, safe_t, S_write.to(state.dtype))
    out = torch.stack(outs, dim=1) * valid.view(B, 1, 1, 1).to(torch.float32)
    return out.reshape(T, Nv, Dv).to(value.dtype)


def _recurrent_gdn_gathered(query, key, value, state, *, ssm_state_indices=None, **kwargs):
    """Run the original recurrent kernel on a contiguous copy of the slots
    named by ``ssm_state_indices`` (any shape; negative = padding) and scatter
    the updated slots back.  graph-safe: mask-driven, no boolean indexing."""
    if state is None or state.is_contiguous() or ssm_state_indices is None:
        return _ORIG_RECURRENT_GDN(query, key, value, state, ssm_state_indices=ssm_state_indices, **kwargs)
    idx = ssm_state_indices.reshape(-1).to(torch.int64)
    valid = (idx >= 0) & (idx < state.shape[0])
    safe = torch.where(valid, idx, torch.zeros_like(idx))
    local_state = state.index_select(0, safe)
    local_idx = torch.arange(idx.numel(), dtype=ssm_state_indices.dtype, device=idx.device)
    local_idx = torch.where(valid, local_idx, torch.full_like(local_idx, -1)).reshape(ssm_state_indices.shape)
    out = _ORIG_RECURRENT_GDN(query, key, value, local_state, ssm_state_indices=local_idx, **kwargs)
    # invalid rows were skipped by the kernel, so they still hold slot 0's
    # content and write it back unchanged.
    state.index_copy_(0, safe, local_state)
    return out


def _recurrent_gdn_dispatch(
    query,
    key,
    value,
    state,
    *,
    beta=None,
    scale=None,
    actual_seq_lengths=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    if (
        num_accepted_tokens is not None
        and gk is None
        and g is not None
        and beta is not None
        and ssm_state_indices is not None
    ):
        # Speculative call shape: the kernel's spec mode needs a bf16 state,
        # this model's is fp32 -- run the step in torch (piece 7c).
        if scale is None:
            scale = key.shape[-1] ** -0.5
        return _gated_delta_rule_spec(
            query,
            key,
            value,
            state,
            beta=beta,
            scale=scale,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
        )
    if (
        num_accepted_tokens is not None
        or gk is not None
        or g is None
        or beta is None
        or actual_seq_lengths is None
        or ssm_state_indices is None
    ):
        # Speculative-decoding call shape: keep the kernel, but address the
        # (strided, see 8) state through a contiguous gather of its slots.
        return _recurrent_gdn_gathered(
            query,
            key,
            value,
            state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
            gk=gk,
        )
    if scale is None:
        scale = key.shape[-1] ** -0.5
    return _gated_delta_rule_decode(
        query,
        key,
        value,
        state,
        beta=beta,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        g=g,
    )


def _override_recurrent_gdn() -> None:
    """Rebind the op on its namespace; idempotent, needs the extension loaded."""
    global _ORIG_RECURRENT_GDN
    if _ORIG_RECURRENT_GDN is not None:
        return
    ns = torch.ops._C_ascend
    _ORIG_RECURRENT_GDN = ns.npu_recurrent_gated_delta_rule
    setattr(ns, "npu_recurrent_gated_delta_rule", _recurrent_gdn_dispatch)


def _own_decode_recurrence() -> None:
    """Install the override the first time a GDN core runs (the extension is
    loaded lazily by the worker, so it cannot be done at import)."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    orig = QwenGatedDeltaNetAttention._forward_core

    def _forward_core(self, *args, **kwargs):
        _override_recurrent_gdn()
        return orig(self, *args, **kwargs)

    QwenGatedDeltaNetAttention._forward_core = _forward_core


# ---------------------------------------------------------------------------
# 8. Conv state addressed through a contiguous gather
# ---------------------------------------------------------------------------
_ORIG_CAUSAL_CONV1D = None


def _causal_conv1d_dispatch(
    output,
    x,
    weight,
    conv_state=None,
    bias_opt=None,
    query_start_loc_opt=None,
    cache_indices_opt=None,
    initial_state_mode_opt=None,
    num_accepted_tokens_opt=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
):
    def call(state, indices):
        return _ORIG_CAUSAL_CONV1D(
            output,
            x,
            weight,
            conv_state=state,
            bias_opt=bias_opt,
            query_start_loc_opt=query_start_loc_opt,
            cache_indices_opt=indices,
            initial_state_mode_opt=initial_state_mode_opt,
            num_accepted_tokens_opt=num_accepted_tokens_opt,
            activation_mode=activation_mode,
            pad_slot_id=pad_slot_id,
            run_mode=run_mode,
        )

    if conv_state is None or conv_state.is_contiguous() or cache_indices_opt is None:
        return call(conv_state, cache_indices_opt)

    # The kernel reads one slot per request from the first column of
    # cache_indices; gather exactly those slots.  graph-safe: mask-driven.
    idx = cache_indices_opt.reshape(cache_indices_opt.shape[0], -1)[:, 0].to(torch.int64)
    valid = (idx >= 0) & (idx != pad_slot_id) & (idx < conv_state.shape[0])
    safe = torch.where(valid, idx, torch.zeros_like(idx))
    local_state = conv_state.index_select(0, safe)  # contiguous [B, state_len, dim]
    local_idx = torch.arange(idx.numel(), dtype=cache_indices_opt.dtype, device=idx.device)
    local_idx = torch.where(valid, local_idx, torch.full_like(local_idx, pad_slot_id))
    out = call(local_state, local_idx)
    # rows the kernel skipped (padding) still hold slot 0's content and are
    # written back unchanged.
    conv_state.index_copy_(0, safe, local_state)
    return out


def _override_causal_conv1d() -> None:
    """Rebind the op on its namespace; idempotent, needs the extension loaded."""
    global _ORIG_CAUSAL_CONV1D
    if _ORIG_CAUSAL_CONV1D is not None:
        return
    ns = torch.ops._C_ascend
    _ORIG_CAUSAL_CONV1D = ns.npu_causal_conv1d_custom
    setattr(ns, "npu_causal_conv1d_custom", _causal_conv1d_dispatch)


def _own_conv_state_addressing() -> None:
    """Install the override the first time a GDN core runs (same timing as 7)."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    orig = QwenGatedDeltaNetAttention._forward_core

    def _forward_core(self, *args, **kwargs):
        _override_causal_conv1d()
        return orig(self, *args, **kwargs)

    QwenGatedDeltaNetAttention._forward_core = _forward_core


# ---------------------------------------------------------------------------
# 10. MTP: the Ascend speculator and a QSA drafter
# ---------------------------------------------------------------------------
def _speculator_accepts_qsa() -> None:
    try:
        from vllm_ascend.worker.v2.spec_decode.autoregressive import speculator as _sp
    except Exception as exc:  # noqa: BLE001  (no V2 spec-decode support in this build)
        logger.debug("piece 10 skipped: %s", exc)
        return
    cls = _sp.AscendAutoRegressiveSpeculator
    orig_set_attn = cls.set_attn

    def set_attn(self, *args, **kwargs):
        try:
            return orig_set_attn(self, *args, **kwargs)
        except ValueError as exc:
            if "Unsupported attention backend" not in str(exc):
                raise
            from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend

            backend = getattr(self, "attn_backend", None)
            if backend is not None and issubclass(backend, Qwen4ExpQSAFlashAttentionBackend):
                self.attn_architecture = "QSA"
                logger.info("MTP drafter attention backend %s classified as QSA", backend.__name__)
                return
            raise

    cls.set_attn = set_attn

    # QSA metadata is rebuilt per draft step by its own builder; the Ascend
    # rewrites below assume GQA/MLA metadata fields and must not touch it.
    for name, skip_value in (
        ("_ascend_update_seq_lens", None),
        ("_update_decode_attn_metadata", None),
        ("_init_decode_draft_attn_metadatas", []),
    ):
        orig = getattr(cls, name, None)
        if orig is None:
            continue

        def _make(orig, skip_value):
            def wrapped(self, *args, **kwargs):
                if getattr(self, "attn_architecture", None) == "QSA":
                    return skip_value
                return orig(self, *args, **kwargs)

            return wrapped

        setattr(cls, name, _make(orig, skip_value))


# ---------------------------------------------------------------------------
# 11. FULL graph replay: parameter-update hook for the QSA impl
# ---------------------------------------------------------------------------
def _qsa_full_graph_update(
    update_stream=None,
    forward_context=None,
    num_tokens=None,
    vllm_config=None,
    speculative_config=None,
    draft_attn_metadatas=None,
):
    """Nothing to patch: QSA reads all per-step quantities from device
    tensors held in the metadata builder's persistent buffers."""
    return None


def _install() -> None:
    try:
        from vllm.models.qwen4_exp.nvidia import qsa as _qsa
    except ImportError:  # model not present in this vLLM build
        return
    impl = _qsa.Qwen4ExpQSAFlashAttentionImpl
    impl.__init__ = _ascend_qsa_init
    impl.do_kv_cache_update = _ascend_qsa_kv_cache_update
    impl.update_graph_params = staticmethod(_qsa_full_graph_update)
    _use_torch_qsa_metadata()
    _disable_fused_pre_indexer()
    _opaque_ngram_embedding()
    _use_ascend_model_state()
    _honour_gated_norm_activation()
    _own_decode_recurrence()
    _own_conv_state_addressing()
    _speculator_accepts_qsa()
    logger.info("Patched Qwen4Exp QSA / PLE / model state / gated norm / decode recurrence / conv state for Ascend")


_install()
