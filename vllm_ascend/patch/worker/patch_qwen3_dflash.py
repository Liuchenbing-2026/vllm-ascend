import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from vllm_ascend import envs
from vllm_ascend.ops.qwen3_dspark_attention import qwen3_dspark_reference_attention

_DSPARK_CONTEXT_DEBUG_MAX_CHUNKS = 256


def _is_tensor_parallel_leader() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() % get_tensor_model_parallel_world_size() == 0


def _cpu_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").clone()


def _should_capture_dspark_backbone(self: DFlashQwen3Model) -> bool:
    return (
        bool(envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH)
        and envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS > 0
        and getattr(self, "_dspark_backbone_debug_enabled", False)
        and hasattr(self, "markov_head")
        and _is_tensor_parallel_leader()
    )


def _capture_dspark_context_chunk(
    self: DFlashQwen3Model,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None,
) -> None:
    if context_slot_mapping is None or not _should_capture_dspark_backbone(self):
        return
    chunks = getattr(self, "_dspark_context_debug_chunks", None)
    if chunks is None:
        chunks = []
        self._dspark_context_debug_chunks = chunks
    if len(chunks) >= _DSPARK_CONTEXT_DEBUG_MAX_CHUNKS:
        return
    chunks.append(
        {
            "context_states": _cpu_snapshot(context_states),
            "context_positions": _cpu_snapshot(context_positions),
            "context_slot_mapping": _cpu_snapshot(context_slot_mapping),
        }
    )


def _capture_dspark_raw_context_chunk(
    self: DFlashQwen3ForCausalLM,
    raw_context_states: torch.Tensor,
) -> None:
    if not _should_capture_dspark_backbone(self.model):
        return
    chunks = getattr(self.model, "_dspark_raw_context_debug_chunks", None)
    if chunks is None:
        chunks = []
        self.model._dspark_raw_context_debug_chunks = chunks
    if len(chunks) >= _DSPARK_CONTEXT_DEBUG_MAX_CHUNKS:
        return
    chunks.append(_cpu_snapshot(raw_context_states))


_ORIGINAL_DFLASH_COMBINE_HIDDEN_STATES = DFlashQwen3ForCausalLM.combine_hidden_states


def _dspark_debug_combine_hidden_states(
    self: DFlashQwen3ForCausalLM,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    _capture_dspark_raw_context_chunk(self, hidden_states)
    return _ORIGINAL_DFLASH_COMBINE_HIDDEN_STATES(self, hidden_states)


DFlashQwen3ForCausalLM.combine_hidden_states = _dspark_debug_combine_hidden_states


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    _capture_dspark_context_chunk(
        self,
        context_states,
        context_positions,
        context_slot_mapping,
    )

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # Pass K as the "query" input and keep the returned rotated tensor.
    # Ascend RoPE is not guaranteed to update its inputs in-place.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    rope_output = self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
    if isinstance(rope_output, (tuple, list)) and rope_output[0] is not None:
        all_k_flat = rope_output[0].reshape_as(all_k_flat)

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    for i in range(L):
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            context_slot_mapping,
        )


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv


def _dspark_debug_model_forward(
    self: DFlashQwen3Model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    input_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    if input_embeds is None:
        input_embeds = self.embed_input_ids(input_ids)

    hidden_states = input_embeds
    residual = None
    layer_records = []
    for layer_index, layer in enumerate(self.layers):
        layer_input = hidden_states if residual is None else hidden_states + residual
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        layer_output = hidden_states + residual
        layer_records.append(
            {
                "layer_index": layer_index,
                "input": _cpu_snapshot(layer_input),
                "attention_residual": _cpu_snapshot(residual),
                "mlp_output": _cpu_snapshot(hidden_states),
                "output": _cpu_snapshot(layer_output),
            }
        )

    final_hidden, _ = self.norm(hidden_states, residual)
    context_chunks = list(getattr(self, "_dspark_context_debug_chunks", []))
    raw_context_chunks = list(
        getattr(self, "_dspark_raw_context_debug_chunks", [])
    )
    self._last_dspark_backbone_debug = {
        "input_ids": _cpu_snapshot(input_ids),
        "positions": _cpu_snapshot(positions),
        "input_embeds": _cpu_snapshot(input_embeds),
        "layers": layer_records,
        "final_hidden": _cpu_snapshot(final_hidden),
        "context_chunks": context_chunks,
        "raw_context_chunks": raw_context_chunks,
        "config": {
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "head_dim": getattr(self.config, "head_dim", None),
            "rms_norm_eps": self.config.rms_norm_eps,
            "rope_parameters": getattr(self.config, "rope_parameters", None),
        },
    }
    return final_hidden


_ORIGINAL_DFLASH_MODEL_FORWARD = DFlashQwen3Model.forward


def _maybe_dspark_debug_model_forward(
    self: DFlashQwen3Model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    input_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    if not _should_capture_dspark_backbone(self):
        return _ORIGINAL_DFLASH_MODEL_FORWARD(
            self,
            input_ids,
            positions,
            input_embeds,
        )
    return _dspark_debug_model_forward(
        self,
        input_ids,
        positions,
        input_embeds,
    )


DFlashQwen3Model.forward = _maybe_dspark_debug_model_forward


def _resolve_layer_attn_metadata(attn_metadata, layer_name):
    """Return the per-layer AscendMetadata for ``layer_name`` if the forward
    context stores a dict, else the metadata object itself."""
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(layer_name)
    return attn_metadata


def _resolve_query_start_loc(attn_metadata, device):
    """Cumulative query start locations ``[0, q0, q0+q1, ...]`` as a tensor."""
    qsl = getattr(attn_metadata, "query_start_loc", None)
    if isinstance(qsl, torch.Tensor):
        return qsl
    # Fall back to the builder-produced cumulative query lengths.
    aslq = getattr(attn_metadata, "actual_seq_lengths_q", None)
    if aslq is None:
        return None
    cumulative = torch.as_tensor(aslq, dtype=torch.int32, device=device)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=device), cumulative])


def _resolve_kv_cache_pair(kv_cache, virtual_engine):
    """Resolve direct, stacked, or virtual-engine wrapped K/V caches."""
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.ndim < 5 or kv_cache.shape[0] != 2:
            raise ValueError(f"Expected stacked K/V cache, got {tuple(kv_cache.shape)}")
        return kv_cache[0], kv_cache[1]

    if not isinstance(kv_cache, (list, tuple)) or not kv_cache:
        raise TypeError(f"Unsupported DSpark KV cache container: {type(kv_cache)!r}")
    if (
        len(kv_cache) == 2
        and all(isinstance(cache, torch.Tensor) for cache in kv_cache)
        and all(cache.ndim == 4 for cache in kv_cache)
    ):
        return kv_cache[0], kv_cache[1]

    selected = kv_cache[virtual_engine]
    if isinstance(selected, torch.Tensor):
        if selected.ndim < 5 or selected.shape[0] != 2:
            raise ValueError(
                f"Expected virtual-engine stacked K/V cache, got {tuple(selected.shape)}"
            )
        return selected[0], selected[1]
    if isinstance(selected, (list, tuple)) and len(selected) == 2:
        return selected[0], selected[1]
    raise TypeError(f"Unsupported virtual-engine KV cache: {type(selected)!r}")


def _dspark_reference_forward(
    self: DFlashQwen3Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic DSpark draft attention via the Torch reference.

    Mirrors ``DFlashQwen3Attention.forward`` up to and including RoPE, then
    computes non-causal draft-block attention with the explicit reference over
    the paged context cache instead of the generic maskless FIA branch. Any
    failure falls back to the original forward so this can never break the
    normal path; it is gated by ``VLLM_ASCEND_DSPARK_REFERENCE_ATTENTION``.
    """
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    q_shape, k_shape = q.shape, k.shape
    q = self.q_norm(q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)).view(q_shape)
    k = self.k_norm(k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)).view(k_shape)
    q, k = self.rotary_emb(positions, q, k)

    forward_context = get_forward_context()
    attn_metadata = _resolve_layer_attn_metadata(
        forward_context.attn_metadata, self.attn.layer_name
    )
    if attn_metadata is None:
        # Profiling / warmup with no metadata: use the normal path.
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    virtual_engine = getattr(forward_context, "virtual_engine", 0)
    key_cache, value_cache = _resolve_kv_cache_pair(
        self.attn.kv_cache, virtual_engine
    )

    block_table = getattr(attn_metadata, "block_tables", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    query_start_loc = _resolve_query_start_loc(attn_metadata, q.device)
    cache_block_size = key_cache.shape[1]

    num_tokens = q.shape[0]
    q3 = q.view(num_tokens, self.num_heads, self.head_dim)
    k3 = k.view(num_tokens, self.num_kv_heads, self.head_dim)
    v3 = v.view(num_tokens, self.num_kv_heads, self.head_dim)

    attn_sink = getattr(self, "attention_sink_bias", None)
    attn_output = qwen3_dspark_reference_attention(
        q3,
        k3,
        v3,
        key_cache,
        value_cache,
        block_table,
        query_start_loc,
        seq_lens,
        self.scaling,
        cache_block_size,
        sliding_window=getattr(self, "sliding_window", None),
        attn_sink=attn_sink,
    )
    attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
    output, _ = self.o_proj(attn_output)
    return output


_ORIGINAL_DFLASH_ATTENTION_FORWARD = DFlashQwen3Attention.forward


def _maybe_reference_forward(
    self: DFlashQwen3Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not envs.VLLM_ASCEND_DSPARK_REFERENCE_ATTENTION:
        return _ORIGINAL_DFLASH_ATTENTION_FORWARD(self, positions, hidden_states)
    try:
        return _dspark_reference_forward(self, positions, hidden_states)
    except Exception as exc:  # noqa: BLE001 - diagnostic path must never break serving
        import traceback

        from vllm.logger import logger

        # NOTE: Ascend's ``_VllmLogger.warning_once`` does not accept ``exc_info``;
        # embed the traceback in the message instead so the fallback never raises.
        logger.warning_once(
            "DSpark reference attention failed; falling back to the normal "
            "DFlash attention path. Set VLLM_ASCEND_DSPARK_REFERENCE_ATTENTION=0 "
            "to silence. First error: "
            + repr(exc)
            + "\n"
            + traceback.format_exc()
        )
        return _ORIGINAL_DFLASH_ATTENTION_FORWARD(self, positions, hidden_states)


DFlashQwen3Attention.forward = _maybe_reference_forward
