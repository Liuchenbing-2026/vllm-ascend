# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Per-layer MonoKernel runtime coordinator for DeepSeek V4.1."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.models.utils import PPMissingLayer

from .ascendc_backend import AscendCFullLayerMonoBackend, is_full_layer_kernel_eligible
from .config import AscendDSV41MonoConfig
from .contracts import (
    DSV41MonoLayerBackend,
    MonoLayerBinding,
    MonoStepPlan,
    PendingPostState,
)
from .seam_kernel import compute_hc_mixes_from_inv_rms, mono_hc_post_collapse_norm


def _is_pure_decode_metadata(attn_metadata: Any) -> tuple[bool, str]:
    """Check whether forward attn_metadata represents a pure-decode batch."""
    if attn_metadata is None:
        return False, "missing attn_metadata (dummy/profile)"
    if isinstance(attn_metadata, dict):
        metas = [meta for meta in attn_metadata.values() if meta is not None]
    else:
        metas = [attn_metadata]
    if not metas:
        return False, "empty attn_metadata"
    for meta in metas:
        num_prefills = getattr(meta, "num_prefills", None)
        num_decodes = getattr(meta, "num_decodes", None)
        num_reqs = getattr(meta, "num_reqs", None)
        if num_prefills is None or num_decodes is None or num_reqs is None:
            return False, "non-v41 attn_metadata"
        if num_prefills > 0 or num_decodes <= 0 or num_decodes != num_reqs:
            return False, f"non-decode batch (prefills={num_prefills}, decodes={num_decodes}, reqs={num_reqs})"
    return True, ""


class AscendTritonMonoBackend:
    """Default in-tree per-layer MonoKernel backend using Triton-Ascend seam fusion."""

    name = "ascend_triton_mono"

    def run_seam_post_collapse_norm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_module: Any,
        norm_4d_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = getattr(norm_module, "bias", None) if getattr(norm_module, "bias_loaded", False) else None
        eps = float(getattr(norm_module, "variance_epsilon", norm_4d_eps))
        return mono_hc_post_collapse_norm(
            x=x,
            residual=residual,
            post=post,
            comb=comb,
            pre=pre,
            norm_weight=norm_module.weight,
            norm_eps=eps,
            norm_4d_eps=norm_4d_eps,
            norm_bias=bias,
        )

    def run_hc_mixes(
        self,
        residual_flat_f32: torch.Tensor,
        inv_rms_4d: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        sinkhorn_iters: int,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compute_hc_mixes_from_inv_rms(
            residual_flat_f32=residual_flat_f32,
            inv_rms_4d=inv_rms_4d,
            hc_fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=hc_eps,
        )


class DSV41AscendMonoRunner:
    """Model-owned coordinator executing eligible decode steps layer-by-layer."""

    def __init__(
        self,
        config: AscendDSV41MonoConfig,
        bindings: dict[int, MonoLayerBinding],
        backend: DSV41MonoLayerBackend | None = None,
        full_layer_backend: AscendCFullLayerMonoBackend | None = None,
    ) -> None:
        self.config = config
        self.bindings = bindings
        self.backend: DSV41MonoLayerBackend = backend or AscendTritonMonoBackend()
        self.full_layer_backend = full_layer_backend or AscendCFullLayerMonoBackend()

    @classmethod
    def from_layers(
        cls,
        config: AscendDSV41MonoConfig,
        layers: Sequence[Any],
        start_layer: int,
        end_layer: int,
        aux_hidden_state_layers: Iterable[int],
        decoder_replay_start: int,
        backend: DSV41MonoLayerBackend | None = None,
    ) -> DSV41AscendMonoRunner:
        aux_set = set(aux_hidden_state_layers)
        bindings: dict[int, MonoLayerBinding] = {}
        local_indices = [idx for idx in range(start_layer, end_layer) if not isinstance(layers[idx], PPMissingLayer)]
        for pos, idx in enumerate(local_indices):
            layer = layers[idx]
            next_idx = local_indices[pos + 1] if pos + 1 < len(local_indices) else None
            next_layer = layers[next_idx] if next_idx is not None else None
            has_next_seam = (
                next_layer is not None
                and next_idx != decoder_replay_start
                and next_idx not in aux_set
                and getattr(next_layer, "engram", None) is None
            )
            bindings[idx] = MonoLayerBinding(
                layer_idx=idx,
                layer=layer,
                can_defer_post=has_next_seam,
            )
        return cls(config=config, bindings=bindings, backend=backend)

    def plan_step(self, positions: torch.Tensor, attn_metadata: Any = None) -> MonoStepPlan:
        """Deterministically choose MonoKernel vs stock path using shared step metadata."""
        num_tokens = int(positions.shape[0])
        if num_tokens <= 0 or num_tokens > self.config.max_rows:
            return MonoStepPlan(active=False, num_tokens=num_tokens, reason="token count out of range")
        if not is_forward_context_available():
            return MonoStepPlan(active=False, num_tokens=num_tokens, reason="no forward context")
        ctx = get_forward_context()
        if ctx.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
            return MonoStepPlan(
                active=False,
                num_tokens=num_tokens,
                reason=f"unsupported cudagraph mode {ctx.cudagraph_runtime_mode}",
            )
        meta = attn_metadata if attn_metadata is not None else ctx.attn_metadata
        is_decode, reason = _is_pure_decode_metadata(meta)
        return MonoStepPlan(active=is_decode, num_tokens=num_tokens, reason=reason)

    @staticmethod
    def flush_pending(
        layer_fallback: Any,
        residual: torch.Tensor,
        pending: PendingPostState | None,
    ) -> torch.Tensor:
        """Materialize a deferred FFN hc_post before a boundary observer reads residual."""
        if pending is None:
            return residual
        return layer_fallback.hc_post(pending.output, residual, pending.post, pending.comb)

    def run_layer(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor | None,
        pending_post: PendingPostState | None,
        *,
        allow_defer_post: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, PendingPostState | None]:
        """Execute one decoder layer under per-layer MonoKernel dispatch."""
        binding = self.bindings[layer_idx]
        layer = binding.layer
        if is_full_layer_kernel_eligible(layer):
            residual = self.flush_pending(layer, hidden_states, pending_post)
            res_out, next_pre = self.full_layer_backend.run_full_layer(layer, positions, residual, pre_mix)
            return res_out, next_pre, None
        residual, x_attn_in, attn_post, attn_comb, attn_pre = self._run_attn_entry(
            layer, hidden_states, pre_mix, pending_post
        )
        x_attn_out = layer.self_attn(positions, x_attn_in, None)
        residual, x_ffn_in, ffn_post, ffn_comb, ffn_pre = self._run_ffn_seam(
            layer, x_attn_out, residual, attn_post, attn_comb, attn_pre
        )
        x_moe_out = layer.mlp(x_ffn_in, input_ids=input_ids, already_sequence_parallel=False)
        if allow_defer_post and binding.can_defer_post:
            return residual, ffn_pre, PendingPostState(x_moe_out, ffn_post, ffn_comb)
        hidden_states = layer.hc_post(x_moe_out, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre, None

    def _run_attn_entry(
        self,
        layer: Any,
        residual: torch.Tensor,
        pre_mix: torch.Tensor,
        pending_post: PendingPostState | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if pending_post is None:
            x_collapsed, attn_post, attn_comb, attn_pre = layer.hc_pre(
                residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, pre_mix
            )
            return residual, layer.input_layernorm(x_collapsed), attn_post, attn_comb, attn_pre

        residual, x_normed, inv_rms_4d = self.backend.run_seam_post_collapse_norm(
            pending_post.output,
            residual,
            pending_post.post,
            pending_post.comb,
            pre_mix,
            layer.input_layernorm,
            layer.norm_eps,
        )
        attn_pre, attn_post, attn_comb = self.backend.run_hc_mixes(
            residual.flatten(-2).float(),
            inv_rms_4d,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.hc_sinkhorn_iters,
            layer.hc_eps,
        )
        return residual, x_normed, attn_post, attn_comb, attn_pre

    def _run_ffn_seam(
        self,
        layer: Any,
        x_attn_out: torch.Tensor,
        residual: torch.Tensor,
        attn_post: torch.Tensor,
        attn_comb: torch.Tensor,
        attn_pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, x_normed, inv_rms_4d = self.backend.run_seam_post_collapse_norm(
            x_attn_out,
            residual,
            attn_post,
            attn_comb,
            attn_pre,
            layer.post_attention_layernorm,
            layer.norm_eps,
        )
        ffn_pre, ffn_post, ffn_comb = self.backend.run_hc_mixes(
            residual.flatten(-2).float(),
            inv_rms_4d,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.hc_sinkhorn_iters,
            layer.hc_eps,
        )
        return residual, x_normed, ffn_post, ffn_comb, ffn_pre


def maybe_build_mono_runner(
    vllm_config: VllmConfig,
    hf_config: Any,
    layers: Sequence[Any],
    start_layer: int,
    end_layer: int,
    aux_hidden_state_layers: Iterable[int],
    decoder_replay_start: int,
) -> DSV41AscendMonoRunner | None:
    """Build DSv4.1 MonoKernel runner at model init if enabled and qualified."""
    mono_cfg = AscendDSV41MonoConfig.from_vllm_config(vllm_config, hf_config)
    if mono_cfg is None:
        return None
    logger.info(
        "DeepSeek V4.1 --monokernel enabled: backend=%s, max_rows=%d",
        mono_cfg.backend_name,
        mono_cfg.max_rows,
    )
    return DSV41AscendMonoRunner.from_layers(
        config=mono_cfg,
        layers=layers,
        start_layer=start_layer,
        end_layer=end_layer,
        aux_hidden_state_layers=aux_hidden_state_layers,
        decoder_replay_start=decoder_replay_start,
    )
