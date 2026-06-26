#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend MoE-LoRA wrapper (v1).

Design (see plan in conversation history):

  - Inherits weight allocation / set_lora / slice helpers from upstream
    FusedMoEWithLoRA. Only the injection mechanism differs: upstream wraps
    Triton modular kernel internals (`TritonExperts.activation` / `moe_sum`),
    which do not exist on Ascend. We instead wrap the per-layer
    `quant_method.apply` and, inside it, temporarily swap the active
    `MoECommMethod._apply_mlp` so the LoRA delta is added on permuted
    activations between the grouped GMMs.

  - Per-layer ownership is critical: `_MoECommMethods` is a module-level
    singleton shared by all 48 MoE layers. If we wrapped `_apply_mlp` at
    init time, layer N+1 would compose on top of layer N's wrapper and
    every forward would stack all layers' LoRA deltas. We bracket the swap
    inside `apply_wrapper` so only the active layer is in effect.

  - v1 scope: unquant + single adapter + no shared experts + no FusedMC2 +
    no dynamic EPLB. Works with TP and with expert parallelism (forced to
    the all-gather comm so the per-local-expert delta is injectable and the
    FULL-decode aclgraph stays capturable). Other paths assert early so
    users get a clear error rather than silently wrong outputs.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.layers.utils import _get_lora_device

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEMlpComputeInput


def _full_decode_graph_enabled() -> bool:
    """True when the cudagraph mode captures a monolithic FULL decode graph.

    Used to auto-select the MoE-LoRA kernel without an env var: grouped_matmul
    is FULL-decode-graph-safe (single adapter); bgmv is faster / multi-adapter
    for eager and PIECEWISE.
    """
    try:
        from vllm.config import get_current_vllm_config

        cm = get_current_vllm_config().compilation_config.cudagraph_mode
        return cm is not None and "FULL" in str(getattr(cm, "name", cm)).upper()
    except Exception:
        return False


def _assert_ascend_moe_lora_supported(base_layer: AscendFusedMoE) -> None:
    """Centralized v1 capability checks. Asserts up-front for clarity."""
    # Expert parallelism IS supported: select_moe_comm_method forces the
    # all-gather comm when LoRA is enabled, which shards experts (EP memory
    # benefit), exposes expanded_row_idx for the per-token adapter lookup,
    # and is cudagraph-capturable. MC2 / fused-MC2 (fused dispatch, no hook
    # point) are avoided by that forced all-gather; FUSED_MC2 also blocked below.
    if getattr(base_layer, "dynamic_eplb", False):
        raise AssertionError(
            "Ascend MoE LoRA v1 is incompatible with dynamic EPLB "
            "(expert migration would break the per-expert LoRA layout)."
        )
    if int(envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2) != 0:
        raise AssertionError(
            "Ascend MoE LoRA v1 cannot patch FusedMC2 path "
            "(dispatch_ffn_combine is a single fused C++ op). "
            "Set VLLM_ASCEND_ENABLE_FUSED_MC2=0."
        )
    # Shared experts (Qwen3.5-MoE) are supported: the fused shared-expert
    # MLP runs outside quant_method.apply, but its own gate/up/down linears
    # are wrapped by the *standard* linear LoRA path (not this MoE wrapper).
    # This wrapper only injects the routed-expert delta; the shared expert
    # is exposed on the wrapper in __init__ for submodule resolution.
    if getattr(base_layer, "multistream_overlap_gate", False):
        raise AssertionError(
            "multistream_overlap_gate=True interleaves quant_method.apply "
            "calls on multiple streams, which breaks the bracketed "
            "comm._apply_mlp swap. Disable it for MoE LoRA."
        )


class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Ascend-native MoE-LoRA wrapper.

    Reuses upstream weight allocation, set_lora, reset_lora, and slicing.
    Overrides only the injection mechanism (`_inject_lora_into_fused_moe`
    is bypassed; we wrap `quant_method.apply` instead).
    """

    def __init__(self, base_layer: AscendFusedMoE) -> None:
        # Skip FusedMoEWithLoRA.__init__: it immediately asserts Triton
        # internals and calls _inject_lora_into_fused_moe which is GPU-only.
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        _assert_ascend_moe_lora_supported(base_layer)
        # Expose the fused shared-expert submodule (Qwen3.5-MoE) so the
        # LoRA manager can resolve ...experts._shared_experts.* after
        # `experts` is replaced by this wrapper. Its linears get LoRA via
        # the standard linear path; this wrapper handles routed experts.
        _se = getattr(base_layer, "_shared_experts", None)
        if _se is not None:
            self._shared_experts = _se
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = _get_lora_device(base_layer)
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        # Per-layer scratch for state captured at apply-time and consumed
        # inside _apply_mlp_with_lora.
        self._moe_state: dict = {}
        # Auto-select MoE-LoRA kernel from cudagraph mode (no env var).
        self._use_grouped = _full_decode_graph_enabled()
        self._inject_lora_into_ascend_fused_moe()

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------
    def _inject_lora_into_ascend_fused_moe(self) -> None:
        """Patch this layer's quant_method.apply to bracket-swap _apply_mlp.

        Bound-method idiom: we replace `quant_method.apply` with a bound
        method that captures `self` (the LoRA wrapper) so each of the 48
        MoE layers has its own wrapper carrying its own stacked LoRA
        weights. The wrapped function does:

            comm = _EXTRA_CTX.moe_comm_method  # picked per-forward
            orig_mlp = comm._apply_mlp
            try:
                comm._apply_mlp = our LoRA-aware version
                return orig_apply(...)         # base path runs as usual,
                                               # _apply_mlp call goes through us
            finally:
                comm._apply_mlp = orig_mlp     # always restore

        This guarantees the swap is strictly bracketed within a single
        layer's forward pass.
        """
        quant_method = self.base_layer.quant_method
        orig_apply = quant_method.apply
        self_ref = self

        def apply_wrapper(qm_self, layer, x, *args, **kwargs):
            comm = _EXTRA_CTX.moe_comm_method
            if comm is None:
                # Without a comm method we cannot reach _apply_mlp; let the
                # base apply run and skip LoRA. This shouldn't happen in
                # practice because ascend_forward_context sets it per fwd.
                return orig_apply(layer, x, *args, **kwargs)
            orig_mlp = comm._apply_mlp
            self_ref._moe_state["expert_map"] = kwargs.get("expert_map")
            try:
                comm._apply_mlp = lambda mlp_input: self_ref._apply_mlp_with_lora(orig_mlp, mlp_input)
                return orig_apply(layer, x, *args, **kwargs)
            finally:
                comm._apply_mlp = orig_mlp

        # Bind as instance attribute on quant_method so each layer has its own.
        # We cannot use __get__ because orig_apply already is a bound method;
        # storing the function directly works because Python looks up instance
        # attrs before class attrs.
        quant_method.apply = apply_wrapper.__get__(quant_method, type(quant_method))  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def set_mapping(self, punica_wrapper):
        # Upstream FusedMoEWithLoRA.set_mapping (vllm v0.22.0+) chains into
        # ``self._moe_kernel.fused_experts.set_lora_context(...)``, but
        # ``_moe_kernel`` is only set by the GPU modular-kernel path that we
        # deliberately skip in __init__. Our injection works through
        # ``punica_wrapper.add_lora_fused_moe(...)`` inside
        # ``_apply_mlp_with_lora``; we only need the base layer to remember
        # the punica wrapper.
        BaseLayerWithLoRA.set_mapping(self, punica_wrapper)

    def create_lora_weights(self, max_loras, lora_config, model_config=None):
        super().create_lora_weights(max_loras, lora_config, model_config)
        # EP fix, 2D wrapper only. The dummy-LoRA loop in LoRAModelManager
        # iterates ``n_slices`` and indexes the flat ``lora_a_stacked``.
        # Upstream derives n_slices from packed_modules_mapping (GLOBAL
        # expert count), but under EP the stacked buffers are local-expert-
        # sized, so the loop would run past the end of lora_a_stacked
        # (IndexError in profile_run). Pin n_slices to the actual flat
        # length; equals the global count when EP is off, so it is a no-op
        # there.
        #
        # The 3D wrapper (Qwen3.5-MoE fuses w1+w3) builds no lora_a_stacked
        # and has its own dummy branch (FusedMoE3DWithLoRA, which reads
        # w13/w2_lora_a_stacked directly and never touches n_slices), so the
        # override is both impossible and unnecessary there -- skip it.
        if hasattr(self, "lora_a_stacked"):
            self.n_slices = len(self.lora_a_stacked)
        # The grouped kernel (FULL-decode aclgraph / EP) folds adapter slot 0
        # only; a request whose adapter lands in slot >= 1 silently gets no
        # delta. Warn (not raise) since max_loras counts slots, not active
        # adapters -- a single-active-adapter server with >1 slots is fine.
        # Multi-active-adapter needs the bgmv path (eager / PIECEWISE + TP).
        if max_loras > 1 and (self._use_grouped or getattr(self.base_layer, "use_ep", False)):
            logger.warning_once(
                "Ascend MoE-LoRA grouped kernel (FULL-decode aclgraph / expert "
                "parallelism) folds a single adapter (slot 0); with max_loras="
                "%d, only the slot-0 adapter receives the LoRA delta. Use eager "
                "+ TP (bgmv) for concurrent multi-adapter.", max_loras,
            )

    # ------------------------------------------------------------------
    # LoRA-aware MLP
    # ------------------------------------------------------------------
    def _grouped_lora_delta(self, y, x, a_stacked, b_stacked, group_list,
                            group_list_type, active, off0=0):
        """ACL-Graph-safe single-adapter MoE-LoRA delta via npu_grouped_matmul.

        The transposed+contiguous LoRA weight copies are built ONCE (lazily, on
        the first eager warmup call before ACL-graph capture) and cached. Doing
        ``.to(dt).transpose(1, 2).contiguous()`` *inside* the captured FULL
        decode graph allocates a fresh weight-sized tensor every decode step,
        which makes the monolithic graph hang / serialize. The cache turns the
        LoRA A/B into persistent graph inputs, exactly like the base experts'
        ``w1``/``w2``.

        Cache is keyed by id(a_stacked); since set_lora copies new adapter
        weights into the SAME stacked buffer in-place, a single static adapter
        is correct. Multi-adapter hot-swap would need invalidation (TODO);
        the bgmv kernel path remains the general multi-adapter route.
        """
        import torch_npu

        dt = x.dtype
        cache = self._moe_state.setdefault("_gT", {})
        key = id(a_stacked)
        AB = cache.get(key)
        if AB is None:
            As, Bs = [], []
            for sl in range(len(a_stacked)):
                As.append(a_stacked[sl][0].to(dt).transpose(1, 2).contiguous())  # [E, in, rank]
                Bs.append(b_stacked[sl][0].to(dt).transpose(1, 2).contiguous())  # [E, rank, slice]
            AB = (As, Bs)
            cache[key] = AB
        As, Bs = AB

        off = off0
        for sl in range(len(As)):
            lx = torch_npu.npu_grouped_matmul(
                x=[x], weight=[As[sl]], split_item=2,
                group_list_type=group_list_type, group_type=0,
                group_list=group_list)[0]
            ld = torch_npu.npu_grouped_matmul(
                x=[lx], weight=[Bs[sl]], split_item=2,
                group_list_type=group_list_type, group_type=0,
                group_list=group_list)[0]
            n = ld.shape[-1]
            y[:, off:off + n] += ld * active
            off += n

    def _apply_mlp_with_lora(self, orig_mlp, mlp_input: MoEMlpComputeInput):
        """LoRA-aware replacement for MoECommMethod._apply_mlp.

        v1 supports only the unquant + AllGather (expanded_row_idx present)
        path. Any other path falls back to the base implementation so the
        forward still produces (non-LoRA-augmented) output.
        """
        # Respect cudagraph LoRA specialization. The cudagraph dispatcher
        # specializes graph variants on forward_context.has_lora; vLLM
        # captures the has_lora=False variant expecting NO LoRA ops in it.
        # Our bracket-swap is otherwise unconditional, so for the no-LoRA
        # variant we must inject nothing and run vLLM's original base MLP
        # (returns the proper (out, before_gmm2_evt) tuple). Skipping this
        # check leaves LoRA ops in the no-LoRA FULL-decode graph, which
        # wedges the worker on a base/no-LoRA request (sample_tokens hang).
        # This also lets cudagraph_specialize_lora stay True (leaner no-LoRA
        # graph) and skips the masked LoRA matmuls on base requests.
        _bd = get_forward_context().batch_descriptor
        if _bd is not None and not _bd.has_lora:
            return orig_mlp(mlp_input)
        if mlp_input.quant.is_quant:
            logger.warning_once(
                "Ascend MoE LoRA on quantized path is not implemented; "
                "running base path only (LoRA delta will be skipped)."
            )
            return orig_mlp(mlp_input)
        if mlp_input.expanded_row_idx is None:
            logger.warning_once(
                "Ascend MoE LoRA requires AllGather comm method "
                "(combine_metadata.expanded_row_idx); current comm method "
                "does not provide it. Skipping LoRA delta."
            )
            return orig_mlp(mlp_input)
        if mlp_input.topk_ids is None:
            logger.warning_once("Ascend MoE LoRA: topk_ids unavailable in MoEMlpComputeInput; skipping LoRA delta.")
            return orig_mlp(mlp_input)

        # Local imports keep the GPU-only test environment importable.
        import torch_npu

        h = mlp_input.hidden_states  # [N_perm, hidden_in]
        gl = mlp_input.group_list
        glt = mlp_input.group_list_type
        w1 = mlp_input.weights.w1
        w2 = mlp_input.weights.w2
        w1_bias = mlp_input.weights.w1_bias
        w2_bias = mlp_input.weights.w2_bias
        # Unquantized MoE always stores w1/w2 as Tensor (the list[Tensor] form
        # is only used by per-channel quantized paths, which we early-out above
        # via mlp_input.quant.is_quant).
        assert isinstance(w1, torch.Tensor) and isinstance(w2, torch.Tensor)
        need_trans = mlp_input.need_trans
        if need_trans:
            # process_weights_after_loading stores w1/w2 already transposed
            # to [num_experts, in, out]; only the legacy unquant path with
            # need_trans=True flips back.
            w1 = w1.transpose(1, 2)
            w2 = w2.transpose(1, 2)

        # ---- per-permuted-row (expert_id, orig_token) (1D, length N_perm) ----
        # npu_moe_init_routing semantics:
        #   sorted_hidden_states[i] corresponds to the original (token, k) pair
        #   indexed by expanded_row_idx[i] (= orig_token * top_k + k), and that
        #   pair was routed to expert id `topk_ids[orig_token, k]`. So both the
        #   expert id and orig token can be recovered with a single gather.
        # NOTE: We deliberately avoid torch.repeat_interleave(arange, gl) here -
        # when `gl` is a device tensor and `output_size` is omitted, PyTorch
        # must sync to read gl.sum() to determine the output shape. That sync
        # is illegal during ACL-graph capture ("not allowed to synchronize
        # captured-stream"). Gathering from topk_ids is pure device-side and
        # graph-capturable.
        top_k = self.base_layer.top_k
        expanded = torch.abs(mlp_input.expanded_row_idx)
        expert_per_row = mlp_input.topk_ids.reshape(-1)[expanded].to(torch.long)
        # token_lora_indices is a 1D device LongTensor sized to
        # max_num_batched_tokens and replicated across EP ranks (all-gather
        # gathers all tokens to every rank), so the per-row adapter slot is
        # a local gather; clamp is a graph-safe no-op.
        orig_token = expanded // top_k
        token_lora_indices = self.punica_wrapper.token_lora_indices
        orig_token = orig_token.clamp_(max=token_lora_indices.numel() - 1)
        lora_per_row = token_lora_indices[orig_token]
        # The grouped kernel is required for the FULL-decode aclgraph and for
        # EP (its local-expert-sized weights don't match bgmv's global expert
        # ids); bgmv otherwise (eager / PIECEWISE, TP, multi-adapter).
        use_grouped = self._use_grouped or getattr(self.base_layer, "use_ep", False)

        _active = (lora_per_row == 0).to(h.dtype).unsqueeze(-1)

        # === Stage 1: gate_up GMM (base) ===
        gate_up = torch_npu.npu_grouped_matmul(
            x=[h],
            weight=[w1],
            bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
            split_item=2,
            group_list_type=glt,
            group_type=0,
            group_list=gl,
        )[0]  # [N_perm, 2*inter] (or [N_perm, inter] when _w13_slices==1)

        # === Stage 2: LoRA delta for w13 ===
        if use_grouped:
            self._grouped_lora_delta(gate_up, h, self.w13_lora_a_stacked,
                                     self.w13_lora_b_stacked, gl, glt, _active)
        else:
            self.punica_wrapper.add_lora_fused_moe(
                y=gate_up,
                x=h,
                lora_a_stacked=self.w13_lora_a_stacked,
                lora_b_stacked=self.w13_lora_b_stacked,
                topk_weights=None,
                sorted_token_ids=None,
                expert_ids=expert_per_row,
                num_tokens_post_padded=None,
                max_lora_rank=self.w13_lora_a_stacked[0].shape[-2],
                top_k_num=1,
                shrink_config={},
                expand_config={},
                adapter_enabled=self.adapter_enabled,
                mul_routed_weight=False,
                fully_sharded=self.fully_sharded,
                offset=0,
                token_lora_mapping=lora_per_row,
            )

        # === Stage 3: activation (SiLU / SwiGLU) ===
        # Match unquant_apply_mlp: activation may arrive as an enum (vllm
        # MoEActivation) or as a raw string; ``getattr(..., "value", ...)``
        # normalizes both.
        act_name = getattr(mlp_input.activation, "value", mlp_input.activation)
        if act_name == "swigluoai":
            silu_out = AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up.view(-1, gate_up.shape[-1]))
        else:
            silu_out = torch_npu.npu_swiglu(gate_up)
        if mlp_input.topk_scales is not None:
            silu_out = silu_out * mlp_input.topk_scales

        # === Stage 4: down GMM (base) ===
        out = torch_npu.npu_grouped_matmul(
            x=[silu_out],
            weight=[w2],
            bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
            split_item=2,
            group_list_type=glt,
            group_type=0,
            group_list=gl,
        )[0]  # [N_perm, hidden_out]

        # === Stage 5: LoRA delta for w2 ===
        if use_grouped:
            self._grouped_lora_delta(out, silu_out, self.w2_lora_a_stacked,
                                     self.w2_lora_b_stacked, gl, glt, _active)
        else:
            self.punica_wrapper.add_lora_fused_moe(
                y=out,
                x=silu_out,
                lora_a_stacked=self.w2_lora_a_stacked,
                lora_b_stacked=self.w2_lora_b_stacked,
                topk_weights=None,
                sorted_token_ids=None,
                expert_ids=expert_per_row,
                num_tokens_post_padded=None,
                max_lora_rank=self.w2_lora_a_stacked[0].shape[-2],
                top_k_num=1,
                shrink_config={},
                expand_config={},
                adapter_enabled=self.adapter_enabled,
                mul_routed_weight=False,
                fully_sharded=self.fully_sharded,
                offset=0,
                token_lora_mapping=lora_per_row,
            )
        # Match MoECommMethod._apply_mlp return contract: (hidden, before_gmm2_evt).
        # Unquant path produces no overlap event (mirrors unquant_apply_mlp's
        # ``return hidden_states, None``); fallback branches above already
        # forward the base ``orig_mlp`` tuple verbatim.
        return out, None

    # ------------------------------------------------------------------
    # Layer-replacement registration
    # ------------------------------------------------------------------
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        del lora_config, model_config
        # AscendSharedFusedMoE inherits from AscendFusedMoE so this isinstance
        # check matches both. _assert_ascend_moe_lora_supported in __init__
        # rejects layers that actually carry shared experts.
        return isinstance(source_layer, AscendFusedMoE) and len(packed_modules_list) == 2


class AscendFusedMoE3DWithLoRA(AscendFusedMoEWithLoRA, FusedMoE3DWithLoRA):
    """For checkpoints that already fuse w1+w3 into a 3D weight (single slice)."""

    def __init__(self, base_layer: AscendFusedMoE) -> None:
        AscendFusedMoEWithLoRA.__init__(self, base_layer)
        # Override: 3D MoE LoRA uses a single w13 slice.
        self._w13_slices = 1

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        del lora_config, model_config
        return isinstance(source_layer, AscendFusedMoE) and len(packed_modules_list) == 1


# ----------------------------------------------------------------------
# Upstream compatibility shim: vllm/lora/model_manager.py:create_dummy_lora
# branches on `module.__class__.__name__ == "FusedMoEWithLoRA"` (and the
# 3D variant). Without this override, our subclasses would skip the
# pack_moe path and hit the generic pack() fallback, which produces a
# flat list of N_experts * 3 sub-LoRAs -- `set_lora` then fails with
# "too many values to unpack (expected 3)".
#
# Overriding only __name__ keeps the actual class object distinct (so
# isinstance / type identity / debugging are unaffected) but lets the
# upstream string compare hit our objects.
# ----------------------------------------------------------------------
AscendFusedMoEWithLoRA.__name__ = "FusedMoEWithLoRA"
AscendFusedMoE3DWithLoRA.__name__ = "FusedMoE3DWithLoRA"
