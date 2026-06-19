# SPDX-License-Identifier: Apache-2.0
# Ascend MoE-LoRA enablement for Qwen3.5-MoE (and other FusedMoE) on vllm-ascend.
#
# Why this exists:
#   vLLM's FusedMoEWithLoRA.__init__ builds a *modular* MoE kernel via
#   quant_method.select_gemm_impl(...), but vllm-ascend deliberately disables the
#   upstream modular kernel (AscendUnquantizedFusedMoEMethod.select_gemm_impl raises
#   ValueError). So FusedMoEWithLoRA can't even be constructed -> MoE LoRA is unsupported.
#
# What we do (correctness-first, torch/torch_npu, no new kernel):
#   1. Override FusedMoEWithLoRA.__init__ on Ascend to SKIP the modular-kernel build.
#      Only the stacked LoRA buffers (created by create_lora_weights/set_lora, which do
#      NOT depend on the kernel) are needed.
#   2. Override set_mapping to stash the WithLoRA layer (holding the stacked buffers +
#      punica_wrapper) onto the base AscendFusedMoE layer as `_ascend_moe_lora`.
#   3. Patch AscendUnquantizedFusedMoEMethod.apply to publish the active (lora_layer,
#      lora_id) for the current batch into a module-global before the experts run.
#   4. Patch unquant_apply_mlp to inject the per-expert LoRA delta with the SAME
#      npu_grouped_matmul + group_list used for the base experts:
#         gate_up += A_w13->B_w13(grouped) applied to GMM1 input (pre-swiglu)
#         down    += A_w2 ->B_w2 (grouped) applied to GMM2 input (post-swiglu)
#      Scaling (alpha/r) is already folded into lora_b by LoRALayerWeights.optimize(),
#      so no extra scale here. Routed-expert grouping aligns automatically because we
#      reuse group_list/group_list_type.
#
# Limitations (correctness-first): assumes a uniform per-batch adapter (one active
# lora id, or base). Mixed-adapter batches on the routed experts are not handled.

import os
import torch

_DEBUG = os.environ.get("MOE_LORA_DEBUG", "0") == "1"


def _log(msg):
    if _DEBUG:
        print(f"[moe-lora] {msg}", flush=True)


# module-global publishing the active routed-expert LoRA for the in-flight experts call
# value: (lora_layer, lora_id:int) or None
_ACTIVE = None


def _set_active(v):
    global _ACTIVE
    _ACTIVE = v


# ----------------------------------------------------------------------------- #
# 1+2. Patch FusedMoEWithLoRA construction / mapping (safe at vllm import time)
# ----------------------------------------------------------------------------- #
def _install_fusedmoe_lora_patches():
    from vllm.lora.layers.base import BaseLayerWithLoRA
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
    from vllm.lora.layers.utils import _get_lora_device

    if getattr(FusedMoEWithLoRA, "_ascend_patched", False):
        return

    def _ascend_init(self, base_layer):
        # Replicate the parts of FusedMoEWithLoRA.__init__ that do NOT need the
        # upstream modular kernel.
        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer
        self._ep_check()
        self.tp_size = base_layer.tp_size
        self.tp_rank = base_layer.tp_rank
        self.device = _get_lora_device(base_layer)
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        # Expose the base layer's fused shared-expert submodule on the wrapper so
        # the LoRA manager's get_submodule("....experts._shared_experts....") still
        # resolves after `experts` is replaced by this wrapper (torch get_submodule
        # walks attributes via getattr).
        _se = getattr(base_layer, "_shared_experts", None)
        if _se is not None:
            self._shared_experts = _se
        self.base_layer.ensure_moe_quant_config_init()
        # SKIP: select_gemm_impl / FusedMoEKernel / supports_lora assert /
        #       _replace_quant_method  (Ascend uses its own apply()).
        self._moe_kernel = None
        # Make sure the runtime experts patches are installed (deferred to here so
        # ops modules are already imported -> no circular import at vllm_ascend init).
        _ensure_runtime_patches()
        _log(f"AscendFusedMoEWithLoRA constructed for {type(base_layer).__name__}")

    def _ascend_set_mapping(self, punica_wrapper):
        self.punica_wrapper = punica_wrapper
        # Stash on the base layer so AscendUnquantizedFusedMoEMethod.apply can find it.
        # Use object.__setattr__ to store a PLAIN attribute: assigning an nn.Module
        # (self) via normal setattr would register it in base_layer._modules, creating
        # a wrapper<->base_layer cycle that breaks named_modules()/get_submodule.
        object.__setattr__(self.base_layer, "_ascend_moe_lora", self)

    FusedMoEWithLoRA.__init__ = _ascend_init
    FusedMoEWithLoRA.set_mapping = _ascend_set_mapping
    FusedMoEWithLoRA._ascend_patched = True
    _log("FusedMoEWithLoRA __init__/set_mapping patched")


# ----------------------------------------------------------------------------- #
# 3+4. Runtime patches (deferred: imported lazily to avoid circular imports)
# ----------------------------------------------------------------------------- #
_RUNTIME_DONE = False


def _ensure_runtime_patches():
    global _RUNTIME_DONE
    if _RUNTIME_DONE:
        return
    try:
        from vllm_ascend.ops.fused_moe import fused_moe as _fm
        from vllm_ascend.ops.fused_moe import moe_mlp as _mm
    except Exception as e:  # pragma: no cover
        _log(f"runtime import not ready: {e}")
        return

    # ---- 3. wrap AscendUnquantizedFusedMoEMethod.apply to publish active lora ----
    Method = _fm.AscendUnquantizedFusedMoEMethod
    if not getattr(Method, "_ascend_lora_wrapped", False):
        _orig_apply = Method.apply

        def _apply(self, *args, **kwargs):
            layer = kwargs.get("layer", args[0] if args else None)
            prev = _ACTIVE
            _set_active(None)
            try:
                ll = getattr(layer, "_ascend_moe_lora", None)
                if ll is not None and getattr(ll, "punica_wrapper", None) is not None:
                    try:
                        tli = ll.punica_wrapper.token_lora_indices
                        lid = int(tli.max().item()) if tli is not None and tli.numel() else -1
                    except Exception:
                        lid = -1
                    if lid >= 0 and int(ll.adapter_enabled[lid].item()) == 1:
                        _set_active((ll, lid))
                        _log(f"active routed-expert LoRA id={lid}")
                return _orig_apply(self, *args, **kwargs)
            finally:
                _set_active(prev)

        Method.apply = _apply
        Method._ascend_lora_wrapped = True

    # ---- 4. patch unquant_apply_mlp to inject the per-expert LoRA delta ----
    if not getattr(_mm, "_ascend_lora_unquant_wrapped", False):
        _orig_unquant = _mm.unquant_apply_mlp

        def _unquant_apply_mlp_lora(hidden_states, w1, w2, group_list,
                                    w1_bias=None, w2_bias=None, activation=None,
                                    group_list_type=1, topk_scales=None, need_trans=True):
            active = _ACTIVE
            if active is None:
                return _orig_unquant(hidden_states, w1, w2, group_list,
                                     w1_bias=w1_bias, w2_bias=w2_bias,
                                     activation=activation, group_list_type=group_list_type,
                                     topk_scales=topk_scales, need_trans=need_trans)
            torch_npu = _mm.torch_npu
            ll, lid = active
            dt = hidden_states.dtype

            def W(t):  # [E,a,b] -> contiguous [E,a,b] cast
                return t.to(dt).contiguous()

            # base GMM1
            w1t = w1.transpose(1, 2) if need_trans else w1
            w2t = w2.transpose(1, 2) if need_trans else w2
            act_name = getattr(activation, "value", activation)
            gmm1_in = hidden_states
            gate_up = torch_npu.npu_grouped_matmul(
                x=[gmm1_in], weight=[w1t],
                bias=[w1_bias.to(torch.float32)] if w1_bias is not None else None,
                split_item=2, group_list_type=group_list_type, group_type=0,
                group_list=group_list)[0]

            # --- inject w13 LoRA on GMM1 input ---
            # Each w13 slice's output width is taken from the LoRA-B tensor itself,
            # so this is correct whether the gate_up LoRA is stored as one fused
            # slice (_w13_slices==1, width=2*inter) or split gate/up (==2, width=inter).
            off = 0
            for sl in range(ll._w13_slices):
                A = W(ll.w13_lora_a_stacked[sl][lid]).transpose(1, 2)  # [E,hidden,r]
                B = W(ll.w13_lora_b_stacked[sl][lid]).transpose(1, 2)  # [E,r,width]
                lx = torch_npu.npu_grouped_matmul(
                    x=[gmm1_in], weight=[A.contiguous()], split_item=2,
                    group_list_type=group_list_type, group_type=0, group_list=group_list)[0]
                ld = torch_npu.npu_grouped_matmul(
                    x=[lx], weight=[B.contiguous()], split_item=2,
                    group_list_type=group_list_type, group_type=0, group_list=group_list)[0]
                n = ld.shape[-1]
                gate_up[:, off:off + n] += ld
                off += n

            # activation
            if act_name == "swigluoai":
                num_experts, _, hidden_size = w1t.shape
                gate_up = _mm.AscendSwigluOAIAndMul.swiglu_oai_forward(
                    gate_up.view(-1, hidden_size))
            else:
                gate_up = torch_npu.npu_swiglu(gate_up)
            if topk_scales is not None:
                gate_up *= topk_scales

            gmm2_in = gate_up
            out = torch_npu.npu_grouped_matmul(
                x=[gmm2_in], weight=[w2t],
                bias=[w2_bias.to(torch.float32)] if w2_bias is not None else None,
                split_item=2, group_list_type=group_list_type, group_type=0,
                group_list=group_list)[0]

            # --- inject w2 LoRA on GMM2 input ---
            A2 = W(ll.w2_lora_a_stacked[0][lid]).transpose(1, 2)  # [E,inter,r]
            B2 = W(ll.w2_lora_b_stacked[0][lid]).transpose(1, 2)  # [E,r,hidden]
            lx2 = torch_npu.npu_grouped_matmul(
                x=[gmm2_in], weight=[A2.contiguous()], split_item=2,
                group_list_type=group_list_type, group_type=0, group_list=group_list)[0]
            ld2 = torch_npu.npu_grouped_matmul(
                x=[lx2], weight=[B2.contiguous()], split_item=2,
                group_list_type=group_list_type, group_type=0, group_list=group_list)[0]
            out += ld2
            _log(f"injected MoE LoRA id={lid} T={hidden_states.shape[0]}")
            return out, None

        _mm.unquant_apply_mlp = _unquant_apply_mlp_lora
        _mm._ascend_lora_unquant_wrapped = True
        _log("unquant_apply_mlp patched")

    _RUNTIME_DONE = True


# --------------------------------------------------------------------------- #
# 5. Drop GDN in_proj from Qwen3.5 packed_modules_mapping.
#    The GatedDeltaNet in_proj_ba (beta/alpha gates) has a tiny output dim; the
#    vllm-ascend v0.21.0rc1 sgmv_expand kernel rejects it ("hidden in should be
#    smaller than hidden out") when the LoRA manager builds dummy adapters for it
#    during profile_run. Our MoE adapters never target in_proj, so removing it
#    from the LoRA packed mapping avoids wrapping those layers entirely.
# --------------------------------------------------------------------------- #
def _strip_inproj_from_packed():
    try:
        from vllm.model_executor.models import qwen3_5 as q
    except Exception as e:
        _log(f"qwen3_5 import for packed-strip failed: {e}")
        return
    for cls_name in (
        "Qwen3_5ForCausalLMBase",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        cls = getattr(q, cls_name, None)
        if cls is None or "packed_modules_mapping" not in cls.__dict__:
            continue
        pmm = dict(cls.packed_modules_mapping or {})
        removed = [k for k in ("in_proj_qkvz", "in_proj_ba") if k in pmm]
        for k in removed:
            del pmm[k]
        if removed:
            cls.packed_modules_mapping = pmm
            _log(f"{cls_name}: removed {removed} from packed_modules_mapping")


# install the safe (vllm-side) patches at import
try:
    _install_fusedmoe_lora_patches()
    _strip_inproj_from_packed()
except Exception as e:  # pragma: no cover
    _log(f"install failed: {e}")
