import vllm.model_executor.layers.mamba.ops.causal_conv1d as _cc1d
import vllm.third_party.flash_linear_attention.ops as fla_ops
import vllm.third_party.flash_linear_attention.ops.layernorm_guard as fla_layernorm_guard
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_fn as _npu_causal_conv1d_fn_impl,
)
from vllm_ascend.ops.causal_conv1d import causal_conv1d_update as _npu_causal_conv1d_update_impl
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn


def _next_power_of_2(n):
    # vLLM #58651's packed_qk_rope kernel evaluates
    # triton.next_power_of_2(rotary_dim) with a tl.constexpr argument.
    # The host helper only understands plain ints, so unwrap first.
    if isinstance(n, tl.constexpr):
        n = n.value
    return next_power_of_2(n)


# Newer Triton tracks JIT kernel dependencies and rejects plain Python
# callables. Keep the host helper callable from constexpr code as well.
if HAS_TRITON and hasattr(triton, "constexpr_function"):
    triton.next_power_of_2 = triton.constexpr_function(next_power_of_2)
else:
    triton.next_power_of_2 = _next_power_of_2


def _npu_causal_conv1d_update(*args, **kwargs):
    for key in (
        "null_block_id",
        "block_idx_first_scheduled_token",
        "block_idx_last_scheduled_token",
        "initial_state_idx",
        "num_computed_tokens",
        "block_size_to_align",
        "validate_data",
        "max_query_len",
        "out",
        "metadata",
    ):
        kwargs.pop(key, None)
    return _npu_causal_conv1d_update_impl(*args, **kwargs)


# npu_kda_causal_conv1d_bind: CUDA causal_conv1d kernels use tl.extra.cuda.gdc_wait
# which Ascend Triton does not provide. Swap the NPU update kernel and a PyTorch
# prefill fn. Models that live in vllm-ascend import the Ascend entry points
# directly and do not rely on this rebind.
_CC1D_FN_DROP = (
    "null_block_id",
    "block_idx_first_scheduled_token",
    "block_idx_last_scheduled_token",
    "initial_state_idx",
    "num_computed_tokens",
    "block_size_to_align",
    "validate_data",
)


def _npu_causal_conv1d_fn(*args, metadata=None, **kwargs):
    for key in _CC1D_FN_DROP:
        kwargs.pop(key, None)
    return _npu_causal_conv1d_fn_impl(*args, **kwargs)


_cc1d.causal_conv1d_update = _npu_causal_conv1d_update
_cc1d.causal_conv1d_fn = _npu_causal_conv1d_fn

fla_layernorm_guard.LayerNormFn = LayerNormFn
fla_ops.chunk_gated_delta_rule = chunk_gated_delta_rule

# GLM-5.3-Flash (and Kimi KDA) import fused_recurrent_kda / chunk_kda_with_fused_gate
# from this FLA module. Swap them for the NPU Triton implementations before the
# model is constructed. Missing on older vLLM trees that predate KDA.
try:
    import vllm.third_party.flash_linear_attention.ops.kda as fla_kda

    from vllm_ascend.ops.triton.kda.kda import (
        chunk_kda as _npu_chunk_kda,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        chunk_kda_with_fused_gate as _npu_chunk_kda_with_fused_gate,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        fused_kda_gate as _npu_fused_kda_gate,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        fused_recurrent_kda as _npu_fused_recurrent_kda,
    )

    fla_kda.fused_recurrent_kda = _npu_fused_recurrent_kda
    fla_kda.chunk_kda = _npu_chunk_kda
    fla_kda.chunk_kda_with_fused_gate = _npu_chunk_kda_with_fused_gate
    fla_kda.fused_kda_gate = _npu_fused_kda_gate
except ImportError:
    pass

# GLM-5.3-Flash vision tower (and any other NVIDIA fused Q/K RMSNorm) launches a
# CUDA Triton kernel that references tl.extra.cuda.gdc_wait. Ascend Triton has
# no such symbol; the AST visitor raises even when launch_pdl is False.
try:
    import torch_npu
    import vllm.models.common.ops as _common_ops
    import vllm.models.common.ops.fused_qk_rmsnorm as _fused_qk_mod

    def _npu_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
        # Same math as the CUDA kernel: RMS in fp32, one cast at store.
        q_out, _ = torch_npu.npu_rms_norm(qr, q_weight, eps)
        kv_out, _ = torch_npu.npu_rms_norm(kv, kv_weight, eps)
        return q_out, kv_out

    _fused_qk_mod.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm
    _common_ops.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm
    # from_import_sweep: the GLM vision tower is imported during model
    # registration, before this patch runs, so its from-import already
    # holds a direct reference to the CUDA kernel. Rebinding the defining
    # module is not enough; rebind every module that captured the name.
    import sys as _sys

    for _m in list(_sys.modules.values()):
        try:
            if getattr(_m, "fused_q_kv_rmsnorm", None) not in (None, _npu_fused_q_kv_rmsnorm):
                _m.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm  # type: ignore[attr-defined]
        except Exception:
            # Lazy loaders, frozen modules and C extensions may refuse the
            # attribute read or the assignment. Skipping them keeps the sweep
            # going for the modules that do hold the CUDA kernel.
            continue
except ImportError:
    pass

# npu_mhc_fused_rmsnorm: the tilelang mHC pre kernel fuses the layer's input
# RMSNorm into layer_input, but mhc_pre_torch (which forward_native/forward_oot
# use on NPU) ignores norm_weight/norm_eps entirely, so every layer would run
# without its input norm. Reapply it here to match the CUDA semantics:
#   layer_input = layer_input * rsqrt(mean(layer_input^2) + norm_eps) * norm_weight
try:
    import torch as _t
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_post_torch as _mhc_post_torch,
    )
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_pre_torch as _mhc_pre_torch,
    )
    from vllm.model_executor.layers import mhc as _mhc_mod

    def _mhc_rms_norm(x, weight, eps):
        xf = x.float()
        var = xf.square().mean(dim=-1, keepdim=True)
        return (xf * _t.rsqrt(var + eps) * weight.float()).to(x.dtype)

    def _mhc_pre_npu(
        self,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        post_mix, comb_mix, layer_input = _mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            layer_input = _mhc_rms_norm(layer_input, norm_weight, norm_eps)
        return post_mix, comb_mix, layer_input

    def _mhc_fused_post_pre_npu(
        self,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        tile_n=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        residual_cur = _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        post_mix_cur, comb_mix_cur, layer_input_cur = _mhc_pre_torch(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            layer_input_cur = _mhc_rms_norm(layer_input_cur, norm_weight, norm_eps)
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    _mhc_mod.MHCPreOp.forward_oot = _mhc_pre_npu
    _mhc_mod.MHCPreOp.forward_native = _mhc_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_oot = _mhc_fused_post_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_native = _mhc_fused_post_pre_npu
except ImportError:
    pass

# npu_kda_causal_conv1d_triton: the PyTorch fallback calls .item() per
# request, which aborts ACL graph capture. The upstream NPU Triton kernel has
# no host sync and accepts the spec-decode kwargs the fallback had to drop.
try:
    from vllm_ascend.ops.triton.mamba.causal_conv1d import (  # type: ignore[attr-defined]
        causal_conv1d_update_npu as _cc1d_update_npu,
    )

    _cc1d.causal_conv1d_update = _cc1d_update_npu
    logger.debug("Bound the NPU Triton causal_conv1d_update for the KDA layers.")
except Exception as _cc1d_err:
    logger.warning(
        "NPU Triton causal_conv1d_update is unavailable (%s); falling back to the"
        " PyTorch implementation, which syncs per request and therefore stalls ACL"
        " graph capture at decode-FULL.",
        _cc1d_err,
    )
