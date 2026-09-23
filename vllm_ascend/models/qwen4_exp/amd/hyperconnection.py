# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities for the AMD model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This AMD variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


# --- round 103: FRACTAL_NZ storage for the HC projections -------------------
# Inert unless QWEN38_HC_NZ is set; see patch_hc_nz.py for the measurement
# that prices it.  Kept in the model file rather than in ops/hc.py because the
# hook that reaches it is the Linear post-load pass, not the glue kernels.
import os as _hc_os

try:  # the delivery image always has torch_npu; keep the import defensive
    import torch_npu as _hc_torch_npu
except ImportError:  # pragma: no cover
    _hc_torch_npu = None

from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod as _HcUnquantizedLinearMethod,
)

_HC_NZ_FORMAT = 29  # ACL_FORMAT_FRACTAL_NZ

# Importing this module registers ``torch.ops.vllm.unquantized_gemm``, the
# opaque F.linear the vendor's own NZ path uses.  It is the same op every other
# Ascend linear already issues, so importing it cannot change those graphs.
try:  # pragma: no cover
    import vllm_ascend.ops.linear as _hc_ascend_linear  # noqa: F401
except Exception:  # pragma: no cover
    _hc_ascend_linear = None


def _hc_nz_level() -> str:
    return _hc_os.environ.get("QWEN38_HC_NZ", "0")


def _hc_maybe_cast_nz(weight, role: str):
    """Return ``weight`` stored FRACTAL_NZ, or ``weight`` untouched.

    ND and NZ are two views of the same matmul, so this is a layout change and
    nothing else (measured max|d| = 0 over the full tensor).  Three shapes are
    exempt because aclnn rejects them outright -- fp32 cannot use NZ, a meta
    tensor has no storage to cast, and ``mat2`` with n == 1 or k == 1 raises
    ``AclNN_Parameter_Error(EZ1001)``.  None of the three HC projections hits
    an exemption at 320/336/10240, so they are safety rails, not policy.
    """
    level = _hc_nz_level()
    if _hc_torch_npu is None or level == "0":
        return weight
    if level == "1" and role != "up":
        return weight
    if weight.dim() < 2 or weight.shape[0] == 1 or weight.shape[1] == 1:
        return weight
    if weight.dtype == torch.float32 or weight.is_meta:
        return weight
    cast = _hc_torch_npu.npu_format_cast(weight, _HC_NZ_FORMAT)
    if _hc_os.environ.get("QWEN38_HC_NZ_VERBOSE") == "1":
        print(
            "[hc_nz] role=%s shape=%s dtype=%s source=%s stored=%s"
            % (
                role,
                tuple(weight.shape),
                weight.dtype,
                _hc_torch_npu.get_npu_format(weight),
                _hc_torch_npu.get_npu_format(cast),
            ),
            flush=True,
        )
    return cast


# --- round 208: int8 (W8A8-dynamic) storage for the HC projections ----------
# Inert unless ``QWEN38_HC_W8A8`` is set, so the champion line is unchanged
# when the flag is absent.  ``hc_gemm_probe.py`` prices the same four variants
# the vendor's own ``AscendW8A8DynamicLinearMethod`` issues, so whatever it
# reports here is the vendor's operator, not an invention of this patch.
#
#   QWEN38_HC_W8A8      "0" off | "1" up only | "2" up + down | "3" down only
#   QWEN38_HC_W8A8_NZ   "1" (default) store the int8 weight FRACTAL_NZ
#
# The two projections carry no checkpoint scale (the index has no
# ``..._scale``/``..._offset`` for them), so the per-output-channel scale is
# computed here from the bf16 weight itself: ``scale = amax(row)/127``,
# ``wq = round(w/scale)`` -- a symmetric int8 cast, which is exactly what the
# probe's V2/V3 do and what the checkpoint's own W8A8 linears were produced
# with.  Activations go through ``torch_npu.npu_dynamic_quant`` per token,
# again the vendor path.  This is a *numeric* change, so it may only be
# delivered behind ``acc_tf.py``'s HC floor.
_HC_W8A8_NZ_FORMAT = 29  # ACL_FORMAT_FRACTAL_NZ


def _hc_w8a8_level() -> str:
    return _hc_os.environ.get("QWEN38_HC_W8A8", "0")


def _hc_w8a8_nz() -> bool:
    return _hc_os.environ.get("QWEN38_HC_W8A8_NZ", "1") == "1"


def _hc_w8a8_role_selected(level: str, role: str) -> bool:
    if level == "1":
        return role == "up"
    if level == "2":
        return role in ("up", "down")
    if level == "3":
        return role == "down"
    return False


def _hc_maybe_quant_int8(weight, role: str):
    """Return ``(wq[k, n], scale[n])`` int8, or ``(None, None)`` untouched.

    Mirrors ``to_hc_gemm_probe``: quantization is per output channel of the
    bf16 weight (``row`` of the ``[n, k]`` checkpoint layout), the stored
    operand is the ``[k, n]`` transpose ``npu_quant_matmul`` wants, and the
    scale keeps the activation dtype the vendor's own method hands over.
    """
    level = _hc_w8a8_level()
    if level == "0" or _hc_torch_npu is None:
        return None, None
    if not _hc_w8a8_role_selected(level, role):
        return None, None
    if weight.dim() != 2 or weight.is_meta:
        return None, None
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return None, None
    wf = weight.detach().float()
    amax = wf.abs().amax(dim=1).clamp_min(1e-8)
    scale = amax / 127.0
    wq = (wf / scale.unsqueeze(1)).round().clamp(-127.0, 127.0)
    wq = wq.to(torch.int8).t().contiguous()  # [k, n]
    if _hc_w8a8_nz():
        wq = _hc_torch_npu.npu_format_cast(wq, _HC_W8A8_NZ_FORMAT)
    if _hc_os.environ.get("QWEN38_HC_W8A8_VERBOSE") == "1":
        print(
            "[hc_w8a8] role=%s shape=%s->%s scale=%s"
            % (role, tuple(weight.shape), tuple(wq.shape), tuple(scale.shape)),
            flush=True,
        )
    return wq, scale.to(weight.dtype)


# --- round 216: re-point the cached custom_op at the method just installed --
# ``AscendLinearBase.__init__`` builds ``custom_op`` and
# ``CustomLinearOp.update_attrs`` snapshots ``layer.quant_method`` into it.
# The three HC projections replace that method after construction (the int8 /
# NZ cast needs a post-load hook), so without this refresh the forward path
# keeps calling the snapshotted ``AscendUnquantizedLinearMethod`` while the
# weight is the int8 operand the replacement produced -- which is exactly the
# ``aclnnMatmulWeightNz ... not implemented for DT_INT8`` crash the round-215
# probe caught at ops/linear.py:129.
def _hc_refresh_op(layer) -> None:
    op = getattr(layer, "custom_op", None)
    if op is not None:
        op.quant_method = layer.quant_method


# --- round 228: K-aligned up projection -------------------------------------
# The up projection is [M, 320] @ [10240, 320]^T -- its K is ``lora_rank`` =
# 320, which is 2.5 x the 128-deep NZ tile.  ``hc_gemm_probe4.py`` measures
# that launch on a free A3 chip at 23.5 us / 266 GB/s; zero-padding K to 512
# costs 1.6x the FLOPs and 1.6x the weight bytes and comes back at 14.4 us /
# 694 GB/s.  The limit was the vendor kernel's tiling, not the arithmetic, and
# because both operands are padded with exact zeros every extra product is 0,
# so the sum is unchanged (the probe reports the deviation of the *padding* as
# max|d| = 0 against the unpadded result; only the pre-existing bf16 rounding
# differs).
#
#   QWEN38_HC_UPK   0 (default, off) | 384 | 512 | 640
#
# The weight pad is a load-time transform and therefore free at run time.  The
# activation pad is one ``F.pad``: 0.2 MB of extra HBM traffic and no state to
# carry across steps, which matters because ``apply`` also runs inside the
# captured decode graph.  probe5 reports max|d| = 0.000e+00 between the K=320
# and the K=512 result on the real weight path.
def _hc_upk_width() -> int:
    try:
        return int(_hc_os.environ.get("QWEN38_HC_UPK", "0"))
    except (TypeError, ValueError):
        return 0


def _hc_pad_k(x, k: int):
    """``x`` widened to ``k`` columns with an exactly-zero tail."""
    return torch.nn.functional.pad(x, (0, k - x.shape[-1]))


class HcNzLinearMethod(_HcUnquantizedLinearMethod):
    """``UnquantizedLinearMethod`` that may store its weight FRACTAL_NZ.

    The cast lives in ``process_weights_after_loading`` because that is the
    only generic hook vLLM offers a module that is not an attention module and
    is not an ``HpcModule``.  ``apply`` changes only for a layer that really
    was cast: an NZ weight has to reach the NPU behind an opaque op, because
    Inductor would otherwise decompose ``F.linear`` into ``mm(x, w.t())`` and
    there is no ``t()`` of an NZ tensor.
    """

    def __init__(self, role: str = "up") -> None:
        super().__init__()
        self.role = role
        self.nz = False
        self.quant = False
        self.k_pad = 0

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        w0 = layer.weight.data
        # round 228: widen K before any cast, so the NZ (and the int8) operand
        # is built from the padded weight and the GEMM itself sees the aligned
        # K.  Inert when QWEN38_HC_UPK is unset.
        kpad = _hc_upk_width()
        if kpad and self.role == "up" and w0.dim() == 2 and 0 < w0.shape[1] < kpad:
            wide = torch.zeros(
                w0.shape[0], kpad, dtype=w0.dtype, device=w0.device
            )
            wide[:, : w0.shape[1]] = w0
            w0 = wide
            self.k_pad = kpad
        # int8 first: a quantized weight is stored as the [k, n] int8 operand
        # and never goes through the bf16 NZ cast.
        wq, wscale = _hc_maybe_quant_int8(w0, self.role)
        if wq is not None:
            self.quant = True
            layer.weight.data = wq
            layer.weight_scale = wscale.flatten()
            return
        cast = _hc_maybe_cast_nz(w0, self.role)
        # Identity, not ``get_npu_format``: the format query is a host-side op
        # and this decision has to be a plain python bool by dictionary-capture
        # time, not a tensor read inside the traced region.
        #
        # The comparison is against ``w0``, the object that was handed to the
        # cast, and *not* against a second ``layer.weight.data``: on a
        # ``Parameter`` (requires_grad True) ``.data`` returns a fresh detached
        # view on every access, so ``cast is layer.weight.data`` would read
        # True even when no cast happened -- which would silently send the
        # control row down the opaque-op path.  ``hc_nz_probe.py`` asserts
        # ``method.nz`` against the tensor's actual format, so that mistake
        # fails the probe instead of the ledger.
        self.nz = cast is not w0
        layer.weight.data = cast

    def apply(self, layer, x, bias=None):
        if self.k_pad and x.dim() == 2 and 0 < x.shape[-1] < self.k_pad:
            x = _hc_pad_k(x, self.k_pad)
        if self.quant:
            qx, pertoken_scale = _hc_torch_npu.npu_dynamic_quant(
                x, dst_type=torch.int8
            )
            if pertoken_scale.dim() == 2:
                qx = qx.squeeze(dim=1)
                pertoken_scale = pertoken_scale.squeeze(dim=1)
            return _hc_torch_npu.npu_quant_matmul(
                qx,
                layer.weight,
                layer.weight_scale,
                pertoken_scale=pertoken_scale,
                output_dtype=x.dtype,
            )
        if self.nz:
            if _hc_ascend_linear is None:
                raise RuntimeError(
                    "QWEN38_HC_NZ produced an FRACTAL_NZ weight but "
                    "torch.ops.vllm.unquantized_gemm is unavailable; refusing "
                    "to hand an NZ tensor to the decomposing path"
                )
            return torch.ops.vllm.unquantized_gemm(x, layer.weight, bias)
        return super().apply(layer, x, bias)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows for
        # alignment and efficient backend dispatch.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )
        # round 103: the three HC projections carry a post-load method so the
        # NZ cast has somewhere to live -- vLLM reaches a module that is not an
        # Attention/HpcModule only through its quant_method
        # (model_executor/model_loader/utils.py:101-107).  The cast itself is
        # env-gated inside the method, so with QWEN38_HC_NZ unset this is one
        # extra inert line per linear and the graphs are unchanged.
        #
        # Three explicit assignments, each carrying a *literal* role, rather
        # than a loop over a table: hc_nz_probe.py's wiring claim reads the
        # roles out of this AST, and a literal is the only form that can be
        # read.  Two of the three are "down" because the merged
        # down+inject GEMM exists only when use_combine is true and the plain
        # down projection only when it is false -- at most two of these three
        # modules exist on any one GatedResidual.
        if getattr(self, "input_mix_weight_down", None) is not None:
            self.input_mix_weight_down.quant_method = HcNzLinearMethod("down")
            _hc_refresh_op(self.input_mix_weight_down)
        if getattr(self, "input_mix_weight_down_block_inject", None) is not None:
            self.input_mix_weight_down_block_inject.quant_method = (
                HcNzLinearMethod("down")
            )
            _hc_refresh_op(self.input_mix_weight_down_block_inject)
        self.input_mix_weight_up.quant_method = HcNzLinearMethod("up")
        _hc_refresh_op(self.input_mix_weight_up)

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
