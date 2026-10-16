#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Pure-PyTorch GDN fallbacks for NPU platforms without an active Triton backend.

``patch_triton`` only performs its Triton-dependent work when Triton is
importable, so its ``if not HAS_TRITON`` fallback could never run: the module
itself was imported from ``vllm_ascend.patch.worker`` under ``if HAS_TRITON``
-- opposite guards that made the fallback unreachable in both states.

This module keeps the fallback on an *unguarded* import path (it is imported
unconditionally by ``vllm_ascend.patch.worker``) and self-guards on
``HAS_TRITON`` here, so the import condition and the fallback condition can
never diverge again. It imports no Triton-backed module at import time and is
therefore safe to import on Triton-less platforms.
"""

import torch
import torch.nn.functional as _F
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON


def _fused_post_conv_prep_pytorch(
    conv_output,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    apply_l2norm=True,
    output_g_exp=False,
):
    L = conv_output.shape[0]
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]

    q = conv_output[:, : H * K].reshape(L, H, K)
    k = conv_output[:, H * K : 2 * H * K].reshape(L, H, K)
    v = conv_output[:, 2 * H * K :].reshape(L, HV, V)

    if apply_l2norm:
        # x / sqrt(sum(x^2) + eps) — matches Triton kernel, in fp32
        def _l2norm(t):
            t_f = t.float()
            return (t_f / torch.sqrt((t_f * t_f).sum(-1, keepdim=True) + 1e-6)).to(t.dtype)

        q, k = _l2norm(q), _l2norm(k)

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    x = (a + dt_bias.unsqueeze(0)).float()
    g = -torch.exp(A_log.float().unsqueeze(0)) * _F.softplus(x)
    if output_g_exp:
        g = torch.exp(g)

    return q, k, v, g, torch.sigmoid(b.float())


def _fused_recurrent_packed_decode_pytorch(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    H = (mixed_qkv.shape[1] - HV * V) // (2 * K)
    ratio = HV // H

    q = mixed_qkv[:, : H * K].reshape(B, H, K)
    k = mixed_qkv[:, H * K : 2 * H * K].reshape(B, H, K)
    v = mixed_qkv[:, 2 * H * K :].reshape(B, HV, V)

    SOFTPLUS_THRESHOLD = 20.0
    x = (a + dt_bias.unsqueeze(0)).float()
    softplus_x = torch.where(x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)
    g = -torch.exp(A_log.float().unsqueeze(0)) * softplus_x  # [B, HV]
    beta = torch.sigmoid(b.float())  # [B, HV]

    for n in range(B):
        state_idx = int(ssm_state_indices[n].item())
        if state_idx <= 0:
            out[n, 0] = 0
            continue

        h = initial_state[state_idx].float()  # [HV, V, K]
        q_n = q[n].float().repeat_interleave(ratio, dim=0)  # [HV, K]
        k_n = k[n].float().repeat_interleave(ratio, dim=0)  # [HV, K]
        v_n = v[n].float()  # [HV, V]

        if use_qk_l2norm_in_kernel:

            def _l2norm(t):
                t_f = t.float()
                return t_f / torch.sqrt((t_f * t_f).sum(-1, keepdim=True) + 1e-6)

            q_n, k_n = _l2norm(q_n), _l2norm(k_n)
        q_n = q_n * scale

        h = h * torch.exp(g[n]).view(HV, 1, 1)
        v_n = v_n - torch.einsum("hvk,hk->hv", h, k_n)
        v_n = v_n * beta[n].view(HV, 1)
        h = h + torch.einsum("hv,hk->hvk", v_n, k_n)
        out[n, 0] = torch.einsum("hvk,hk->hv", h, q_n).to(out.dtype)
        initial_state[state_idx] = h.to(initial_state.dtype)

    return out, initial_state


def apply_patch() -> None:
    """Install the pure-PyTorch GDN fallbacks on the FLA ops.

    Both the ``ops`` package re-export and the ``fused_recurrent`` submodule are
    updated. ``qwen_gdn_linear_attn`` imports
    ``fused_recurrent_gated_delta_rule_packed_decode`` from the package while the
    kernel is defined on the submodule, so binding only one of them leaves the
    consumer pointing at the Triton kernel.
    """
    import vllm.third_party.flash_linear_attention.ops as fla_ops
    import vllm.third_party.flash_linear_attention.ops.fused_recurrent as fla_fused_recurrent

    fla_ops.fused_post_conv_prep = _fused_post_conv_prep_pytorch
    fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = (
        _fused_recurrent_packed_decode_pytorch
    )
    fla_ops.fused_recurrent_gated_delta_rule_packed_decode = _fused_recurrent_packed_decode_pytorch


if not HAS_TRITON:
    try:
        apply_patch()
    except ImportError as _exc:  # pragma: no cover - depends on the vLLM tree
        logger.warning(
            "No-Triton GDN PyTorch fallback could not be installed (%s); the"
            " Triton GDN kernels will be used if they are available.",
            _exc,
        )
