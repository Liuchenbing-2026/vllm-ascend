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

"""Pure-PyTorch fallbacks for the GDN multi-step decode / prefill helpers.

Platforms without an active Triton backend (e.g. 310P) cannot import
``patch_triton``: that module rebinds Triton kernels and is therefore only
imported while ``HAS_TRITON`` is true (see ``patch/worker/__init__.py``).
Keeping the fallback next to those Triton patches made it unreachable, so it
lives in this standalone module, which is imported exactly when Triton is
missing.

The helpers are installed on the ``flash_linear_attention.ops`` package and on
every module that already imported them by name, because
``qwen_gdn_linear_attn`` uses ``from ...ops import fused_post_conv_prep``, which
captures the original object in the importing module's namespace at import
time.
"""

import sys

import torch
import torch.nn.functional as F
import vllm.third_party.flash_linear_attention.ops as fla_ops
import vllm.third_party.flash_linear_attention.ops.fused_recurrent as fla_fused_recurrent


def fused_post_conv_prep_pytorch(
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
        # x / sqrt(sum(x^2) + eps) - matches the Triton kernel, in fp32
        def _l2norm(t):
            t_f = t.float()
            return (t_f / torch.sqrt((t_f * t_f).sum(-1, keepdim=True) + 1e-6)).to(t.dtype)

        q, k = _l2norm(q), _l2norm(k)

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    x = (a + dt_bias.unsqueeze(0)).float()
    g = -torch.exp(A_log.float().unsqueeze(0)) * F.softplus(x)
    if output_g_exp:
        g = torch.exp(g)

    return q, k, v, g, torch.sigmoid(b.float())


def fused_recurrent_gated_delta_rule_packed_decode_pytorch(
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


def _rebind(name: str, original, replacement) -> None:
    """Point the modules that imported ``original`` at ``replacement``."""
    for module in list(sys.modules.values()):
        try:
            if getattr(module, name, None) is not original:
                continue
            setattr(module, name, replacement)
        except Exception:
            # Lazy loaders, frozen modules and C extensions may refuse the
            # attribute read. Skipping them keeps the sweep going for the
            # modules that do hold the imported callable.
            continue


def apply() -> None:
    original_post_conv_prep = fla_ops.fused_post_conv_prep
    original_packed_decode = fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode
    fla_ops.fused_post_conv_prep = fused_post_conv_prep_pytorch
    fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = (
        fused_recurrent_gated_delta_rule_packed_decode_pytorch
    )
    _rebind("fused_post_conv_prep", original_post_conv_prep, fused_post_conv_prep_pytorch)
    _rebind(
        "fused_recurrent_gated_delta_rule_packed_decode",
        original_packed_decode,
        fused_recurrent_gated_delta_rule_packed_decode_pytorch,
    )


apply()
