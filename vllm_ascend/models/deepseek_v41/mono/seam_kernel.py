# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Triton-Ascend fused mHC seam and Sinkhorn kernels for DSv4.1 MonoKernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_DEFAULT_BLOCK_D = 1024


@triton.jit(do_not_specialize=["total_rows", "hidden_size"])
def _mono_hc_post_collapse_norm_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    norm_w_ptr,
    norm_b_ptr,
    res_out_ptr,
    norm_out_ptr,
    inv_rms_4d_ptr,
    stride_xm,
    stride_rm,
    stride_rl,
    stride_pm,
    stride_cm,
    stride_prem,
    stride_rom,
    stride_rol,
    stride_om,
    norm_eps,
    norm_4d_eps,
    total_rows,
    hidden_size,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    off_d_base = tl.arange(0, BLOCK_D)
    rows_per_core = tl.cdiv(total_rows, num_cores)
    start_row = pid * rows_per_core
    end_row = tl.minimum(start_row + rows_per_core, total_rows)

    for row in tl.range(start_row, end_row):
        pb, kbase, cb = row * stride_pm, row * stride_prem, row * stride_cm
        p0, p1 = tl.load(post_ptr + pb).to(tl.float32), tl.load(post_ptr + pb + 1).to(tl.float32)
        p2, p3 = tl.load(post_ptr + pb + 2).to(tl.float32), tl.load(post_ptr + pb + 3).to(tl.float32)
        k0, k1 = tl.load(pre_ptr + kbase).to(tl.float32), tl.load(pre_ptr + kbase + 1).to(tl.float32)
        k2, k3 = tl.load(pre_ptr + kbase + 2).to(tl.float32), tl.load(pre_ptr + kbase + 3).to(tl.float32)
        c00, c01 = tl.load(comb_ptr + cb).to(tl.float32), tl.load(comb_ptr + cb + 1).to(tl.float32)
        c02, c03 = tl.load(comb_ptr + cb + 2).to(tl.float32), tl.load(comb_ptr + cb + 3).to(tl.float32)
        c10, c11 = tl.load(comb_ptr + cb + 4).to(tl.float32), tl.load(comb_ptr + cb + 5).to(tl.float32)
        c12, c13 = tl.load(comb_ptr + cb + 6).to(tl.float32), tl.load(comb_ptr + cb + 7).to(tl.float32)
        c20, c21 = tl.load(comb_ptr + cb + 8).to(tl.float32), tl.load(comb_ptr + cb + 9).to(tl.float32)
        c22, c23 = tl.load(comb_ptr + cb + 10).to(tl.float32), tl.load(comb_ptr + cb + 11).to(tl.float32)
        c30, c31 = tl.load(comb_ptr + cb + 12).to(tl.float32), tl.load(comb_ptr + cb + 13).to(tl.float32)
        c32, c33 = tl.load(comb_ptr + cb + 14).to(tl.float32), tl.load(comb_ptr + cb + 15).to(tl.float32)

        sum_sq_u, sum_sq_4d = 0.0, 0.0
        rm, rom, xm, om = row * stride_rm, row * stride_rom, row * stride_xm, row * stride_om

        for d_start in tl.range(0, hidden_size, BLOCK_D):
            d_off = d_start + off_d_base
            mask_d = d_off < hidden_size
            xv = tl.load(x_ptr + xm + d_off, mask=mask_d, other=0.0).to(tl.float32)
            r0 = tl.load(res_ptr + rm + d_off, mask=mask_d, other=0.0).to(tl.float32)
            r1 = tl.load(res_ptr + rm + stride_rl + d_off, mask=mask_d, other=0.0).to(tl.float32)
            r2 = tl.load(res_ptr + rm + 2 * stride_rl + d_off, mask=mask_d, other=0.0).to(tl.float32)
            r3 = tl.load(res_ptr + rm + 3 * stride_rl + d_off, mask=mask_d, other=0.0).to(tl.float32)

            y0 = (p0 * xv + c00 * r0 + c10 * r1 + c20 * r2 + c30 * r3).to(tl.bfloat16)
            y1 = (p1 * xv + c01 * r0 + c11 * r1 + c21 * r2 + c31 * r3).to(tl.bfloat16)
            y2 = (p2 * xv + c02 * r0 + c12 * r1 + c22 * r2 + c32 * r3).to(tl.bfloat16)
            y3 = (p3 * xv + c03 * r0 + c13 * r1 + c23 * r2 + c33 * r3).to(tl.bfloat16)
            tl.store(res_out_ptr + rom + d_off, y0, mask=mask_d)
            tl.store(res_out_ptr + rom + stride_rol + d_off, y1, mask=mask_d)
            tl.store(res_out_ptr + rom + 2 * stride_rol + d_off, y2, mask=mask_d)
            tl.store(res_out_ptr + rom + 3 * stride_rol + d_off, y3, mask=mask_d)

            f0, f1, f2, f3 = y0.to(tl.float32), y1.to(tl.float32), y2.to(tl.float32), y3.to(tl.float32)
            sum_sq_4d += tl.sum(tl.where(mask_d, f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3, 0.0), axis=0)
            u_tile = (k0 * f0 + k1 * f1 + k2 * f2 + k3 * f3).to(tl.bfloat16)
            tl.store(norm_out_ptr + om + d_off, u_tile, mask=mask_d)
            uf = u_tile.to(tl.float32)
            sum_sq_u += tl.sum(tl.where(mask_d, uf * uf, 0.0), axis=0)

        inv_rms_u = tl.rsqrt(sum_sq_u / hidden_size + norm_eps)
        tl.store(inv_rms_4d_ptr + row, tl.rsqrt(sum_sq_4d / (4 * hidden_size) + norm_4d_eps))

        for d_start in tl.range(0, hidden_size, BLOCK_D):
            d_off = d_start + off_d_base
            mask_d = d_off < hidden_size
            uf = tl.load(norm_out_ptr + om + d_off, mask=mask_d, other=0.0).to(tl.float32)
            w = tl.load(norm_w_ptr + d_off, mask=mask_d, other=0.0).to(tl.float32)
            ov = uf * inv_rms_u * w
            if HAS_BIAS:
                ov += tl.load(norm_b_ptr + d_off, mask=mask_d, other=0.0).to(tl.float32)
            tl.store(norm_out_ptr + om + d_off, ov.to(tl.bfloat16), mask=mask_d)


def mono_hc_post_collapse_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    norm_4d_eps: float,
    norm_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused NPU Triton kernel for hc_post + hc_collapse + RMSNorm + 4D inv_rms."""
    if x.ndim != 2 or residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError(f"Expected x [M, D] and residual [M, 4, D], got {x.shape} and {residual.shape}")
    total_rows, hidden_size = x.shape
    res_out = torch.empty_like(residual)
    norm_out = torch.empty_like(x)
    inv_rms_4d = torch.empty((total_rows, 1), dtype=torch.float32, device=x.device)
    if total_rows == 0:
        return res_out, norm_out, inv_rms_4d

    post_c, comb_c, pre_c = post.contiguous(), comb.contiguous(), pre.contiguous()
    w_c = norm_weight.contiguous()
    has_bias = norm_bias is not None
    b_c = norm_bias.contiguous() if norm_bias is not None else w_c

    grid = (min(get_vectorcore_num(), total_rows),)
    _mono_hc_post_collapse_norm_kernel[grid](
        x,
        residual,
        post_c,
        comb_c,
        pre_c,
        w_c,
        b_c,
        res_out,
        norm_out,
        inv_rms_4d,
        x.stride(0),
        residual.stride(0),
        residual.stride(1),
        post_c.stride(0),
        comb_c.stride(0),
        pre_c.stride(0),
        res_out.stride(0),
        res_out.stride(1),
        norm_out.stride(0),
        float(norm_eps),
        float(norm_4d_eps),
        total_rows,
        hidden_size,
        HAS_BIAS=has_bias,
        BLOCK_D=_DEFAULT_BLOCK_D,
    )
    return res_out, norm_out, inv_rms_4d


@triton.jit(do_not_specialize=["total_rows", "sinkhorn_iters"])
def _mono_hc_sinkhorn_kernel(
    mixes_ptr,
    inv_rms_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    stride_m,
    total_rows,
    sinkhorn_iters,
    hc_eps,
):
    row = tl.program_id(0)
    if row >= total_rows:
        return
    inv_rms = tl.load(inv_rms_ptr + row).to(tl.float32)
    s0 = tl.load(scale_ptr).to(tl.float32)
    s1 = tl.load(scale_ptr + 1).to(tl.float32)
    s2 = tl.load(scale_ptr + 2).to(tl.float32)

    off4 = tl.arange(0, 4)
    base_m = row * stride_m
    p_raw = tl.load(mixes_ptr + base_m + off4).to(tl.float32) * inv_rms * s0 + tl.load(base_ptr + off4).to(tl.float32)
    po_raw = tl.load(mixes_ptr + base_m + 4 + off4).to(tl.float32) * inv_rms * s1 + tl.load(base_ptr + 4 + off4).to(
        tl.float32
    )
    tl.store(pre_ptr + row * 4 + off4, 1.0 / (1.0 + tl.exp(-p_raw)) + hc_eps)
    tl.store(post_ptr + row * 4 + off4, 2.0 / (1.0 + tl.exp(-po_raw)))

    r = tl.arange(0, 4)[:, None]
    c = tl.arange(0, 4)[None, :]
    c_off = r * 4 + c
    cm = tl.load(mixes_ptr + base_m + 8 + c_off).to(tl.float32) * inv_rms * s2 + tl.load(base_ptr + 8 + c_off).to(
        tl.float32
    )
    e = tl.exp(cm - tl.max(cm, axis=1)[:, None])
    cm = e / tl.sum(e, axis=1)[:, None] + hc_eps
    cm = cm / (tl.sum(cm, axis=0)[None, :] + hc_eps)
    for _ in tl.range(1, sinkhorn_iters):
        cm = cm / (tl.sum(cm, axis=1)[:, None] + hc_eps)
        cm = cm / (tl.sum(cm, axis=0)[None, :] + hc_eps)
    tl.store(comb_ptr + row * 16 + c_off, cm)


def _triton_hc_sinkhorn(
    mixes: torch.Tensor,
    inv_rms_4d: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = mixes.shape[0]
    pre = torch.empty((total_rows, 4), dtype=torch.float32, device=mixes.device)
    post = torch.empty((total_rows, 4), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((total_rows, 4, 4), dtype=torch.float32, device=mixes.device)
    if total_rows == 0:
        return pre, post, comb
    scale_c, base_c = hc_scale.contiguous(), hc_base.contiguous()
    _mono_hc_sinkhorn_kernel[(total_rows,)](
        mixes,
        inv_rms_4d,
        scale_c,
        base_c,
        pre,
        post,
        comb,
        mixes.stride(0),
        total_rows,
        max(int(sinkhorn_iters), 1),
        float(hc_eps),
    )
    return pre, post, comb


def compute_hc_mixes_from_inv_rms(
    residual_flat_f32: torch.Tensor,
    inv_rms_4d: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (pre, post, comb) mHC coefficients reusing Pass-1 ``inv_rms_4d``."""
    mixes = F.linear(residual_flat_f32, hc_fn)
    if mixes.device.type == "npu":
        return _triton_hc_sinkhorn(mixes, inv_rms_4d, hc_scale, hc_base, sinkhorn_iters, hc_eps)
    mixes = mixes * inv_rms_4d
    pre_raw, post_raw, comb_raw = mixes.split([4, 4, 16], dim=-1)
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:4]) + hc_eps
    post = 2.0 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[4:8])
    comb = comb_raw.unflatten(-1, (4, 4)) * hc_scale[2] + hc_base[8:].view(4, 4)
    comb = comb.softmax(dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, comb
