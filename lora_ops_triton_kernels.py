"""Triton bgmv/sgmv kernels (validated bit-exact vs AscendC on 910B4).

Adapted from /tmp/bgmv-triton + /tmp/sgmv_repo (tail-block mask fix).
expand kernels support slice semantics (y_out[:, offset:offset+size] += x @ w[:, offset:offset+size]).
"""
import triton
import triton.language as tl

# ---- bgmv ----

GRID_HINT_B = ["B"]

_WIN: tl.constexpr = 11776  # native TILE_LENGTH


@triton.jit
def bgmv_shrink(X, lora_a, indices, y, scale,
                H: tl.constexpr, R: tl.constexpr, L: tl.constexpr):
    b = tl.program_id(0)
    idx = tl.load(indices + b)

    r = tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)

    if idx >= 0:
        x_base = X + b * H
        a_base = lora_a + idx * (R * H)
        # Match native bgmv_shrink reduction structure: H is split into
        # TILE_LENGTH=11776 windows; each window's Mul+ReduceSum is independent
        # and window partials accumulate in fp32 (sequential, in order). Within
        # a window a sequential 64-chunk tl.sum is bit-exact vs the hardware
        # ReduceSum (verified for all H <= 11776); continuing the fp32
        # accumulation straight across a window boundary is NOT bit-exact
        # (down_proj H=17408 = 11776+5632 needs the per-window restart).
        for w in range(0, H // _WIN):
            acc_w = tl.zeros([R], dtype=tl.float32)
            for h0 in range(0, _WIN, 64):
                hs = w * _WIN + h0 + tl.arange(0, 64)
                x = tl.load(x_base + hs).to(tl.float32)
                a = tl.load(a_base + r[:, None] * H + hs[None, :]).to(tl.float32)
                acc_w += tl.sum(a * x[None, :], axis=1)
            acc += acc_w
        tail = (H // _WIN) * _WIN
        acc_t = tl.zeros([R], dtype=tl.float32)
        for h0 in range(tail, H, 64):
            hs = h0 + tl.arange(0, 64)
            m = hs < H
            x = tl.load(x_base + hs, mask=m, other=0.0).to(tl.float32)
            a = tl.load(a_base + r[:, None] * H + hs[None, :],
                        mask=m[None, :], other=0.0).to(tl.float32)
            acc_t += tl.sum(a * x[None, :], axis=1)
        acc += acc_t
        acc = acc * scale

    # AscendC: idx < 0 => skip row, y left unchanged
    old = tl.load(y + b * R + r)
    out = tl.where(idx >= 0, acc, old)
    tl.store(y + b * R + r, out)


@triton.jit
def bgmv_expand(
    y_ptr,
    lora_b_ptr,
    indices_ptr,
    y_in_ptr,
    y_out_ptr,
    R: tl.constexpr,
    Ho: tl.constexpr,        # weight slice dim (lora_b already narrowed by caller)
    L: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    Y_HO: tl.constexpr,      # full output dim of y tensors
    SLICE_OFF: tl.constexpr, # column offset of this slice in y
):
    b = tl.program_id(0)

    idx = tl.load(indices_ptr + b)
    safe_idx = tl.minimum(tl.maximum(idx, 0), L - 1)

    r_offs = tl.arange(0, R)
    y_row = tl.load(y_ptr + b * R + r_offs)  # [R] fp32

    for ho_start in range(0, Ho, BLOCK_HO):
        ho_offs = ho_start + tl.arange(0, BLOCK_HO)
        ho_mask = ho_offs < Ho

        lora_offs = safe_idx * (Ho * R) + ho_offs[:, None] * R + r_offs[None, :]
        lora_vals = tl.load(lora_b_ptr + lora_offs, mask=ho_mask[:, None], other=0.0)

        prod = lora_vals.to(tl.float32) * y_row[None, :]
        acc = tl.sum(prod, axis=1)  # [BLOCK_HO] fp32
        acc = tl.where(idx >= 0, acc, 0.0)

        out_offs = b * Y_HO + SLICE_OFF + ho_offs
        y_in = tl.load(y_in_ptr + out_offs, mask=ho_mask, other=0.0)
        res = (y_in.to(tl.float32) + acc).to(y_in.dtype)

        tl.store(y_out_ptr + out_offs, res, mask=ho_mask)


# ---- sgmv ----

_BLOCK_H: tl.constexpr = 64


@triton.jit
def sgmv_shrink_kernel(
    X,            # [B, H]    fp16/bf16 activations, one row per token
    lora_a,       # [L, R, H] fp16/bf16 LoRA A weights (H contiguous)
    indices,      # [NR] int32, LoRA index per request (-1 => skip whole request)
    token_nums,   # [NR] int32, token count per request (sum == B)
    y,            # [B, R] fp32 output
    scale: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    NR: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- token row -> request mapping (prefix sum of token_nums), NR is a power of 2 (1,2,4,8) ----
    offs_nr = tl.arange(0, NR)
    cnt = tl.load(token_nums + offs_nr).to(tl.int32)          # [NR]
    idxs = tl.load(indices + offs_nr).to(tl.int32)            # [NR]

    lower = offs_nr[None, :] < offs_nr[:, None]               # [NR,NR] strictly-lower triangle
    start = tl.sum(tl.where(lower, cnt[None, :], 0), axis=1)  # exclusive prefix sum, [NR]
    end = start + cnt

    in_seg = (pid >= start) & (pid < end)                     # exactly one True when sum==B
    # max over segments: rows beyond the token total yield -1 => skip row
    idx = tl.max(tl.where(in_seg, idxs, -1), axis=0)          # scalar LoRA index (-1 => skip)

    keep = tl.where(idx >= 0, 1.0, 0.0)                       # scalar fp32 gate
    idx_safe = tl.maximum(idx, 0)                             # keep all loads in bounds

    offs_r = tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)

    a_base = idx_safe * (R * H)
    x_base = pid * H

    # Same native-window reduction structure as bgmv_shrink (TILE_LENGTH=11776).
    for w in range(0, H // _WIN):
        acc_w = tl.zeros([R], dtype=tl.float32)
        for h0 in range(0, _WIN, _BLOCK_H):
            offs_h = w * _WIN + h0 + tl.arange(0, _BLOCK_H)
            x_blk = tl.load(X + x_base + offs_h).to(tl.float32)          # [BH]
            a_blk = tl.load(
                lora_a + a_base + offs_r[:, None] * H + offs_h[None, :],
            ).to(tl.float32)                                             # [R, BH]
            acc_w += tl.sum(a_blk * x_blk[None, :], axis=1)
        acc += acc_w
    tail = (H // _WIN) * _WIN
    acc_t = tl.zeros([R], dtype=tl.float32)
    for h0 in range(tail, H, _BLOCK_H):
        offs_h = h0 + tl.arange(0, _BLOCK_H)
        hmask = offs_h < H
        x_blk = tl.load(X + x_base + offs_h, mask=hmask, other=0.0).to(tl.float32)   # [BH]
        a_blk = tl.load(
            lora_a + a_base + offs_r[:, None] * H + offs_h[None, :],
            mask=hmask[None, :],
            other=0.0,
        ).to(tl.float32)                                                             # [R, BH]
        acc_t += tl.sum(a_blk * x_blk[None, :], axis=1)
    acc += acc_t

    # AscendC: idx < 0 => skip row, y left unchanged
    old = tl.load(y + pid * R + offs_r)
    out = acc * scale * keep + old * (1.0 - keep)
    tl.store(y + pid * R + offs_r, out)


@triton.jit
def sgmv_expand(
    X_ptr,            # fp32 [B, R]
    lora_b_ptr,       # fp16/bf16 [L, Ho, R]  (Ho = slice dim, already narrowed by caller)
    indices_ptr,      # int32 [NR]
    token_nums_ptr,   # int32 [NR]
    y_in_ptr,         # fp16/bf16 [B, Y_HO]
    y_out_ptr,        # fp16/bf16 [B, Y_HO]
    R: tl.constexpr,
    Ho: tl.constexpr,
    L: tl.constexpr,
    NR: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    Y_HO: tl.constexpr,
    SLICE_OFF: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- map token row -> request via exclusive prefix sum of token_nums ----
    r_off = tl.arange(0, NR)
    cnts = tl.load(token_nums_ptr + r_off).to(tl.float32)      # [NR]
    idxs = tl.load(indices_ptr + r_off).to(tl.float32)         # [NR]
    tri = (r_off[None, :] < r_off[:, None]).to(tl.float32)     # [NR, NR]
    starts = tl.sum(tri * cnts[None, :], axis=1)               # [NR] exclusive prefix
    pf = pid.to(tl.float32)
    in_seg = (pf >= starts) & (pf < starts + cnts)
    sel = tl.max(tl.where(in_seg, idxs, -1.0), axis=0)         # scalar fp32
    lora_id = sel.to(tl.int32)
    safe_id = tl.maximum(lora_id, 0)
    skip_scale = tl.where(sel >= 0.0, 1.0, 0.0)

    # ---- fp32 shrink-output row ----
    r_idx = tl.arange(0, R)
    x = tl.load(X_ptr + pid * R + r_idx)                       # fp32 [R]

    base = safe_id * (Ho * R)
    for h0 in range(0, Ho, BLOCK_HO):
        ho = h0 + tl.arange(0, BLOCK_HO)
        ho_mask = ho < Ho
        w = tl.load(lora_b_ptr + base + ho[:, None] * R + r_idx[None, :],
                    mask=ho_mask[:, None], other=0.0)
        acc = tl.sum(w.to(tl.float32) * x[None, :], axis=1)    # fp32 [BLOCK_HO]
        out_offs = pid * Y_HO + SLICE_OFF + ho
        yi = tl.load(y_in_ptr + out_offs, mask=ho_mask, other=0.0).to(tl.float32)
        out = yi + acc * skip_scale
        tl.store(y_out_ptr + out_offs, out.to(y_out_ptr.dtype.element_ty),
                 mask=ho_mask)
