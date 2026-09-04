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


# ==========================================================================
# v2 kernels
# ==========================================================================

_BLK_DEFAULT: tl.constexpr = 512   # divides 5120, 6144, 11776 and 17408-11776


@triton.jit
def sgmv_shrink_v2(X, lora_a, indices, token_nums, y,
                   scale: tl.constexpr,
                   H: tl.constexpr, R: tl.constexpr, L: tl.constexpr,
                   NR: tl.constexpr, BLK: tl.constexpr, NJ: tl.constexpr,
                   EXACT: tl.constexpr):
    """grid (B,).  Same [R, .] tile shape as v1 but BLK (=512) wide instead of
    64, so each lora_a row moves in 1 KB bursts instead of 128 B ones.

    EXACT=1 reproduces v1's summation order exactly (reduce each 64-wide group,
    then add the NJ group results back in sequence) and is bit-identical to the
    AscendC kernel; it costs ~2.5x.  EXACT=0 reduces the whole BLK block at once
    -- measured max relative deviation 4.4e-7, i.e. 1e-4 of one bf16 ulp.
    """
    pid = tl.program_id(0)

    offs_nr = tl.arange(0, NR)
    cnt = tl.load(token_nums + offs_nr).to(tl.int32)
    idxs = tl.load(indices + offs_nr).to(tl.int32)
    lower = offs_nr[None, :] < offs_nr[:, None]
    start = tl.sum(tl.where(lower, cnt[None, :], 0), axis=1)
    in_seg = (pid >= start) & (pid < start + cnt)
    idx = tl.max(tl.where(in_seg, idxs, -1), axis=0)
    idx_safe = tl.maximum(idx, 0)

    offs_r = tl.arange(0, R)
    jr = tl.arange(0, NJ)
    acc = tl.zeros([R], dtype=tl.float32)
    a_base = idx_safe * (R * H)
    x_base = pid * H

    for w in range(0, H // _WIN):
        acc_w = tl.zeros([R], dtype=tl.float32)
        for h0 in range(0, _WIN, BLK):
            oh = w * _WIN + h0 + tl.arange(0, BLK)
            xb = tl.load(X + x_base + oh).to(tl.float32)
            ab = tl.load(lora_a + a_base + offs_r[:, None] * H + oh[None, :]).to(tl.float32)
            pr = ab * xb[None, :]
            if EXACT:
                p = tl.sum(tl.reshape(pr, (R, NJ, 64)), axis=2)
                if NJ == 8:
                    # peel the 8 group sums apart with tl.split (pure register
                    # shuffles) and add them back in original column order --
                    # same association as the masked loop below, ~free instead
                    # of 8 full-width selects.
                    e0, o0 = tl.split(tl.reshape(p, (R, 4, 2)))
                    ee, eo = tl.split(tl.reshape(e0, (R, 2, 2)))
                    oe, oo = tl.split(tl.reshape(o0, (R, 2, 2)))
                    g0, g4 = tl.split(ee)
                    g2, g6 = tl.split(eo)
                    g1, g5 = tl.split(oe)
                    g3, g7 = tl.split(oo)
                    acc_w += g0
                    acc_w += g1
                    acc_w += g2
                    acc_w += g3
                    acc_w += g4
                    acc_w += g5
                    acc_w += g6
                    acc_w += g7
                else:
                    for j in tl.static_range(NJ):
                        acc_w += tl.sum(tl.where(jr[None, :] == j, p, 0.0), axis=1)
            else:
                acc_w += tl.sum(pr, axis=1)
        acc += acc_w

    tail = (H // _WIN) * _WIN
    acc_t = tl.zeros([R], dtype=tl.float32)
    for h0 in range(tail, H, BLK):
        oh = h0 + tl.arange(0, BLK)
        xb = tl.load(X + x_base + oh).to(tl.float32)
        ab = tl.load(lora_a + a_base + offs_r[:, None] * H + oh[None, :]).to(tl.float32)
        pr = ab * xb[None, :]
        if EXACT:
            p = tl.sum(tl.reshape(pr, (R, NJ, 64)), axis=2)
            if NJ == 8:
                e0, o0 = tl.split(tl.reshape(p, (R, 4, 2)))
                ee, eo = tl.split(tl.reshape(e0, (R, 2, 2)))
                oe, oo = tl.split(tl.reshape(o0, (R, 2, 2)))
                g0, g4 = tl.split(ee)
                g2, g6 = tl.split(eo)
                g1, g5 = tl.split(oe)
                g3, g7 = tl.split(oo)
                acc_t += g0
                acc_t += g1
                acc_t += g2
                acc_t += g3
                acc_t += g4
                acc_t += g5
                acc_t += g6
                acc_t += g7
            else:
                for j in tl.static_range(NJ):
                    acc_t += tl.sum(tl.where(jr[None, :] == j, p, 0.0), axis=1)
        else:
            acc_t += tl.sum(pr, axis=1)
    acc += acc_t

    # AscendC skips idx<0 rows entirely (y untouched); a masked store matches
    # that bit for bit, where the old keep-blend rewrote the row and could
    # flip a -0.0 to +0.0.
    tl.store(y + pid * R + offs_r, acc * scale, mask=(offs_r >= 0) & (idx >= 0))


@triton.jit
def sgmv_shrink_v2s(X, lora_a, indices, token_nums, y,
                    scale: tl.constexpr,
                    H: tl.constexpr, R: tl.constexpr, L: tl.constexpr,
                    NR: tl.constexpr, BLK: tl.constexpr, NJ: tl.constexpr,
                    EXACT: tl.constexpr):
    """grid (B*R,): one program per (token, rank), fully contiguous loads.

    Decode variant: on a (B,) grid a B<=8 batch runs 1..8 programs on 40 AIV
    cores; this runs B*R (16..128) of them.  The per-element summation order
    is identical to sgmv_shrink_v2 (64-wide group tree, groups added in
    sequence, 11776-element windows), so the two are bit-interchangeable and
    the B dispatch in the wrapper cannot change results.
    """
    pid = tl.program_id(0)
    b = pid // R
    r = pid % R

    offs_nr = tl.arange(0, NR)
    cnt = tl.load(token_nums + offs_nr).to(tl.int32)
    idxs = tl.load(indices + offs_nr).to(tl.int32)
    lower = offs_nr[None, :] < offs_nr[:, None]
    start = tl.sum(tl.where(lower, cnt[None, :], 0), axis=1)
    in_seg = (b >= start) & (b < start + cnt)
    idx = tl.max(tl.where(in_seg, idxs, -1), axis=0)
    idx_safe = tl.maximum(idx, 0)

    jr = tl.arange(0, NJ)
    acc = tl.zeros([1], dtype=tl.float32)
    a_base = idx_safe * (R * H) + r * H
    x_base = b * H

    for w in range(0, H // _WIN):
        acc_w = tl.zeros([1], dtype=tl.float32)
        for h0 in range(0, _WIN, BLK):
            oh = w * _WIN + h0 + tl.arange(0, BLK)
            xb = tl.load(X + x_base + oh).to(tl.float32)
            ab = tl.load(lora_a + a_base + oh).to(tl.float32)
            pr = ab * xb
            if EXACT:
                p = tl.sum(tl.reshape(pr, (1, NJ, 64)), axis=2)
                if NJ == 8:
                    e0, o0 = tl.split(tl.reshape(p, (1, 4, 2)))
                    ee, eo = tl.split(tl.reshape(e0, (1, 2, 2)))
                    oe, oo = tl.split(tl.reshape(o0, (1, 2, 2)))
                    g0, g4 = tl.split(ee)
                    g2, g6 = tl.split(eo)
                    g1, g5 = tl.split(oe)
                    g3, g7 = tl.split(oo)
                    acc_w += g0
                    acc_w += g1
                    acc_w += g2
                    acc_w += g3
                    acc_w += g4
                    acc_w += g5
                    acc_w += g6
                    acc_w += g7
                else:
                    for j in tl.static_range(NJ):
                        acc_w += tl.sum(tl.where(jr[None, :] == j, p, 0.0), axis=1)
            else:
                acc_w += tl.sum(tl.reshape(pr, (1, BLK)), axis=1)
        acc += acc_w

    tail = (H // _WIN) * _WIN
    acc_t = tl.zeros([1], dtype=tl.float32)
    for h0 in range(tail, H, BLK):
        oh = h0 + tl.arange(0, BLK)
        xb = tl.load(X + x_base + oh).to(tl.float32)
        ab = tl.load(lora_a + a_base + oh).to(tl.float32)
        pr = ab * xb
        if EXACT:
            p = tl.sum(tl.reshape(pr, (1, NJ, 64)), axis=2)
            if NJ == 8:
                e0, o0 = tl.split(tl.reshape(p, (1, 4, 2)))
                ee, eo = tl.split(tl.reshape(e0, (1, 2, 2)))
                oe, oo = tl.split(tl.reshape(o0, (1, 2, 2)))
                g0, g4 = tl.split(ee)
                g2, g6 = tl.split(eo)
                g1, g5 = tl.split(oe)
                g3, g7 = tl.split(oo)
                acc_t += g0
                acc_t += g1
                acc_t += g2
                acc_t += g3
                acc_t += g4
                acc_t += g5
                acc_t += g6
                acc_t += g7
            else:
                for j in tl.static_range(NJ):
                    acc_t += tl.sum(tl.where(jr[None, :] == j, p, 0.0), axis=1)
        else:
            acc_t += tl.sum(tl.reshape(pr, (1, BLK)), axis=1)
    acc += acc_t

    o = b * R + r + tl.zeros([1], dtype=tl.int32)
    tl.store(y + o, acc * scale, mask=(o >= 0) & (idx >= 0))


@triton.jit
def sgmv_expand_v2(X_ptr, lora_b_ptr, indices_ptr, token_nums_ptr,
                   y_in_ptr, y_out_ptr,
                   R: tl.constexpr, Ho: tl.constexpr, L: tl.constexpr,
                   NR: tl.constexpr, BLOCK_HO: tl.constexpr,
                   Y_HO: tl.constexpr, SLICE_OFF: tl.constexpr,
                   NCHUNK: tl.constexpr, TB: tl.constexpr):
    """grid (ceil(B/TB)*NCHUNK,).

    Two changes vs v1, which used grid (B,) and looped the whole Ho inside one
    program:
      * the Ho axis is spread over programs, so decode (B=4..8) fills the 40
        AIV cores instead of 4..8 of them;
      * TB token rows share one loaded weight tile, cutting lora_b traffic TBx,
        and the [BLOCK_HO, R] tile is transposed so the sum runs over the outer
        axis (R long vector adds) instead of over a 16-wide contiguous axis.
    """
    pid = tl.program_id(0)
    g = pid // NCHUNK
    c = pid % NCHUNK
    rows = g * TB + tl.arange(0, TB)

    r_off = tl.arange(0, NR)
    cnts = tl.load(token_nums_ptr + r_off).to(tl.int32)
    idxs = tl.load(indices_ptr + r_off).to(tl.int32)
    # the row count is the segment-length sum -- NEVER take it as a runtime
    # scalar arg: triton specializes an int arg that equals 1 at compile time
    # out of the signature, and the flat-buffer launcher would then misplace
    # the grid triple by 4 bytes (measured: program 0 re-run ~20x, 36x device
    # time at B=64, wrong sums).
    B = tl.sum(cnts, axis=0)
    row_ok = rows < B
    tri = r_off[None, :] < r_off[:, None]
    starts = tl.sum(tl.where(tri, cnts[None, :], 0), axis=1)
    in_seg = (rows[:, None] >= starts[None, :]) & (rows[:, None] < (starts + cnts)[None, :])
    lid = tl.max(tl.where(in_seg, idxs[None, :], -1), axis=1)
    keep = tl.where((lid >= 0) & row_ok, 1.0, 0.0)
    base_id = tl.max(tl.where(keep > 0.0, lid, 0), axis=0)
    keep = tl.where(lid == base_id, keep, 0.0)

    r_idx = tl.arange(0, R)
    x = tl.load(X_ptr + rows[:, None] * R + r_idx[None, :],
                mask=row_ok[:, None], other=0.0)
    xk = x * keep[:, None]
    h0 = c * BLOCK_HO
    flat = tl.arange(0, BLOCK_HO * R)
    w = tl.reshape(tl.load(lora_b_ptr + base_id * (Ho * R) + h0 * R + flat),
                   (BLOCK_HO, R))
    pv = w.to(tl.float32)[None, :, :] * xk[:, None, :]   # [TB, BH, R]
    # A tl.sum over the CONTIGUOUS inner rank axis lowers to the hardware
    # block reduce (BlockReduceSum + PairReduceSum for R=16), which is the
    # very reduction the AscendC kernel uses -- verified bit-identical on
    # the full sweep (all shapes x B x NR x two magnitudes, ndiff=0).
    # A transposed outer-axis reduce is ~30% faster at prefill sizes but
    # differs on ~1/50k rounding-tie elements; bit-exactness wins here.
    acc = tl.sum(pv, axis=2)

    ho = h0 + tl.arange(0, BLOCK_HO)
    out_offs = rows[:, None] * Y_HO + SLICE_OFF + ho[None, :]
    # AscendC skips lid<0 (no-lora) rows; adding acc==+/-0.0 instead would
    # flip a -0.0 y element to +0.0.  A combined (row_ok & lid>=0) DMA mask
    # crashes bishengir (scalar UB OOB), so keep the proven row_ok mask and do
    # the skip as a value select: for skipped rows store the original bf16
    # bits back unchanged (the bf16->fp32->bf16 round trip is the identity,
    # -0.0 included).
    yb = tl.load(y_in_ptr + out_offs, mask=row_ok[:, None], other=0.0)
    yi = yb.to(tl.float32)
    upd = (yi + acc).to(y_out_ptr.dtype.element_ty)
    val = tl.where((lid >= 0)[:, None], upd, yb)
    tl.store(y_out_ptr + out_offs, val, mask=row_ok[:, None])
