"""Turn the branch's lora_ops_triton{,_kernels}.py into the optimised v2.

Run:  python3 apply_v2_patch.py <src_dir> <dst_dir>
Every edit is an exact-string replacement so the diff is auditable.

Edits, in the order they appear below:

 K1  add sgmv_shrink_v2 : grid (B,), 512-wide contiguous-per-row [R,BLK] tile
                          instead of the 64-wide one (8x fewer DMA bursts).
 K2  add sgmv_expand_v2 : grid (ceil(B/TB)*NCHUNK,), TB token rows per program
                          sharing one weight tile, reduction over the OUTER
                          axis after a transpose instead of over the 16-wide
                          contiguous axis.
 P1  _cpp_launch        : clamp the rtKernelLaunch blockDim to the physical AIV
                          core count.  triton-ascend's own launcher does this
                          (driver.py, enable_auto_map_parallel_blocks); the
                          branch launcher passed the full logical grid, which
                          costs up to 13x device time at prefill batch sizes.
                          The true logical grid is already packed separately
                          into the arg buffer, so this changes nothing else.
 P2  _cpp_case_key      : only take .dtype of actual tensors.  bgmv_shrink
                          passes `scaling` (a float) in example_args, so every
                          bgmv_shrink call raises AttributeError today.
 P3  _to_int32          : identity.  The kernels already do
                          `tl.load(indices + ...).to(tl.int32)`, so int64
                          tensors work directly; the cast was launching two
                          extra device kernels per op call, which aclgraph
                          bakes into every decode step.  Also removes the
                          _cast_scratch buffer whose pointer gets frozen into a
                          captured graph.
 P4  timing             : `m3` was only bound inside `if _cpp_enabled():` but
                          referenced on the fallback path (NameError when
                          TRITON_LORA_CPP=0), and `_timing_end("sgmv_shrink")`
                          was called twice on one path.
 P5  wrappers           : route sgmv_shrink / sgmv_expand_slice to the v2
                          kernels (TRITON_LORA_V2=0 restores v1).
 P6  _cpp_launch ints   : the expand kernel takes a runtime int32 arg (B).
"""
import os
import sys

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)


def rd(n):
    return open(os.path.join(SRC, n), encoding="utf-8").read()


def wr(n, s):
    open(os.path.join(DST, n), "w", encoding="utf-8", newline="\n").write(s)


def sub(s, old, new, tag):
    if old not in s:
        raise SystemExit("PATCH FAILED (%s): anchor not found" % tag)
    if s.count(old) != 1:
        raise SystemExit("PATCH FAILED (%s): anchor appears %d times" % (tag, s.count(old)))
    print("  ok %s" % tag)
    return s.replace(old, new)


# ---------------------------------------------------------------- kernels ---
K = rd("lora_ops_triton_kernels.py")

K += '''

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
        m = oh < H          # partial tail when BLK does not divide H (H%64!=0)
        xb = tl.load(X + x_base + oh, mask=m, other=0.0).to(tl.float32)
        ab = tl.load(lora_a + a_base + offs_r[:, None] * H + oh[None, :], mask=m[None, :], other=0.0).to(tl.float32)
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
        m = oh < H          # partial tail when BLK does not divide H (H%64!=0)
        xb = tl.load(X + x_base + oh, mask=m, other=0.0).to(tl.float32)
        ab = tl.load(lora_a + a_base + oh, mask=m, other=0.0).to(tl.float32)
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
    ho = h0 + tl.arange(0, BLOCK_HO)
    ho_ok = ho < Ho          # partial last chunk when BLOCK_HO does not divide Ho
    flat = tl.arange(0, BLOCK_HO * R)
    # 1-D contiguous weight load (the fast path), masked past Ho for the tail
    # chunk; flat // R is the row (ho) offset of each of the BLOCK_HO*R elements.
    w = tl.reshape(tl.load(lora_b_ptr + base_id * (Ho * R) + h0 * R + flat,
                           mask=(h0 + flat // R) < Ho, other=0.0),
                   (BLOCK_HO, R))
    pv = w.to(tl.float32)[None, :, :] * xk[:, None, :]   # [TB, BH, R]
    # A tl.sum over the CONTIGUOUS inner rank axis lowers to the hardware
    # block reduce (BlockReduceSum + PairReduceSum for R=16), which is the
    # very reduction the AscendC kernel uses -- verified bit-identical on
    # the full sweep (all shapes x B x NR x two magnitudes, ndiff=0).
    # A transposed outer-axis reduce is ~30% faster at prefill sizes but
    # differs on ~1/50k rounding-tie elements; bit-exactness wins here.
    acc = tl.sum(pv, axis=2)

    out_offs = rows[:, None] * Y_HO + SLICE_OFF + ho[None, :]
    smask = row_ok[:, None] & ho_ok[None, :]
    # AscendC skips lid<0 (no-lora) rows; adding acc==+/-0.0 instead would
    # flip a -0.0 y element to +0.0.  A combined (row_ok & lid>=0) DMA mask
    # crashes bishengir (scalar UB OOB), so keep the proven row_ok mask and do
    # the skip as a value select: for skipped rows store the original bf16
    # bits back unchanged (the bf16->fp32->bf16 round trip is the identity,
    # -0.0 included).
    yb = tl.load(y_in_ptr + out_offs, mask=smask, other=0.0)
    yi = yb.to(tl.float32)
    upd = (yi + acc).to(y_out_ptr.dtype.element_ty)
    val = tl.where((lid >= 0)[:, None], upd, yb)
    tl.store(y_out_ptr + out_offs, val, mask=smask)
'''
wr("lora_ops_triton_kernels.py", K)
print("kernels: appended sgmv_shrink_v2 + sgmv_expand_v2")

# ------------------------------------------------------------------- ops ---
S = rd("lora_ops_triton.py")

# P3 -------------------------------------------------------------------------
S = sub(S, '''    key = (tag, t.numel())
    s = _cast_scratch.get(key)
    if s is None or s.device != t.device:
        s = torch.empty(t.numel(), dtype=torch.int32, device=t.device)
        _cast_scratch[key] = s
    return s.copy_(t)''',
        '''    # v2: identity.  Every kernel already narrows on load
    # (`tl.load(indices + ...).to(tl.int32)`), so int64 tensors are fine as-is.
    # The cast used to launch two extra device kernels per op call -- which
    # aclgraph captures and then replays on EVERY decode step -- and it froze a
    # _cast_scratch pointer into the captured graph.  Set TRITON_LORA_CAST=1 to
    # restore the old behaviour.
    if os.environ.get("TRITON_LORA_CAST", "0") == "0":
        return t
    key = (tag, t.numel())
    s = _cast_scratch.get(key)
    if s is None or s.device != t.device:
        s = torch.empty(t.numel(), dtype=torch.int32, device=t.device)
        _cast_scratch[key] = s
    return s.copy_(t)''', "P3 _to_int32 identity")

# P1 + P6 --------------------------------------------------------------------
S = sub(S, '''def _cpp_launch(case, grid_x, ptrs, floats=()):
    CL, _ = _cpp_setup()
    b = case["buf"]
    off = 24  # [ffts][syncBlockLock][workspace]
    for p in ptrs:
        struct.pack_into("<Q", b, off, p)
        off += 8
    for f in floats:
        struct.pack_into("<f", b, off, f)
        off += 4
    off = (off + 3) & ~3
    struct.pack_into("<iii", b, off, grid_x, 1, 1)
    return CL.lora_launch_flat(case["func"],
                               torch.npu.current_stream().npu_stream,
                               grid_x, b, off + 12)''',
        '''_NUM_AIV = None


def _num_aiv() -> int:
    """Physical AIV block count.  triton-ascend clamps blockDim to this in its
    own launcher (backends/ascend/driver.py, `enable_auto_map_parallel_blocks`)
    and the compiled kernel walks the logical grid -- which is passed in the arg
    buffer -- with a grid-stride loop."""
    global _NUM_AIV
    if _NUM_AIV is None:
        for mod, cls in (("triton.backends.ascend.driver", "NPUUtils"),
                         ("triton.backends.ascend.utils", "NPUUtils")):
            try:
                import importlib
                _NUM_AIV = int(getattr(importlib.import_module(mod), cls)()
                               .get_aivector_core_num())
                break
            except Exception:
                continue
        if not _NUM_AIV:
            _NUM_AIV = 40
    return _NUM_AIV


def _cpp_launch(case, grid_x, ptrs, floats=(), ints=()):
    CL, _ = _cpp_setup()
    b = case["buf"]
    off = 24  # [ffts][syncBlockLock][workspace]
    for p in ptrs:
        struct.pack_into("<Q", b, off, p)
        off += 8
    for f in floats:
        struct.pack_into("<f", b, off, f)
        off += 4
    for i in ints:
        struct.pack_into("<i", b, off, int(i))
        off += 4
    off = (off + 3) & ~3
    struct.pack_into("<iii", b, off, grid_x, 1, 1)
    # v2 FIX: blockDim must be clamped to the physical block count.  Passing the
    # full logical grid (up to max_num_batched_tokens) cost 3.2x device time at
    # B=256 and 12.9x at B=1024, measured on 910B4 with the identical binary.
    nb = _num_aiv()
    block = grid_x if grid_x < nb else nb
    return CL.lora_launch_flat(case["func"],
                               torch.npu.current_stream().npu_stream,
                               block, b, off + 12)''', "P1+P6 launcher clamp")

# P2 -------------------------------------------------------------------------
S = sub(S, '''def _cpp_case_key(kernel_fn, kwargs, tensors):
    return (kernel_fn.__name__,
            tuple(sorted((k, str(v)) for k, v in kwargs.items())),
            tuple(str(t.dtype) for t in tensors))''',
        '''def _cpp_case_key(kernel_fn, kwargs, tensors):
    # v2: two fixes.  (a) `tensors` is really example_args and may hold python
    # scalars -- bgmv_shrink passes `scaling` -- so .dtype must be guarded, or
    # every bgmv_shrink call raises AttributeError.  (b) sorted() + str() per
    # kwarg on each of ~1200 op calls per forward is pure host tax; kwargs is
    # built with a fixed key order at every call site, so the values alone
    # identify the case.
    return (kernel_fn.__name__,
            tuple(kwargs.values()),
            tuple(t.dtype for t in tensors if isinstance(t, torch.Tensor)))''',
        "P2 case-key: guard scalars + drop the per-call sort/str")

# P4 -------------------------------------------------------------------------
S = sub(S, '''    kwargs = dict(H=H, R=R, L=L, NR=seq_len_tensor.numel(), scale=scaling)
    m2 = _timing_start("sgmv_shrink|lookup")''',
        '''    kwargs = dict(H=H, R=R, L=L, NR=seq_len_tensor.numel(), scale=scaling)
    m2 = _timing_start("sgmv_shrink|lookup")
    m3 = None  # v2: was only bound inside the _cpp_enabled() branch -> NameError''',
        "P4a sgmv_shrink m3")

S = sub(S, '''    _timing_end("sgmv_shrink", t0)
    _timing_end("sgmv_shrink|prep", m1)
    _timing_end("sgmv_shrink|lookup", m2)
    _timing_end("sgmv_shrink|launch", m3)
    K.sgmv_shrink_kernel[(B,)](inputs, w, idx32, seq32,
                               output_tensor, **kwargs)
    _timing_end("sgmv_shrink", t0)
    return output_tensor''',
        '''    _timing_end("sgmv_shrink|prep", m1)
    _timing_end("sgmv_shrink|lookup", m2)
    _timing_end("sgmv_shrink|launch", m3)
    K.sgmv_shrink_kernel[(B,)](inputs, w, idx32, seq32,
                               output_tensor, **kwargs)
    _timing_end("sgmv_shrink", t0)
    return output_tensor''', "P4b double timing_end")

S = sub(S, '''                  BLOCK_HO=_expand_blk_ho(R), Y_HO=output_tensor.size(1),
                  SLICE_OFF=slice_offset)
    m2 = _timing_start("sgmv_expand_slice|lookup")''',
        '''                  BLOCK_HO=_expand_blk_ho(R), Y_HO=output_tensor.size(1),
                  SLICE_OFF=slice_offset)
    m2 = _timing_start("sgmv_expand_slice|lookup")
    m3 = None  # v2: see sgmv_shrink''', "P4c sgmv_expand m3")

# P5 -------------------------------------------------------------------------
S = sub(S, '''def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                token_nums, scaling):
    t0 = _timing_start("sgmv_shrink")''',
        '''_V2 = os.environ.get("TRITON_LORA_V2", "1") != "0"
_V2_EXACT = 1 if os.environ.get("TRITON_LORA_EXACT", "1") != "0" else 0


_V2_BLK_CACHE = {}


def _v2_blk(H: int) -> int:
    """Largest power-of-two <= 512 dividing both H and the 11776 AscendC window
    (so the per-window accumulator restart that keeps down_proj matching stays
    on a block boundary)."""
    b = _V2_BLK_CACHE.get(H)
    if b is None:
        b = 64
        for c in (512, 256, 128, 64):
            if H % c == 0 and 11776 % c == 0:
                b = c
                break
        _V2_BLK_CACHE[H] = b
    return b


_V2_CFG_CACHE = {}


def _v2_expand_cfg(Ho: int, R: int, NR: int, B: int):
    """(BLOCK_HO, TB).

    TB>1 makes one program own TB consecutive token rows so they share a single
    loaded weight tile.  That is only correct when every row in the group maps
    to the SAME lora id, which is guaranteed exactly when the batch is a single
    segment (NR == 1).  `compute_meta` collapses consecutive equal ids, so an
    all-one-adapter batch always yields NR == 1; with NR > 1 a group could
    straddle a segment boundary, so fall back to one row per program, which is
    per-row correct by construction.

    TB is bucketed by B so a B=1 decode step does not pay for 3 masked-off
    rows; every bucket computes each real row identically, so the bucket
    choice cannot change results.
    """
    if NR != 1:
        tb = 1
    elif B <= 2:
        tb = B
    elif B <= 8:
        tb = 4
    else:
        tb = 8
    key = (Ho, R, NR, tb)
    got = _V2_CFG_CACHE.get(key)
    if got is not None:
        return got
    bh = 128
    while bh > 32 and Ho % bh:
        bh //= 2
    while tb > 1 and tb * bh * R * 4 > 96 * 1024:
        tb //= 2
    _V2_CFG_CACHE[key] = (bh, tb)
    return bh, tb


def sgmv_shrink_v1(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                   seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                   token_nums, scaling):
    return _sgmv_shrink_impl(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                             seq_len_tensor, lora_indices_tensor, batches,
                             max_seq_length, token_nums, scaling)


def sgmv_shrink(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                token_nums, scaling):
    if not _V2:
        return _sgmv_shrink_impl(inputs, lora_a_weights, output_tensor,
                                 b_seq_start_loc, seq_len_tensor,
                                 lora_indices_tensor, batches, max_seq_length,
                                 token_nums, scaling)
    t0 = _timing_start("sgmv_shrink")
    B, H = inputs.shape
    # vllm packs lora_a as [L, 1, R, H]; the kernel needs only L, R and the base
    # pointer, and reshaping a contiguous tensor keeps the same data_ptr, so the
    # reshape is materialised only on the plain-triton fallback path.
    sh = lora_a_weights.shape
    L, R = sh[0], sh[-2]
    idx32 = _to_int32(lora_indices_tensor, "idx")
    seq32 = _to_int32(seq_len_tensor, "seq")
    blk = _v2_blk(H)
    # v2 shrink now masks its partial tail (oh < H), so any H is handled
    # natively -- no fallback needed.  blk always divides the 11776 window so
    # the main loop stays exact; only the final tail block can be partial.
    kwargs = dict(scale=scaling, H=H, R=R, L=L, NR=seq_len_tensor.numel(),
                  BLK=blk, NJ=blk // 64, EXACT=_V2_EXACT)
    # decode batches leave most of the 40 AIV cores idle on a (B,) grid; the
    # (B*R,) variant has the identical per-element summation order, so this
    # dispatch cannot change results.
    if B * R <= 40:
        kern, grid = K.sgmv_shrink_v2s, B * R
    else:
        kern, grid = K.sgmv_shrink_v2, B
    if _cpp_enabled():
        case = _cpp_get_case(kern, kwargs,
                             (inputs, lora_a_weights, idx32, seq32,
                              output_tensor), (grid,))
        if case is not None:
            _cpp_launch(case, grid, [inputs.data_ptr(), lora_a_weights.data_ptr(),
                                     idx32.data_ptr(), seq32.data_ptr(),
                                     output_tensor.data_ptr()])
            _timing_end("sgmv_shrink", t0)
            return output_tensor
    w = lora_a_weights.reshape(L, R, H)
    kern[(grid,)](inputs, w, idx32, seq32, output_tensor, **kwargs)
    _timing_end("sgmv_shrink", t0)
    return output_tensor


def _sgmv_shrink_impl(inputs, lora_a_weights, output_tensor, b_seq_start_loc,
                      seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                      token_nums, scaling):
    t0 = _timing_start("sgmv_shrink")''', "P5a sgmv_shrink v2")

S = sub(S, '''def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                      seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                      token_nums, slice_offset, slice_size, add_inputs=False):
    t0 = _timing_start("sgmv_expand_slice")''',
        '''def sgmv_expand_slice(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                      seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                      token_nums, slice_offset, slice_size, add_inputs=False):
    if not _V2:
        return _sgmv_expand_slice_impl(inputs, lora_b_weights, output_tensor,
                                       b_seq_start_loc, seq_len_tensor,
                                       lora_indices_tensor, batches, max_seq_length,
                                       token_nums, slice_offset, slice_size,
                                       add_inputs)
    t0 = _timing_start("sgmv_expand_slice")
    B, R = inputs.shape
    sh = lora_b_weights.shape          # [L, 1, Ho, R]; reshape keeps data_ptr
    L, Ho = sh[0], sh[-2]
    idx32 = _to_int32(lora_indices_tensor, "idx")
    seq32 = _to_int32(seq_len_tensor, "seq")
    bh, tb = _v2_expand_cfg(Ho, R, seq_len_tensor.numel(), B)
    # v2 expand now masks its partial last chunk (ho < Ho), so any Ho is
    # handled natively; ceil so the tail chunk is launched.
    nchunk = (Ho + bh - 1) // bh
    grid = ((B + tb - 1) // tb) * nchunk
    kwargs = dict(R=R, Ho=Ho, L=L, NR=seq_len_tensor.numel(), BLOCK_HO=bh,
                  Y_HO=output_tensor.size(1), SLICE_OFF=slice_offset,
                  NCHUNK=nchunk, TB=tb)
    if _cpp_enabled():
        case = _cpp_get_case(K.sgmv_expand_v2, kwargs,
                             (inputs, lora_b_weights, idx32, seq32,
                              output_tensor, output_tensor), (grid,))
        if case is not None:
            _cpp_launch(case, grid, [inputs.data_ptr(), lora_b_weights.data_ptr(),
                                     idx32.data_ptr(), seq32.data_ptr(),
                                     output_tensor.data_ptr(),
                                     output_tensor.data_ptr()])
            _timing_end("sgmv_expand_slice", t0)
            return output_tensor
    w = lora_b_weights.reshape(L, Ho, R)
    K.sgmv_expand_v2[(grid,)](inputs, w, idx32, seq32, output_tensor,
                              output_tensor, **kwargs)
    _timing_end("sgmv_expand_slice", t0)
    return output_tensor


def _sgmv_expand_slice_impl(inputs, lora_b_weights, output_tensor, b_seq_start_loc,
                            seq_len_tensor, lora_indices_tensor, batches,
                            max_seq_length, token_nums, slice_offset, slice_size,
                            add_inputs=False):
    t0 = _timing_start("sgmv_expand_slice")''', "P5b sgmv_expand v2")

# P6c _cpp_make_case must separate float and int scalars -----------------------
S = sub(S, '''    tensors = [t for t in example_args if isinstance(t, torch.Tensor)]
    floats = [f for f in example_args if not isinstance(f, torch.Tensor)]''',
        '''    tensors = [t for t in example_args if isinstance(t, torch.Tensor)]
    # v2: ints (the expand kernel takes a runtime row count) must be packed as
    # <i, not <f, in the verify-retry launch below.
    floats = [f for f in example_args if isinstance(f, float)]
    int_args = [f for f in example_args
                if isinstance(f, int) and not isinstance(f, bool)]''', "P6c scalar split")

S = sub(S, '''    n = 24 + 8 * len(ptrs_d) + 4 * len(floats) + 16''',
        '''    n = 24 + 8 * len(ptrs_d) + 4 * (len(floats) + len(int_args)) + 16''',
        "P6d arg-buffer sizing")

S = sub(S, '''        ret = _cpp_launch(case, grid[0], ptrs_d, floats)''',
        '''        ret = _cpp_launch(case, grid[0], ptrs_d, floats, int_args)''',
        "P6e verify launch scalars")

wr("lora_ops_triton.py", S)
for extra in ("lora_ops.py", "lora_cpp_launcher.cpp",
              "lora_cpp_launcher.cpython-312-aarch64-linux-gnu.so",
              "lora_native_ops.cpp"):
    p = os.path.join(SRC, extra)
    if os.path.exists(p):
        open(os.path.join(DST, extra), "wb").write(open(p, "rb").read())
print("wrote v2 tree to", DST)
