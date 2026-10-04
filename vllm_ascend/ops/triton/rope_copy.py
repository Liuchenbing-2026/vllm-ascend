"""Experimental NeoX RoPE with contiguous outputs from strided Q/K inputs.

No weights, attention, cache update or sampling changes. This prototype only
accepts full rotary dimensions and densely packed per-token heads.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_copy(Q, K, C, P, OQ, OK, T: tl.constexpr, QS: tl.constexpr,
               KS: tl.constexpr, CS: tl.constexpr, QH: tl.constexpr,
               KH: tl.constexpr, D: tl.constexpr, QB: tl.constexpr,
               KB: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.cdiv(T, tl.num_programs(0))
    begin = pid * rows
    end = tl.minimum(begin + rows, T)
    half = tl.arange(0, D // 2)
    qh = tl.arange(0, QB)
    kh = tl.arange(0, KB)
    for row in range(begin, end):
        pos = tl.load(P + row).to(tl.int32)
        cos = tl.load(C + pos * CS + half).to(tl.float32)
        sin = tl.load(C + pos * CS + half + D // 2).to(tl.float32)
        qi = qh[:, None] * D + half[None, :]
        qm = qh[:, None] < QH
        q1 = tl.load(Q + row * QS + qi, qm, 0).to(tl.float32)
        q2 = tl.load(Q + row * QS + qi + D // 2, qm, 0).to(tl.float32)
        a = q1 * cos[None, :] - q2 * sin[None, :]
        b = q2 * cos[None, :] + q1 * sin[None, :]
        tl.store(OQ + row * QH * D + qi, a, qm)
        tl.store(OQ + row * QH * D + qi + D // 2, b, qm)
        ki = kh[:, None] * D + half[None, :]
        km = kh[:, None] < KH
        k1 = tl.load(K + row * KS + ki, km, 0).to(tl.float32)
        k2 = tl.load(K + row * KS + ki + D // 2, km, 0).to(tl.float32)
        a = k1 * cos[None, :] - k2 * sin[None, :]
        b = k2 * cos[None, :] + k1 * sin[None, :]
        tl.store(OK + row * KH * D + ki, a, km)
        tl.store(OK + row * KH * D + ki + D // 2, b, km)


def rope_copy_forward(q, k, cache, positions, vector_cores):
    t, qh, d = q.shape
    oq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ok = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _rope_copy[(min(t, vector_cores),)](
        q, k, cache, positions, oq, ok, t, q.stride(0), k.stride(0),
        cache.stride(0), qh, k.shape[1], d,
        triton.next_power_of_2(qh), triton.next_power_of_2(k.shape[1]))
    return oq, ok
