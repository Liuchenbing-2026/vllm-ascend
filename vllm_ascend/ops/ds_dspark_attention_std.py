# SPDX-License-Identifier: Apache-2.0
"""Torch-only paged-gather-attend helper for the DSpark draft ("root cause B").

ORIGINAL, PURE-TORCH. There is NO fused AscendC kernel and NO .so dependency here
anymore: the draft block attention reads context/block K==V straight out of the
vLLM PAGED SWA KV-cache (through the bit-exact-verified dspark_swa_indices slot
table) and runs the user's VALIDATED sparse attention math. This is cheap for the
draft-eager round -- each block is tiny (block_size queries x <=window+block keys
x head_dim, a SINGLE shared kv head) next to the 3 MHC/MoE layers.

`sparse_attn` below is LIFTED VERBATIM from staging/ds_dspark_math.sparse_attn (the
offline-validated reference, cos~=1 vs DeepSeek): K == V shared, softmax_scale =
head_dim**-0.5, attn_sink added in the DENOMINATOR ONLY (unscaled, NOT multiplied
into the softmax numerator), per-block, non-causal -- every query row of a block
attends to the SAME gathered slot list. Do not "improve" it; it is the contract.

Window numbers / mask modes stay in ds_dspark_meta_std so the (now two) call sites
-- the metadata builder and this helper -- cannot drift.
"""
from __future__ import annotations

import torch


def _unwrap_single_kv_cache(kv_cache):
    while isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 1:
        kv_cache = kv_cache[0]
    return kv_cache


# ---------------------------------------------------------------------------
# VERBATIM from ds_dspark_math.sparse_attn -- shared K==V, attn_sink in the
# DENOMINATOR only (unscaled softmax). q:[b,m,h,d]  kv:[b,n,d]  topk_idx:[b,m,k].
# ---------------------------------------------------------------------------
def sparse_attn(q, kv, attn_sink, topk_idx, scale):
    b, m, h, d = q.shape
    valid = topk_idx != -1
    idx = topk_idx.clamp(min=0).long()
    kv_g = torch.gather(kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2,
                        idx.unsqueeze(-1).expand(b, m, idx.size(-1), d))
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), kv_g.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    smax = scores.amax(-1, keepdim=True)
    e = torch.exp(scores - smax)
    denom = e.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - smax.squeeze(-1))
    return (torch.einsum("bmhk,bmkd->bmhd", e, kv_g.float()) / denom.unsqueeze(-1)).to(q.dtype)


# ---------------------------------------------------------------------------
# Paged gather + attend. Per draft block: take the block's shared swa_indices row
# (every row of a block is identical -> use row 0), drop -1 pads via swa_lens,
# gather the K==V rows from the flat paged cache, run the validated sparse_attn.
# ---------------------------------------------------------------------------
def paged_gather_attend(q, kv_cache, swa_indices, swa_lens, block_size,
                        cache_block_size, attn_sink, scale):
    """q:[T, n_heads, head_dim] TND. Returns out:[T, n_heads, head_dim].

    kv_cache is the raw paged SWA cache tensor. TODO(serve-verify): confirm its
    layout is [num_blocks, cache_block_size, num_kv_heads=1, head_dim]; we flatten
    the leading (block, offset) pair to a single slot axis so slot_id ==
    block*cache_block_size + offset indexes it directly (matches the meta builder
    and dsa_kv_compress_scatter's slot convention). cache_block_size is passed for
    that contract check only -- the flat view makes the read layout-agnostic.
    """
    del cache_block_size  # implied by the flat view; kept for signature/contract clarity
    cache = _unwrap_single_kv_cache(kv_cache)
    # Fix 5: stay defensive to BOTH plausible on-device paged SWA layouts --
    # [num_blocks, cache_block_size, num_kv_heads=1, head_dim] AND the squeezed
    # [num_blocks, cache_block_size, head_dim]. Do NOT hard-assume a rank: fold the
    # leading (block, offset) pair into a single slot axis so slot_id ==
    # block*cache_block_size + offset indexes rows directly; the per-slot gather
    # below reshapes to [L, head_dim], folding any trailing kv_head=1.
    # TODO(serve-verify): confirm the exact on-device layout / rank of this cache.
    if cache.dim() >= 3:
        flat = cache.reshape(-1, *cache.shape[2:])         # [nb*cbs, (1,) head_dim]
    else:
        flat = cache                                       # already flat [num_slots, head_dim]
    out = torch.zeros_like(q)
    T = q.shape[0]
    for off in range(0, T, block_size):
        end = min(off + block_size, T)
        L = int(swa_lens[off])
        if L <= 0:
            continue
        slots = swa_indices[off, 0, :L].to(torch.long)
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            continue
        kv = flat.index_select(0, slots).reshape(slots.numel(), -1)  # [L, head_dim] (folds kv_head=1)
        blk = end - off
        q_block = q[off:end].unsqueeze(0)                  # [1, blk, n_heads, head_dim]
        # non-causal: every row of the block attends to the SAME gathered slot list.
        topk = torch.arange(slots.numel(), device=q.device).view(1, 1, -1).expand(1, blk, -1)
        o = sparse_attn(q_block, kv.unsqueeze(0), attn_sink, topk, float(scale))  # [1, blk, h, d]
        out[off:end] = o.squeeze(0)
    return out
