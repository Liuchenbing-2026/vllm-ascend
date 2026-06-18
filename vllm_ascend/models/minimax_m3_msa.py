"""MiniMax-M3 Sparse Attention (MSA): index branch + block-sparse selection.

Authoritative algorithm: official MSA paper (arXiv 2606.13392), matched to the
M3 checkpoint weights/config. Verified against the pure-torch reference
ref_msa.py / ref_msa_unit.py: when the sequence fits within
sparse_topk_blocks * sparse_block_size tokens (all blocks selected), MSA reduces
EXACTLY to dense GQA full attention (max abs diff ~3e-5, fp32 noise).

MSA = GQA backbone + a lightweight Index Branch:
  * One index query head per GQA group (H_kv=4), one shared index key head.
  * score S_ij^(r) = (Qidx_i^(r) . Kidx_j) / sqrt(d_idx), no activation, causal.
  * block max-pool (sparse_score_type='max'), block size 128.
  * top-k (k=16) blocks per (query, group); the LOCAL block containing the query
    is always selected; sparse_init_block (=0 for M3) initial blocks forced too.
  * Main branch: the 16 query heads of a group share the group's selected blocks
    and run exact softmax over only those blocks.

This module provides the math as free functions (plain-tensor, CPU/NPU, unit
testable) plus an nn.Module wiring the index projections/norms. The fused NPU
op (npu_sparse_flash_attention, block-native via sparse_block_size) is the perf
fast-path used in the MSA attention backend; this gather+SDPA path is the
correctness/eager-prefill reference equivalent.

NOTE(index_rope): whether RoPE is applied to index q/k is NOT specified in the
paper and no public modeling.py exists. It affects long-range block-selection
QUALITY, not mechanism correctness. Exposed as `index_rope` flag, default False;
resolve empirically (needle eval) once running on NPU.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def msa_block_scores(iq: torch.Tensor, ik: torch.Tensor, block_size: int) -> torch.Tensor:
    """Index-branch block scores.

    iq: [S, G, d_idx] (per-group index query, already normed/roped)
    ik: [S, d_idx]    (shared index key, already normed/roped)
    returns M: [G, S, nb]  block max-pooled causal scores (-inf where unreachable)
    """
    S, G, d = iq.shape
    scale = d ** -0.5
    isc = torch.einsum("igd,jd->gij", iq, ik) * scale          # [G,S,S]
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=iq.device), 1)
    isc = isc.masked_fill(causal[None], float("-inf"))
    nb = (S + block_size - 1) // block_size
    bid = (torch.arange(S, device=iq.device) // block_size)
    M = torch.full((G, S, nb), float("-inf"), device=iq.device, dtype=isc.dtype)
    for b in range(nb):
        cols = bid == b
        M[:, :, b] = isc[:, :, cols].amax(dim=2)
    return M


def msa_select_mask(M: torch.Tensor, S: int, block_size: int, topk_blocks: int,
                    local_blocks: int = 1, init_blocks: int = 0) -> torch.Tensor:
    """Top-k block selection -> per-key boolean mask.

    M: [G, S, nb]. returns allow: [G, S, S] (True where key j may be attended by
    query i for that group), already causal.
    """
    G, _, nb = M.shape
    dev = M.device
    bstart = torch.arange(nb, device=dev) * block_size
    reach = bstart[None, :] <= torch.arange(S, device=dev)[:, None]          # [S,nb]
    Mr = M.masked_fill(~reach[None], float("-inf"))
    k = min(topk_blocks, nb)
    topv, topi = Mr.topk(k, dim=-1)                                          # [G,S,k]
    blocksel = torch.zeros(G, S, nb, dtype=torch.bool, device=dev)
    blocksel.scatter_(2, topi, topv > float("-inf"))                        # ignore -inf picks
    qblk = (torch.arange(S, device=dev) // block_size)
    # force local block(s): the block containing i and (local_blocks-1) before it
    for off in range(local_blocks):
        lb = (qblk - off).clamp(min=0)
        blocksel[:, torch.arange(S, device=dev), lb] = True
    # force initial/sink blocks (M3: init_blocks=0 -> no-op), respecting reachability
    for b in range(init_blocks):
        canb = (bstart[b] <= torch.arange(S, device=dev))
        blocksel[:, canb, b] = True
    keyblk = (torch.arange(S, device=dev) // block_size)                    # [S]
    keymask = blocksel[:, :, keyblk]                                        # [G,S,S]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=dev))
    return keymask & causal[None]


def msa_block_sparse_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          allow: torch.Tensor, scale: float) -> torch.Tensor:
    """Exact softmax over selected blocks (correctness/eager path).

    q: [S, nH, hd]; k,v: [S, G, hd]; allow: [G, S, S]. returns [S, nH, hd].
    """
    S, nH, hd = q.shape
    G = k.shape[1]
    rep = nH // G
    out = torch.zeros(S, nH, hd, device=q.device, dtype=q.dtype)
    for r in range(G):
        qg = q[:, r * rep:(r + 1) * rep, :]                                # [S,rep,hd]
        kk = k[:, r, :]; vv = v[:, r, :]                                   # [S,hd]
        sc = torch.einsum("shd,td->sht", qg, kk) * scale                   # [S,rep,S]
        sc = sc.masked_fill(~allow[r][:, None, :], float("-inf"))
        aw = torch.softmax(sc, dim=-1)
        out[:, r * rep:(r + 1) * rep, :] = torch.einsum("sht,td->shd", aw, vv)
    return out


def msa_attention(q, k, v, iq, ik, *, block_size, topk_blocks, local_blocks,
                  init_blocks, scale):
    """Full MSA forward (eager/correctness path). Shapes:
    q:[S,nH,hd] k,v:[S,G,hd] iq:[S,G,d_idx] ik:[S,d_idx]. returns [S,nH,hd]."""
    S = q.shape[0]
    M = msa_block_scores(iq, ik, block_size)
    allow = msa_select_mask(M, S, block_size, topk_blocks, local_blocks, init_blocks)
    return msa_block_sparse_sdpa(q, k, v, allow, scale)


def msa_decode_attn(q1, kf, vf, iq1, ikf, *, block_size, topk_blocks,
                    local_blocks, init_blocks, scale, positions,
                    return_sel=False, forced_sel=None):
    """Decode-step MSA. If forced_sel (list[G] of block-id lists) is given, the
    indexer is skipped and that selection is reused (current local block always
    added) -- this is the IndexCache fast path. If return_sel, also returns the
    selected block ids of the first query (for caching)."""
    import torch
    Tq, nH, hd = q1.shape
    S, G, _ = kf.shape
    d = iq1.shape[-1]
    rep = nH // G
    out = torch.zeros(Tq, nH, hd, device=q1.device, dtype=q1.dtype)
    sel_all = None
    for t in range(Tq):
        L = int(positions[t]) + 1
        nb = (L + block_size - 1) // block_size
        qblk = (L - 1) // block_size
        if forced_sel is not None:
            chosen_g = []
            for r in range(G):
                ch = {b for b in forced_sel[r] if b < nb}
                ch.add(qblk)
                chosen_g.append(ch)
        else:
            isc = torch.einsum('gd,ld->gl', iq1[t], ikf[:L]) * (d ** -0.5)
            M = torch.full((G, nb), float('-inf'), device=q1.device, dtype=isc.dtype)
            for b in range(nb):
                lo = b * block_size; hi = min(lo + block_size, L)
                M[:, b] = isc[:, lo:hi].amax(dim=1)
            chosen_g = []
            for r in range(G):
                kk_ = min(topk_blocks, nb)
                ch = set(torch.topk(M[r], kk_).indices.tolist())
                ch.add(qblk)
                for b in range(min(init_blocks, nb)):
                    ch.add(b)
                chosen_g.append(ch)
        if return_sel and t == 0:
            sel_all = [sorted(c) for c in chosen_g]
        for r in range(G):
            cols = []
            for b in sorted(chosen_g[r]):
                lo = b * block_size; hi = min(lo + block_size, L)
                cols.extend(range(lo, hi))
            cols = torch.tensor(cols, device=q1.device, dtype=torch.long)
            ksel = kf[cols, r, :]; vsel = vf[cols, r, :]
            for hh in range(rep):
                head = r * rep + hh
                sc = (q1[t, head] @ ksel.T) * scale
                out[t, head] = torch.softmax(sc, dim=-1) @ vsel
    if return_sel:
        return out, sel_all
    return out
