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


def msa_decode_select(iq, seq_lens, block_tables, idxk_cache, *,
                      block_size, topk_blocks, init_blocks=0):
    """IndexCache: compute ONLY the indexer block selection (the expensive part --
    scoring the new query against ALL cached index keys). Returns top-k block
    positions [B, G, K] (long), future-masked. msa_decode_vec / msa_decode_fia take
    this as `topk_idx=` to skip the per-step indexer when a cached selection is reused
    (MM3_INDEX_TOPK_FREQ > 1). Numerically identical to their inline selection."""
    import torch
    iq = iq.float()
    B, G, d = iq.shape
    MAXB = block_tables.shape[1]
    BS = block_size
    dev = iq.device
    bt = block_tables.to(torch.long).clamp(0, idxk_cache.shape[0] - 1)
    L = MAXB * BS
    ikf = idxk_cache[bt].reshape(B, L, d).float()
    pos = torch.arange(L, device=dev)
    sl = seq_lens.to(dev)
    valid = pos[None, :] < sl[:, None]
    NEG = torch.finfo(torch.float32).min
    isc = torch.einsum('bgd,bld->bgl', iq, ikf) * (d ** -0.5)
    isc = isc.masked_fill(~valid[:, None, :], NEG)
    M = isc.reshape(B, G, MAXB, BS).amax(dim=-1)
    qblk = ((sl - 1) // BS).clamp(min=0)
    blk = torch.arange(MAXB, device=dev)
    M = M.masked_fill(blk[None, None, :] > qblk[:, None, None], NEG)
    K = min(topk_blocks, MAXB)
    return M.topk(K, dim=-1).indices  # [B, G, K]


def msa_decode_vec(q, iq, seq_lens, block_tables, kc, vc, idxk_cache, *,
                   block_size, topk_blocks, scale, local_blocks=1, init_blocks=0,
                   topk_idx=None):
    """Fixed-shape, sync-free, branchless decode MSA (cudagraph/FULL-capturable).
    q:[B,nH,hd] iq:[B,G,d] seq_lens:[B](long,device) block_tables:[B,MAXB](long)
    kc/vc:[num_blocks,BS,nKV,hd] idxk_cache:[num_blocks,BS,d]. Returns [B,nH,hd].
    Numerically == msa_decode_attn recompute path (verified maxabsdiff 0). No
    IndexCache (always recompute). Gathers in native dtype then upcasts the small
    gathered tensors (never casts the whole paged cache)."""
    import torch
    q = q.float(); iq = iq.float()
    B, nH, hd = q.shape
    G, d = iq.shape[1], iq.shape[2]
    MAXB = block_tables.shape[1]; BS = block_size
    nKV = kc.shape[2]; rep = nH // nKV; repg = nH // G
    L = MAXB * BS; dev = q.device
    bt = block_tables.to(torch.long).clamp(0, kc.shape[0] - 1)  # clamp padding to valid block (masked out by `valid`); avoids OOB gather under graph
    kf = kc[bt].reshape(B, L, nKV, hd).float()
    vf = vc[bt].reshape(B, L, nKV, hd).float()
    pos = torch.arange(L, device=dev)
    valid = pos[None, :] < seq_lens.to(dev)[:, None]
    NEG = torch.finfo(torch.float32).min
    if topk_idx is None:
        # IndexCache miss/disabled: run the indexer and select fresh.
        ikf = idxk_cache[bt].reshape(B, L, d).float()
        isc = torch.einsum('bgd,bld->bgl', iq, ikf) * (d ** -0.5)
        isc = isc.masked_fill(~valid[:, None, :], NEG)
        M = isc.reshape(B, G, MAXB, BS).amax(dim=-1)
        K = min(topk_blocks, MAXB)
        topk_idx = M.topk(K, dim=-1).indices
    sel_blk = torch.zeros(B, G, MAXB, dtype=torch.bool, device=dev)
    sel_blk.scatter_(2, topk_idx, True)
    qblk = ((seq_lens.to(dev) - 1) // BS).clamp(min=0)
    sel_blk.scatter_(2, qblk[:, None, None].expand(B, G, 1), True)
    for b in range(init_blocks):
        sel_blk[:, :, b] = True
    sel_tok = (sel_blk.unsqueeze(-1).expand(B, G, MAXB, BS).reshape(B, G, L)
               & valid[:, None, :])
    k_e = kf.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
    v_e = vf.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
    sc = torch.einsum('bhd,bhld->bhl', q, k_e) * scale
    sel_h = sel_tok.repeat_interleave(repg, dim=1)
    sc = sc.masked_fill(~sel_h, NEG)
    p = torch.softmax(sc, dim=-1)
    out = torch.einsum('bhl,bhld->bhd', p, v_e)
    return out


def _msa_rewrite_blocktable(topk_idx, seq_lens, block_tables, *, block_size,
                            topk_blocks, init_blocks=0):
    """Graph-safe (no python loop / .tolist / host sync) vectorized rewrite of pure
    top-k block ids into a FIA-ready physical block_table (selected blocks ascending,
    LOCAL block LAST, padded with the local block) + actual_seq_lengths_kv. CPU-proven
    bit-identical to the msa_decode_fia python-loop rewrite (tq_dsv4_scripts/qv/validate_fg.py)."""
    import torch
    dev = topk_idx.device
    B = topk_idx.shape[0]
    MAXB = block_tables.shape[1]
    Kp = topk_blocks + 1
    sl = seq_lens.to(dev)
    qblk = ((sl - 1) // block_size).clamp(min=0)
    numBlocks = ((sl + block_size - 1) // block_size).clamp(min=1)
    order = torch.arange(MAXB, device=dev)
    src = (topk_idx < numBlocks[:, None]).to(torch.float32)
    sel = torch.zeros(B, MAXB, device=dev)
    sel.scatter_(1, topk_idx.clamp(0, MAXB - 1), src)
    sel[torch.arange(B, device=dev), qblk] = 1.0
    for ib in range(init_blocks):
        sel[:, ib] = torch.maximum(sel[:, ib], (numBlocks > ib).to(torch.float32))
    sel_mask = sel > 0.5
    masked = torch.where(sel_mask, order[None, :].expand(B, MAXB),
                         torch.full((B, MAXB), MAXB, device=dev, dtype=torch.long))
    flog = masked.sort(dim=1).values[:, :Kp]
    nsel = (flog < MAXB).sum(dim=1).clamp(min=1)
    flog = torch.where(flog < MAXB, flog, qblk[:, None])
    fbt = block_tables.to(torch.long).gather(1, flog).to(torch.int32)
    partial = (sl - qblk * block_size).clamp(min=1, max=block_size)
    kv_lens = ((nsel - 1) * block_size + partial).to(torch.int32)
    return fbt, kv_lens


def msa_decode_fia_graph(query, iq, seq_lens, block_tables, kc, vc, idxk_cache, *,
                         block_size, topk_blocks, scale, num_heads, num_kv_heads,
                         local_blocks=1, init_blocks=0, topk_idx_buffer=None, skip_topk=False):
    """Graph-safe (cudagraph-capturable) MSA decode: vectorized indexer selection +
    vectorized block_table rewrite + standard FIA. No python loops / .tolist / host
    sync. Optional IndexCache via a persistent topk_idx_buffer (skip_topk reads it;
    else compute + in-place copy_), mirroring GLM/DSA. Numerically == msa_decode_fia."""
    import torch
    import torch_npu
    B, nH, hd = query.shape
    G, d = iq.shape[1], iq.shape[2]
    MAXB = block_tables.shape[1]
    nblk = kc.shape[0]
    dev = query.device
    K = min(topk_blocks, MAXB)
    sl = seq_lens.to(dev)
    bt = block_tables.to(torch.long).clamp(0, nblk - 1)
    numBlocks = ((sl + block_size - 1) // block_size).clamp(min=1)
    if skip_topk and topk_idx_buffer is not None:
        topk_idx = topk_idx_buffer[:B]
    else:
        L = MAXB * block_size
        ikf = idxk_cache[bt].reshape(B, L, d).float()
        pos = torch.arange(L, device=dev)
        valid = pos[None, :] < sl[:, None]
        NEG = torch.finfo(torch.float32).min
        isc = torch.einsum('bgd,bld->bgl', iq.float(), ikf) * (d ** -0.5)
        isc = isc.masked_fill(~valid[:, None, :], NEG)
        M = isc.reshape(B, G, MAXB, block_size).amax(-1)[:, 0, :]
        blk = torch.arange(MAXB, device=dev)[None, :]
        M = M.masked_fill(blk >= numBlocks[:, None], NEG)
        topk_idx = M.topk(K, dim=-1).indices
        if topk_idx_buffer is not None:
            topk_idx_buffer[:B].copy_(topk_idx)
    fbt, kv_lens = _msa_rewrite_blocktable(topk_idx, sl, bt, block_size=block_size,
                                           topk_blocks=topk_blocks, init_blocks=init_blocks)
    key = kc.view(nblk, block_size, -1)
    value = vc.view(nblk, block_size, -1)
    asl_q = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query, key=key, value=value, block_table=fbt,
        input_layout="TND", block_size=block_size,
        actual_seq_lengths=asl_q, actual_seq_lengths_kv=kv_lens,
        num_key_value_heads=num_kv_heads, num_heads=nH, scale=scale, sparse_mode=0)
    return attn_out.view(B, nH, hd)


def msa_decode_fia_opgraph(query, iq, seq_lens, block_tables, kc, vc, idxk_cache, *,
                           block_size, topk_blocks, scale, num_heads, num_kv_heads,
                           local_blocks=1, init_blocks=0, sel_buffer=None):
    """FULL-cudagraph-safe MSA decode via the custom AscendC selection op
    torch.ops._C_ascend.msa_dist_top_k (single graph-safe kernel: indexer score +
    block max-pool + force-local + top-k), then a static gather to the physical
    block_table + standard FIA. Replaces the pure-torch einsum/topk/broadcast path
    that hit a BroadcastTo MTE OOB under FULL graph replay. kv_lens derives purely
    from seq_len (no data-dependent dedup): the op pads with the local block id, and
    those duplicate slots fall beyond kv_lens so FIA never reads them twice.
    Numerically == msa_decode_fia / msa_decode_vec."""
    import torch
    import torch_npu
    B, nH, hd = query.shape
    G = iq.shape[1]
    d = iq.shape[2]
    MAXB = block_tables.shape[1]
    nblk = kc.shape[0]
    dev = query.device
    topk_total = topk_blocks + local_blocks + init_blocks
    sl = seq_lens.to(torch.int32)
    # Clamp the block_table fed to the selection op to the idxk side-cache block
    # range. The kernel reads idxk_cache[block_table[b]] for b<numBlocks where
    # numBlocks=ceil(seq_len/128); under FULL-graph replay a stale/garbage seq_len
    # (or padding slots beyond the real sequence length) yields out-of-range phys
    # ids -> MTE DDR OOB. Clamping guarantees every gathered phys is in-range; valid
    # entries (< num_gpu_blocks << idxk blocks) are unaffected, so correctness holds
    # when seq_len is right and we merely avoid the crash when it is not.
    _nblk_idxk = idxk_cache.shape[0]
    bt_i32 = block_tables.clamp(0, _nblk_idxk - 1).to(torch.int32).contiguous()
    iqb = iq.to(torch.bfloat16).contiguous()
    ikc = idxk_cache.view(idxk_cache.shape[0], block_size, d).to(torch.bfloat16)
    if sel_buffer is None:
        sel_buffer = torch.zeros(B, G, topk_total, dtype=torch.int32, device=dev)
    sel = torch.ops._C_ascend.msa_dist_top_k(
        iqb, ikc, sl, bt_i32, sel_buffer,
        block_size, topk_blocks, local_blocks, init_blocks)
    sel_logical = sel[:, 0, :].to(torch.long).clamp(0, MAXB - 1)
    fbt = block_tables.to(torch.long).gather(1, sel_logical).to(torch.int32)
    sll = sl.to(torch.long)
    numBlocks = ((sll + block_size - 1) // block_size).clamp(min=1)
    nsel = numBlocks.clamp(max=topk_blocks)
    partial = (sll - (numBlocks - 1) * block_size).clamp(min=1, max=block_size)
    kv_lens = ((nsel - 1) * block_size + partial).to(torch.int32)
    key = kc.view(nblk, block_size, -1)
    value = vc.view(nblk, block_size, -1)
    asl_q = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query, key=key, value=value, block_table=fbt,
        input_layout="TND", block_size=block_size,
        actual_seq_lengths=asl_q, actual_seq_lengths_kv=kv_lens,
        num_key_value_heads=num_kv_heads, num_heads=nH, scale=scale, sparse_mode=0)
    return attn_out.view(B, nH, hd)


def msa_decode_fia(query, iq, seq_lens, block_tables, kc, vc, idxk_cache, *,
                   block_size, topk_blocks, scale, num_heads, num_kv_heads,
                   local_blocks=1, init_blocks=0, topk_idx=None):
    """Phase A: torch block-selection -> filtered (rewritten) block_table whose
    LAST entry is the local/partial block (all earlier entries full blocks) ->
    standard torch_npu.npu_fused_infer_attention_score over only selected blocks
    (KV gather done inside the FIA kernel; sparse_mode=0 + per-seq
    actual_seq_lengths_kv truncates the partial local block for causality).
    Selection is eager (PIECEWISE) here; Phase C replaces it with a fused op.
    Returns [B, nH, hd]. Numerically == msa_decode_vec."""
    import torch, torch_npu
    B, nH, hd = query.shape
    G, d = iq.shape[1], iq.shape[2]
    MAXB = block_tables.shape[1]
    nblk = kc.shape[0]
    dev = query.device
    Kp = topk_blocks + 1
    bt = block_tables.to(torch.long).clamp(0, nblk - 1)
    L = MAXB * block_size
    pos = torch.arange(L, device=dev)
    sl_dev = seq_lens.to(dev)
    qblk = ((sl_dev - 1) // block_size).clamp(min=0)
    if topk_idx is None:
        # IndexCache miss/disabled: run the indexer and select fresh.
        ikf = idxk_cache[bt].reshape(B, L, d).float()
        valid = pos[None, :] < sl_dev[:, None]
        NEG = torch.finfo(torch.float32).min
        isc = torch.einsum('bgd,bld->bgl', iq.float(), ikf) * (d ** -0.5)
        isc = isc.masked_fill(~valid[:, None, :], NEG)
        M = isc.reshape(B, G, MAXB, block_size).amax(-1)[:, 0, :]   # [B,MAXB] (per-rank G=1)
        blk_idx = torch.arange(MAXB, device=dev)[None, :]
        M = M.masked_fill(blk_idx > qblk[:, None], NEG)             # mask future blocks
        K = min(topk_blocks, MAXB)
        topk_idx = M.topk(K, dim=-1).indices                       # [B,K] block positions
    elif topk_idx.dim() == 3:
        topk_idx = topk_idx[:, 0, :]                               # [B,G,K] -> [B,K] (G=1)
    fbt = torch.zeros(B, Kp, dtype=torch.int32, device=dev)
    kv_lens = []
    qblk_l = qblk.tolist(); sl_l = seq_lens.tolist(); tk = topk_idx.tolist()
    for b in range(B):
        ql = int(qblk_l[b]); sl = int(sl_l[b])
        sel = {x for x in tk[b] if x <= ql}
        sel.add(ql)
        for ib in range(min(init_blocks, ql + 1)):
            sel.add(ib)
        sel = sorted(sel)                                      # ascending -> local (ql) LAST
        nsel = len(sel)
        phys = bt[b, torch.tensor(sel, device=dev, dtype=torch.long)].to(torch.int32)
        fbt[b, :nsel] = phys
        if nsel < Kp:
            fbt[b, nsel:] = phys[-1]
        partial = sl - ql * block_size                         # tokens in local block (1..block_size)
        kv_lens.append((nsel - 1) * block_size + partial)
    key = kc.view(nblk, block_size, -1)
    value = vc.view(nblk, block_size, -1)
    asl_q = list(range(1, B + 1))
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query, key=key, value=value, block_table=fbt,
        input_layout="TND", block_size=block_size,
        actual_seq_lengths=asl_q, actual_seq_lengths_kv=kv_lens,
        num_key_value_heads=num_kv_heads, num_heads=nH, scale=scale, sparse_mode=0,
    )
    return attn_out.view(B, nH, hd)
