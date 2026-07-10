# SPDX-License-Identifier: Apache-2.0
"""DSpark PAGED attention metadata: non-causal swa_indices slot table + per-pass
metadata dataclasses + window formulas + draft SWA kv-cache group registration.

Original implementation (own structure/naming/comments), semantically equal to
the DSPARK_SPEC 5 contract. This is the *single source of truth* for:
  * the window numbers (ori_win_left / ori_win_right),
  * the aligned slot-table width,
  * the vectorized non-causal slot table `build_dspark_swa_indices`,
  * the draft SWA kv-cache spec / group registration that makes the draft own a
    real paged cache (kv_cache_groups > 0, the structural switch vs the eager
    private-cache path which registered ZERO groups).

The ops wrapper (ops/ds_dspark_attention_std.py) and the draft attention module
(models/ds_dspark_draft_std.py) both import the window constants + builder from
here so the numbers can never drift between the three call sites.

Two independent "block" concepts -- never conflate:
  query_block  = num_speculative_tokens (semi-AR draft block)  = 5
  cache_block  = paged KV block = cache_config.block_size       = 128
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# --- AscendC sparse-attn mask modes (must match the recompiled device kernel) --
DSPARK_SAS_ORI_MASK_MODE = 4
DSPARK_SAS_CMP_MASK_MODE = 3
_WIDTH_ALIGN = 128


# ===========================================================================
# 1. Window / width formulas (SPEC 5).
# ===========================================================================
def dspark_window(window_size: int, block_size: int) -> tuple[int, int]:
    """(ori_win_left, ori_win_right).

    ori_win_left  = window_size + block_size - 1   (real 128+5-1 = 132)
    ori_win_right = block_size - 1                 (real 5-1 = 4, > 0 == NON-causal)

    NOTE the argument order is (window_size, block_size). Passing them swapped is
    the exact bug that yields ori_win_right = window-1 = 127 and re-introduces a
    causal band -- always call with window first.
    """
    return window_size + block_size - 1, block_size - 1


def draft_causal_band(window_size: int) -> tuple[int, int]:
    """(window-1, 0): the CAUSAL scheduling band only. NOT true visibility --
    get_draft_swa_window returns this, and it is deliberately NOT used by the
    sparse op (the slot table is authoritative). Kept for the fallback gate."""
    return window_size - 1, 0


def aligned_index_width(window_size: int, block_size: int, align: int = _WIDTH_ALIGN) -> int:
    """Device-side slot-table width; >= window+block, 128-aligned.

    REAL config 128+5 = 133 -> 256.  (The fake unit test 7+5 = 12 -> 128; the
    staging DSPARK_SWA_WIDTH=128 constant was correct ONLY for that test and is
    WRONG for the paged real path -- use this function instead.)
    """
    need = int(window_size) + int(block_size)
    return ((need + align - 1) // align) * align


# ===========================================================================
# 2. Canonical non-causal visible-slot table (root cause B).
#    Vectorized over requests; every query row of a request's block gets the
#    SAME trailing-window slot list; slot ids resolved through the paged block
#    table; tail-padded to index_width with -1.
# ===========================================================================
def build_dspark_swa_indices(
    *,
    block_table: torch.Tensor,                       # [num_reqs, max_blocks] int
    query_start_loc: torch.Tensor,                   # [num_reqs+1] cu_seqlens_q
    seq_lens: torch.Tensor,                          # [num_reqs] KV len incl. current draft block
    slot_mapping: torch.Tensor | None = None,        # [num_query_tokens] paged write slots (validity)
    token_to_req_indices: torch.Tensor | None = None,  # [num_query_tokens] -> req; None => contiguous
    block_size: int = 5,                             # draft semi-AR query block (num_spec_tokens)
    window_size: int = 128,                          # draft SWA window
    cache_block_size: int = 128,                     # paged block (cache_config.block_size)
    num_query_tokens: int | None = None,
    index_width: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dspark_swa_indices[num_q,1,W] int32 pad -1, dspark_swa_lens[num_q] int32).

    Per req r (vectorized):
        query_len   = query_start_loc[r+1] - query_start_loc[r]      # block_size for a full block
        start_pos   = clamp(seq_lens[r] - query_len - window_size, 0)
        visible_len = seq_lens[r] - start_pos                        # == min(seq_len, window+block) when query_len==block
        visible positions = [start_pos, seq_lens[r])
        slot_id(pos) = block_table[r, pos//cache_block_size]*cache_block_size + pos%cache_block_size
    All rows of one req's block get an IDENTICAL slot list (pure non-causal).
    Invalid rows (token_to_req out of range OR slot_mapping < 0) -> len 0, all -1.

    The (seq_len - query_len - window) form (not the eager min(seq_len,window+block))
    is the one that stays correct under chunked prefill where query_len != block.
    """
    dev = device or block_table.device
    if index_width is None:
        index_width = aligned_index_width(window_size, block_size)
    if index_width < window_size + block_size:
        raise ValueError(f"index_width {index_width} < window+block {window_size + block_size}")
    if num_query_tokens is None:
        num_query_tokens = int(token_to_req_indices.numel()) if token_to_req_indices is not None \
            else int(query_start_loc[-1].item())

    indices = torch.full((num_query_tokens, 1, index_width), -1, dtype=torch.int32, device=dev)
    lens = torch.zeros((num_query_tokens,), dtype=torch.int32, device=dev)
    if num_query_tokens == 0:
        return indices, lens

    req_count = max(int(query_start_loc.numel()) - 1, 0)
    req_count = min(req_count, int(block_table.shape[0]), int(seq_lens.numel()))
    if req_count <= 0 or int(block_table.shape[1]) <= 0:
        return indices, lens

    # ---- per-row req id: explicit map, else contiguous [q_start[r], q_start[r+1]).
    if token_to_req_indices is not None:
        row_req = token_to_req_indices[:num_query_tokens].to(dev, torch.long)
        valid_rows = (row_req >= 0) & (row_req < req_count)
    else:
        row_req = torch.zeros(num_query_tokens, dtype=torch.long, device=dev)
        qsl_long = query_start_loc.to(dev, torch.long)
        for r in range(req_count):
            row_req[qsl_long[r]:qsl_long[r + 1]] = r
        valid_rows = torch.ones(num_query_tokens, dtype=torch.bool, device=dev)
    if slot_mapping is not None:
        sm = slot_mapping[:num_query_tokens].to(dev)
        valid_rows &= (sm >= 0) if sm.ndim == 1 else torch.all(sm >= 0, dim=-1)
    if not bool(valid_rows.any()):
        return indices, lens

    # ---- per-req trailing window start / visible length.
    qsl = query_start_loc[:req_count + 1].to(dev, torch.long)
    query_lens = qsl[1:] - qsl[:-1]                                  # [req_count]
    req_seq = seq_lens[:req_count].to(dev, torch.long)
    start_pos = torch.clamp(req_seq - query_lens - int(window_size), min=0)   # [req_count]
    vis_len = req_seq - start_pos                                    # [req_count]
    if int(vis_len.max()) > index_width:
        raise ValueError(f"visible len {int(vis_len.max())} exceeds index_width {index_width}")

    # ---- per-req slot table [req_count, W]; column j -> position start_pos[r]+j.
    off = torch.arange(index_width, dtype=torch.long, device=dev)
    vis_pos = start_pos.unsqueeze(1) + off.unsqueeze(0)             # [req_count, W]
    vis_mask = off.unsqueeze(0) < vis_len.unsqueeze(1)
    blk_num = (vis_pos // int(cache_block_size)).clamp(0, int(block_table.shape[1]) - 1)
    blk_id = block_table[:req_count].to(dev, torch.long).gather(1, blk_num)
    slot = (blk_id * int(cache_block_size) + vis_pos % int(cache_block_size)).to(torch.int32)
    slot.masked_fill_(~vis_mask, -1)                               # [req_count, W]

    # ---- scatter identical per-req rows to every query token; null invalid rows.
    clamp_req = row_req.clamp(0, req_count - 1)
    indices[:, 0, :] = slot.index_select(0, clamp_req)
    lens.copy_(vis_len.index_select(0, clamp_req).to(torch.int32))
    indices[~valid_rows] = -1
    lens[~valid_rows] = 0
    return indices, lens


# ===========================================================================
# 3. Per-pass metadata dataclasses (fields per Component D contract).
#    dspark_swa_indices / dspark_swa_lens live on BOTH decode and prefill; the
#    attention forward pushes them into the kernel as ori_sparse_indices ONLY
#    when non-None. sin/cos/sas_metadata may be None this (eager) round: the
#    attention re-applies RoPE from cad.positions and the ops wrapper lazily
#    builds sas_metadata when it is None.
# ===========================================================================
@dataclass
class DSparkDraftDecodeMeta:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int                       # PAGED cache block (=128)
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    ori_win_left: int
    ori_win_right: int
    dspark_swa_indices: torch.Tensor | None
    dspark_swa_lens: torch.Tensor | None
    sin: torch.Tensor | None = None       # TODO(serve-verify): get_cos_and_sin_dsa cache for full-graph
    cos: torch.Tensor | None = None
    sas_metadata: Any | None = None       # None => ops wrapper builds it lazily this round


@dataclass
class DSparkDraftPrefillMeta:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    ori_win_left: int
    ori_win_right: int
    dspark_swa_indices: torch.Tensor | None
    dspark_swa_lens: torch.Tensor | None
    sin: torch.Tensor | None = None
    cos: torch.Tensor | None = None
    sas_metadata: Any | None = None


@dataclass
class DSparkDraftAttnMetadata:
    """Top-level per-layer metadata handed to a draft attention layer via
    forward_context.draft_attn_metadatas. Mirrors AscendDSAMetadata's shape:
    a decode sub-struct and/or a prefill sub-struct. For the decode-drafting
    round only `decode` is populated."""
    num_actual_tokens: int
    num_input_tokens: int
    block_tables: torch.Tensor | None = None
    decode: DSparkDraftDecodeMeta | None = None
    prefill: DSparkDraftPrefillMeta | None = None


# ===========================================================================
# 4. Standalone drafting metadata builder.
#    In-tree, the AttentionGroup's builder is the dsa_v1 DSA builder; its
#    build_for_drafting must be patched to call build_dspark_swa_indices and
#    attach the swa fields (see FRAMEWORK_TOUCHPOINTS). This class is the
#    original reference implementation of that contract and can be wired as the
#    group's builder directly for a first bring-up.
# ===========================================================================
class DsparkPagedMetaBuilder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.block_size = int(kv_cache_spec.block_size)             # PAGED cache block (=128)
        hf = vllm_config.model_config.hf_config
        self.window_size = int(getattr(hf, "sliding_window", 128))
        spec = vllm_config.speculative_config
        self.query_block = int(getattr(spec, "num_speculative_tokens", 0)
                               or getattr(getattr(spec, "draft_model_config", None), "hf_config", None)
                               and getattr(spec.draft_model_config.hf_config, "dspark_block_size", 5) or 5)
        self.index_width = aligned_index_width(self.window_size, self.query_block)

    def _is_noncausal_draft(self, cad) -> bool:
        # Slot table only applies once the proposer flipped cad.causal to False.
        return getattr(cad, "causal", True) is False

    def _swa(self, cad, slot_mapping, num_tokens):
        if not self._is_noncausal_draft(cad):
            return None, None
        block_table = getattr(cad, "block_table_tensor", None)
        if block_table is None:
            return None, None
        num_reqs = int(getattr(cad, "num_reqs", 0) or 0)
        positions = cad.positions[:num_tokens]
        sm = slot_mapping[:num_tokens] if slot_mapping is not None else None
        return build_dspark_swa_indices(
            block_table=block_table[:num_reqs],
            query_start_loc=cad.query_start_loc[:num_reqs + 1],
            seq_lens=cad.seq_lens[:num_reqs],
            slot_mapping=sm,
            token_to_req_indices=getattr(cad, "token_to_req_indices", None),
            block_size=self.query_block,
            window_size=self.window_size,
            cache_block_size=self.block_size,
            num_query_tokens=int(positions.numel()),
            index_width=self.index_width,
            device=positions.device,
        )

    def build_for_drafting(self, cad, draft_index: int = 1, block_size: int | None = None):
        """Decode-drafting build (the semi-AR block is a decode-shaped pass).
        Returns DSparkDraftAttnMetadata carrying the swa slot table + win params.
        block_size arg is the PAGED cache block passed by the proposer (=128)."""
        del draft_index
        cache_block = int(block_size) if block_size else self.block_size
        num_tokens = int(getattr(cad, "num_input_tokens", 0) or 0) \
            or int(getattr(cad, "num_actual_tokens", 0) or 0)
        num_reqs = int(getattr(cad, "num_reqs", 0) or 0)
        slot_mapping = getattr(cad, "slot_mapping", None)
        swa_i, swa_l = self._swa(cad, slot_mapping, num_tokens)
        has_idx = swa_i is not None
        wl, wr = dspark_window(self.window_size, self.query_block) if has_idx \
            else draft_causal_band(self.window_size)
        decode = DSparkDraftDecodeMeta(
            block_table=getattr(cad, "block_table_tensor", None),
            slot_mapping=slot_mapping[:num_tokens] if slot_mapping is not None else None,
            block_size=cache_block,
            seq_lens=cad.seq_lens[:num_reqs],
            query_start_loc=cad.query_start_loc[:num_reqs + 1],
            ori_win_left=wl, ori_win_right=wr,
            dspark_swa_indices=swa_i, dspark_swa_lens=swa_l,
            # TODO(serve-verify): sin/cos via get_cos_and_sin_dsa(use_cache=True,
            #   draft_index=draft_index) + sas_metadata via the device metadata op
            #   are only required once the draft runs in a captured graph.
            sin=None, cos=None, sas_metadata=None)
        return DSparkDraftAttnMetadata(
            num_actual_tokens=num_tokens, num_input_tokens=num_tokens,
            block_tables=getattr(cad, "block_table_tensor", None),
            decode=decode, prefill=None)


# ===========================================================================
# 5. Draft SWA kv-cache GROUP registration (kv_cache_groups > 0 for the draft).
#    This is the switch that distinguishes the paged path from the eager
#    attention-free path (which left draft_attn_groups empty / groups == 0).
# ===========================================================================
def draft_swa_layer_names(draft_model) -> list[str]:
    """One prefix per draft decoder layer's registered SWA Attention module.
    Mirrors model.get_draft_kv_cache_layer_names()."""
    return [layer.self_attn.dsa_attn.swa_cache_layer.prefix
            for layer in draft_model.layers.values()]


def make_draft_swa_kv_cache_spec(vllm_config, cache_block_size: int):
    """What each draft SWA Attention layer's get_kv_cache_spec() must return so
    the 3 draft layers form their OWN kv-cache group, separate from the target's
    43 full-attention MLA layers: an MLA spec that carries the sliding window so
    the coordinator will not merge it upward into the target group.

    TODO(serve-verify): confirm the exact vllm-ascend v0.23 spec class + ctor.
    AscendSlidingWindowMLASpec is imported by deepseek_v4.py (see tp_diffs) and is
    the most likely fit; if it lacks a sliding_window field the window must be
    tagged on the Attention layer and read by the group splitter instead.
    """
    from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec  # noqa
    hf = vllm_config.model_config.hf_config
    return AscendSlidingWindowMLASpec(          # TODO(serve-verify): exact kwargs
        block_size=int(cache_block_size),        # paged block (=128)
        num_kv_heads=1,                          # MLA latent / MQA
        head_size=int(hf.head_dim),              # 512
        dtype=vllm_config.model_config.dtype,
        sliding_window=int(hf.sliding_window),   # 128 -> forces a distinct group
        # compress_ratio=1  (DSpark uncompressed; builder reads getattr(spec,'compress_ratio',0))
    )


def register_draft_swa_groups(proposer, kv_cache_config) -> None:
    """Partition kv_cache_config.kv_cache_groups, keep the groups owning the draft
    SWA layers, build one AttentionGroup per (backend, spec) bucket. Sets
    proposer.draft_attn_groups / kv_cache_gid / kernel_block_size. Raises if no
    draft group is present (paged path requires them)."""
    from collections import defaultdict
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    from vllm.v1.worker.utils import AttentionGroup

    draft_names = set(draft_swa_layer_names(proposer.model))
    proposer._draft_attn_layer_names = draft_names
    proposer.attn_layer_names = sorted(draft_names)
    proposer.piece_all_attn_layer_name = [
        list(proposer.attn_layer_names) for _ in range(proposer.num_speculative_tokens)]
    proposer.draft_attn_groups = []
    layers = get_layers_from_vllm_config(proposer.vllm_config, AttentionLayerBase)
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        keep = [n for n in group.layer_names if n in draft_names]
        if not keep:
            continue
        buckets: dict = defaultdict(list)
        specs: dict = {}
        for name in keep:
            backend = layers[name].get_attn_backend()
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            key = (backend.full_cls_name(), spec)
            specs[key] = (backend, spec)
            buckets[key].append(name)
        for key, names in buckets.items():
            backend, spec = specs[key]
            builder = backend.get_builder_cls()(spec, names, proposer.vllm_config, proposer.device)
            proposer.draft_attn_groups.append(
                AttentionGroup(backend, names, spec, gid, [builder]))
    if not proposer.draft_attn_groups:
        raise RuntimeError(
            "DSpark paged path requires registered draft SWA kv-cache groups; "
            "set VLLM_ASCEND_DSPARK_USE_PRIVATE_CACHE=1 to fall back to the eager path.")
    proposer.kv_cache_gid = proposer.draft_attn_groups[0].kv_cache_group_id
    proposer.kernel_block_size = int(proposer.draft_attn_groups[0].kv_cache_spec.block_size)


# ===========================================================================
# 6. Self-test: reproduce the oracle unit-test numbers (adapted to the
#    vectorized builder). All rows of a block identical; -1 pad; correct start.
# ===========================================================================
def _expected_slots(block_table, req_idx, cache_block_size, start, end):
    return torch.tensor(
        [int(block_table[req_idx, p // cache_block_size]) * cache_block_size + p % cache_block_size
         for p in range(start, end)], dtype=torch.int32)


def _selftest():
    # Case 1: single req, window=7 block=5 seq_len=15 -> visible_len 12, start 3.
    assert dspark_window(7, 5) == (11, 4)
    bt = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    idx, lens = build_dspark_swa_indices(
        block_table=bt, query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([15], dtype=torch.int32),
        token_to_req_indices=torch.zeros(5, dtype=torch.int32),
        block_size=5, window_size=7, cache_block_size=64, index_width=128)
    assert torch.equal(lens, torch.full((5,), 12, dtype=torch.int32)), lens
    exp = _expected_slots(bt, 0, 64, 3, 15)
    for row in range(5):
        assert torch.equal(idx[row, 0, :12], exp)
        assert torch.all(idx[row, 0, 12:] == -1)

    # Case 2: two reqs interleaved, window=4 block=2, seq_lens=[12,22] -> vis 6.
    bt2 = torch.tensor([[10], [20]], dtype=torch.int32)
    idx2, lens2 = build_dspark_swa_indices(
        block_table=bt2, query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([12, 22], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        block_size=2, window_size=4, cache_block_size=64, index_width=128)
    assert torch.equal(lens2, torch.full((4,), 6, dtype=torch.int32)), lens2
    exp0 = _expected_slots(bt2, 0, 64, 6, 12)
    exp1 = _expected_slots(bt2, 1, 64, 16, 22)
    for row in (0, 2):
        assert torch.equal(idx2[row, 0, :6], exp0)
    for row in (1, 3):
        assert torch.equal(idx2[row, 0, :6], exp1)

    # Real width sanity: 128+5 -> 256 (NOT 128).
    assert aligned_index_width(128, 5) == 256
    print("ds_dspark_meta_std self-test PASS (matches oracle unit-test numbers; real width 256)")


if __name__ == "__main__":
    _selftest()
