#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Prefix cache retention monkey-patches for DSv4 SWA cache.

Backports vllm PR #43447 ("Selective prefix-cache retention for sliding-window
KV cache") to the v0.20.2rc vllm base used by vllm-ascend.

The PR has TWO mechanisms; both are required for full hit-rate recovery:

  A. **Free-queue ordering** (covers heavy-concurrency 1M-context scenarios):
     Sliding-window blocks freed at window-slide time are split by cached-ness:
     uncached scratch -> prepend (reused first), cached -> append (survive).

  B. **Selective retention checkpoints** (covers 16k-class single-replay
     scenarios): instead of densely caching every block-size-aligned tail,
     keep tails only at sparse checkpoint boundaries (1024 / 2048 / 4096 / ...).
     This reduces the cached block count so future scratch eviction pressure
     can't sweep them all out, even at low concurrency.

A alone is NOT enough at 16k: dense caching produces too many cached blocks
relative to the small sliding window, so even the back-of-queue cached blocks
get flushed by a single follow-up request. B sparsifies them so each request
holds only ~5 checkpoint tails (one per power-of-2 token boundary), which
survives the next request's scratch churn comfortably.

On DSv4 trace replay (vllm main repo, A+B together), this lifts
prefix_cache_hit from 0% to 74% under heavy 1M-context concurrency.

Scope of this patch:
  - Pure monkey-patch; no signature changes to public APIs.
  - Two env gates (both default off, preserving current behavior bit-for-bit):
       VLLM_ASCEND_ENABLE_PREFIX_CACHE_RETENTION    -> activates A (free queue)
       VLLM_ASCEND_PREFIX_CACHE_RETENTION_INTERVAL  -> activates B (checkpoint)
  - Applies only to ``SlidingWindowManager`` paths (DSv4's
    ``AscendDeepseekV4Compressor`` / ``AscendDeepseekV4SWACache`` register
    ``SlidingWindowMLASpec`` which inherits ``SlidingWindowSpec``).
  - Does NOT touch ``CompressAttentionManager`` (DSv4 indexer path).

What it patches:
  1. ``FreeKVCacheBlockQueue.prepend_n``       -- new method (queue head insert).
  2. ``BlockPool.free_blocks``                 -- adds ``prepend=False`` kwarg.
  3. ``BlockPool.cache_full_blocks``           -- adds ``block_mask=None`` kwarg
     for selective retention (B).
  4. ``SlidingWindowManager.cache_blocks``     -- routes through reachable_block_mask
     when retention interval env is set (B).
  5. ``SlidingWindowManager.reachable_block_mask`` -- new classmethod, the core
     algorithm of PR #43447: which blocks to actually cache (B).
  6. ``SlidingWindowManager.remove_skipped_blocks`` -- splits cached vs uncached (A).
  7. ``SlidingWindowManager.free``             -- same split on request free (A).
"""

from __future__ import annotations

from typing import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from vllm_ascend import envs as _envs_ascend


def _enabled() -> bool:
    """Single source of truth for the env gate, evaluated lazily per call."""
    return _envs_ascend.VLLM_ASCEND_ENABLE_PREFIX_CACHE_RETENTION


def _get_retention_interval() -> int | None:
    """Resolve ``VLLM_ASCEND_PREFIX_CACHE_RETENTION_INTERVAL`` to an int.

    Returns ``None`` when retention is not set or the main retention gate is
    off. Returns ``0`` to mean "keep only the latest replay boundary".
    Returns a positive int (in tokens) for fixed-interval checkpoint retention.
    """
    if not _enabled():
        return None
    raw = _envs_ascend.VLLM_ASCEND_PREFIX_CACHE_RETENTION_INTERVAL
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip().lower() == "auto":
            # Until upstream PR #43447 is merged with the canonical multi-tier
            # 1024/2048/.../32768 schedule, "auto" maps to the smallest useful
            # interval (1024 tokens) which already captures most of the win
            # for the typical 16k-128k regime.
            return 1024
        try:
            return int(raw)
        except ValueError:
            return None
    return int(raw)


def _contiguous_blocks_for_hit(window_size: int, block_size: int,
                               use_eagle: bool) -> int:
    """Number of contiguous cached blocks needed for an SWA prefix cache hit.

    Matches the formula used by ``SlidingWindowManager.find_longest_cache_hit``
    in v0.20.2rc: ``cdiv(window_size - 1, block_size)`` plus one extra block
    when EAGLE is enabled (the proposer peeks one block past the boundary).
    """
    need = cdiv(window_size - 1, block_size)
    if use_eagle:
        need += 1
    return need


# ---------------------------------------------------------------------------
# 1. FreeKVCacheBlockQueue.prepend_n  --  insert blocks at the head of the
#    doubly-linked free list. Mirrors the structure of the existing
#    ``append_n`` so the queue remains a valid intrusive doubly-linked list.
# ---------------------------------------------------------------------------

def _prepend_n(self, blocks):
    """Put a list of blocks at the front of the free list."""
    if len(blocks) == 0:
        return
    first_block = self.fake_free_list_head.next_free_block
    assert first_block is not None, (
        "next_free_block of fake_free_list_head must always exist"
    )
    prev_block = self.fake_free_list_head
    for block in blocks:
        block.prev_free_block = prev_block
        prev_block.next_free_block = block
        prev_block = block
    prev_block.next_free_block = first_block
    first_block.prev_free_block = prev_block
    self.num_free_blocks += len(blocks)


# Only attach if not already present (forward-compat: upstream might add it).
if not hasattr(FreeKVCacheBlockQueue, "prepend_n"):
    FreeKVCacheBlockQueue.prepend_n = _prepend_n


# ---------------------------------------------------------------------------
# 2. BlockPool.free_blocks  --  add ``prepend`` kwarg. Keeps the default
#    behavior identical to the original (prepend=False -> append_n).
# ---------------------------------------------------------------------------

_orig_free_blocks = BlockPool.free_blocks


def _free_blocks_with_prepend(self, ordered_blocks, prepend: bool = False) -> None:
    """Drop-in replacement for BlockPool.free_blocks with optional prepend."""
    # Materialize the iterable to allow multiple passes.
    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1
    freed_blocks = [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
    if prepend and freed_blocks:
        # prepend_n is guaranteed to exist after the patch above.
        self.free_block_queue.prepend_n(freed_blocks)
    elif freed_blocks:
        self.free_block_queue.append_n(freed_blocks)


BlockPool.free_blocks = _free_blocks_with_prepend


# ---------------------------------------------------------------------------
# 2b. BlockPool.cache_full_blocks  --  add ``block_mask`` kwarg so the
#     SlidingWindowManager can opt out of caching individual blocks under
#     selective retention. block_mask[i] = True means cache block i, False
#     means skip it (still write the hash on the block object but do NOT
#     insert into the global hash->block dict, so it can't be a hit target).
#     ``None`` means cache all blocks (legacy behavior).
# ---------------------------------------------------------------------------

_orig_cache_full_blocks = BlockPool.cache_full_blocks


def _cache_full_blocks_with_mask(
    self,
    request,
    blocks,
    num_cached_blocks: int,
    num_full_blocks: int,
    block_size: int,
    kv_cache_group_id: int,
    block_mask: list[bool] | None = None,
) -> None:
    """Drop-in replacement for BlockPool.cache_full_blocks that honors a
    per-new-block boolean mask. ``block_mask=None`` -> original behavior."""
    if block_mask is None:
        return _orig_cache_full_blocks(
            self, request, blocks, num_cached_blocks, num_full_blocks,
            block_size, kv_cache_group_id,
        )

    if num_cached_blocks >= num_full_blocks:
        return
    num_new = num_full_blocks - num_cached_blocks
    if len(block_mask) != num_new:
        # Defensive: fall back to legacy behavior on shape mismatch rather than
        # silently mis-caching.
        return _orig_cache_full_blocks(
            self, request, blocks, num_cached_blocks, num_full_blocks,
            block_size, kv_cache_group_id,
        )

    # Replicate the head of the original function (block hash computation,
    # block_hashes selection) up to the point where we apply the mask.
    # We do this by calling the original on a contracted [start, end) range
    # one masked-block at a time would be slow; instead we inline the logic
    # but skip non-mask blocks. To keep the surface area small we call the
    # original function on each *contiguous run* of mask=True blocks.
    run_start: int | None = None
    for i in range(num_new):
        if block_mask[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                _orig_cache_full_blocks(
                    self, request, blocks,
                    num_cached_blocks + run_start,
                    num_cached_blocks + i,
                    block_size, kv_cache_group_id,
                )
                run_start = None
    if run_start is not None:
        _orig_cache_full_blocks(
            self, request, blocks,
            num_cached_blocks + run_start,
            num_full_blocks,
            block_size, kv_cache_group_id,
        )


BlockPool.cache_full_blocks = _cache_full_blocks_with_mask


# ---------------------------------------------------------------------------
# 3. SlidingWindowManager.reachable_block_mask  +  selective cache_blocks.
#    Implements PR #43447's selective-retention core.
# ---------------------------------------------------------------------------

def _reachable_block_mask(
    cls,
    start_block: int,
    end_block: int,
    alignment_tokens: int,
    kv_cache_spec,
    use_eagle: bool,
    retention_interval: int | None = None,
    num_prompt_tokens: int | None = None,
):
    """Per-block boolean mask for cache_full_blocks under selective retention.

    Returns ``None`` to mean "cache everything" (the fast path). Otherwise
    returns a list of length ``end_block - start_block`` where ``True`` marks
    blocks worth caching.

    Algorithm: identical to vllm PR #43447's SlidingWindowManager.reachable_block_mask.
      (1) At every ``segment_tokens``-aligned boundary, keep a tail of ``need``
          contiguous blocks (so a future prefix hit landing on that boundary
          can grab a window's worth of contiguous cached blocks).
      (2) Additionally keep the tail at the latest replayable prompt boundary
          (`(num_prompt_tokens - 1) // alignment_tokens * alignment_tokens`)
          even if sparse retention skipped it.
    """
    assert isinstance(kv_cache_spec, SlidingWindowSpec)
    block_size = kv_cache_spec.block_size
    if alignment_tokens % block_size != 0:
        # Alignment must be a whole number of blocks for the mask math to work.
        return None
    need = _contiguous_blocks_for_hit(
        window_size=kv_cache_spec.sliding_window,
        block_size=block_size,
        use_eagle=use_eagle,
    )
    shift = 1 if use_eagle else 0

    mask = [False] * (end_block - start_block)

    # (1) Segment-boundary tails.
    #   retention_interval = None -> dense (every alignment boundary).
    #   retention_interval = 0    -> no dense tails; only the replay boundary.
    #   retention_interval > 0    -> tail once per retention_interval tokens.
    segment_tokens = (
        alignment_tokens if retention_interval is None
        else (None if retention_interval == 0 else retention_interval)
    )
    if segment_tokens is not None:
        per_segment = segment_tokens // block_size
        if per_segment > 0:
            if need >= per_segment:
                # Tails are denser than segments — cache everything.
                return None
            for i in range(start_block, end_block):
                if i >= shift and (i - shift) % per_segment >= per_segment - need:
                    mask[i - start_block] = True

    # (2) Latest replay-boundary tail.
    if retention_interval is not None and num_prompt_tokens is not None:
        latest = (num_prompt_tokens - 1) // alignment_tokens * alignment_tokens
        prompt_end_block = latest // block_size + shift
        for i in range(
            max(start_block, prompt_end_block - need),
            min(end_block, prompt_end_block),
        ):
            mask[i - start_block] = True

    return mask


SlidingWindowManager.reachable_block_mask = classmethod(_reachable_block_mask)


_orig_swm_cache_blocks = SlidingWindowManager.cache_blocks


def _swm_cache_blocks_retention(self, request, num_tokens: int) -> None:
    """Drop-in replacement that adds selective retention when env is set.

    Signature stays identical to upstream (request, num_tokens) so all callers
    (including ascend's own coordinator that doesn't pass retention_interval)
    continue to work. We read the retention interval from env lazily.
    """
    retention_interval = _get_retention_interval()
    if retention_interval is None:
        # No selective retention -> behave exactly like the original.
        return _orig_swm_cache_blocks(self, request, num_tokens)

    num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
    num_full_blocks = num_tokens // self.block_size
    if num_cached_blocks >= num_full_blocks:
        return

    # Use scheduler_block_size if available (hybrid groups); fall back to this
    # manager's own block_size (single-group case).
    alignment_tokens = getattr(self, "scheduler_block_size", self.block_size)
    if alignment_tokens is None or alignment_tokens <= 0:
        alignment_tokens = self.block_size

    block_mask = self.reachable_block_mask(
        start_block=num_cached_blocks,
        end_block=num_full_blocks,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=self.kv_cache_spec,
        use_eagle=getattr(self, "use_eagle", False),
        retention_interval=retention_interval,
        num_prompt_tokens=request.num_prompt_tokens,
    )

    # mask=None means "cache everything" -> upstream fast path.
    if block_mask is None:
        return _orig_swm_cache_blocks(self, request, num_tokens)

    # Cache only the masked blocks. We call cache_full_blocks once per kv cache
    # group with the boolean mask; the patched cache_full_blocks above handles
    # the contiguous-run dispatch.
    for group_id in self.kv_cache_group_ids:
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=group_id,
            block_mask=block_mask,
        )
    self.num_cached_block[request.request_id] = num_full_blocks


SlidingWindowManager.cache_blocks = _swm_cache_blocks_retention


# ---------------------------------------------------------------------------
# 4. SlidingWindowManager.remove_skipped_blocks  --  split cached vs uncached
#    when sliding-window blocks roll out. Cached -> append (best-effort prefix
#    cache retention), uncached scratch -> prepend (reuse first).
#
#    This is the actual hit-rate win: in baseline behavior, concurrent
#    requests' scratch blocks land at the back of the queue and push out
#    older cached blocks.
# ---------------------------------------------------------------------------

_orig_remove_skipped_blocks = SlidingWindowManager.remove_skipped_blocks


def _remove_skipped_blocks_retention(self, request_id: str,
                                     total_computed_tokens: int) -> None:
    """Same as upstream remove_skipped_blocks but splits the freed blocks by
    cached-ness and prepends the uncached scratch group so they get reused
    first.

    Parameter name matches vllm main's ``SlidingWindowManager.remove_skipped_blocks``
    signature ``(request_id, total_computed_tokens)``.
    """
    if not _enabled():
        return _orig_remove_skipped_blocks(self, request_id, total_computed_tokens)

    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    if num_skipped_tokens <= 0:
        # All tokens within attention window — nothing to free. Matches the
        # fast-return in the upstream implementation.
        return

    blocks = self.req_to_blocks[request_id]
    num_skipped_blocks = num_skipped_tokens // self.block_size
    # ``last_computed_tokens`` may overshoot the request's currently-allocated
    # range (see upstream comment); clamp to actual blocks.
    num_skipped_blocks = min(num_skipped_blocks, len(blocks))

    cached_blocks = []
    uncached_blocks = []
    for i in range(num_skipped_blocks - 1, -1, -1):
        if blocks[i] == self._null_block:
            # Once we hit a null block (already padded by a previous call),
            # everything before it must also be null - stop.
            break
        if blocks[i].block_hash is None:
            uncached_blocks.append(blocks[i])
        else:
            cached_blocks.append(blocks[i])
        blocks[i] = self._null_block

    # Cached blocks: append (back of queue) -> survive longer for prefix hits.
    # Uncached scratch: prepend (front of queue) -> reused first by next req.
    self.block_pool.free_blocks(cached_blocks)
    self.block_pool.free_blocks(uncached_blocks, prepend=True)


SlidingWindowManager.remove_skipped_blocks = _remove_skipped_blocks_retention


# ---------------------------------------------------------------------------
# 5. SlidingWindowManager.free  --  same cached/uncached split when a request
#    finishes and releases all its blocks at once.
# ---------------------------------------------------------------------------

_orig_swm_free = SlidingWindowManager.free


def _free_retention(self, request_id: str) -> None:
    """Drop-in replacement that splits a request's freed blocks by cached-ness.

    Reverse iteration matches upstream's ordered-by-eviction-priority semantic
    (the last allocated block has the highest eviction priority).
    """
    if not _enabled():
        return _orig_swm_free(self, request_id)

    req_blocks = self.req_to_blocks.pop(request_id, [])
    if req_blocks:
        cached_blocks = []
        uncached_blocks = []
        for block in reversed(req_blocks):
            if block.block_hash is None:
                uncached_blocks.append(block)
            else:
                cached_blocks.append(block)
        self.block_pool.free_blocks(cached_blocks)
        self.block_pool.free_blocks(uncached_blocks, prepend=True)

    self.num_cached_block.pop(request_id, None)


SlidingWindowManager.free = _free_retention
