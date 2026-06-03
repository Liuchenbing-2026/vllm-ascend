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

Backports the core mechanism of vllm PR #43447 ("Selective prefix-cache
retention for sliding-window KV cache") to the v0.20.2rc vllm base used by
vllm-ascend.

Mechanism: when a request's sliding-window blocks slide out of the window,
the manager hands them back to the free queue. The default behavior appends
them all to the back, where they get evicted last - but in concurrent long-
context workloads, the *uncached* scratch blocks of a new request flush away
the *cached* prefix blocks of older requests because both sit in the same
queue and the cached ones are older. PR #43447's fix:

  - Put **uncached** (block_hash is None) blocks at the *front* of the free
    queue -> they get reused first.
  - Keep **cached** blocks at the *back* -> they survive concurrent scratch
    allocations and remain available for future prefix-cache hits.

On DSv4 trace replay (vllm main repo), this lifts prefix_cache_hit from 0% to
~74% under heavy 1M-context concurrency.

Scope of this patch:
  - Pure monkey-patch; no signature changes to public APIs.
  - Gated by ``VLLM_ASCEND_ENABLE_PREFIX_CACHE_RETENTION``; default 0 to
    preserve current behavior bit-for-bit.
  - Applies only to ``SlidingWindowManager`` paths (DSv4's
    ``AscendDeepseekV4Compressor`` / ``AscendDeepseekV4SWACache`` register
    ``SlidingWindowMLASpec`` which inherits ``SlidingWindowSpec``).
  - Does NOT touch ``CompressAttentionManager`` (DSv4 indexer path) which
    uses ``MLAAttentionSpec`` and is out of scope for PR #43447.

What it patches:
  1. ``FreeKVCacheBlockQueue.prepend_n``  -- new method (insert at queue head).
  2. ``BlockPool.free_blocks``            -- adds ``prepend=False`` kwarg.
  3. ``SlidingWindowManager.remove_skipped_blocks``  -- splits cached vs
     uncached when sliding-window blocks roll out.
  4. ``SlidingWindowManager.free``        -- same split when a request finishes.

Selective retention (``VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` env, sparse
checkpointing of SWA tails) is intentionally NOT ported in this first pass;
the free-queue ordering change alone captures most of the hit-rate win.
"""

from __future__ import annotations

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

from vllm_ascend import envs as _envs_ascend


def _enabled() -> bool:
    """Single source of truth for the env gate, evaluated lazily per call."""
    return _envs_ascend.VLLM_ASCEND_ENABLE_PREFIX_CACHE_RETENTION


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
# 3. SlidingWindowManager.remove_skipped_blocks  --  split cached vs uncached
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
# 4. SlidingWindowManager.free  --  same cached/uncached split when a request
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
