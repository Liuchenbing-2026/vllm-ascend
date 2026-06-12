# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DSv4 partial compressed prefix-cache hooks.

These two monkey-patches are vllm-ascend-side hooks that vLLM does not provide
upstream (they are not part of vLLM PR #43447 either). They attach the DSv4
partial-cache cleanup to BlockPool eviction, and forward partial-hit copy block
ids from Scheduler to the model runner.

The underlying vLLM is assumed to already provide the PR #43447 APIs
(``FreeKVCacheBlockQueue.prepend_n``, ``BlockPool.free_blocks(prepend=...)``,
``SlidingWindowManager.free(prepend=True)`` and
``SingleTypeKVCacheManager.remove_skipped_blocks(prepend=True)``) — partial
cache code calls them directly.
"""

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm_ascend.core.single_type_kv_cache_manager import (
    remove_partial_cache_entries_for_block,
)


def _patch_partial_prefix_cache_cleanup() -> None:
    current = BlockPool._maybe_evict_cached_block
    if getattr(current, "_vllm_ascend_partial_prefix_cache_cleanup_patch", False):
        return

    original_maybe_evict = current

    def _maybe_evict_cached_block(self: BlockPool, block: KVCacheBlock) -> bool:
        evicted = original_maybe_evict(self, block)
        remove_partial_cache_entries_for_block(self, block.block_id)
        return evicted

    _maybe_evict_cached_block._vllm_ascend_partial_prefix_cache_cleanup_patch = True
    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block
    logger.debug("Patched BlockPool partial prefix-cache cleanup.")


def _patch_scheduler_copy_blocks() -> None:
    try:
        from vllm.v1.core.sched import scheduler as scheduler_mod
    except ImportError:
        return

    scheduler_classes = [scheduler_mod.Scheduler]
    for module_name, class_name in (
        ("vllm_ascend.core.scheduler_profiling_chunk", "ProfilingChunkScheduler"),
        ("vllm_ascend.core.scheduler_dynamic_batch", "SchedulerDynamicBatch"),
        ("vllm_ascend.patch.platform.patch_balance_schedule", "BalanceScheduler"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            scheduler_classes.append(getattr(module, class_name))
        except Exception:
            continue

    for scheduler_cls in scheduler_classes:
        current = scheduler_cls.schedule
        if getattr(current, "_vllm_ascend_copy_blocks_patch", False):
            continue
        original_schedule = current

        def schedule(self, *args, __original_schedule=original_schedule, **kwargs):
            scheduler_output = __original_schedule(self, *args, **kwargs)
            coordinator = getattr(self.kv_cache_manager, "coordinator", None)
            take_copy_block_ids = getattr(coordinator, "take_copy_block_ids", None)
            if take_copy_block_ids is not None:
                copy_block_ids = take_copy_block_ids()
                if copy_block_ids:
                    scheduler_output.new_block_ids_to_copy = copy_block_ids
            return scheduler_output

        schedule._vllm_ascend_copy_blocks_patch = True
        scheduler_cls.schedule = schedule
    logger.debug("Patched Scheduler.schedule to forward KV copy blocks.")


_patch_partial_prefix_cache_cleanup()
_patch_scheduler_copy_blocks()
