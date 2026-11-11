# SPDX-License-Identifier: Apache-2.0
"""Runtime port of FlashStates hybrid-state frontier checkpoints.

The upstream implementation lives in vLLM core (v0.33); vllm-ascend is a
plugin, so this module applies the same scheduler / KV-cache hooks at import
time. It is gated by ``additional_config["flashstate"]`` and is inert unless
enabled.

This is the first (exact-end frontier) stage:
- a finished request's recurrent state at its exact end is cached under a
  frontier key;
- the next request extending that prefix resumes there;
- GreedyDual-Size eviction is optional.

Decode-boundary snapshots, in-chunk prefill export, speculative compact
state and the Mooncake store tier are not ported here.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId, KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.hybrid_checkpoint.cost import PrefillCostModel
from vllm_ascend.hybrid_checkpoint.manager import HybridCheckpointManager

logger = logging.getLogger(__name__)

_FRONTIER_HIT_ATTR = "_flashstate_pending_hit"


def _settings(vllm_config: Any) -> dict[str, Any]:
    additional = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        return {}
    settings = additional.get("flashstate", {})
    return settings if isinstance(settings, dict) else {}


def _enabled(vllm_config: Any) -> bool:
    return bool(_settings(vllm_config).get("enable_frontier", False))


# ---------------------------------------------------------------------------
# BlockPool: keys that are not the hash of the block's own token range.
# ---------------------------------------------------------------------------
def _cache_block_under_key(
    self: BlockPool,
    block: KVCacheBlock,
    block_hash_with_group_id: BlockHashWithGroupId,
    num_tokens: int,
    request: Any = None,
    parent_block_hash: BlockHash | None = None,
) -> None:
    assert not block.is_null
    self._insert_block_hash(block_hash_with_group_id, block, num_tokens=num_tokens)


def _uncache_key(
    self: BlockPool,
    block: KVCacheBlock,
    block_hash_with_group_id: BlockHashWithGroupId,
) -> None:
    keys = self.cached_block_hashes_by_block[block.block_id]
    keys.remove(block_hash_with_group_id)
    if not keys:
        del self.cached_block_hashes_by_block[block.block_id]
    self.cached_block_hash_to_block.pop(block_hash_with_group_id, block.block_id)
    self._notify_keys_removed([block_hash_with_group_id])


def _notify_keys_removed(
    self: BlockPool, block_hashes: list[BlockHashWithGroupId]
) -> None:
    listener = getattr(self, "key_removal_listener", None)
    if listener is not None and block_hashes:
        listener(block_hashes)


def _install_block_pool_patches() -> None:
    if not hasattr(BlockPool, "cache_block_under_key"):
        BlockPool.cache_block_under_key = _cache_block_under_key  # type: ignore[attr-defined]
    if not hasattr(BlockPool, "uncache_key"):
        BlockPool.uncache_key = _uncache_key  # type: ignore[attr-defined]
    if not hasattr(BlockPool, "_notify_keys_removed"):
        BlockPool._notify_keys_removed = _notify_keys_removed  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# KVCacheManager: extend a block-aligned hit with a frontier.
# ---------------------------------------------------------------------------
_original_get_computed_blocks = KVCacheManager.get_computed_blocks
_original_allocate_slots = KVCacheManager.allocate_slots


def _get_computed_blocks(self: KVCacheManager, request: Any):
    blocks, num_new_computed_tokens, shared_prefix_boundary = (
        _original_get_computed_blocks(self, request)
    )
    checkpoints = getattr(self, "checkpoints", None)
    if checkpoints is None:
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    hit = checkpoints.frontier_hit_tokens(request, num_new_computed_tokens)
    if hit:
        extended, num_new_computed_tokens = checkpoints.extend_hit(
            request,
            blocks.blocks,
            num_new_computed_tokens,
            request.num_tokens - 1,
        )
        blocks = KVCacheBlocks(extended)

    # ``allocated`` follows once the scheduler allocates the blocks; remember
    # the local hit length for it.
    setattr(request, _FRONTIER_HIT_ATTR, num_new_computed_tokens)
    checkpoints.admitted(request, num_new_computed_tokens, blocks, 0)
    return blocks, num_new_computed_tokens, shared_prefix_boundary


def _allocate_slots(self: KVCacheManager, request: Any, *args, **kwargs):
    out = _original_allocate_slots(self, request, *args, **kwargs)
    checkpoints = getattr(self, "checkpoints", None)
    if checkpoints is not None:
        hit = getattr(request, _FRONTIER_HIT_ATTR, None)
        if hit is not None:
            delattr(request, _FRONTIER_HIT_ATTR)
            checkpoints.allocated(request, hit)
    return out


# ---------------------------------------------------------------------------
# Scheduler: create the manager and publish on request finish.
# ---------------------------------------------------------------------------
_original_scheduler_init = Scheduler.__init__
_original_free_request = Scheduler._free_request


def _scheduler_init(self: Scheduler, *args, **kwargs) -> None:
    _original_scheduler_init(self, *args, **kwargs)

    vllm_config = getattr(self, "vllm_config", None)
    if vllm_config is None or not _enabled(vllm_config):
        self.mamba_checkpoints = None
        return

    cache = self.cache_config
    if not cache.enable_prefix_caching or cache.mamba_cache_mode != "align":
        raise ValueError(
            "flashstate frontier checkpoints require --enable-prefix-caching "
            "and --mamba-cache-mode align"
        )
    if not self.use_v2_model_runner:
        raise ValueError(
            "flashstate frontier checkpoints require Model Runner V2 "
            "(VLLM_USE_V2_MODEL_RUNNER=1)"
        )

    settings = _settings(vllm_config)
    eviction_cost = None
    if settings.get("prefix_cache_eviction", "lru") == "gds":
        coefficients = settings.get("prefill_cost_coefficients")
        if not coefficients:
            raise ValueError(
                'flashstate additional_config: prefix_cache_eviction="gds" '
                "requires prefill_cost_coefficients"
            )
        eviction_cost = PrefillCostModel(*coefficients)

    from vllm.utils.hashing import get_hash_fn_by_name

    self.mamba_checkpoints = HybridCheckpointManager(
        self.kv_cache_manager.coordinator,
        get_hash_fn_by_name(self.cache_config.prefix_caching_hash_algo),
        waiting=lambda: self.waiting,
        eviction_cost=eviction_cost,
        eviction_leaf_last=bool(settings.get("prefix_cache_leaf_last", False)),
        eviction_leaf_idle_s=settings.get("prefix_cache_leaf_idle_seconds"),
        speculative_decoding=False,
        unit_events=False,
    )
    self.kv_cache_manager.checkpoints = self.mamba_checkpoints
    logger.info(
        "flashstate: frontier checkpoints enabled (gds=%s, leaf_last=%s)",
        eviction_cost is not None,
        bool(settings.get("prefix_cache_leaf_last", False)),
    )


def _free_request(self: Scheduler, request: Any, delay_free_blocks: bool = False):
    checkpoints = getattr(self, "mamba_checkpoints", None)
    if checkpoints is not None:
        try:
            checkpoints.request_finished(request)
        except Exception:
            logger.exception("flashstate: failed to publish frontier checkpoint")
    return _original_free_request(self, request, delay_free_blocks)


def _install() -> None:
    _install_block_pool_patches()
    KVCacheManager.get_computed_blocks = _get_computed_blocks  # type: ignore[method-assign]
    KVCacheManager.allocate_slots = _allocate_slots  # type: ignore[method-assign]
    Scheduler.__init__ = _scheduler_init  # type: ignore[method-assign]
    Scheduler._free_request = _free_request  # type: ignore[method-assign]


_install()