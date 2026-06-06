# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projectx
import os
import sys
from math import lcm
from typing import Literal

import vllm
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.single_type_kv_cache_manager import (
    CompressAttentionManager,
    get_manager_for_kv_cache_spec,
)

# vllm PR #43447 added prefix-cache local KV retention (sliding-window
# checkpoint tails). The retention constants and SlidingWindowManager class
# live in upstream single_type_kv_cache_manager. Fall back to no-op symbols on
# older vllm without PR #43447 so this patch keeps working.
try:
    from vllm.v1.core.single_type_kv_cache_manager import (
        AUTO_RETENTION_BASE,
        AUTO_RETENTION_INTERVAL,
    )

    _HAS_LOCAL_KV_RETENTION = True
except ImportError:  # pragma: no cover - older vllm without PR #43447
    AUTO_RETENTION_BASE = 1024
    AUTO_RETENTION_INTERVAL = 32768
    _HAS_LOCAL_KV_RETENTION = False

USE_MULTI_GROUPS_KV_CACHE = True


def _read_prefix_cache_retention_interval() -> int | None:
    value = os.getenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL")
    if value is None:
        return None
    return int(value)


def _validate_prefix_cache_retention_interval(
    retention_interval: int | Literal["auto"] | None,
    alignment_tokens: int,
    kv_cache_config: KVCacheConfig,
) -> None:
    if retention_interval is None or retention_interval == "auto":
        return
    if not any(
        isinstance(group.kv_cache_spec, SlidingWindowSpec)
        for group in kv_cache_config.kv_cache_groups
    ):
        raise ValueError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL is set but this model has "
            "no sliding-window KV cache group."
        )
    if retention_interval < 0 or retention_interval % alignment_tokens != 0:
        raise ValueError(
            f"VLLM_PREFIX_CACHE_RETENTION_INTERVAL ({retention_interval}) "
            "must be non-negative and a multiple of the prefix-cache "
            f"alignment ({alignment_tokens})."
        )


def _install_prefix_cache_retention_patch() -> None:
    if not hasattr(FreeKVCacheBlockQueue, "prepend_n"):

        def prepend_n(self: FreeKVCacheBlockQueue, blocks: list[KVCacheBlock]) -> None:
            if len(blocks) == 0:
                return
            first_block = self.fake_free_list_head.next_free_block
            assert first_block is not None, (
                "next_free_block of fake_free_list_head should always exist"
            )
            prev_block = self.fake_free_list_head
            for block in blocks:
                block.prev_free_block = prev_block
                prev_block.next_free_block = block
                prev_block = block
            prev_block.next_free_block = first_block
            first_block.prev_free_block = prev_block
            self.num_free_blocks += len(blocks)

        FreeKVCacheBlockQueue.prepend_n = prepend_n  # type: ignore[attr-defined]

    if not hasattr(BlockPool, "_ascend_orig_cache_full_blocks"):
        BlockPool._ascend_orig_cache_full_blocks = BlockPool.cache_full_blocks  # type: ignore[attr-defined]

        def cache_full_blocks(
            self: BlockPool,
            request,
            blocks: list[KVCacheBlock],
            num_cached_blocks: int,
            num_full_blocks: int,
            block_size: int,
            kv_cache_group_id: int,
            block_mask: list[bool] | None = None,
        ) -> None:
            orig_cache_full_blocks = self._ascend_orig_cache_full_blocks  # type: ignore[attr-defined]
            if block_mask is None:
                return orig_cache_full_blocks(
                    request,
                    blocks,
                    num_cached_blocks,
                    num_full_blocks,
                    block_size,
                    kv_cache_group_id,
                )

            new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
            assert len(block_mask) == len(new_full_blocks)
            masked_blocks: list[KVCacheBlock] = []
            for keep, block in zip(block_mask, new_full_blocks):
                if not keep and not block.is_null:
                    block.is_null = True
                    masked_blocks.append(block)
            try:
                return orig_cache_full_blocks(
                    request,
                    blocks,
                    num_cached_blocks,
                    num_full_blocks,
                    block_size,
                    kv_cache_group_id,
                )
            finally:
                for block in masked_blocks:
                    block.is_null = False

        BlockPool.cache_full_blocks = cache_full_blocks  # type: ignore[assignment]

    if not hasattr(BlockPool, "_ascend_orig_free_blocks"):
        BlockPool._ascend_orig_free_blocks = BlockPool.free_blocks  # type: ignore[attr-defined]

        def free_blocks(
            self: BlockPool,
            ordered_blocks,
            prepend: bool = False,
        ) -> None:
            blocks_list = list(ordered_blocks)
            for block in blocks_list:
                block.ref_cnt -= 1
            freed_blocks = [
                block
                for block in blocks_list
                if block.ref_cnt == 0 and not block.is_null
            ]
            if prepend:
                self.free_block_queue.prepend_n(freed_blocks)
            else:
                self.free_block_queue.append_n(freed_blocks)

        BlockPool.free_blocks = free_blocks  # type: ignore[assignment]

    if not hasattr(SingleTypeKVCacheManager, "_ascend_orig_cache_blocks"):
        SingleTypeKVCacheManager._ascend_orig_cache_blocks = SingleTypeKVCacheManager.cache_blocks  # type: ignore[attr-defined]

        @classmethod
        def reachable_block_mask(
            cls,
            start_block: int,
            end_block: int,
            alignment_tokens: int | None,
            kv_cache_spec: KVCacheSpec,
            use_eagle: bool,
            retention_interval: int | Literal["auto"] | None = None,
            num_prompt_tokens: int | None = None,
        ) -> list[bool] | None:
            return None

        def cache_blocks(
            self: SingleTypeKVCacheManager,
            request,
            num_tokens: int,
            retention_interval: int | Literal["auto"] | None = None,
            alignment_tokens: int | None = None,
        ) -> None:
            num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
            num_full_blocks = num_tokens // self.block_size
            if num_cached_blocks >= num_full_blocks:
                return
            alignment = alignment_tokens
            if alignment is None:
                alignment = getattr(self, "_prefix_cache_alignment_tokens", None)
            block_mask = self.reachable_block_mask(
                start_block=num_cached_blocks,
                end_block=num_full_blocks,
                alignment_tokens=alignment,
                kv_cache_spec=self.kv_cache_spec,
                use_eagle=getattr(self, "_prefix_cache_use_eagle", False),
                retention_interval=retention_interval,
                num_prompt_tokens=getattr(request, "num_prompt_tokens", None),
            )
            self.block_pool.cache_full_blocks(
                request=request,
                blocks=self.req_to_blocks[request.request_id],
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=self.block_size,
                kv_cache_group_id=self.kv_cache_group_id,
                block_mask=block_mask,
            )
            self.num_cached_block[request.request_id] = num_full_blocks

        def remove_skipped_blocks(
            self: SingleTypeKVCacheManager,
            request_id: str,
            total_computed_tokens: int,
        ) -> None:
            num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
            if num_skipped_tokens <= 0:
                return
            blocks = self.req_to_blocks[request_id]
            num_skipped_blocks = min(
                num_skipped_tokens // self.block_size,
                len(blocks),
            )
            removed_cached_blocks: list[KVCacheBlock] = []
            removed_uncached_blocks: list[KVCacheBlock] = []
            for i in range(num_skipped_blocks - 1, -1, -1):
                if blocks[i] == self._null_block:
                    break
                if blocks[i].block_hash is None:
                    removed_uncached_blocks.append(blocks[i])
                else:
                    removed_cached_blocks.append(blocks[i])
                blocks[i] = self._null_block
            self.block_pool.free_blocks(removed_cached_blocks)
            self.block_pool.free_blocks(removed_uncached_blocks, prepend=True)

        SingleTypeKVCacheManager.reachable_block_mask = reachable_block_mask  # type: ignore[attr-defined]
        SingleTypeKVCacheManager.cache_blocks = cache_blocks  # type: ignore[assignment]
        SingleTypeKVCacheManager.remove_skipped_blocks = remove_skipped_blocks  # type: ignore[assignment]

    if not hasattr(SlidingWindowManager, "_ascend_prefix_retention_patch"):

        @classmethod
        def _contiguous_blocks_for_hit(
            cls,
            window_size: int,
            block_size: int,
            use_eagle: bool,
        ) -> int:
            need = cdiv(window_size - 1, block_size)
            if use_eagle:
                need += 1
            return need

        @classmethod
        def reachable_block_mask(
            cls,
            start_block: int,
            end_block: int,
            alignment_tokens: int | None,
            kv_cache_spec: KVCacheSpec,
            use_eagle: bool,
            retention_interval: int | Literal["auto"] | None = None,
            num_prompt_tokens: int | None = None,
        ) -> list[bool] | None:
            assert isinstance(kv_cache_spec, SlidingWindowSpec)
            if alignment_tokens is None:
                return None
            assert alignment_tokens % kv_cache_spec.block_size == 0

            block_size = kv_cache_spec.block_size
            need = cls._contiguous_blocks_for_hit(
                window_size=kv_cache_spec.sliding_window,
                block_size=block_size,
                use_eagle=use_eagle,
            )
            shift = 1 if use_eagle else 0
            mask = [False] * (end_block - start_block)

            segment_tokens: int | None
            if retention_interval is None:
                segment_tokens = alignment_tokens
            elif retention_interval == "auto":
                segment_tokens = AUTO_RETENTION_INTERVAL
            elif retention_interval == 0:
                segment_tokens = None
            else:
                segment_tokens = retention_interval

            if segment_tokens is not None:
                per_segment = segment_tokens // block_size
                if need >= per_segment:
                    return None
                for i in range(start_block, end_block):
                    if i >= shift and (i - shift) % per_segment >= per_segment - need:
                        mask[i - start_block] = True

            if retention_interval is not None and num_prompt_tokens is not None:
                latest = (num_prompt_tokens - 1) // alignment_tokens * alignment_tokens
                prompt_end_block = latest // block_size + shift
                for i in range(
                    max(start_block, prompt_end_block - need),
                    min(end_block, prompt_end_block),
                ):
                    mask[i - start_block] = True

            return mask

        def free(self: SlidingWindowManager, request_id: str) -> None:
            req_blocks = self.req_to_blocks.pop(request_id, [])
            if req_blocks:
                cached_blocks: list[KVCacheBlock] = []
                uncached_blocks: list[KVCacheBlock] = []
                for block in reversed(req_blocks):
                    if block.block_hash is None:
                        uncached_blocks.append(block)
                    else:
                        cached_blocks.append(block)
                self.block_pool.free_blocks(cached_blocks)
                self.block_pool.free_blocks(uncached_blocks, prepend=True)
            self.num_cached_block.pop(request_id, None)

        SlidingWindowManager._contiguous_blocks_for_hit = _contiguous_blocks_for_hit  # type: ignore[attr-defined]
        SlidingWindowManager.reachable_block_mask = reachable_block_mask  # type: ignore[assignment]
        SlidingWindowManager.free = free  # type: ignore[assignment]
        SlidingWindowManager._ascend_prefix_retention_patch = True  # type: ignore[attr-defined]

    if "_ascend_orig_mamba_cache_blocks" not in MambaManager.__dict__:
        MambaManager._ascend_orig_mamba_cache_blocks = MambaManager.cache_blocks  # type: ignore[attr-defined]

        def mamba_cache_blocks(
            self: MambaManager,
            request,
            num_tokens: int,
            retention_interval: int | Literal["auto"] | None = None,
            alignment_tokens: int | None = None,
        ) -> None:
            return self._ascend_orig_mamba_cache_blocks(request, num_tokens)  # type: ignore[attr-defined]

        MambaManager.cache_blocks = mamba_cache_blocks  # type: ignore[assignment]

    if "_ascend_orig_compress_cache_blocks" not in CompressAttentionManager.__dict__:
        CompressAttentionManager._ascend_orig_compress_cache_blocks = CompressAttentionManager.cache_blocks  # type: ignore[attr-defined]

        def compress_cache_blocks(
            self: CompressAttentionManager,
            request,
            num_tokens: int,
            retention_interval: int | Literal["auto"] | None = None,
            alignment_tokens: int | None = None,
        ) -> None:
            num_tokens //= self.compress_ratio
            return super(CompressAttentionManager, self).cache_blocks(
                request,
                num_tokens,
                retention_interval=retention_interval,
                alignment_tokens=alignment_tokens,
            )

        CompressAttentionManager.cache_blocks = compress_cache_blocks  # type: ignore[assignment]


_install_prefix_cache_retention_patch()
_HAS_LOCAL_KV_RETENTION = True


class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        eagle_attn_layer_names: list[str] | None = None,
        metrics_collector: KVCacheMetricsCollector | None = None,
        max_num_batched_tokens: int | None = None,
        # vllm PR #43447: optional prefix-cache local KV retention. The
        # scheduler passes ``CacheConfig.prefix_cache_retention_interval``
        # through ``get_kv_cache_coordinator`` -> here.
        local_kv_retention_interval: int | Literal["auto"] | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # Fall back to `max_model_len` when unset so the recycling-aware
        # admission cap (vLLM PR #40946) collapses to the prior uncapped
        # behavior. The scheduler always supplies the real value at runtime.
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(g.kv_cache_spec.block_size % hash_block_size == 0 for g in kv_cache_config.kv_cache_groups), (
                "block_size must be divisible by hash_block_size"
            )
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

        # vllm PR #43447: store the retention interval and mirror upstream's
        # one-time info log. The inherited ``HybridKVCacheCoordinator.cache_blocks``
        # consults ``self.local_kv_retention_interval`` + ``self.eagle_lookup_group_ids``
        # (set by ``verify_and_split_kv_cache_groups`` below) to route sliding-window
        # groups through ``cache_blocks_at_boundaries``.
        if local_kv_retention_interval is None:
            local_kv_retention_interval = _read_prefix_cache_retention_interval()
        self.local_kv_retention_interval = local_kv_retention_interval
        _validate_prefix_cache_retention_interval(
            self.local_kv_retention_interval,
            self.lcm_block_size,
            kv_cache_config,
        )
        self._init_prefix_cache_retention_metadata()
        if self.local_kv_retention_interval is not None and _HAS_LOCAL_KV_RETENTION:
            has_sliding_window_group = any(
                isinstance(manager, SlidingWindowManager) for manager in self.single_type_managers
            )
            if has_sliding_window_group:
                if self.local_kv_retention_interval == "auto":
                    logger.info(
                        "Using prefix-cache local KV retention strategy: retain "
                        "sliding-window checkpoint tails at powers of 2 from %d to "
                        "%d tokens, then every %d tokens, plus the latest replayable "
                        "prompt boundary.",
                        AUTO_RETENTION_BASE,
                        AUTO_RETENTION_INTERVAL,
                        AUTO_RETENTION_INTERVAL,
                    )
                elif self.local_kv_retention_interval == 0:
                    logger.info(
                        "Using prefix-cache local KV retention strategy: retain only "
                        "the latest replayable prompt boundary."
                    )
                else:
                    logger.info(
                        "Using prefix-cache local KV retention strategy: retain "
                        "sliding-window checkpoint tails at the configured "
                        "%d-token interval after prefix-cache alignment, plus "
                        "the latest replayable prompt boundary.",
                        self.local_kv_retention_interval,
                    )

        self.use_eagle = use_eagle

    def _init_prefix_cache_retention_metadata(self) -> None:
        for manager in self.single_type_managers:
            manager._prefix_cache_alignment_tokens = self.lcm_block_size
            manager._prefix_cache_use_eagle = False
        for idx, (_, group_ids, _) in enumerate(self.attention_groups):
            use_eagle = idx in self.eagle_attn_group_indices
            for group_id in group_ids:
                self.single_type_managers[group_id]._prefix_cache_use_eagle = use_eagle

    def cache_blocks(self, request, num_computed_tokens: int) -> None:
        for manager in self.single_type_managers:
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.local_kv_retention_interval,
            )

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, "HybridKVCacheCoordinator requires at least two attention groups."

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # Attention-group indices (into ``self.attention_groups``) that
        # contain at least one EAGLE/MTP KV cache group.
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }
        # vllm PR #43447: per-group ids of EAGLE/MTP groups, consumed by the
        # inherited ``cache_blocks`` to decide whether a sliding-window manager
        # needs to retain one extra local block past the replay boundary.
        self.eagle_lookup_group_ids: set[int] = {
            gid
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if i in self.eagle_attn_group_indices
            for gid in group_ids
        }

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        # NOTE: use 16k as the alignment tokens for model with compress ratio
        block_sizes = [spec.block_size * getattr(spec, "compress_ratio", 1) for spec, _, _ in self.attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, kv_cache_spec.block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = curr_hit_length // spec.block_size * spec.block_size
                    continue

                use_eagle = idx in self.eagle_attn_group_indices and idx not in eagle_verified

                _max_length = curr_hit_length
                if use_eagle:
                    # Eagle needs to match one more block and then pop the last.
                    _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                _new_hit_length = len(hit_blocks[0]) * spec.block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                compress_ratio = getattr(spec, "compress_ratio", 1)
                curr_hit_length = len(hit_blocks[0]) * spec.block_size * max(compress_ratio, 1)
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // spec.block_size
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
    # vllm PR #43447: KVCacheManager passes this through from
    # ``CacheConfig.prefix_cache_retention_interval``. Default keeps the
    # pre-PR behavior, and the kwarg keeps us compatible with older vllm
    # versions that don't forward it.
    local_kv_retention_interval: int | Literal["auto"] | None = None,
) -> KVCacheCoordinator:
    return AscendHybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_num_batched_tokens=max_num_batched_tokens,
        local_kv_retention_interval=local_kv_retention_interval,
    )


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]

# `kv_cache_manager` imports `get_kv_cache_coordinator` with
# `from ... import ...`, so if it was loaded before this patch runs
# (for example through the recompute scheduler path), it keeps the
# old function object. Update that cached binding as well.
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
