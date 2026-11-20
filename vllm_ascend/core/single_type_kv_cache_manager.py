# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend-specific single-type KV cache managers."""

from collections.abc import Sequence
from typing import Any

from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, BlockHashListWithBlockSize, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec


def _resolve_block_hashes_compat(
    block_hashes: BlockHashList,
    hash_block_size: int,
    block_size: int,
    *,
    supports_fine_grained_hash_lookup: bool,
    alignment_tokens: int,
) -> BlockHashList:
    """Resolve block hashes across pre- and post-resolve_block_hashes vLLM pins."""
    resolve_fn = getattr(kv_cache_utils, "resolve_block_hashes", None)
    if callable(resolve_fn):
        return resolve_fn(
            block_hashes,
            hash_block_size,
            block_size,
            supports_fine_grained_hash_lookup=supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
    if block_size == hash_block_size or isinstance(block_hashes, BlockHashListWithBlockSize):
        return block_hashes
    assert block_size % hash_block_size == 0
    return BlockHashListWithBlockSize(block_hashes, hash_block_size, block_size)


def _extract_swa_reach(spec_or_config: Any) -> tuple[int, int]:
    """Return ``(num_retained_tokens, block_size_or_zero)`` across spec/config APIs."""
    params = getattr(spec_or_config, "params", None)
    if params is not None:
        sliding_window = params.sliding_window
        extra_retained = getattr(params, "extra_retained_tokens", 0)
        return sliding_window - 1 + extra_retained, 0
    sliding_window = spec_or_config.sliding_window
    extra_retained = getattr(spec_or_config, "extra_retained_tokens", 0)
    return sliding_window - 1 + extra_retained, getattr(spec_or_config, "block_size", 0)


class AscendDFlashSlidingWindowManager(SlidingWindowManager):
    """Tail-only sliding-window KV cache manager for DFlash/DSpark drafters.

    Draft attention only reads keys projected below the latest target tokens.
    Allocations keep leading blocks below ``num_tokens_main_model - tail_tokens``
    as null blocks so long prefill chunks and P/D loads back only each tail.
    """

    def __init__(
        self,
        kv_cache_spec: Any,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            block_pool=block_pool,
            enable_caching=enable_caching,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            needs_kv_cache_zeroing=needs_kv_cache_zeroing,
            max_admission_blocks_per_request=max_admission_blocks_per_request,
            **kwargs,
        )
        manager_cfg = getattr(self, "manager_config", None)
        params = getattr(manager_cfg, "params", None) if manager_cfg is not None else None
        if params is not None and hasattr(params, "tail_tokens"):
            self.tail_tokens = int(params.tail_tokens)
        elif hasattr(kv_cache_spec, "tail_tokens"):
            self.tail_tokens = int(kv_cache_spec.tail_tokens)
        else:
            num_spec = getattr(kv_cache_spec, "num_speculative_tokens", 0)
            max_steps = getattr(kv_cache_spec, "max_in_flight_steps", 1)
            self.tail_tokens = int(self.sliding_window + num_spec * max_steps)
        if not hasattr(self, "extra_retained_tokens"):
            self.extra_retained_tokens = getattr(kv_cache_spec, "extra_retained_tokens", 0)
        self._num_null_prefix_blocks: dict[str, int] = {}

    def _num_unbacked_leading_blocks(self, num_tokens_main_model: int) -> int:
        return max(num_tokens_main_model - self.tail_tokens, 0) // self.block_size

    def get_num_blocks_to_reserve(self, request_id: str, num_tokens: int) -> int | None:
        """Bound admission reservation by counting only live non-null blocks."""
        if self.max_admission_blocks_per_request is None:
            return None
        blocks = self.req_to_blocks.get(request_id, ())
        start = self._num_null_prefix_blocks.get(request_id, 0)
        # Chunks reserve non-null tails above earlier null gaps, so scan past start.
        num_held_blocks = sum(not blocks[i].is_null for i in range(start, len(blocks)))
        num_required_blocks = min(
            cdiv(num_tokens, self.block_size),
            self.max_admission_blocks_per_request,
        )
        return max(num_required_blocks - num_held_blocks, 0)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        del prefill_end
        if apply_admission_cap and self.max_admission_blocks_per_request is not None:
            reservation = self.get_num_blocks_to_reserve(request_id, num_tokens)
            assert reservation is not None
            return reservation

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
        num_unbacked_blocks = self._num_unbacked_leading_blocks(num_tokens_main_model)
        if request_id in self.num_cached_block:
            assert len(new_computed_blocks) == 0
            return max(num_required_blocks - max(num_req_blocks, num_unbacked_blocks), 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        num_skipped_blocks = num_skipped_tokens // self.block_size
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks, num_unbacked_blocks),
            0,
        )
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:],
        )
        has_partial_hit = getattr(self, "_has_partial_local_hit", None)
        if callable(has_partial_hit) and has_partial_hit(new_computed_blocks, num_local_computed_tokens):
            num_new_blocks += 1
        return num_new_blocks + num_evictable_blocks

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
        num_tokens_main_model: int | None = None,
    ) -> None:
        num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        req_blocks = self.req_to_blocks[request_id]
        num_computed_blocks = cdiv(num_total_computed_tokens, self.block_size)
        # Pre-#64109 coordinators omit num_tokens_main_model; fall back to computed span.
        effective_main_tokens = (
            num_total_computed_tokens if num_tokens_main_model is None else num_tokens_main_model
        )
        num_null_blocks = min(
            self._num_unbacked_leading_blocks(effective_main_tokens),
            num_computed_blocks,
        ) - len(req_blocks)
        req_blocks.extend([self._null_block] * max(num_null_blocks, 0))
        num_new_blocks = max(0, num_computed_blocks - len(req_blocks))
        allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(allocated_blocks)
        if getattr(self, "_record_new_block_ids", False) and record_for_zeroing:
            self.new_block_ids.extend(block.block_id for block in allocated_blocks)

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        cow_blocks: list[KVCacheBlock] = []
        partial_hit_reqs = getattr(self, "_partial_hit_reqs", None)
        if partial_hit_reqs is not None and request_id in partial_hit_reqs:
            block_idx, source_block = partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)

        req_blocks = self.req_to_blocks[request_id]
        null_blocks = [self._null_block] * max(
            self._num_unbacked_leading_blocks(num_tokens_main_model) - len(req_blocks),
            0,
        )
        req_blocks.extend(null_blocks)
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return null_blocks + cow_blocks
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        if getattr(self, "_record_new_block_ids", False):
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return null_blocks + cow_blocks + new_blocks

    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        assert first_block == 0
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None:
            return
        last_block = min(last_block, len(blocks))
        start = self._num_null_prefix_blocks.get(request_id, 0)
        freed: list[KVCacheBlock] = []
        # Earlier chunks' tails sit above null gaps, so scan through nulls down to start.
        for i in range(last_block - 1, start - 1, -1):
            if not blocks[i].is_null:
                freed.append(blocks[i])
                blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)
        self._num_null_prefix_blocks[request_id] = max(start, last_block)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._num_null_prefix_blocks.pop(request_id, None)
        return [block for block in super().pop_blocks_for_free(request_id) if not block.is_null]

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: Any,
        drop_eagle_block: Any,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        num_request_tokens: int | None = None,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Find longest hit whose blocks in ``[L - num_retained_tokens, hit)`` exist."""
        assert dcp_world_size == 1 and pcp_world_size == 1
        # Tolerate both pre-#61452 (kv_cache_spec, drop_eagle_block) and post-#61452 args.
        if isinstance(kv_cache_spec, int):
            block_size = kv_cache_spec
            num_retained_tokens, _ = _extract_swa_reach(drop_eagle_block)
        else:
            assert isinstance(kv_cache_spec, SlidingWindowSpec), (
                "AscendDFlashSlidingWindowManager can only be used for sliding window groups"
            )
            num_retained_tokens, block_size = _extract_swa_reach(kv_cache_spec)
        alignment_tokens = alignment_tokens or block_size
        assert alignment_tokens % block_size == 0

        block_hashes = _resolve_block_hashes_compat(
            block_hashes,
            block_pool.hash_block_size,
            block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        if num_request_tokens is None:
            num_request_tokens = max_length
        max_num_blocks = max_length // block_size
        first_block = max(num_request_tokens - num_retained_tokens, 0) // block_size
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * max_num_blocks for _ in kv_cache_group_ids
        )
        num_blocks = min(first_block, max_num_blocks)
        while num_blocks < max_num_blocks and (
            cached_block := block_pool.get_cached_block(block_hashes[num_blocks], kv_cache_group_ids)
        ):
            for computed, cached in zip(computed_blocks, cached_block):
                computed[num_blocks] = cached
            num_blocks += 1
        num_blocks = num_blocks * block_size // alignment_tokens * (alignment_tokens // block_size)
        for computed in computed_blocks:
            del computed[num_blocks:]
        return computed_blocks, num_blocks * block_size

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: Any,
        use_eagle: Any = False,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
        final_segment_end_block: int | None = None,
    ) -> list[bool] | None:
        del use_eagle, dcp_world_size, final_segment_end_block
        if retention_interval is None or alignment_tokens is None:
            return None
        if isinstance(kv_cache_spec, int):
            block_size = kv_cache_spec
            num_retained_tokens, _ = _extract_swa_reach(use_eagle)
        else:
            num_retained_tokens, block_size = _extract_swa_reach(kv_cache_spec)
        mask = [False] * (end_block - start_block)
        for boundary_tokens in reachable_boundaries:
            hit_tokens = boundary_tokens // alignment_tokens * alignment_tokens
            first_read = max(boundary_tokens + 1 - num_retained_tokens, 0)
            for j in range(
                max(start_block, first_read // block_size),
                min(end_block, hit_tokens // block_size),
            ):
                mask[j - start_block] = True
        return mask


class AscendCircularBufferManager(CircularBufferManager):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size,
            pcp_world_size,
            needs_kv_cache_zeroing,
            max_admission_blocks_per_request,
        )
        # Ascend zeroes every physical backing of a recycled block ID. Ring
        # allocations must report IDs too, even when the last owner was an
        # attention group. The upstream ring manager excludes these IDs.
        self._record_new_block_ids = needs_kv_cache_zeroing
