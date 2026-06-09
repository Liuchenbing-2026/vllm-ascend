# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


class ComputedBlockList(list[KVCacheBlock]):
    """KV blocks plus the logical token length they cover."""

    def __init__(
        self,
        blocks: Sequence[KVCacheBlock] = (),
        logical_hit_length: int | None = None,
    ) -> None:
        super().__init__(blocks)
        self.logical_hit_length = logical_hit_length


_PARTIAL_BLOCK_HASH_TO_BLOCK: dict[BlockHashWithGroupId, KVCacheBlock] = {}
_PARTIAL_BLOCK_ID_TO_HASHES: defaultdict[int, set[BlockHashWithGroupId]] = defaultdict(set)


def _hash_range(
    block_hashes: BlockHashList,
    hash_block_size: int,
    start_token: int,
    end_token: int,
) -> BlockHash | None:
    if end_token <= start_token:
        return None
    if start_token % hash_block_size != 0 or end_token % hash_block_size != 0:
        return None
    start = start_token // hash_block_size
    end = end_token // hash_block_size
    if end > len(block_hashes):
        return None
    return BlockHash(b"".join(block_hashes[start:end]))


def _insert_partial_cache(
    block_hash: BlockHash,
    kv_cache_group_id: int,
    block: KVCacheBlock,
) -> None:
    key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
    old_block = _PARTIAL_BLOCK_HASH_TO_BLOCK.get(key)
    if old_block is not None and old_block.block_id != block.block_id:
        _PARTIAL_BLOCK_ID_TO_HASHES[old_block.block_id].discard(key)
    _PARTIAL_BLOCK_HASH_TO_BLOCK[key] = block
    _PARTIAL_BLOCK_ID_TO_HASHES[block.block_id].add(key)


def get_partial_cached_block(block_hash: BlockHash, kv_cache_group_id: int) -> KVCacheBlock | None:
    key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
    block = _PARTIAL_BLOCK_HASH_TO_BLOCK.get(key)
    if block is None:
        return None
    return block


def remove_partial_cache_entries_for_block(block_id: int) -> None:
    keys = _PARTIAL_BLOCK_ID_TO_HASHES.pop(block_id, set())
    for key in keys:
        block = _PARTIAL_BLOCK_HASH_TO_BLOCK.get(key)
        if block is not None and block.block_id == block_id:
            _PARTIAL_BLOCK_HASH_TO_BLOCK.pop(key, None)


class CompressAttentionManager(FullAttentionManager):
    def __init__(self, kv_cache_spec: MLAAttentionSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block
        self.copy_block_ids: list[tuple[int, int, int]] = []
        self._copy_src_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

    def _num_partial_hit_blocks(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
    ) -> int:
        compressed_tokens = total_computed_tokens // self.compress_ratio
        num_full_hit_blocks = compressed_tokens // self.block_size
        return max(0, len(new_computed_blocks) - num_full_hit_blocks)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        # assert isinstance(self.kv_cache_spec, (CompressAttentionSpec, C4IndexerSpec))

        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio
        total_computed_tokens //= self.compress_ratio

        num_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        # Partial compressed hits are copied into a private destination block
        # before the request resumes. The source block may also need to be
        # pinned if it is an eviction candidate, which super() already counts.
        return num_blocks + self._num_partial_hit_blocks(
            new_computed_blocks,
            total_computed_tokens * self.compress_ratio,
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.
        3. (Optional) For KV connectors, allocate new blocks for external computed
            tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # It should not have any new computed blocks.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_logical_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_total_computed_tokens = num_total_logical_tokens // self.compress_ratio
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        num_full_hit_blocks = num_total_computed_tokens // self.block_size
        if len(new_computed_blocks) > num_full_hit_blocks:
            full_blocks = list(new_computed_blocks[:num_full_hit_blocks])
            partial_src_blocks = list(new_computed_blocks[num_full_hit_blocks:])
            if self.enable_caching:
                self.block_pool.touch(partial_src_blocks)
                self._copy_src_blocks[request_id].extend(partial_src_blocks)
            partial_dst_blocks = self.block_pool.get_new_blocks(len(partial_src_blocks))
            self.copy_block_ids.extend(
                (self.kv_cache_group_id, src.block_id, dst.block_id)
                for src, dst in zip(partial_src_blocks, partial_dst_blocks)
            )
            new_computed_blocks = [*full_blocks, *partial_dst_blocks]

        # Touch the computed full blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks[:num_full_hit_blocks])
        else:
            assert not any(new_computed_blocks), "Computed blocks should be empty when prefix caching is disabled"

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = min(len(req_blocks), num_full_hit_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        num_tokens //= self.compress_ratio
        ## TODO: check spec decode
        num_tokens_main_model //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        compressed_tokens = num_tokens // self.compress_ratio
        num_full_blocks = compressed_tokens // self.block_size
        logical_block_size = self.block_size * self.compress_ratio

        if num_cached_blocks < num_full_blocks:
            self.block_pool.cache_full_blocks(
                request=request,
                blocks=self.req_to_blocks[request.request_id],
                num_cached_blocks=num_cached_blocks,
                num_full_blocks=num_full_blocks,
                block_size=logical_block_size,
                kv_cache_group_id=self.kv_cache_group_id,
            )
            self.num_cached_block[request.request_id] = num_full_blocks

        self._cache_partial_block_boundaries(request, compressed_tokens)

    def _cache_partial_block_boundaries(self, request: Request, compressed_tokens: int) -> None:
        req_blocks = self.req_to_blocks[request.request_id]
        if compressed_tokens <= 0 or not req_blocks:
            return

        hash_block_size = self.block_pool.hash_block_size
        logical_block_size = self.block_size * self.compress_ratio
        max_logical_tokens = compressed_tokens * self.compress_ratio
        num_blocks_with_tokens = min(cdiv(compressed_tokens, self.block_size), len(req_blocks))

        for block_idx in range(num_blocks_with_tokens):
            block = req_blocks[block_idx]
            if block.is_null:
                continue
            block_start = block_idx * logical_block_size
            block_end = min(block_start + logical_block_size, max_logical_tokens)
            boundary = block_start + hash_block_size
            while boundary <= block_end:
                # Full-block boundaries are already represented in the normal
                # prefix cache hash table.
                if boundary - block_start != logical_block_size:
                    block_hash = _hash_range(
                        request.block_hashes,
                        hash_block_size,
                        block_start,
                        boundary,
                    )
                    if block_hash is not None:
                        _insert_partial_cache(block_hash, self.kv_cache_group_id, block)
                boundary += hash_block_size

    def take_copy_block_ids(self) -> list[tuple[int, int, int]]:
        copy_block_ids = self.copy_block_ids
        self.copy_block_ids = []
        return copy_block_ids

    def free(self, request_id: str) -> None:
        pinned_src_blocks = self._copy_src_blocks.pop(request_id, [])
        super().free(request_id)
        if pinned_src_blocks:
            self.block_pool.free_blocks(reversed(pinned_src_blocks))

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        # assert isinstance(
        #     kv_cache_spec, Compress4AttentionSpec | Compress128AttentionSpec | C4IndexerSpec
        # ), (
        #     "CompressAttentionManager can only be used for compressor attention groups"
        # )
        computed_blocks: tuple[ComputedBlockList, ...] = tuple(
            ComputedBlockList() for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        compress_ratio = kv_cache_spec.compress_ratio
        logical_block_size = block_size * compress_ratio
        hash_block_size = block_pool.hash_block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
            logical_block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // logical_block_size
        full_block_hashes = (
            block_hashes
            if logical_block_size == hash_block_size
            else [_hash_range(block_hashes, hash_block_size, i * logical_block_size, (i + 1) * logical_block_size)
                  for i in range(max_num_blocks)]
        )
        for block_hash in itertools.islice(full_block_hashes, max_num_blocks):
            if block_hash is None:
                break
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()

        logical_hit_length = len(computed_blocks[0]) * logical_block_size
        partial_start = logical_hit_length
        candidate = min(max_length // alignment_tokens * alignment_tokens, partial_start + logical_block_size - alignment_tokens)
        while candidate > partial_start:
            partial_hash = _hash_range(block_hashes, hash_block_size, partial_start, candidate)
            if partial_hash is None:
                candidate -= alignment_tokens
                continue
            partial_blocks: list[KVCacheBlock] = []
            for group_id in kv_cache_group_ids:
                block = get_partial_cached_block(partial_hash, group_id)
                if block is None:
                    partial_blocks = []
                    break
                partial_blocks.append(block)
            if partial_blocks:
                for computed, cached in zip(computed_blocks, partial_blocks):
                    computed.append(cached)
                    computed.logical_hit_length = candidate
                logical_hit_length = candidate
                break
            candidate -= alignment_tokens

        while (
            logical_block_size != alignment_tokens  # Faster for common case.
            and logical_hit_length % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
                computed.logical_hit_length = len(computed) * logical_block_size
            logical_hit_length = len(computed_blocks[0]) * logical_block_size
        if logical_hit_length and computed_blocks[0].logical_hit_length is None:
            for computed in computed_blocks:
                computed.logical_hit_length = logical_hit_length
        return computed_blocks


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """Build the per-spec KV cache manager.

    For DSv4 / DSA path (``MLAAttentionSpec`` with ``compress_ratio>1``), align
    the runtime admission gate with the startup pool-sizing bound the same way
    vLLM PR #40946 does for ``SlidingWindowSpec`` / ``ChunkedLocalAttentionSpec``.
    Without this cap, an admitted request can demand more blocks than the pool
    was sized to back, and ``allocate_slots`` silently returns ``None`` from
    the ``full_sequence_must_fit`` branch, leaving long-input requests stuck
    in the waiting queue (see vLLM issue #40863, observed on DSv4 + MTP with
    cc>=1 and prompt>=32K).

    The compressed-MLA peak per request is bounded by
    ``cdiv(max_model_len // compress_ratio, block_size)`` (it does not shrink
    via recycling like SWA, but neither does it ever exceed this). Capping at
    this value matches the pool sizer and makes admission consistent with the
    block budget actually held.
    """
    manager_class = spec_manager_map[type(kv_cache_spec)]
    if isinstance(kv_cache_spec, MLAAttentionSpec) and kv_cache_spec.compress_ratio > 1:
        manager_class = CompressAttentionManager
        if max_model_len is not None:
            # Compressed-MLA peak in blocks: ceil(max_model_len/compress/block).
            compress_ratio = kv_cache_spec.compress_ratio
            block_size = kv_cache_spec.block_size
            max_compressed_tokens = max_model_len // compress_ratio
            kwargs["max_admission_blocks_per_request"] = cdiv(max_compressed_tokens, block_size) + 1
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        # Replicate the upstream PR #40946 cap setting for recycling specs.
        # We override the vLLM factory above, so the upstream block that does
        # this lives in dead code (never reached); without re-applying it here
        # SlidingWindowMLASpec / ChunkedLocalAttentionSpec groups have no cap
        # and ``full_sequence_must_fit`` admission reserves the full
        # ``max_model_len`` worth of blocks per request, exhausting the pool
        # at cc>=2 on DSv4 (see vLLM issue #40863).
        if max_num_batched_tokens is not None and max_model_len is not None:
            kwargs["max_admission_blocks_per_request"] = kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
