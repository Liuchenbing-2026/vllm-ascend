from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from functools import lru_cache, partial
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast
from weakref import proxy

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_bytes,
    get_block_hashes,
    get_group_layer_specs,
    get_reusable_prefix_length,
    hashed_prefix_length,
    infer_cacheable_group_ids,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
else:
    KVCacheGroupSpec = Any

_CACHE_MISSING = object()
_MANAGER_CLASS_CACHE_ATTR = "_manager_class_cache"
_REACHABLE_MASK_CACHE_SIZE = 32

# kwargs every reachable_block_mask implementation must accept. Used when the
# manager's signature cannot be introspected or accepts **kwargs.
_REACHABLE_MASK_BASE_KWARGS = frozenset(("start_block", "end_block", "alignment_tokens", "kv_cache_spec", "use_eagle"))
_REACHABLE_MASK_OPTIONAL_KWARGS = frozenset(("retention_interval", "num_prompt_tokens", "reachable_boundaries"))
_FIND_HIT_BASE_KWARGS = frozenset(
    (
        "block_hashes",
        "max_length",
        "kv_cache_group_ids",
        "block_pool",
        "kv_cache_spec",
        "drop_eagle_block",
        "alignment_tokens",
    )
)
# manager class -> accepted parameter names per static method.
_REACHABLE_MASK_KWARGS_CACHE: dict[type[SingleTypeKVCacheManager], frozenset[str]] = {}
_FIND_HIT_KWARGS_CACHE: dict[type[SingleTypeKVCacheManager], frozenset[str]] = {}

# Per-group pool query used by the shared reachable hit lookup: returns the
# subset of ``group_block_hashes`` present in the pool (all replicas valid).
# A None ``lookup_mask`` means the group has no reachability limits.
GroupHitQuery = Callable[[int, Sequence[BlockHash | str], Sequence[bool] | None], Iterable[BlockHash | str]]


class _HBMBlockHash(bytes):
    """A hash whose complete hash block is already computed locally."""


class _HBMPrefixBlockHashes(Sequence[BlockHash]):
    """Mark local hashes on access, without expanding per-group hit sets.

    Managers select each group's terminal hash without copying it. That hash
    is local exactly when the complete group block ends within the HBM prefix;
    an interior hash cannot make a larger, partially local block hit.
    """

    def __init__(self, block_hashes: Sequence[BlockHash], num_hbm_hashes: int) -> None:
        self._block_hashes = block_hashes
        self._num_hbm_hashes = num_hbm_hashes

    def __len__(self) -> int:
        return len(self._block_hashes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block_hash = self._block_hashes[index]
        if index < self._num_hbm_hashes:
            return cast(BlockHash, _HBMBlockHash(block_hash_to_bytes(block_hash)))
        return block_hash


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by external AscendStore key existence."""

    def __init__(
        self,
        hash_block_size: int,
        exists: set[tuple[int, bytes]] | None = None,
    ) -> None:
        # exists=None is used for load/store masks where hit length has already
        # been decided and each manager only needs to apply its own reachability.
        self._exists = exists
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        if self._exists is None or isinstance(block_hash, _HBMBlockHash):
            return [self._present_block] * len(group_ids)
        h = block_hash_to_bytes(block_hash)
        if all((group_id, h) in self._exists for group_id in group_ids):
            return [self._present_block] * len(group_ids)
        return None


class AscendStoreCoordinator:
    """Hybrid cache-hit/mask coordinator for AscendStore external KV Pool.

    This mirrors vLLM's external KV coordinator but uses AscendStore's external
    key granularity. Compressed specs already expose raw-token block sizes,
    while transfer addresses remain in cache-domain blocks.
    """

    def __init__(
        self,
        kv_cache_groups: Sequence[KVCacheGroupSpec | Any],
        scheduler_block_size: int,
        hash_block_size: int,
        group_block_sizes: list[int],
        group_cache_families: list[str],
        use_eagle: bool = False,
        retention_interval: int | None = None,
        num_hash_lookahead_tokens: int = 0,
    ) -> None:
        assert len(kv_cache_groups) == len(group_block_sizes)
        assert len(kv_cache_groups) == len(group_cache_families)
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of hash_block_size ({hash_block_size})"
        )

        self.kv_cache_groups = list(kv_cache_groups)
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.num_hash_lookahead_tokens = max(0, int(num_hash_lookahead_tokens))
        # Lookahead hashes already protect trailing drafter tokens; only enable
        # legacy EAGLE block dropping when lookahead hashing is disabled.
        self.use_eagle = bool(use_eagle) and self.num_hash_lookahead_tokens == 0
        self.retention_interval = retention_interval
        self.group_block_sizes = group_block_sizes
        self.group_cache_families = group_cache_families
        self.group_effective_block_sizes = list(group_block_sizes)
        self.cacheable_group_ids = infer_cacheable_group_ids(kv_cache_groups)
        for group_id in self.cacheable_group_ids:
            effective_block_size = self.group_effective_block_sizes[group_id]
            assert effective_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
            assert scheduler_block_size % effective_block_size == 0, (
                "scheduler_block_size must be a multiple of each group's effective block_size"
            )

        if self.num_hash_lookahead_tokens > 0:
            self.eagle_group_ids: set[int] = set()
        else:
            self.eagle_group_ids = {
                i for i, group in enumerate(kv_cache_groups) if getattr(group, "is_eagle_group", False)
            }
            if self.use_eagle and not self.eagle_group_ids:
                self.eagle_group_ids = set(range(len(kv_cache_groups)))

        self._verify_and_split_kv_cache_groups()
        # Do not keep the coordinator alive through its cache while GC is frozen.
        self._cached_reachable_masks = lru_cache(maxsize=_REACHABLE_MASK_CACHE_SIZE)(
            partial(type(self)._reachable_masks, proxy(self))
        )

    def _verify_and_split_kv_cache_groups(self) -> None:
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager], Any]] = []
        self.group_effective_specs: list[KVCacheSpec] = []
        self.group_manager_configs: list[Any] = []

        for group_id, group in enumerate(self.kv_cache_groups):
            spec = _group_spec(group)
            effective_spec = _copy_spec_with_block_size(spec, self.group_effective_block_sizes[group_id])
            manager_config = getattr(group, "manager_config", None)
            self.group_effective_specs.append(effective_spec)
            self.group_manager_configs.append(manager_config)
            if group_id not in self.cacheable_group_ids:
                continue
            manager_cls = _get_group_manager_class(group, spec)

            for existing_spec, group_ids, existing_cls, existing_cfg in attention_groups:
                if existing_spec == effective_spec and existing_cfg == manager_config:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(group_id)
                    break
            else:
                attention_groups.append((effective_spec, [group_id], manager_cls, manager_config))

        self.attention_groups = sorted(
            attention_groups,
            key=lambda item: not isinstance(item[0], FullAttentionSpec),
        )
        self.eagle_attn_group_indices: set[int] = {
            index
            for index, (_, group_ids, _, _) in enumerate(self.attention_groups)
            if any(group_id in self.eagle_group_ids for group_id in group_ids)
        }
        if self.use_eagle and not self.eagle_attn_group_indices:
            self.eagle_attn_group_indices = set(range(len(self.attention_groups)))
        self.eagle_reachable_group_ids: set[int] = {
            group_id for index in self.eagle_attn_group_indices for group_id in self.attention_groups[index][1]
        }

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[bool], ...], int]:
        effective_max_length = max_length
        if self.num_hash_lookahead_tokens > 0:
            effective_max_length = min(
                max_length,
                hashed_prefix_length(block_hashes, self.hash_block_size),
            )
        blocks_per_group, hit_length = self._find_hit_blocks(
            block_hashes,
            effective_max_length,
            cached_block_pool,
            apply_eagle=apply_eagle,
        )
        masks = tuple([block is not cached_block_pool.null_block for block in blocks] for blocks in blocks_per_group)
        return masks, hit_length

    def load_mask(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        masks, _ = self.find_longest_cache_hit(
            block_hashes,
            token_len,
            ExternalCachedBlockPool(self.hash_block_size),
            apply_eagle=False,
        )
        if self.num_hash_lookahead_tokens <= 0:
            return masks
        # Pad trailing unhashed blocks with False so out-of-range checks never
        # treat unhashed suffix blocks as allowed.
        padded: list[list[bool]] = []
        for group_id, mask in enumerate(masks):
            if group_id not in self.cacheable_group_ids:
                padded.append(list(mask))
                continue
            expected_blocks = token_len // self.group_effective_block_sizes[group_id]
            group_mask = list(mask)
            if len(group_mask) < expected_blocks:
                group_mask.extend([False] * (expected_blocks - len(group_mask)))
            padded.append(group_mask)
        return tuple(padded)

    def _reachable_masks(
        self,
        aligned_token_len: int,
        retention_interval: int | None,
        num_prompt_tokens: int | None,
        num_hash_lookahead_tokens: int = 0,
    ) -> list[tuple[int, list[bool] | None]]:
        assert aligned_token_len % self.lcm_block_size == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of lcm_block_size ({self.lcm_block_size})"
        )
        # Stores retain the request's replay checkpoint. Lookups must probe
        # every possible checkpoint: a shorter request may have published a
        # boundary that is not the current request's replay/segment boundary.
        if num_prompt_tokens is not None:
            if num_hash_lookahead_tokens > 0:
                resend_boundary = get_reusable_prefix_length(
                    num_prompt_tokens,
                    num_hash_lookahead_tokens=num_hash_lookahead_tokens,
                )
                reachable_boundaries: Sequence[int] = (resend_boundary, num_prompt_tokens)
            else:
                reachable_boundaries = (max(0, num_prompt_tokens - 1),)
        else:
            reachable_boundaries = range(self.lcm_block_size, aligned_token_len + 1, self.lcm_block_size)
        masks: list[tuple[int, list[bool] | None]] = []
        for group_id, spec in enumerate(self.group_effective_specs):
            if group_id not in self.cacheable_group_ids:
                masks.append((0, []))
                continue
            group_block_size = self.group_effective_block_sizes[group_id]
            num_chunks = aligned_token_len // group_block_size
            if not _uses_reachable_mask(self.group_cache_families[group_id]):
                masks.append((num_chunks, None))
                continue
            group = self.kv_cache_groups[group_id]
            manager_cls = _get_group_manager_class(group, _group_spec(group))
            mask = _reachable_block_mask(
                manager_cls,
                start_block=0,
                end_block=num_chunks,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                block_size=group_block_size,
                manager_config=self.group_manager_configs[group_id],
                use_eagle=group_id in self.eagle_reachable_group_ids,
                retention_interval=retention_interval,
                num_prompt_tokens=num_prompt_tokens,
                reachable_boundaries=reachable_boundaries,
            )
            masks.append((num_chunks, mask))
        return masks

    def _apply_hash_coverage_bound(
        self,
        masks: list[tuple[int, list[bool] | None]],
        num_block_hashes: int | None,
    ) -> list[tuple[int, list[bool] | None]]:
        if num_block_hashes is None:
            return masks
        hashed_tokens = num_block_hashes * self.hash_block_size
        bounded: list[tuple[int, list[bool] | None]] = []
        for group_id, (num_chunks, mask) in enumerate(masks):
            if group_id not in self.cacheable_group_ids:
                bounded.append((num_chunks, mask))
                continue
            hashed_blocks = hashed_tokens // self.group_effective_block_sizes[group_id]
            if hashed_blocks >= num_chunks:
                bounded.append((num_chunks, mask))
                continue
            group_mask = [True] * num_chunks if mask is None else mask.copy()
            for idx in range(max(0, hashed_blocks), num_chunks):
                group_mask[idx] = False
            bounded.append((num_chunks, group_mask))
        return bounded

    def store_mask(
        self,
        aligned_token_len: int,
        num_prompt_tokens: int | None = None,
        num_block_hashes: int | None = None,
    ) -> tuple[list[bool], ...]:
        masks = self._cached_reachable_masks(
            aligned_token_len,
            self.retention_interval,
            num_prompt_tokens,
            self.num_hash_lookahead_tokens,
        )
        effective_hash_bound = num_block_hashes if self.num_hash_lookahead_tokens > 0 else None
        bounded = self._apply_hash_coverage_bound(masks, effective_hash_bound)
        return tuple([True] * num_chunks if mask is None else mask.copy() for num_chunks, mask in bounded)

    def lookup_mask(
        self,
        aligned_token_len: int,
        num_block_hashes: int | None = None,
    ) -> tuple[list[bool] | None, ...]:
        # Probe candidate replay boundaries as well as segment boundaries;
        # find_longest_cache_hit still requires all necessary states to exist.
        masks = self._cached_reachable_masks(
            aligned_token_len,
            self.retention_interval,
            None,
            self.num_hash_lookahead_tokens,
        )
        effective_hash_bound = num_block_hashes if self.num_hash_lookahead_tokens > 0 else None
        bounded = self._apply_hash_coverage_bound(masks, effective_hash_bound)
        for num_chunks, mask in bounded:
            if mask is not None:
                assert len(mask) == num_chunks
        return tuple(None if mask is None or all(mask) else mask.copy() for _, mask in bounded)

    def find_reachable_hit_tokens(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
        query_group_hits: GroupHitQuery,
        *,
        log_context: str = "reachable_lookup",
        hbm_hit_tokens: int = 0,
    ) -> int:
        """Reachability-aware external hit lookup shared by both transfer paths.

        Queries, per group, only the blocks the group's lookup mask allows —
        the subset the save paths persist for reachability-limited groups —
        and derives the hit length via find_longest_cache_hit so sparsely
        stored pools still yield correct prefix hits. Both the scheduler-side
        layerwise lookup and the worker-side non-layerwise lookup delegate
        here so the hit semantics cannot drift between them.

        ``query_group_hits(group_id, group_block_hashes, lookup_mask)`` must
        return the subset of ``group_block_hashes`` present in the pool for
        that group, using whichever key layout and query backend the caller
        owns. When ``hbm_hit_tokens`` is provided, the callback only returns
        external hits; locally computed complete blocks are marked lazily for
        the existing managers, including each group's terminal-hash selection.
        """
        effective_token_len = token_len
        num_block_hashes: int | None = None
        if self.num_hash_lookahead_tokens > 0:
            effective_token_len = min(token_len, hashed_prefix_length(block_hashes, self.hash_block_size))
            if effective_token_len <= 0:
                logger.debug("%s: token_len=%d no lookahead-covered blocks found", log_context, token_len)
                return 0
            num_block_hashes = len(block_hashes)

        aligned_len = cdiv(effective_token_len, self.lcm_block_size) * self.lcm_block_size
        lookup_masks = self.lookup_mask(aligned_len, num_block_hashes=num_block_hashes)
        exists: set[tuple[int, bytes]] = set()
        block_hashes_to_check = block_hashes[: effective_token_len // self.hash_block_size]

        for group_id in self.cacheable_group_ids:
            group_block_size = self.group_effective_block_sizes[group_id]
            group_block_hashes = get_block_hashes(block_hashes_to_check, group_block_size, self.hash_block_size)
            hits = query_group_hits(group_id, group_block_hashes, lookup_masks[group_id])
            exists.update((group_id, block_hash_to_bytes(hit)) for hit in hits)

        if not exists and hbm_hit_tokens <= 0:
            logger.debug("%s: token_len=%d no pooled blocks found", log_context, effective_token_len)
            return 0
        hit_hashes: Sequence[BlockHash] = block_hashes
        if hbm_hit_tokens > 0:
            num_hbm_hashes = min(hbm_hit_tokens // self.hash_block_size, effective_token_len // self.hash_block_size)
            hit_hashes = _HBMPrefixBlockHashes(block_hashes, num_hbm_hashes)
        _, hit_length = self.find_longest_cache_hit(
            hit_hashes,
            effective_token_len,
            ExternalCachedBlockPool(self.hash_block_size, exists),
            apply_eagle=False,
        )
        logger.debug("%s: token_len=%d hit_tokens=%d", log_context, effective_token_len, hit_length)
        return hit_length

    def block_hashes_for_spec(self, block_hashes: Sequence[BlockHash], spec: KVCacheSpec) -> BlockHashList:
        # Managers accept lazy sequences and preserve their terminal hashes
        # when resolving each group's block-size view.
        return cast(BlockHashList, block_hashes)

    def _find_hit_blocks(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        eagle_indices = self.eagle_attn_group_indices if apply_eagle else set()
        if len(self.attention_groups) == 1:
            spec, group_ids, manager_cls, manager_config = self.attention_groups[0]
            hashes = self.block_hashes_for_spec(block_hashes, spec)
            hit_blocks, hit_length = _find_longest_cache_hit(
                manager_cls,
                block_hashes=hashes,
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                block_size=spec.block_size,
                manager_config=manager_config,
                drop_eagle_block=0 in eagle_indices,
                alignment_tokens=spec.block_size,
            )
            blocks_by_group: list[list[KVCacheBlock]] = [[] for _ in range(len(self.kv_cache_groups))]
            for group_id, blocks in zip(group_ids, hit_blocks, strict=True):
                blocks_by_group[group_id] = blocks
            return tuple(blocks_by_group), hit_length

        hit_length = max_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * len(self.kv_cache_groups)
        hit_length_by_group: list[int] = [0] * len(self.kv_cache_groups)
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for index, (spec, group_ids, manager_cls, manager_config) in enumerate(self.attention_groups):
                first_group_id = group_ids[0]
                cached = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    curr_hit_length = min(curr_hit_length, hit_length_by_group[first_group_id])
                    continue

                drop_eagle_block = index in eagle_indices and index not in eagle_verified
                max_group_length = curr_hit_length
                if drop_eagle_block:
                    max_group_length = min(curr_hit_length + spec.block_size, max_length)
                hashes = self.block_hashes_for_spec(block_hashes, spec)
                hit_blocks, new_hit_length = _find_longest_cache_hit(
                    manager_cls,
                    block_hashes=hashes,
                    max_length=max_group_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    block_size=spec.block_size,
                    manager_config=manager_config,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self.lcm_block_size,
                )
                if drop_eagle_block:
                    eagle_verified.add(index)
                elif new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks, strict=True):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = new_hit_length

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        spec0, group_ids0, _, _ = self.attention_groups[0]
        if isinstance(spec0, FullAttentionSpec):
            num_blocks = cdiv(hit_length, spec0.block_size)
            for group_id in group_ids0:
                full_blocks = hit_blocks_by_group[group_id]
                assert full_blocks is not None
                del full_blocks[num_blocks:]
                hit_length_by_group[group_id] = hit_length

        return (
            tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group),
            hit_length,
        )


def _group_spec(group: Any) -> KVCacheSpec:
    specs = get_group_layer_specs(group)
    assert specs, f"KV cache group {group!r} has no KVCacheSpec entries"
    return cast(KVCacheSpec, specs[0])


def _get_group_manager_class(group: Any, spec: KVCacheSpec) -> type[SingleTypeKVCacheManager]:
    manager_config = getattr(group, "manager_config", None)
    manager_cls = getattr(manager_config, "manager_cls", None)
    if inspect.isclass(manager_cls):
        return cast(type[SingleTypeKVCacheManager], manager_cls)
    return _get_manager_class(spec)


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def _copy_spec_with_block_size(spec: KVCacheSpec, block_size: int) -> KVCacheSpec:
    if spec.block_size == block_size:
        return spec
    copy_with_new_block_size = getattr(spec, "copy_with_new_block_size", None)
    if copy_with_new_block_size is not None:
        return copy_with_new_block_size(block_size)
    return replace(spec, block_size=block_size)


def _get_manager_class_cache() -> dict[str, Any]:
    cache = getattr(_get_manager_class, _MANAGER_CLASS_CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(_get_manager_class, _MANAGER_CLASS_CACHE_ATTR, cache)
    return cast(dict[str, Any], cache)


def _get_manager_class(spec: KVCacheSpec) -> type[SingleTypeKVCacheManager]:
    cache = _get_manager_class_cache()
    registry = cache.get("registry", _CACHE_MISSING)
    if registry is _CACHE_MISSING:
        try:
            registry_module = import_module("vllm.v1.kv_cache_spec_registry")
            registry = getattr(registry_module, "KVCacheSpecRegistry", None)
        except ImportError:
            registry = None
        cache["registry"] = registry

    if registry is not None:
        manager_cls = registry.get_manager_class(spec)
        if manager_cls is not None:
            return manager_cls

    spec_manager_map = cache.get("spec_manager_map", _CACHE_MISSING)
    if spec_manager_map is _CACHE_MISSING:
        try:
            manager_module = import_module("vllm.v1.core.single_type_kv_cache_manager")
            spec_manager_map = vars(manager_module)["spec_manager_map"]
        except Exception as exc:
            raise AssertionError(f"No manager registered for KVCacheSpec {type(spec)}") from exc
        cache["spec_manager_map"] = spec_manager_map

    try:
        manager_cls = spec_manager_map[type(spec)]
    except Exception as exc:
        raise AssertionError(f"No manager registered for KVCacheSpec {type(spec)}") from exc
    return manager_cls


def _accepted_manager_kwargs(
    cache: dict[type[SingleTypeKVCacheManager], frozenset[str]],
    manager_cls: type[SingleTypeKVCacheManager],
    method: Any,
    fallback_kwargs: frozenset[str],
) -> frozenset[str]:
    """Return parameter names accepted by a ``SingleTypeKVCacheManager`` classmethod."""
    cached = cache.get(manager_cls, _CACHE_MISSING)
    if cached is not _CACHE_MISSING:
        return cast(frozenset[str], cached)

    accepted = fallback_kwargs
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        pass
    else:
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            accepted = fallback_kwargs
        else:
            accepted = frozenset(parameters)
    cache[manager_cls] = accepted
    return accepted


def _filter_manager_kwargs(
    manager_cls: type[SingleTypeKVCacheManager],
    method_name: str,
    cache: dict[type[SingleTypeKVCacheManager], frozenset[str]],
    method: Any,
    fallback_kwargs: frozenset[str],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    accepted = _accepted_manager_kwargs(cache, manager_cls, method, fallback_kwargs)
    unsupported = set(kwargs) - accepted
    if unsupported:
        logger.debug(
            "KV cache manager %s does not accept %s kwargs %s; ignoring them.",
            manager_cls.__name__,
            method_name,
            sorted(unsupported),
        )
    return {name: value for name, value in kwargs.items() if name in accepted}


def _find_longest_cache_hit(
    manager_cls: type[SingleTypeKVCacheManager],
    **kwargs: Any,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    find_longest = manager_cls.find_longest_cache_hit
    filtered = _filter_manager_kwargs(
        manager_cls,
        "find_longest_cache_hit",
        _FIND_HIT_KWARGS_CACHE,
        find_longest,
        _FIND_HIT_BASE_KWARGS,
        kwargs,
    )
    hit_result = find_longest(**filtered)
    return cast(tuple[tuple[list[KVCacheBlock], ...], int], hit_result)


def _reachable_mask_accepted_kwargs(
    manager_cls: type[SingleTypeKVCacheManager],
    reachable_block_mask: Any,
) -> frozenset[str]:
    """Parameter names a manager's ``reachable_block_mask`` accepts.

    Falls back to the base contract when introspection is unavailable or the
    implementation swallows everything via ``**kwargs``.
    """
    return _accepted_manager_kwargs(
        _REACHABLE_MASK_KWARGS_CACHE,
        manager_cls,
        reachable_block_mask,
        _REACHABLE_MASK_BASE_KWARGS | _REACHABLE_MASK_OPTIONAL_KWARGS,
    )


def _reachable_block_mask(
    manager_cls: type[SingleTypeKVCacheManager],
    **kwargs: Any,
) -> list[bool] | None:
    reachable_block_mask = getattr(manager_cls, "reachable_block_mask", None)
    if reachable_block_mask is None:
        return None
    # Filter by signature instead of sniffing TypeError text: the old fallback
    # keyed on a keyword no manager accepts and therefore dropped
    # retention_interval on every call, silently degrading every sparse-retention
    # mask to the dense default.
    filtered = _filter_manager_kwargs(
        manager_cls,
        "reachable_block_mask",
        _REACHABLE_MASK_KWARGS_CACHE,
        reachable_block_mask,
        _REACHABLE_MASK_BASE_KWARGS | _REACHABLE_MASK_OPTIONAL_KWARGS,
        kwargs,
    )
    return reachable_block_mask(**filtered)


def _uses_reachable_mask(cache_family: str | None) -> bool:
    return cache_family in (None, "default", "c1")
