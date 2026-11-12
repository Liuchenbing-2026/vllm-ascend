# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for vLLM versions that do not expose these symbols.

Only used by the optional speculative/export paths, which are disabled in
this Ascend port; they keep the frontier manager importable on vLLM 0.30.
"""
from typing import NamedTuple


class MambaStateGather(NamedTuple):
    column: int
    group_ids: tuple[int, ...]
    dst_block_ids: tuple[int, ...]
    src_block_ids: tuple[int, ...]


class RecoveryUnitStored:
    def __init__(self, *args, **kwargs):
        pass


class RecoveryUnitRemoved:
    def __init__(self, *args, **kwargs):
        pass


def install_block_hash_map_compat() -> None:
    """Add ``BlockHashToBlockMap.get_all_blocks`` on vLLM 0.30.

    The frontier manager asks the prefix-cache map for every block that a hash
    maps to (a hash may be shared by several blocks).  That helper only exists
    in later vLLM releases, so on 0.30 every publish raised AttributeError and
    was dropped.  The 0.30 map uses the same ``_cache`` layout, so the shim is
    a straight re-implementation.
    """
    from vllm.v1.core.block_pool import BlockHashToBlockMap

    if hasattr(BlockHashToBlockMap, "get_all_blocks"):
        return

    def get_all_blocks(self, key):
        blocks = self._cache.get(key)
        if blocks is None:
            return []
        if isinstance(blocks, dict):
            return list(blocks.values())
        return [blocks]

    BlockHashToBlockMap.get_all_blocks = get_all_blocks  # type: ignore[attr-defined]
