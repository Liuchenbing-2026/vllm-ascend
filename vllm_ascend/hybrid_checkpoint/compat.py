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
