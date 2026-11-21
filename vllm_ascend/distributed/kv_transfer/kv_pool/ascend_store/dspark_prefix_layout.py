# SPDX-License-Identifier: Apache-2.0
"""DSpark prefix ownership, available to the planner without store imports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_group_layer_specs

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
else:
    KVCacheGroupSpec = Any


def draft_only_group_ids(groups: Sequence[KVCacheGroupSpec | Any]) -> tuple[int, ...]:
    """Find groups whose prefix objects live entirely in the draft companion.

    K3.1 mixes target MLA/indexer and draft layers, so its groups still need
    ordinary target hit keys. Other layouts can have separate draft groups,
    for which no ordinary target object is saved. ``is_eagle_group`` marks
    groups containing any draft layer and cannot distinguish these cases.

    Run before scheduler conversion selects one representative layer spec.
    Inspect global specs even when this PP rank has no local layer names.
    """
    result = []
    for group_id, group in enumerate(groups):
        specs = getattr(group, "distinct_specs", None) or get_group_layer_specs(group)
        if specs and all(getattr(layer_spec, "is_draft", False) for layer_spec in specs):
            result.append(group_id)
    return tuple(result)
