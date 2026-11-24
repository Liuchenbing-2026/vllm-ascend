# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Allow mixed page sizes in layer-compact KV cache layouts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import vllm.v1.core.kv_cache_planning as kv_cache_planning
from vllm.utils.math_utils import round_up


def group_layers_by_spec(group: Any) -> dict[Any, list[str]]:
    """Group layer names by spec in first-appearance placement order."""
    layers_by_spec: defaultdict[Any, list[str]] = defaultdict(list)
    per_layer_specs = getattr(group, "kv_cache_specs", None)
    if isinstance(per_layer_specs, Mapping):
        for layer_name, spec in per_layer_specs.items():
            layers_by_spec[spec].append(layer_name)
        return dict(layers_by_spec)

    group_spec = group.kv_cache_spec
    nested_specs = getattr(group_spec, "kv_cache_specs", None)
    layer_names = getattr(group, "layer_names", None)
    if layer_names is None:
        layer_names = list(nested_specs) if nested_specs is not None else []
    for layer_name in layer_names:
        spec = nested_specs[layer_name] if nested_specs is not None else group_spec
        layers_by_spec[spec].append(layer_name)
    return dict(layers_by_spec)


def get_layer_block_stride(spec: Any) -> int:
    """Return one layer's block stride in a layer-outer layout."""
    alignment_fn = getattr(spec, "get_block_stride_alignment", None)
    alignment = alignment_fn() if callable(alignment_fn) else 1
    return round_up(spec.page_size_bytes, alignment)


def overlaid_layers_match(kv_cache_groups: Sequence[Any]) -> bool:
    """Whether groups can overlay each other in a layer-compact layout."""
    strides_per_group = [
        [get_layer_block_stride(spec) for spec, layer_names in group_layers_by_spec(group).items() for _ in layer_names]
        for group in kv_cache_groups
    ]
    longest = max(strides_per_group, key=len, default=[])
    return all(strides == longest[: len(strides)] for strides in strides_per_group)


def validate_kv_cache_layout_for_mixed_pages(
    layout: Any,
    kv_cache_groups: Sequence[Any],
) -> None:
    """Validate that ``layout`` can express mixed page sizes."""
    page_sizes = {spec.page_size_bytes for group in kv_cache_groups for spec in group_layers_by_spec(group)}
    if len(page_sizes) <= 1:
        # Empty or uniform page sizes always have a rectangular layer dim.
        return

    if not layout.is_block_compact or (layout.is_layer_compact and not overlaid_layers_match(kv_cache_groups)):
        raise ValueError(
            f"KV cache layout {layout.name} cannot express this model's "
            f"mixed page sizes ({sorted(page_sizes)}); a backend should "
            "declare block-outermost supported layouts (e.g. BLHNC), or "
            "set VLLM_KV_CACHE_LAYOUT=BLHNC."
        )


def apply_kv_cache_layout_patch() -> bool:
    """Install the mixed-page layer-compact validation patch if needed."""
    if hasattr(kv_cache_planning, "_overlaid_layers_match"):
        return False
    kv_cache_planning._group_layers_by_spec = group_layers_by_spec  # type: ignore[attr-defined]
    kv_cache_planning._overlaid_layers_match = overlaid_layers_match  # type: ignore[attr-defined]
    kv_cache_planning.validate_kv_cache_layout = validate_kv_cache_layout_for_mixed_pages  # type: ignore[assignment]
    return True


apply_kv_cache_layout_patch()
