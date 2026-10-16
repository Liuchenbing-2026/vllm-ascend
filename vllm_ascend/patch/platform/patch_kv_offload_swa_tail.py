# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Retain the trailing sliding-window page that a CPU-offload tail store needs.

A sliding-window prefix-cache hit is a *run* of contiguous cached pages: the
lookup walks a group's pages from the newest one backwards and only accepts a
match once ``_contiguous_blocks_for_hit`` pages in a row are present, plus one
more page when the EAGLE block drop is active. ``extra_retained_tokens`` is the
engine's mechanism for keeping the trailing edge of the window alive below the
window; upstream only uses it for multi-module MTP.

That is not enough for a hybrid model whose window is exactly one page wide.
DeepSeek V4.1 compresses several layer families into sliding-window pages of
128, 32 and 8 tokens while the model's ``sliding_window`` is 128, so the two
128-token page groups hold a single page at a time. ``SimpleCPUOffloadConnector``
stores a page only on the scheduler step *after* its KV is committed, and by
then the next page has already slid in and freed the previous one. The offload
cache therefore never accumulates the tail run a later hit has to match, and a
long-context (32K) prefix collapses to a short or empty external hit while a
short (16K) one still hits, because only the short one keeps its pages out of
reach of the allocator's reuse.

Retaining one extra page below the window closes exactly that gap: the store
sees the complete ``need``-page tail while it is still addressable, and the
blocks are never attended, so attention results are unchanged. The budget is
one page per sliding-window group per request.

vLLM 0.31 reads the retained edge from the *spec* on both sides of the offload
path -- ``SlidingWindowManager`` copies it into the block manager, and the
offloading scheduler uses it to widen the store mask (``_reachable_store_block_mask``)
and the load window (``get_sliding_window_size_in_chunks``). The spec is
therefore widened where the engine core finalises the KV cache configuration,
and the manager keeps a fallback for pins that hand the manager a stale spec.
"""

import functools
import os
from dataclasses import replace

from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)


def _cpu_offload_enabled() -> bool:
    """Whether the simple CPU KV offload connector is driving the store path."""
    value = os.getenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "0")
    return value not in ("", "0")


def _retained_edge_tokens(spec: SlidingWindowSpec, dcp_world_size: int) -> int:
    """One page, or one rank-local page when DCP shards the page across ranks."""
    page = spec.block_size
    if getattr(spec, "dcp_sharded", False):
        page *= max(1, dcp_world_size)
    return page


def _widen_spec(spec, dcp_world_size: int):
    """Return ``spec`` with at least one extra page retained below the window."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        nested = spec.kv_cache_specs
        widened = {name: _widen_spec(sub_spec, dcp_world_size) for name, sub_spec in nested.items()}
        if all(widened[name] is nested[name] for name in nested):
            return spec
        return replace(spec, kv_cache_specs=widened)
    if not isinstance(spec, SlidingWindowSpec):
        return spec
    edge = _retained_edge_tokens(spec, dcp_world_size)
    if edge <= 0 or spec.extra_retained_tokens >= edge:
        return spec
    logger.info_once(
        "CPU KV offload: retaining one extra %d-token sliding-window page "
        "below the window so the offload store can publish the full hit run.",
        edge,
    )
    return replace(spec, extra_retained_tokens=edge)


def _dcp_world_size(vllm_config) -> int:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    size = getattr(parallel_config, "decode_context_parallel_size", 1)
    return max(1, int(size or 1))


def _widen_configs(configs, vllm_config) -> int:
    """Widen every sliding-window spec the offload path will look at."""
    dcp_world_size = _dcp_world_size(vllm_config)
    widened = 0
    for config in configs:
        for group in getattr(config, "kv_cache_groups", None) or ():
            new_spec = _widen_spec(group.kv_cache_spec, dcp_world_size)
            if new_spec is not group.kv_cache_spec:
                group.kv_cache_spec = new_spec
                widened += 1
    return widened


def _patch_kv_cache_configs() -> None:
    """Widen the finalised specs, the object both offload sides read."""
    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils.get_kv_cache_configs
    if getattr(original, "_vllm_ascend_offload_tail_widened", False):
        return

    @functools.wraps(original)
    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        configs = original(vllm_config, kv_cache_specs, available_memory)
        if _cpu_offload_enabled():
            _widen_configs(configs, vllm_config)
        return configs

    get_kv_cache_configs._vllm_ascend_offload_tail_widened = True  # type: ignore[attr-defined]
    kv_cache_utils.get_kv_cache_configs = get_kv_cache_configs
    # The engine core imports the symbol by name, so rebind it there as well.
    try:
        from vllm.v1.engine import core as engine_core
    except Exception:  # pragma: no cover - module layout guard
        return
    if getattr(engine_core, "get_kv_cache_configs", None) is original:
        engine_core.get_kv_cache_configs = get_kv_cache_configs


def _patch_sliding_window_offload_tail() -> None:
    """Fallback for a manager built from a spec that was not widened in time."""
    original = SlidingWindowManager.__init__
    if getattr(original, "_vllm_ascend_offload_tail_patched", False):
        return

    @functools.wraps(original)
    def __init__(self, kv_cache_spec, *args, **kwargs):
        original(self, kv_cache_spec, *args, **kwargs)
        if not _cpu_offload_enabled():
            return
        # DCP shards a page across ranks, so the retained edge has to cover the
        # rank-local page the connector stores, not the logical one.
        block_size = kv_cache_spec.block_size * getattr(self, "dcp_world_size", 1)
        if block_size > 0 and self.extra_retained_tokens < block_size:
            self.extra_retained_tokens = block_size
            logger.info_once(
                "CPU KV offload: retaining one extra %d-token sliding-window page (manager fallback).",
                block_size,
            )

    __init__._vllm_ascend_offload_tail_patched = True  # type: ignore[attr-defined]
    SlidingWindowManager.__init__ = __init__


_patch_kv_cache_configs()
_patch_sliding_window_offload_tail()
