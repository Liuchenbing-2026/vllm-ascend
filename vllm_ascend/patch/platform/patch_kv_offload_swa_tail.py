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
"""
import functools
import os
from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
logger = init_logger(__name__)


def _cpu_offload_enabled() -> bool:
    """Whether the simple CPU KV offload connector is driving the store path."""
    return bool(os.getenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "0")) and os.getenv(
        "VLLM_USE_SIMPLE_KV_OFFLOAD", "0"
    ) != "0"


def _patch_sliding_window_offload_tail() -> None:
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
                "CPU KV offload: retaining one extra %d-token sliding-window page "
                "below the window so the offload store can publish the full hit run.",
                block_size,
            )

    __init__._vllm_ascend_offload_tail_patched = True  # type: ignore[attr-defined]
    SlidingWindowManager.__init__ = __init__


_patch_sliding_window_offload_tail()
