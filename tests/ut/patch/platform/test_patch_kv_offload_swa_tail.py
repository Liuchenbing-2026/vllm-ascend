# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""CPU-offload sliding-window tail retention.

A sliding-window prefix-cache hit needs a run of ``_contiguous_blocks_for_hit``
pages (plus one more when the EAGLE block drop is active), so the offload store
has to publish the trailing page while it is still addressable. The patch widens
``SlidingWindowManager.extra_retained_tokens`` to one page, and only when the
simple CPU offload connector is driving the store path.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from vllm_ascend.patch.platform import patch_kv_offload_swa_tail as module

OFFLOAD_ENV = "VLLM_USE_SIMPLE_KV_OFFLOAD"
BLOCK_SIZE = 128


@pytest.fixture
def window_manager_init():
    """The module patches at import; every test here installs its own stand-in."""
    saved = SlidingWindowManager.__init__
    yield
    SlidingWindowManager.__init__ = saved


def _upstream_init(extra_retained_tokens: int = 0, dcp_world_size: int = 1):
    """Stand-in for upstream ``__init__``: it owns the two attributes the patch
    reads and writes, and nothing else."""

    def __init__(self, kv_cache_spec, **kwargs):
        self.extra_retained_tokens = extra_retained_tokens
        self.dcp_world_size = dcp_world_size

    return __init__


def _spec(block_size: int = BLOCK_SIZE) -> SimpleNamespace:
    return SimpleNamespace(block_size=block_size)


def _build(monkeypatch, env: str | None, **stub_kwargs) -> SimpleNamespace:
    if env is None:
        monkeypatch.delenv(OFFLOAD_ENV, raising=False)
    else:
        monkeypatch.setenv(OFFLOAD_ENV, env)
    SlidingWindowManager.__init__ = _upstream_init(**stub_kwargs)
    module._patch_sliding_window_offload_tail()
    manager = SimpleNamespace()
    SlidingWindowManager.__init__(manager, _spec())
    return manager


def test_offload_on_retains_one_extra_page(monkeypatch, window_manager_init):
    manager = _build(monkeypatch, "1")
    assert manager.extra_retained_tokens == BLOCK_SIZE


def test_offload_off_leaves_the_window_untouched(monkeypatch, window_manager_init):
    for value in ("0", None):
        manager = _build(monkeypatch, value)
        assert manager.extra_retained_tokens == 0


def test_retained_edge_covers_the_rank_local_page_under_dcp(monkeypatch, window_manager_init):
    """DCP shards a page across ranks, so the retained edge has to cover the
    rank-local page the connector stores, not the logical one."""
    manager = _build(monkeypatch, "1", dcp_world_size=2)
    assert manager.extra_retained_tokens == 2 * BLOCK_SIZE


def test_an_existing_wider_edge_is_never_narrowed(monkeypatch, window_manager_init):
    """Multi-module MTP already asks for a wider edge than one page; the patch
    must not shrink it to a single page."""
    manager = _build(monkeypatch, "1", extra_retained_tokens=4 * BLOCK_SIZE)
    assert manager.extra_retained_tokens == 4 * BLOCK_SIZE


def test_patching_twice_does_not_compound(monkeypatch, window_manager_init):
    monkeypatch.setenv(OFFLOAD_ENV, "1")
    SlidingWindowManager.__init__ = _upstream_init()
    module._patch_sliding_window_offload_tail()
    module._patch_sliding_window_offload_tail()
    manager = SimpleNamespace()
    SlidingWindowManager.__init__(manager, _spec())
    assert manager.extra_retained_tokens == BLOCK_SIZE
    assert SlidingWindowManager.__init__._vllm_ascend_offload_tail_patched


def test_upstream_still_carries_the_field_the_patch_widens():
    """Pin guard: the field is what the patch widens, and the hit rule below is
    what it has to satisfy. A pin bump that renames or drops either one has to
    fail here instead of silently turning the patch into a no-op."""
    assert "extra_retained_tokens" in SlidingWindowSpec.__dataclass_fields__
    # One page wide window plus the EAGLE block drop is exactly why the tail run
    # is two pages long, and why one extra retained page closes the gap.
    assert SlidingWindowManager._contiguous_blocks_for_hit(BLOCK_SIZE, BLOCK_SIZE, True) == 2
    assert SlidingWindowManager._contiguous_blocks_for_hit(BLOCK_SIZE, BLOCK_SIZE, False) == 1
