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
has to publish the trailing page while it is still addressable. vLLM 0.31 reads
the retained edge from the KV cache *spec*, so the patch widens the spec, and
only when the simple CPU offload connector is driving the store path.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    get_sliding_window_size_in_chunks,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.patch.platform import patch_kv_offload_swa_tail as module

OFFLOAD_ENV = "VLLM_USE_SIMPLE_KV_OFFLOAD"
BLOCK_SIZE = 128
WINDOW = 128


def _swa_spec(extra_retained_tokens: int = 0, dcp_sharded: bool = False) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        extra_retained_tokens=extra_retained_tokens,
        dcp_sharded=dcp_sharded,
    )


@pytest.fixture
def window_manager_init():
    """The module patches at import; every test here installs its own stand-in."""
    saved = SlidingWindowManager.__init__
    yield
    SlidingWindowManager.__init__ = saved


def _upstream_init(extra_retained_tokens: int = 0, dcp_world_size: int = 1):
    def __init__(self, kv_cache_spec, **kwargs):
        self.extra_retained_tokens = extra_retained_tokens
        self.dcp_world_size = dcp_world_size

    return __init__


def _build_manager(monkeypatch, env, **stub_kwargs) -> SimpleNamespace:
    if env is None:
        monkeypatch.delenv(OFFLOAD_ENV, raising=False)
    else:
        monkeypatch.setenv(OFFLOAD_ENV, env)
    SlidingWindowManager.__init__ = _upstream_init(**stub_kwargs)
    module._patch_sliding_window_offload_tail()
    manager = SimpleNamespace()
    SlidingWindowManager.__init__(manager, SimpleNamespace(block_size=BLOCK_SIZE))
    return manager


#
# Spec widening: the mechanism vLLM 0.31 reads on both sides of the offload path.
#


def test_spec_keeps_one_extra_page():
    widened = module._widen_spec(_swa_spec(), dcp_world_size=1)
    assert widened.extra_retained_tokens == BLOCK_SIZE


def test_spec_keeps_one_rank_local_page_under_dcp():
    widened = module._widen_spec(_swa_spec(dcp_sharded=True), dcp_world_size=2)
    assert widened.extra_retained_tokens == 2 * BLOCK_SIZE


def test_spec_is_left_alone_when_the_edge_is_already_wider():
    spec = _swa_spec(extra_retained_tokens=4 * BLOCK_SIZE)
    assert module._widen_spec(spec, dcp_world_size=1) is spec


def test_spec_is_left_alone_for_other_attention_types():
    spec = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    assert module._widen_spec(spec, dcp_world_size=1) is spec


def test_spec_widening_reaches_the_offload_load_window():
    """The offload scheduler sizes its load window from the spec; the widened
    spec has to be the one it reads."""
    plain = _swa_spec()
    widened = module._widen_spec(plain, dcp_world_size=1)
    tokens_per_chunk = 64
    assert get_sliding_window_size_in_chunks(widened, tokens_per_chunk) > get_sliding_window_size_in_chunks(
        plain, tokens_per_chunk
    )


def test_uniform_specs_are_widened_in_place_of_their_members():
    uniform = UniformTypeKVCacheSpecs(
        block_size=BLOCK_SIZE,
        kv_cache_specs={"layer_a": _swa_spec(), "layer_b": _swa_spec()},
    )
    widened = module._widen_spec(uniform, dcp_world_size=1)
    assert widened is not uniform
    assert all(spec.extra_retained_tokens == BLOCK_SIZE for spec in widened.kv_cache_specs.values())
    assert uniform.kv_cache_specs["layer_a"].extra_retained_tokens == 0


def _config_with_swa_group():
    group = KVCacheGroupSpec(layer_names=["layer_a"], kv_cache_spec=_swa_spec())
    return SimpleNamespace(kv_cache_groups=[group]), group


def _fake_vllm_config(dcp_world_size: int = 1):
    return SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_world_size))


def test_widen_configs_widens_swa_groups():
    config, group = _config_with_swa_group()
    assert module._widen_configs([config], _fake_vllm_config()) == 1
    assert group.kv_cache_spec.extra_retained_tokens == BLOCK_SIZE


def test_wrapper_widens_only_when_the_offload_switch_is_on(monkeypatch):
    """The switch is read once per engine start, so the wrapper has to gate on
    it rather than widening every sliding window on the box."""
    from vllm.v1.core import kv_cache_utils

    saved = kv_cache_utils.get_kv_cache_configs

    def unpatched(vllm_config, kv_cache_specs, available_memory):
        config, _ = _config_with_swa_group()
        return [config]

    try:
        kv_cache_utils.get_kv_cache_configs = unpatched
        module._patch_kv_cache_configs()
        wrapped = kv_cache_utils.get_kv_cache_configs
        assert wrapped is not unpatched

        monkeypatch.setenv(OFFLOAD_ENV, "0")
        group = wrapped(_fake_vllm_config(), {}, 0)[0].kv_cache_groups[0]
        assert group.kv_cache_spec.extra_retained_tokens == 0

        monkeypatch.setenv(OFFLOAD_ENV, "1")
        group = wrapped(_fake_vllm_config(), {}, 0)[0].kv_cache_groups[0]
        assert group.kv_cache_spec.extra_retained_tokens == BLOCK_SIZE

        # Patching twice must not stack another wrapper on top.
        module._patch_kv_cache_configs()
        assert kv_cache_utils.get_kv_cache_configs is wrapped
    finally:
        kv_cache_utils.get_kv_cache_configs = saved


#
# Manager fallback and the manager's own attributes.
#


def test_manager_fallback_retains_one_extra_page(monkeypatch, window_manager_init):
    manager = _build_manager(monkeypatch, "1")
    assert manager.extra_retained_tokens == BLOCK_SIZE


def test_manager_fallback_is_inert_without_offload(monkeypatch, window_manager_init):
    for value in ("0", None):
        manager = _build_manager(monkeypatch, value)
        assert manager.extra_retained_tokens == 0


def test_manager_fallback_covers_the_rank_local_page_under_dcp(monkeypatch, window_manager_init):
    manager = _build_manager(monkeypatch, "1", dcp_world_size=2)
    assert manager.extra_retained_tokens == 2 * BLOCK_SIZE


def test_manager_fallback_never_narrows_a_wider_edge(monkeypatch, window_manager_init):
    manager = _build_manager(monkeypatch, "1", extra_retained_tokens=4 * BLOCK_SIZE)
    assert manager.extra_retained_tokens == 4 * BLOCK_SIZE


def test_manager_fallback_is_idempotent(monkeypatch, window_manager_init):
    monkeypatch.setenv(OFFLOAD_ENV, "1")
    SlidingWindowManager.__init__ = _upstream_init()
    module._patch_sliding_window_offload_tail()
    module._patch_sliding_window_offload_tail()
    manager = SimpleNamespace()
    SlidingWindowManager.__init__(manager, SimpleNamespace(block_size=BLOCK_SIZE))
    assert manager.extra_retained_tokens == BLOCK_SIZE
    assert SlidingWindowManager.__init__._vllm_ascend_offload_tail_patched


def test_upstream_still_carries_the_field_the_patch_widens():
    """Pin guard: the field is what the patch widens, and the hit rule below is
    what it has to satisfy. A pin bump that renames or drops either one has to
    fail here instead of silently turning the patch into a no-op."""
    assert "extra_retained_tokens" in SlidingWindowSpec.__dataclass_fields__
    # One page wide window plus the EAGLE block drop is exactly why the tail run
    # is two pages long, and why one extra retained page closes the gap.
    assert SlidingWindowManager._contiguous_blocks_for_hit(WINDOW, BLOCK_SIZE, True) == 2
    assert SlidingWindowManager._contiguous_blocks_for_hit(WINDOW, BLOCK_SIZE, False) == 1
