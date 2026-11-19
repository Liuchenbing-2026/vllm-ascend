# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU state contracts for native hybrid frontier checkpoints on Ascend."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import CircularBufferSpec


def test_frontier_tail_preserves_ascend_ring_storage():
    """A resumable tail must keep both FP32 planes and token slot writes."""
    from vllm_ascend.core.kv_cache_interface import (
        AscendFrontierKPoolTailSpec,
        AscendIndexerKPoolTailSpec,
    )

    kwargs = dict(
        block_size=4,
        compress_ratio=4,
        ring_capacity=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=4,
    )
    original = AscendIndexerKPoolTailSpec(**kwargs)
    frontier = AscendFrontierKPoolTailSpec(**kwargs)
    assert isinstance(frontier, CircularBufferSpec)
    assert isinstance(frontier, AscendIndexerKPoolTailSpec)
    assert frontier.page_size_bytes == original.page_size_bytes == 256
    assert frontier.storage_block_size == original.storage_block_size == 4
    assert frontier.uses_slot_mapping == original.uses_slot_mapping
    assert frontier.group_size == 4 and frontier.resume_catchup == "decode"
    assert not frontier.prefix_cacheable
    assert not frontier.is_uniform_with_collection({"old": original})
    assert AscendFrontierKPoolTailSpec.merge([frontier, frontier]) == frontier


def test_frontier_ring_hit_is_copied_into_private_zeroed_block():
    """Restoring a tail retains the source until the scheduled device copy."""
    from vllm_ascend.core.kv_cache_interface import AscendFrontierKPoolTailSpec
    from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager

    spec = AscendFrontierKPoolTailSpec(
        block_size=4,
        compress_ratio=4,
        ring_capacity=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=4,
    )
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=4)
    manager = AscendCircularBufferManager(spec, pool, True, 0, 64, needs_kv_cache_zeroing=True)
    source = manager.allocate_new_blocks("first", 3, 3)[0]
    manager.take_new_block_ids()
    manager.takes_frontier_hits = True
    manager.add_local_computed_blocks("next", [source], 3, 0)
    target = manager.allocate_new_blocks("next", 4, 4)[0]
    assert target is not source
    assert manager.take_new_block_ids() == [target.block_id]
    assert manager.take_pending_cow_copies() == [(source, target)]
    assert source.ref_cnt == 2 and target.ref_cnt == 2


def test_frontier_tail_copy_preserves_keys_gates_and_other_pages():
    """Exercise the actual Ascend page view and shared device-copy implementation."""
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

    from vllm_ascend.core.kv_cache_interface import AscendFrontierKPoolTailSpec
    from vllm_ascend.models.glm5next.cache_views import view_kpool_tail_cache
    from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace

    spec = AscendFrontierKPoolTailSpec(
        block_size=4,
        compress_ratio=4,
        ring_capacity=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=4,
    )
    num_blocks = 4
    # Padding makes the physical view non-contiguous, as in a shared small page.
    page_elements = spec.page_size_bytes // 4 + 8
    storage = torch.arange(num_blocks * page_elements, dtype=torch.float32)
    tail = view_kpool_tail_cache("tail", spec, storage.view(torch.int8), num_blocks)
    before = storage.view(num_blocks, page_elements).clone()
    copy_kv_cache_blocks_inplace(
        [tail],
        num_blocks,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=2), KVCacheBlockCopy(src_block_id=2, dst_block_id=1)],
    )
    after = storage.view(num_blocks, page_elements)
    tail_elements = spec.unpadded_page_size_bytes // 4
    assert torch.equal(after[1, :tail_elements], before[2, :tail_elements])
    assert torch.equal(after[2, :tail_elements], before[1, :tail_elements])
    assert torch.equal(after[:, tail_elements:], before[:, tail_elements:])
    assert torch.equal(after[[0, 3]], before[[0, 3]])


@pytest.mark.parametrize(
    "mode",
    [
        "recompute_scheduler_enable",
        "dyntra_lb_config",
        "enable_balance_scheduling",
    ],
)
def test_frontier_rejects_pinned_schedulers_before_platform_setup(mode):
    from vllm_ascend.platform import _setup_worker_and_scheduler

    config = SimpleNamespace(cache_config=SimpleNamespace(mamba_frontier_checkpoint=True))
    scheduler = SimpleNamespace(
        recompute_scheduler_enable=False,
        dyntra_lb_config=SimpleNamespace(enabled=False),
        enable_balance_scheduling=False,
    )
    setattr(scheduler, mode, SimpleNamespace(enabled=True) if mode.endswith("_config") else True)
    with pytest.raises(ValueError, match=mode):
        _setup_worker_and_scheduler(config, SimpleNamespace(scheduler_config=scheduler))


def test_frontier_rejects_custom_scheduler_cls_and_allows_compatible():
    from vllm_ascend.platform import _frontier_unsupported_scheduler_cls, _setup_worker_and_scheduler

    class _CustomScheduler:
        pass

    ascend_cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            recompute_scheduler_enable=False,
            dyntra_lb_config=SimpleNamespace(enabled=False),
            enable_balance_scheduling=False,
        )
    )
    for bad_cls in ("custom.pkg.MyScheduler", _CustomScheduler):
        bad_vllm_cfg = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_frontier_checkpoint=True),
            scheduler_config=SimpleNamespace(scheduler_cls=bad_cls),
        )
        with pytest.raises(ValueError, match="scheduler_cls"):
            _setup_worker_and_scheduler(bad_vllm_cfg, ascend_cfg)

    for ok_cls in (
        None,
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "vllm_ascend.core.batch_job_aware_scheduler.BatchJobAwareAsyncScheduler",
        "vllm_ascend.core.short_request_first_scheduler.ShortRequestFirstAsyncScheduler",
    ):
        ok_vllm_cfg = SimpleNamespace(
            scheduler_config=SimpleNamespace(scheduler_cls=ok_cls),
        )
        assert _frontier_unsupported_scheduler_cls(ok_vllm_cfg) is None
