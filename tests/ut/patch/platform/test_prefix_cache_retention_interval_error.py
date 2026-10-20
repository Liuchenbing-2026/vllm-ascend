# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    _augment_retention_interval_error,
)


def _alignment_error(interval: int, block_size: int) -> ValueError:
    return ValueError(
        f"VLLM_PREFIX_CACHE_RETENTION_INTERVAL ({interval}) must be "
        f"non-negative and a multiple of scheduler_block_size ({block_size})."
    )


def test_reports_next_valid_multiple_of_effective_block_size():
    exc = _alignment_error(4096, 1536)

    enriched = _augment_retention_interval_error(exc, 4096, 1536)

    assert "must be non-negative and a multiple" in str(enriched)
    assert "for example 4608" in str(enriched)


def test_leaves_valid_interval_error_untouched():
    exc = _alignment_error(4608, 1536)

    assert _augment_retention_interval_error(exc, 4608, 1536) is exc


def test_leaves_negative_interval_error_untouched():
    exc = _alignment_error(-1, 1536)

    assert _augment_retention_interval_error(exc, -1, 1536) is exc


@pytest.mark.parametrize("scheduler_block_size", [None, 0])
def test_leaves_error_untouched_without_scheduler_block_size(scheduler_block_size):
    exc = _alignment_error(4096, 4096)

    assert _augment_retention_interval_error(exc, 4096, scheduler_block_size) is exc
