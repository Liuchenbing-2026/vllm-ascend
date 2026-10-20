# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.patch.platform import patch_kv_cache_coordinator as coordinator_patch


class _FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class _MambaSpec:
    def __init__(self, block_size):
        self.block_size = block_size


def _kv_cache_config(*specs):
    groups = [SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    return SimpleNamespace(kv_cache_groups=groups)


def _upstream_raises(message):
    def validate(retention_interval, scheduler_block_size, kv_cache_config):
        raise ValueError(message)

    return MagicMock(side_effect=validate)


def test_multiple_mismatch_reports_group_breakdown_and_next_valid_interval():
    config = _kv_cache_config(_FullAttentionSpec(32), _MambaSpec(1536))
    validate = _upstream_raises(
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL (4096) must be non-negative and "
        "a multiple of scheduler_block_size (1536)."
    )

    with pytest.raises(ValueError) as excinfo:
        coordinator_patch._validate_retention_interval_with_hint(4096, 1536, config, validate)

    message = str(excinfo.value)
    assert "must be non-negative and a multiple of scheduler_block_size (1536)" in message
    assert "_FullAttentionSpec=32" in message
    assert "_MambaSpec=1536" in message
    assert "Set VLLM_PREFIX_CACHE_RETENTION_INTERVAL to 4608" in message
    assert excinfo.value.__cause__ is not None
    validate.assert_called_once_with(4096, 1536, config)


def test_valid_interval_is_delegated_unchanged():
    config = _kv_cache_config(_FullAttentionSpec(32), _MambaSpec(1536))
    validate = MagicMock()

    coordinator_patch._validate_retention_interval_with_hint(4608, 1536, config, validate)

    validate.assert_called_once_with(4608, 1536, config)


def test_non_multiple_validation_errors_are_not_rewritten():
    config = _kv_cache_config(_FullAttentionSpec(32))
    message = (
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL is set but this model has no "
        "sliding-window or Mamba KV cache group, so retention has no effect."
    )
    validate = _upstream_raises(message)

    with pytest.raises(ValueError) as excinfo:
        coordinator_patch._validate_retention_interval_with_hint(4096, 32, config, validate)

    assert str(excinfo.value) == message
    assert "LCM" not in str(excinfo.value)
    validate.assert_called_once_with(4096, 32, config)


def test_zero_interval_is_not_hinted():
    config = _kv_cache_config(_FullAttentionSpec(32), _MambaSpec(1536))
    validate = _upstream_raises("unrelated failure")

    with pytest.raises(ValueError, match="unrelated failure"):
        coordinator_patch._validate_retention_interval_with_hint(0, 1536, config, validate)


def test_group_summary_skips_specs_without_block_size():
    config = _kv_cache_config(_FullAttentionSpec(32), SimpleNamespace())

    assert coordinator_patch._kv_group_block_size_summary(config) == "_FullAttentionSpec=32"
