# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the unified DSpark SWA Triton wrapper and helpers."""

from unittest.mock import patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton import dspark_swa_indices as ds


def test_normalize_output_mode():
    assert ds.normalize_dspark_swa_output_mode() == ds.OUTPUT_MODE_PHYSICAL
    assert ds.normalize_dspark_swa_output_mode(use_logical_indices=True) == ds.OUTPUT_MODE_LOGICAL
    assert ds.normalize_dspark_swa_output_mode("logical", use_logical_indices=True) == ds.OUTPUT_MODE_LOGICAL
    with pytest.raises(ValueError, match="Unsupported DSpark SWA output_mode"):
        ds.normalize_dspark_swa_output_mode("bad")
    with pytest.raises(ValueError, match="Conflicting DSpark SWA mode arguments"):
        ds.normalize_dspark_swa_output_mode("physical", use_logical_indices=True)


def test_build_dspark_swa_indices_triton_requires_triton():
    with (
        patch.object(ds, "HAS_TRITON", False),
        pytest.raises(RuntimeError, match="Triton is required"),
    ):
        ds.build_dspark_swa_indices_triton(
            block_table=torch.ones((1, 2), dtype=torch.int32),
            num_speculative_tokens=3,
            window_size=16,
            block_size=128,
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
        )


def test_dspark_swa_indices_supported_removes_pow2_and_8192_limits():
    with patch.object(ds, "HAS_TRITON", False):
        assert ds.dspark_swa_indices_supported(128, 16) is False
    if not HAS_TRITON:
        return
    assert ds.dspark_swa_indices_supported(96, 16384, output_mode=ds.OUTPUT_MODE_PHYSICAL) is True
    assert ds.dspark_swa_indices_supported(100, 10000, output_mode=ds.OUTPUT_MODE_PHYSICAL) is True
    assert ds.dspark_swa_indices_supported(0, 16, output_mode=ds.OUTPUT_MODE_PHYSICAL) is False
    assert ds.dspark_swa_indices_supported(128, 0, output_mode=ds.OUTPUT_MODE_PHYSICAL) is False
    assert ds.dspark_swa_indices_supported(0, 0, output_mode=ds.OUTPUT_MODE_LOGICAL) is True


@pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")
@pytest.mark.parametrize("block_w", [8, 128, 1024])
@pytest.mark.parametrize("block_size", [1, 3, 12, 48, 96, 100, 128, 2048])
def test_bt_win_pow2_bounds_any_window_and_table_width(block_w, block_size):
    bt_win = ds._compute_bt_win_pow2(block_w, block_size, False)
    assert bt_win >= 2
    assert (bt_win & (bt_win - 1)) == 0
    num_blocks = 20_000
    for start_pos in (0, 1, block_size - 1, block_size, block_size + 1, 15000 * block_size - 1):
        blk_base = min(max(start_pos // block_size, 0), num_blocks - 1)
        blk_max = min(max((start_pos + block_w - 1) // block_size, 0), num_blocks - 1)
        assert 0 <= blk_max - blk_base < bt_win


@pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")
def test_triton_wrapper_rejects_host_tensors():
    with pytest.raises(ValueError, match="requires an accelerator device"):
        ds.build_dspark_swa_indices_triton(
            block_table=torch.ones((1, 4), dtype=torch.int32),
            num_speculative_tokens=3,
            window_size=16,
            block_size=96,
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
        )
