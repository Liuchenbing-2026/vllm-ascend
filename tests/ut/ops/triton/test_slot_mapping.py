# SPDX-License-Identifier: Apache-2.0
"""Precision + boundary tests for the Triton slot_mapping operators.

Run modes:
  TRITON_INTERPRET=1 python -m pytest tests/ut/ops/triton/test_slot_mapping.py   # CPU semantic check
  python -m pytest tests/ut/ops/triton/test_slot_mapping.py                       # on NPU
"""
import numpy as np
import pytest
import torch

from vllm_ascend.ops.triton.slot_mapping import (
    compute_positions_and_slot_mapping,
    compute_slot_mapping,
    slot_mapping_reference,
)


def _dev():
    return "npu" if torch.npu.is_available() else "cpu"


def _rand_case(n, num_reqs, block_size, max_blocks, seed=0):
    rng = np.random.default_rng(seed)
    block_table = rng.integers(0, 100000, size=(num_reqs, max_blocks), dtype=np.int32)
    req = rng.integers(0, num_reqs, size=n, dtype=np.int32)
    pos = rng.integers(0, max_blocks * block_size - 1, size=n).astype(np.int64)
    return req, pos, block_table


def _run_slot(req_np, pos_np, bt_np, block_size):
    dev = _dev()
    req = torch.from_numpy(req_np).to(dev)
    pos = torch.from_numpy(pos_np).to(dev)
    bt = torch.from_numpy(bt_np).to(dev).contiguous()
    out = torch.empty(req_np.shape[0], dtype=torch.int32, device=dev)
    compute_slot_mapping(req, pos, bt, out, block_size)
    return out.cpu().numpy()


def test_pr_11931_reference_case():
    """The exact #11931 UT vector (bs>4-token sanity for the general path)."""
    block_size = 128
    bt = np.zeros((2, 8), dtype=np.int32)
    bt[0, 0], bt[0, 1], bt[1, 0], bt[1, 1] = 0, 1, 4, 5
    req = np.array([0, 0, 1, 1], dtype=np.int32)
    pos = np.array([0, 129, 0, 130], dtype=np.int64)
    expected = np.array([0, 129, 512, 642], dtype=np.int32)
    np.testing.assert_array_equal(_run_slot(req, pos, bt, block_size), expected)


@pytest.mark.parametrize(
    "name,n,num_reqs,block_size,max_blocks",
    [
        ("bs1", 1, 1, 128, 64),
        ("bs4_tiny", 4, 4, 128, 64),
        ("non_pow2", 257, 8, 128, 64),
        ("large_gt_cores", 50000, 64, 128, 512),
        ("block64", 8000, 32, 64, 300),
        ("single_req_many_tok", 4096, 1, 128, 64),
    ],
)
def test_slot_matches_reference(name, n, num_reqs, block_size, max_blocks):
    req, pos, bt = _rand_case(n, num_reqs, block_size, max_blocks, seed=hash(name) & 0xFFFF)
    ref = slot_mapping_reference(req, pos, bt, block_size)
    np.testing.assert_array_equal(_run_slot(req, pos, bt, block_size), ref)


@pytest.mark.parametrize(
    "name,n,num_reqs,block_size,max_blocks",
    [("bs1", 1, 1, 128, 64), ("mixed", 300, 8, 128, 64), ("large", 20000, 64, 128, 512)],
)
def test_fused_positions_and_slot(name, n, num_reqs, block_size, max_blocks):
    """positions = num_computed_tokens[req] + query_pos, then slot from positions."""
    rng = np.random.default_rng(hash(name) & 0xFFFF)
    bt = rng.integers(0, 100000, size=(num_reqs, max_blocks), dtype=np.int32)
    req = rng.integers(0, num_reqs, size=n, dtype=np.int32)
    # keep num_computed_tokens + query_pos within table range
    ctx = rng.integers(0, max_blocks * block_size // 2, size=num_reqs).astype(np.int64)
    qpos = rng.integers(0, max_blocks * block_size // 2 - 1, size=n).astype(np.int64)
    pos_ref = ctx[req] + qpos
    slot_ref = slot_mapping_reference(req, pos_ref, bt, block_size)

    dev = _dev()
    positions = torch.empty(n, dtype=torch.int64, device=dev)
    slots = torch.empty(n, dtype=torch.int32, device=dev)
    p, s = compute_positions_and_slot_mapping(
        torch.from_numpy(req).to(dev),
        torch.from_numpy(ctx).to(dev),
        torch.from_numpy(qpos).to(dev),
        torch.from_numpy(bt).to(dev).contiguous(),
        positions,
        slots,
        block_size,
    )
    np.testing.assert_array_equal(p.cpu().numpy(), pos_ref)
    np.testing.assert_array_equal(s.cpu().numpy(), slot_ref)
