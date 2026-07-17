# SPDX-License-Identifier: Apache-2.0
"""Optimized Triton slot mapping for regular Ascend decode.

The upstream kernel launches one program per request plus a padding program that
writes the unused tail of the full slot-mapping buffer. This implementation
uses device-resident request indices, fuses position generation with slot
mapping, keeps index arithmetic in int32, and sizes the tile to the token count.
"""
from __future__ import annotations

import numpy as np
import torch
from vllm.triton_utils import tl, triton

# Token tile per program iteration. 1024 keeps UB tiny (~1024 * 36 B ~= 36 KB
# << 192 KB) while giving each core enough work to hide load latency.
_BLOCK_SIZE = 1024


@triton.jit(do_not_specialize=["n_tokens", "bt_row_stride"])
def _slot_mapping_kernel(
    req_indices_ptr,  # int32 [n_tokens] (device)
    positions_ptr,  # int64 [n_tokens] (device)
    block_table_ptr,  # int32 [num_reqs * bt_row_stride] (device, flattened)
    slot_mapping_ptr,  # int32 [n_tokens] (device, output)
    n_tokens,  # runtime scalar; do_not_specialize -> no recompile
    bt_row_stride,  # = max_num_blocks_per_req * blocks_per_phys_block
    BLOCK_SIZE_KV: tl.constexpr,  # KV cache block size (e.g. 128)
    BLOCK_SIZE: tl.constexpr,  # token tile per iteration
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    n_blocks = tl.cdiv(n_tokens, BLOCK_SIZE)
    # A single program walks additional tiles for large prefills.
    for block_id in range(pid, n_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_tokens
        req = tl.load(req_indices_ptr + offsets, mask=mask, other=0)  # int32
        # Cast positions to int32 up front and keep ALL index math in int32.
        # Ascend has no native int64 vector ALU, so int64 //, %, * get emulated
        # scalar-wise (per element) and the kernel goes scalar-bound. pos,
        # bt_idx and slot all fit int32 for realistic configs, so this is safe;
        # it also makes the block_table gather use int32 (not int64) addressing.
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        logical_block = pos // BLOCK_SIZE_KV  # int32
        bt_idx = req * bt_row_stride + logical_block  # int32 index
        block_number = tl.load(block_table_ptr + bt_idx, mask=mask, other=0)  # int32
        slot = block_number * BLOCK_SIZE_KV + pos % BLOCK_SIZE_KV  # int32
        tl.store(slot_mapping_ptr + offsets, slot, mask=mask)


@triton.jit(do_not_specialize=["n_tokens", "bt_row_stride"])
def _fused_position_slot_mapping_kernel(
    req_indices_ptr,  # int32 [n_tokens]
    num_computed_tokens_ptr,  # int64 [num_reqs] (per-request context length)
    query_pos_ptr,  # int64 [n_tokens] (token offset within its query)
    block_table_ptr,  # int32 [num_reqs * bt_row_stride]
    positions_ptr,  # int64 [n_tokens] (output)
    slot_mapping_ptr,  # int32 [n_tokens] (output)
    n_tokens,
    bt_row_stride,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    n_blocks = tl.cdiv(n_tokens, BLOCK_SIZE)
    for block_id in range(pid, n_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_tokens
        req = tl.load(req_indices_ptr + offsets, mask=mask, other=0)  # int32
        # positions = num_computed_tokens[req] + query_pos. Cast to int32 so
        # all index math stays on the vector ALU (int64 -> scalar-emulated).
        ctx = tl.load(num_computed_tokens_ptr + req, mask=mask, other=0).to(tl.int32)
        qp = tl.load(query_pos_ptr + offsets, mask=mask, other=0).to(tl.int32)
        pos = ctx + qp  # int32
        # positions output is int64 (vLLM convention): a contiguous vector
        # store+cast, not scalar arithmetic.
        tl.store(positions_ptr + offsets, pos.to(tl.int64), mask=mask)
        logical_block = pos // BLOCK_SIZE_KV  # int32
        bt_idx = req * bt_row_stride + logical_block  # int32 index
        block_number = tl.load(block_table_ptr + bt_idx, mask=mask, other=0)  # int32
        slot = block_number * BLOCK_SIZE_KV + pos % BLOCK_SIZE_KV  # int32
        tl.store(slot_mapping_ptr + offsets, slot, mask=mask)


def compute_slot_mapping(
    req_indices: torch.Tensor,  # int32 [n_tokens] (device)
    positions: torch.Tensor,  # int64 [n_tokens] (device)
    block_table: torch.Tensor,  # int32 [num_reqs, bt_row_stride] (device, contig)
    slot_mapping: torch.Tensor,  # int32 [>= n_tokens] output (device)
    block_size: int,
) -> torch.Tensor:
    """Compute slot mapping entirely on device without padding the unused tail."""
    n_tokens = req_indices.shape[0]
    bt_row_stride = block_table.shape[1]
    tile_size = min(_BLOCK_SIZE, triton.next_power_of_2(n_tokens))
    num_programs = 1
    _slot_mapping_kernel[(num_programs,)](
        req_indices,
        positions,
        block_table,
        slot_mapping,
        n_tokens,
        bt_row_stride,
        BLOCK_SIZE_KV=block_size,
        BLOCK_SIZE=tile_size,
    )
    return slot_mapping


def compute_positions_and_slot_mapping(
    req_indices: torch.Tensor,  # int32 [n_tokens]
    num_computed_tokens: torch.Tensor,  # int64 [num_reqs]
    query_pos: torch.Tensor,  # int64 [n_tokens]
    block_table: torch.Tensor,  # int32 [num_reqs, bt_row_stride]
    positions: torch.Tensor,  # int64 [n_tokens] output
    slot_mapping: torch.Tensor,  # int32 [n_tokens] output
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused positions + slot_mapping in one launch (one HBM round-trip saved)."""
    n_tokens = req_indices.shape[0]
    bt_row_stride = block_table.shape[1]
    tile_size = min(_BLOCK_SIZE, triton.next_power_of_2(n_tokens))
    num_programs = 1
    _fused_position_slot_mapping_kernel[(num_programs,)](
        req_indices,
        num_computed_tokens,
        query_pos,
        block_table,
        positions,
        slot_mapping,
        n_tokens,
        bt_row_stride,
        BLOCK_SIZE_KV=block_size,
        BLOCK_SIZE=tile_size,
    )
    return positions, slot_mapping


def slot_mapping_reference(
    req_indices: np.ndarray,
    positions: np.ndarray,
    block_table: np.ndarray,  # [num_reqs, bt_row_stride]
    block_size: int,
) -> np.ndarray:
    """Pure-numpy reference for precision cross-check."""
    logical = positions // block_size
    blk = block_table[req_indices, logical]
    return (blk.astype(np.int64) * block_size + positions % block_size).astype(np.int32)
