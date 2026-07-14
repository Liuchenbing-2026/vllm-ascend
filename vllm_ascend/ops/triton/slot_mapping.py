# SPDX-License-Identifier: Apache-2.0
"""Host-cheap, ACLGraph-capturable Triton slot_mapping for Ascend NPU.

slot_mapping[i] = block_table[req_indices[i], positions[i] // block_size] * block_size
                  + positions[i] % block_size

The problem this fixes (#7640 host-bound regression)
----------------------------------------------------
#7640 moved slot_mapping onto a Triton kernel launched as::

    _compute_slot_mapping_kernel[(num_reqs + 1,)](num_tokens, ..., query_start_loc, ...)

Two properties of that launch make it *host-bound* on the decode critical path:

  1. **Data-dependent grid** ``(num_reqs + 1,)`` — the launch config changes
     every step, so the host recomputes the grid + re-dispatches the kernel on
     the critical path between "positions ready" and the attention kernel, and
     the launch can't be hidden behind NPU work. For tiny-compute decode steps
     the host launch dominates -> the NPU starves.
  2. A per-step, per-shape launch is unfriendly to ACLGraph capture, so the
     launch cost is paid on every replay instead of being captured once.

#11931's CPU fallback only *sidesteps* this for bs=1 — it is a temporary
workaround, not a fix, and it leaves the host doing the work.

The fix (this module): make the kernel host-cheap and capturable
----------------------------------------------------------------
  * **Fixed grid** = ``vectorcore_num`` (a build-time device constant), never
    a function of ``num_reqs``. A core-internal strided loop covers any token
    count, so one stable launch config serves bs=1 and bs=large alike.
  * **``do_not_specialize``** on the runtime scalars (``n_tokens``,
    ``bt_row_stride``) so a changing token count never triggers a host-side
    recompile.
  * **Pure device, no host sync** — all inputs are device tensors; no
    ``.item()``, no ``copy_to_gpu`` round-trip, no ``event.synchronize()``.
  * Result: the launch is constant + side-effect-free, so it is captured once
    in the ACLGraph and the per-step host launch cost amortizes to ~0 on
    replay. No CPU/size dispatch needed — short and long are both fast.

A fused variant (``compute_positions_and_slot_mapping``) also folds the
``positions = num_computed_tokens[req] + query_pos`` compute into the same
kernel, removing one launch and one HBM round-trip of ``positions``.
"""
from __future__ import annotations

import numpy as np
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

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
    num_programs = tl.num_programs(axis=0)  # == fixed grid (vectorcore_num)
    n_blocks = tl.cdiv(n_tokens, BLOCK_SIZE)
    # Core-internal strided loop: one fixed launch config covers any n_tokens.
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
    """Single host-cheap kernel path — no CPU/size dispatch.

    Fixed grid + do_not_specialize + all-device makes this ACLGraph-capturable;
    the per-step host launch amortizes to ~0 on graph replay, so bs=1 decode
    and large prefill are both fast without any fallback branch.
    """
    n_tokens = req_indices.shape[0]
    bt_row_stride = block_table.shape[1]
    num_programs = get_vectorcore_num()  # fixed grid, independent of num_reqs
    _slot_mapping_kernel[(num_programs,)](
        req_indices,
        positions,
        block_table.reshape(-1),
        slot_mapping,
        n_tokens,
        bt_row_stride,
        BLOCK_SIZE_KV=block_size,
        BLOCK_SIZE=_BLOCK_SIZE,
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
    num_programs = get_vectorcore_num()
    _fused_position_slot_mapping_kernel[(num_programs,)](
        req_indices,
        num_computed_tokens,
        query_pos,
        block_table.reshape(-1),
        positions,
        slot_mapping,
        n_tokens,
        bt_row_stride,
        BLOCK_SIZE_KV=block_size,
        BLOCK_SIZE=_BLOCK_SIZE,
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
