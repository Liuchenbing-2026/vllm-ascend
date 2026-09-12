# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

_HIDDEN_SIZE = 2048
_TOP_K = 8
_NUM_EXPERTS = 256
_DUMMY_ROWS = _NUM_EXPERTS // _TOP_K
_MAX_TOKENS = 8192
_A2_VECTOR_CORES = 40
_COPY_BLOCK = 4096
_ROUTING_BLOCK = 256


@triton.jit
def _prepare_local_partial_kernel(
    hidden,
    ids,
    weights,
    active,
    padded_hidden,
    padded_ids,
    padded_weights,
    padded_active,
    tokens,
    HAS_ACTIVE: tl.constexpr,
    PREAPPLY_ACTIVE: tl.constexpr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    EXPERTS: tl.constexpr,
    DUMMY_ROWS: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    ROUTING_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    rows = tokens + DUMMY_ROWS
    for block in range(pid, tl.cdiv(rows * HIDDEN, COPY_BLOCK), programs):
        offsets = block * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
        values = tl.load(hidden + offsets, offsets < tokens * HIDDEN, other=1.0)
        tl.store(padded_hidden + offsets, values, offsets < rows * HIDDEN)

    for block in range(pid, tl.cdiv(rows * TOP_K, ROUTING_BLOCK), programs):
        offsets = block * ROUTING_BLOCK + tl.arange(0, ROUTING_BLOCK)
        real = offsets < tokens * TOP_K
        expert = tl.load(ids + offsets, real, other=0)
        expert = tl.where(real, expert, (offsets - tokens * TOP_K) % EXPERTS)
        if PREAPPLY_ACTIVE and HAS_ACTIVE:
            is_active = tl.load(active + offsets // TOP_K, real, other=1) != 0
            expert = tl.where(is_active, expert, EXPERTS)
        # Widening BF16 probabilities is exact. The dummy probability 1/8
        # is representable in both BF16 and FP32.
        probability = tl.load(weights + offsets, real, other=1.0 / TOP_K).to(tl.float32)
        tl.store(padded_ids + offsets, expert, offsets < rows * TOP_K)
        tl.store(padded_weights + offsets, probability, offsets < rows * TOP_K)

    if not PREAPPLY_ACTIVE:
        for block in range(pid, tl.cdiv(rows, ROUTING_BLOCK), programs):
            offsets = block * ROUTING_BLOCK + tl.arange(0, ROUTING_BLOCK)
            if HAS_ACTIVE:
                mask = tl.load(active + offsets, offsets < tokens, other=1)
            else:
                mask = tl.full((ROUTING_BLOCK,), 1, tl.int8)
            tl.store(padded_active + offsets, mask, offsets < rows)


def prepare_cann_megamoe_local_partial(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_active_mask: torch.Tensor | None,
    num_experts: int,
    ep_rank_id: int,
    ep_world_size: int,
    dummy_cache: dict | None = None,
    *,
    preapply_active_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """Prepare replicated A2 BF16 local-partial inputs in one device launch.

    The local-partial layout appends all 32 sentinel rows on every rank, so
    callers pass sentinel rank/world 0/1, irrespective of the TP rank. Returned
    probabilities are already FP32 for CANN. Every output owns fresh storage:
    CANN may modify routing IDs, and queued calls must remain independent.
    ``dummy_cache`` is accepted for the reference helper's calling convention;
    this kernel writes the constants directly and does not need a cache.

    With ``preapply_active_mask``, inactive real routes receive expert ID 256,
    exactly as A2 MegaMoe's ApplyXActiveMask does before sorting. The returned
    mask is None so CANN skips its scalar mask traversal. Sentinel rows stay
    active. This mode is specific to the local-partial CANN layout.
    """
    if (num_experts, ep_rank_id, ep_world_size) != (_NUM_EXPERTS, 0, 1):
        raise ValueError("Local-partial preparation requires 256 experts and sentinel rank/world 0/1.")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != _HIDDEN_SIZE:
        raise ValueError("Local-partial preparation requires hidden size 2048.")
    tokens = hidden_states.shape[0]
    if not 1 <= tokens <= _MAX_TOKENS:
        raise ValueError("Local-partial preparation supports 1 through 8192 real tokens.")
    if topk_ids.shape != (tokens, _TOP_K) or topk_weights.shape != topk_ids.shape:
        raise ValueError("Local-partial preparation requires eight routes per real token.")
    if hidden_states.dtype != torch.bfloat16 or topk_ids.dtype != torch.int32:
        raise ValueError("Local-partial preparation requires BF16 hidden states and INT32 routing IDs.")
    if topk_weights.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Local-partial preparation requires BF16 or FP32 routing probabilities.")
    inputs = (hidden_states, topk_ids, topk_weights)
    if x_active_mask is not None:
        if x_active_mask.shape != (tokens,) or x_active_mask.dtype not in (torch.bool, torch.int8):
            raise ValueError("Active mask must contain one BOOL or INT8 value per real token.")
        inputs += (x_active_mask,)
    if hidden_states.device.type != "npu" or any(
        x.device != hidden_states.device or not x.is_contiguous() for x in inputs
    ):
        raise ValueError("Local-partial preparation requires contiguous tensors on the same NPU.")

    rows = tokens + _DUMMY_ROWS
    padded_hidden = torch.empty((rows, _HIDDEN_SIZE), dtype=torch.bfloat16, device=hidden_states.device)
    padded_ids = torch.empty((rows, _TOP_K), dtype=torch.int32, device=hidden_states.device)
    padded_weights = torch.empty((rows, _TOP_K), dtype=torch.float32, device=hidden_states.device)
    padded_active = (
        None
        if preapply_active_mask
        else torch.empty(
            rows, dtype=torch.int8 if x_active_mask is None else x_active_mask.dtype, device=hidden_states.device
        )
    )
    programs = min(triton.cdiv(rows * _HIDDEN_SIZE, _COPY_BLOCK), _A2_VECTOR_CORES)
    _prepare_local_partial_kernel[(programs,)](
        hidden_states,
        topk_ids,
        topk_weights,
        hidden_states if x_active_mask is None else x_active_mask,
        padded_hidden,
        padded_ids,
        padded_weights,
        padded_hidden if padded_active is None else padded_active,
        tokens,
        HAS_ACTIVE=x_active_mask is not None,
        PREAPPLY_ACTIVE=preapply_active_mask,
        HIDDEN=_HIDDEN_SIZE,
        TOP_K=_TOP_K,
        EXPERTS=_NUM_EXPERTS,
        DUMMY_ROWS=_DUMMY_ROWS,
        COPY_BLOCK=_COPY_BLOCK,
        ROUTING_BLOCK=_ROUTING_BLOCK,
    )
    return padded_hidden, padded_ids, padded_weights, padded_active, tokens
