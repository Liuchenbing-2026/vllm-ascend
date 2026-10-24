# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

_HIDDEN_SIZE = 2048
_TOP_K = 8
_NUM_EXPERTS = 256
_MAX_TOKENS = 8192


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
    """Prepare full A2 BF16 local-partial inputs without sentinel rows.

    CANN accepts zero-row local experts, so appending one sentinel per expert
    only adds copies. Hidden states stay zero-copy. Routing IDs always own
    independent storage because CANN may apply the active mask in place.
    Probabilities are widened to FP32 for CANN.

    With ``preapply_active_mask``, inactive real routes receive expert ID 256
    in the cloned IDs and the returned mask is None. Otherwise the returned
    active mask has the INT8 dtype required by CANN.
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

    topk_ids = topk_ids.clone()
    if preapply_active_mask:
        if x_active_mask is not None:
            topk_ids.masked_fill_(~x_active_mask.bool().unsqueeze(-1), _NUM_EXPERTS)
        active_mask = None
    elif x_active_mask is None:
        active_mask = torch.ones(tokens, dtype=torch.int8, device=hidden_states.device)
    else:
        active_mask = x_active_mask.to(torch.int8)
    return hidden_states, topk_ids, topk_weights.to(torch.float32), active_mask, tokens
