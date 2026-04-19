# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

_SLOT_MAPPING_IMPL = os.environ.get("VLLM_SLOT_MAPPING_IMPL", "triton")


def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    if _SLOT_MAPPING_IMPL == "ascendc":
        _update_ascendc(
            num_computed_tokens, num_accepted_tokens,
            prev_positions, valid_sampled_token_count,
            prev_num_draft_tokens, cpu_num_computed_tokens,
        )
    else:
        _update_torch(
            num_computed_tokens, num_accepted_tokens,
            prev_positions, valid_sampled_token_count,
            prev_num_draft_tokens, cpu_num_computed_tokens,
        )


def _update_torch(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    # Clamp because prev_positions can be -1 for new requests
    gather_indices = prev_positions.clamp(min=0)

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(torch.where(participating, corrected, cpu_num_computed_tokens))
    num_accepted_tokens.copy_(torch.where(participating, valid_counts, num_accepted_tokens))


def _update_ascendc(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    n = prev_positions.shape[0]
    # AscendC kernel works with int32 and modifies outputs in-place.
    # Cast inputs (read-only) as needed; outputs must already be int32.
    prev_pos_i32 = prev_positions[:n].to(torch.int32)
    valid_i32 = valid_sampled_token_count.to(torch.int32)
    draft_i32 = prev_num_draft_tokens.to(torch.int32)
    cpu_val_i32 = cpu_num_computed_tokens[:n].to(torch.int32)

    # Create int32 output buffers for in-place kernel modification
    out_computed = num_computed_tokens[:n].to(torch.int32).contiguous()
    out_accepted = num_accepted_tokens[:n].to(torch.int32).contiguous()

    torch.ops._C_ascend.npu_update_num_computed_tokens(
        prev_pos_i32,
        valid_i32,
        draft_i32,
        cpu_val_i32,
        out_computed,
        out_accepted,
    )

    # Copy results back to original tensors (handles dtype conversion)
    num_computed_tokens[:n].copy_(out_computed)
    num_accepted_tokens[:n].copy_(out_accepted)
