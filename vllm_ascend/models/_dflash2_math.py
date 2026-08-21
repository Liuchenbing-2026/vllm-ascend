#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Pure-torch DFlash2 math (grouped conv + candidate edge scoring).

Kept free of vllm imports so the CPU reference tests run standalone.
Adapted from vllm-project/vllm PR #52816.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Grouped dynamic depthwise convolution over the draft block.

    ``out[i,c] = sum_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c]`` with taps
    cut at the block boundary. Pure torch ops; runs on NPU as-is.
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


def score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Score every adjacent candidate transition in one parallel einsum."""
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)


def selector_walk(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic greedy walk over ``[B, steps, K, K]`` edge scores.

    ``scores[b, s, p, c]`` is the transition score from predecessor candidate
    ``p`` (any p row at step 0; they all share the anchor predecessor) to
    candidate ``c``. Ties break to the smallest candidate index, so the path
    is a deterministic function of the score bits. Returns ``[B, steps]``
    token ids.
    """
    batch_size, num_steps, top_k, _ = scores.shape
    if out is None:
        out = candidate_ids.new_empty((batch_size, num_steps), dtype=torch.int64)
    row_indices = torch.arange(batch_size, device=scores.device)
    previous = 0
    for step in range(num_steps):
        # Aligned advanced indexing (no cross-product): row_indices selects the
        # batch dim and ``previous`` selects the predecessor slot per row.
        step_scores = scores[row_indices, step, previous]  # [B, K]
        index = step_scores.argmax(dim=-1)  # [B]
        out[:, step] = candidate_ids[:batch_size, step].gather(1, index.unsqueeze(-1)).squeeze(-1)
        previous = index
    return out


def selector_sample_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a selector path and return its realized sparse distribution.

    Random rows use inverse-CDF sampling over the top-K lattice. Greedy rows
    take the argmax path and return a point mass, so one captured graph serves
    mixed greedy and random batches. ``q_rows`` is the proposal distribution
    required by lossless rejection sampling.
    """
    top_k = candidate_ids.shape[-1]
    temperatures = temperatures.view(-1, 1)

    initial_probs = torch.softmax(scores[:, 0, 0].float() / temperatures, dim=-1)
    initial_indices = uniforms[:, :1].ge(initial_probs.cumsum(dim=-1)).sum(dim=-1).clamp_max(top_k - 1)

    transition_probs = torch.softmax(
        scores[:, 1:].float() / temperatures[:, :, None, None],
        dim=-1,
    )
    local_maps = uniforms[:, 1:, None, None].ge(transition_probs.cumsum(dim=-1)).sum(dim=-1).clamp_max(top_k - 1)

    initial_indices = torch.where(
        greedy_mask,
        scores[:, 0, 0].argmax(dim=-1),
        initial_indices,
    )
    local_maps = torch.where(
        greedy_mask[:, None, None],
        scores[:, 1:].argmax(dim=-1),
        local_maps,
    )

    previous = initial_indices
    path = [previous]
    for step in range(scores.shape[1] - 1):
        previous = local_maps[:, step].gather(-1, previous[:, None])[:, 0]
        path.append(previous)
    path_indices = torch.stack(path, dim=1)

    tokens = candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]
    realized_rows = transition_probs.gather(
        2,
        path_indices[:, :-1, None, None].expand(-1, -1, 1, top_k),
    )[:, :, 0]
    q_rows = torch.cat((initial_probs.unsqueeze(1), realized_rows), dim=1)
    q_rows = torch.where(
        greedy_mask[:, None, None],
        F.one_hot(path_indices, top_k).float(),
        q_rows,
    )
    return tokens, q_rows


def densify_selector_probs(
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter sparse top-K q rows into flattened verification order.

    ``candidate_ids`` and ``q_rows`` must already follow the verifier's
    flattened request-major order. This keeps request-state bookkeeping out of
    the tensor-only helper.

    ``out`` must be zero-initialized (or cleared after its previous use). The
    returned candidate ids let the caller clear only the columns it touched.
    """
    dense = out[: candidate_ids.shape[0]]
    dense.scatter_(1, candidate_ids, q_rows)
    return dense, candidate_ids


__all__ = [
    "densify_selector_probs",
    "grouped_conv",
    "score_edges",
    "selector_sample_path",
    "selector_walk",
]
