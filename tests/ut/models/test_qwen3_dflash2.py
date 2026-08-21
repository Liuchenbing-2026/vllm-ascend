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

"""CPU reference tests for the DFlash2 model-side math (upstream vllm#52816)."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.models._dflash2_math import (
    densify_selector_probs,
    grouped_conv,
    score_edges,
    selector_sample_path,
    selector_walk,
)


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int) -> None:
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (base[tap] + delta[:, position, tap, :, None]) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_grouped_conv_is_bit_deterministic() -> None:
    torch.manual_seed(3)
    hidden = torch.randn(16, 8)
    delta = torch.randn(16, 2, 2)
    base = torch.randn(2, 8)
    first = grouped_conv(hidden, delta, base, block_size=8, num_groups=2, group_size=4, taps=2)
    for _ in range(4):
        again = grouped_conv(hidden, delta, base, block_size=8, num_groups=2, group_size=4, taps=2)
        assert torch.equal(first, again)


def test_grouped_conv_two_tap_cuts_block_boundary() -> None:
    # Block of 4: only position 0 of each block loses the predecessor tap;
    # positions 1..3 keep the intra-block dependency.
    hidden = torch.randn(8, 4)  # two blocks
    delta = torch.zeros(8, 2, 1)
    base = torch.zeros(2, 4)
    base[1] = 1.0  # only the predecessor tap
    out = grouped_conv(hidden, delta, base, block_size=4, num_groups=1, group_size=4, taps=2)
    assert torch.equal(out[0], torch.zeros(4))  # block 0 head: boundary cut
    assert torch.equal(out[1:4], hidden[0:3])  # intra-block predecessors kept
    assert torch.equal(out[4], torch.zeros(4))  # block 1 head: boundary cut
    assert torch.equal(out[5:8], hidden[4:7])


def test_selector_edges_match_sequential_reference() -> None:
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = anchors[:, None].expand(-1, top_k) if step == 0 else candidate_ids[:, step - 1]
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def test_score_edges_is_bit_deterministic() -> None:
    torch.manual_seed(5)
    vocab, rank, top_k = 11, 4, 3
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (2, 3, top_k))
    unary = torch.randn(2, 3, top_k)
    hidden = torch.randn(2, 3, rank)
    anchors = torch.randint(vocab, (2,))
    first = score_edges(predecessors, successors, candidate_ids, unary, hidden, anchors, top_k)
    for _ in range(4):
        again = score_edges(predecessors, successors, candidate_ids, unary, hidden, anchors, top_k)
        assert torch.equal(first, again)


def _brute_walk(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    batch_size, num_steps, top_k, _ = scores.shape
    tokens = torch.empty((batch_size, num_steps), dtype=torch.int64)
    previous = torch.zeros(batch_size, dtype=torch.long)
    for step in range(num_steps):
        step_scores = scores[:, step]
        for b in range(batch_size):
            row = step_scores[b, previous[b]]
            index = int(row.argmax().item())
            tokens[b, step] = candidate_ids[b, step, index]
            previous[b] = index
    return tokens


def test_selector_walk_matches_sequential_reference() -> None:
    torch.manual_seed(2)
    batch, steps, top_k = 3, 5, 4
    candidate_ids = torch.randint(30, (batch, steps, top_k))
    scores = torch.randn(batch, steps, top_k, top_k)
    actual = selector_walk(candidate_ids, scores)
    expected = _brute_walk(candidate_ids, scores)
    assert torch.equal(actual, expected)


def test_selector_walk_tie_breaks_to_smallest_index() -> None:
    candidate_ids = torch.tensor([[[7, 3, 9, 1]]], dtype=torch.int64)  # [1,1,4]
    scores = torch.zeros(1, 1, 4, 4)  # all transitions tie
    out = selector_walk(candidate_ids, scores)
    assert out[0, 0].item() == 7  # argmax ties to index 0


def test_selector_walk_is_bit_deterministic() -> None:
    torch.manual_seed(9)
    candidate_ids = torch.randint(30, (2, 4, 3))
    scores = torch.randn(2, 4, 3, 3)
    first = selector_walk(candidate_ids, scores)
    for _ in range(4):
        assert torch.equal(first, selector_walk(candidate_ids, scores))


def test_selector_sample_path_greedy_matches_selector_walk() -> None:
    torch.manual_seed(10)
    batch, steps, top_k = 3, 5, 4
    candidate_ids = torch.randint(30, (batch, steps, top_k))
    scores = torch.randn(batch, steps, top_k, top_k)

    tokens, q_rows = selector_sample_path(
        candidate_ids,
        scores,
        uniforms=torch.rand(batch, steps),
        temperatures=torch.ones(batch),
        greedy_mask=torch.ones(batch, dtype=torch.bool),
    )

    expected = selector_walk(candidate_ids, scores)
    assert torch.equal(tokens, expected)
    selected_indices = q_rows.argmax(dim=-1)
    assert torch.equal(
        candidate_ids.gather(-1, selected_indices.unsqueeze(-1))[:, :, 0],
        tokens,
    )
    assert torch.all((q_rows == 0) | (q_rows == 1))
    assert torch.equal(q_rows.sum(dim=-1), torch.ones(batch, steps))


def test_selector_sample_path_returns_realized_transition_probs() -> None:
    candidate_ids = torch.tensor(
        [[[10, 11], [20, 21], [30, 31]]],
        dtype=torch.int64,
    )
    scores = torch.tensor(
        [
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[3.0, 0.0], [0.0, 3.0]],
            ]
        ]
    )
    uniforms = torch.tensor([[0.80, 0.10, 0.90]])

    tokens, q_rows = selector_sample_path(
        candidate_ids,
        scores,
        uniforms=uniforms,
        temperatures=torch.ones(1),
        greedy_mask=torch.zeros(1, dtype=torch.bool),
    )

    initial = torch.softmax(scores[:, 0, 0], dim=-1)
    first_index = uniforms[:, 0:1].ge(initial.cumsum(dim=-1)).sum(dim=-1)
    second_probs = torch.softmax(scores[:, 1, first_index[0]], dim=-1)
    second_index = uniforms[:, 1:2].ge(second_probs.cumsum(dim=-1)).sum(dim=-1)
    third_probs = torch.softmax(scores[:, 2, second_index[0]], dim=-1)
    third_index = uniforms[:, 2:3].ge(third_probs.cumsum(dim=-1)).sum(dim=-1)
    indices = torch.stack((first_index, second_index, third_index), dim=1)

    assert torch.equal(tokens, candidate_ids.gather(-1, indices.unsqueeze(-1))[:, :, 0])
    torch.testing.assert_close(q_rows[:, 0], initial)
    torch.testing.assert_close(q_rows[:, 1], second_probs)
    torch.testing.assert_close(q_rows[:, 2], third_probs)
    torch.testing.assert_close(q_rows.sum(dim=-1), torch.ones(1, 3))


def test_densify_selector_probs_uses_verification_order() -> None:
    candidate_ids = torch.tensor(
        [[1, 3], [2, 4], [0, 6]],
        dtype=torch.int64,
    )
    q_rows = torch.tensor([[0.25, 0.75], [0.60, 0.40], [0.80, 0.20]])
    out = torch.zeros(6, 8)

    dense, selected_ids = densify_selector_probs(
        candidate_ids,
        q_rows,
        out=out,
    )

    assert torch.equal(selected_ids, torch.tensor([[1, 3], [2, 4], [0, 6]]))
    torch.testing.assert_close(dense.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(dense[0, [1, 3]], q_rows[0])
    torch.testing.assert_close(dense[1, [2, 4]], q_rows[1])
    torch.testing.assert_close(dense[2, [0, 6]], q_rows[2])
    assert torch.count_nonzero(dense).item() == 6
