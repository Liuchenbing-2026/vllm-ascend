# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen3_dflash2 import CandidateSelector, _grouped_conv, _score_edges


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int) -> None:
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)

    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (base[tap] + delta[:, position, tap, :, None]) * hidden_blocks[:, position - tap]
    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


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

    actual = _score_edges(predecessors, successors, candidate_ids, unary, hidden, anchors, top_k)

    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = anchors[:, None].expand(-1, top_k) if step == 0 else candidate_ids[:, step - 1]
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )
    torch.testing.assert_close(actual, expected)


def test_candidate_selector_projects_flattened_hidden_states() -> None:
    """Regression: NPU unquantized_gemm's Dynamo fake is 2D-only, so the selector
    projects [num_reqs * num_steps, H] and restores the leading dimensions."""
    torch.manual_seed(2)
    batch, steps, top_k, hidden_size, rank = 2, 4, 3, 7, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden_states = torch.randn(batch, steps, hidden_size)
    anchors = torch.randint(vocab, (batch,))

    def project(flat_hidden_states):
        assert flat_hidden_states.shape == (batch * steps, hidden_size)
        return flat_hidden_states[:, :rank]

    selector = SimpleNamespace(
        hidden_projection=project,
        predecessor_codebook=predecessors,
        successor_codebook=successors,
        top_k=top_k,
    )

    actual = CandidateSelector.forward(selector, candidate_ids, unary, hidden_states, anchors)

    expected = _score_edges(predecessors, successors, candidate_ids, unary, hidden_states[..., :rank], anchors, top_k)
    torch.testing.assert_close(actual, expected)
