"""CPU reference tests for the DFlash2 model-side math (upstream vllm#52816)."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.models._dflash2_math import grouped_conv, score_edges, selector_walk


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


def testgrouped_conv_is_bit_deterministic() -> None:
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
