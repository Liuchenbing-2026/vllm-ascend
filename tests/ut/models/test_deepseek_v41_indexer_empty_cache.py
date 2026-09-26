# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Regression UT for the DeepSeek V4.1 indexer empty compressed-cache check.

The original implementation short-circuited on the host scalar
``source_metadata.max_cache_seq_len == 0``.  During FULL_DECODE_ONLY graph
capture the framework always runs a dummy batch whose compressed cache is
empty, so that host branch was resolved once at capture time and baked into the
graph: every replay then published ``-1`` sparse indices even for long
contexts.  The check therefore has to stay a device-side tensor predicate, so
that the same callable yields different indices for different inputs.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch

from vllm_ascend.models.deepseek_v41 import indexer as indexer_module
from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_indexer(compress_ratio: int = 2, index_topk: int = 4) -> DeepseekV41Indexer:
    indexer = DeepseekV41Indexer.__new__(DeepseekV41Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.compress_ratio = compress_ratio
    indexer.index_topk = index_topk
    return indexer


def _source_metadata(cache_seq_lens):
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        cache_seq_lens=cache_seq_lens,
        cmp_residual=None,
        max_query_len=3,
        qli_metadata=object(),
        block_table=torch.zeros((1, 4), dtype=torch.int32),
    )


class _Run:
    """Result of one stubbed ``select_projected`` call."""

    def __init__(self, selected, candidates, prepared, op_candidates):
        self.selected = selected
        self.candidates = candidates
        self.prepared = prepared
        self.op_candidates = op_candidates


def _run_select_projected(
    cache_seq_lens,
    *,
    is_candidate_source: bool,
    uses_candidate_filter: bool = False,
) -> _Run:
    """Call select_projected with every npu/triton op replaced by a stub."""
    tokens, topk = 2, 4
    indexer = _make_indexer(index_topk=topk)
    query = torch.zeros((tokens, 4, 8), dtype=torch.float16)
    weights = torch.zeros((tokens, 4), dtype=torch.float32)
    positions = torch.arange(tokens, dtype=torch.int32)
    key = torch.zeros((16, 1, 8), dtype=torch.int8)
    key_scale = torch.zeros((16, 1, 1), dtype=torch.float32)
    candidates = torch.full((tokens, 1, topk), 7, dtype=torch.int32)
    prepared = torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk)
    op_selected = torch.zeros((tokens, 1, topk), dtype=torch.int32)
    op_candidates = torch.full((tokens, 1, topk), 9, dtype=torch.int32)

    with (
        patch.object(
            indexer_module,
            "quantize_indexer_query",
            return_value=(query, torch.ones((tokens,), dtype=torch.float32)),
        ),
        patch.object(
            indexer_module,
            "prepare_indexer_indices",
            return_value=prepared,
        ),
        patch.object(indexer_module, "wait_for_device_metadata", return_value=None),
        patch.object(torch.ops, "_C_ascend", create=True) as ascend_ops,
    ):
        ascend_ops.npu_quant_lightning_indexer_v3.return_value = (
            op_selected,
            None,
            op_candidates,
        )
        selected, out_candidates = indexer.select_projected(
            query,
            weights,
            positions,
            (key, key_scale),
            _source_metadata(cache_seq_lens),
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=topk,
            candidate_block_size=16,
            candidates=candidates,
        )
    return _Run(selected, out_candidates, prepared, op_candidates)


class TestEmptyCompressedCache:
    def test_all_zero_cache_lengths_publish_minus_one_indices(self):
        """An empty source plane must still be detected at replay time."""
        run = _run_select_projected(
            torch.zeros((2,), dtype=torch.int32),
            is_candidate_source=True,
        )

        expected = torch.full_like(run.prepared, -1)
        assert torch.equal(run.selected, expected)
        assert torch.equal(run.candidates, torch.full_like(run.op_candidates, -1))

    def test_non_empty_cache_lengths_keep_the_selected_indices(self):
        """The device predicate must not clip the regular long-context path."""
        run = _run_select_projected(
            torch.tensor([6, 6], dtype=torch.int32),
            is_candidate_source=True,
        )

        assert torch.equal(run.selected, run.prepared)
        assert torch.equal(run.candidates, run.op_candidates)

    def test_one_shot_callable_yields_both_outcomes(self):
        """Regression guard: a host branch would freeze one outcome forever.

        Graph capture runs the dummy (empty cache) batch first, so a capture
        time decision would keep the ``-1`` result for every later replay.
        """
        empty = _run_select_projected(
            torch.zeros((2,), dtype=torch.int32),
            is_candidate_source=True,
        )
        non_empty = _run_select_projected(
            torch.tensor([6, 6], dtype=torch.int32),
            is_candidate_source=True,
        )

        assert torch.equal(empty.selected, torch.full_like(empty.prepared, -1))
        assert torch.equal(non_empty.selected, non_empty.prepared)
        assert not torch.equal(empty.selected, non_empty.selected)

    def test_missing_or_empty_cache_lengths_are_ignored(self):
        """Eager paths without a device copy keep the previous behaviour."""
        for cache_seq_lens in (None, torch.zeros((0,), dtype=torch.int32)):
            run = _run_select_projected(
                cache_seq_lens,
                is_candidate_source=True,
            )

            assert torch.equal(run.selected, run.prepared)
            assert torch.equal(run.candidates, run.op_candidates)


def test_indexer_no_longer_branches_on_host_cache_length():
    """The empty-cache decision must not be made on a host scalar again."""
    code = (REPO_ROOT / "vllm_ascend/models/deepseek_v41/indexer.py").read_text()

    assert "source_metadata.max_cache_seq_len" not in code
    compares = {
        ast.unparse(node)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Compare)
    }
    assert "cache_lens.max() == 0" in compares
