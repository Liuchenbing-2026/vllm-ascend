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

from types import SimpleNamespace

import torch

from vllm_ascend.models._dflash2_math import selector_walk
from vllm_ascend.spec_decode.dflash2_proposer import AscendDflash2Proposer


def test_greedy_draft_uses_argmax_walk_without_proposal_probs() -> None:
    proposer = AscendDflash2Proposer.__new__(AscendDflash2Proposer)
    proposer._enable_probabilistic_draft_probs = False
    proposer._selector_tokens = torch.empty(2, 3, dtype=torch.int64)

    candidate_ids = torch.tensor(
        [
            [[3, 4], [5, 6], [7, 8]],
            [[13, 14], [15, 16], [17, 18]],
        ],
        dtype=torch.int64,
    )
    scores = torch.randn(2, 3, 2, 2)

    actual = proposer._sample_selector_path(candidate_ids, scores, batch_size=2)

    assert actual.data_ptr() == proposer._selector_tokens.data_ptr()
    assert torch.equal(actual, selector_walk(candidate_ids, scores))
    assert proposer.prepare_draft_probs(None) is None  # type: ignore[arg-type]


def test_prepare_draft_probs_tracks_reordered_requests() -> None:
    proposer = AscendDflash2Proposer.__new__(AscendDflash2Proposer)
    proposer._enable_probabilistic_draft_probs = True
    proposer.num_speculative_tokens = 2
    proposer._selector_req_ids = ("request-a", "request-b")
    proposer._selector_candidate_ids = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=torch.int64,
    )
    proposer._selector_q_rows = torch.tensor(
        [
            [[0.25, 0.75], [0.60, 0.40]],
            [[0.10, 0.90], [0.80, 0.20]],
        ],
    )
    proposer._draft_probs = None
    proposer._active_draft_prob_candidate_ids = None
    proposer.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))
    proposer.model = SimpleNamespace(
        model=SimpleNamespace(candidate_selector=SimpleNamespace(predecessor_codebook=torch.empty(9, 1)))
    )
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["request-b", "request-a"]))
    metadata = SimpleNamespace(
        num_draft_tokens=[1, 2],
        draft_token_ids=torch.empty(3, dtype=torch.int64),
    )

    draft_probs = proposer.prepare_draft_probs(metadata)

    assert draft_probs is not None
    torch.testing.assert_close(draft_probs[0, [5, 6]], torch.tensor([0.10, 0.90]))
    torch.testing.assert_close(draft_probs[1, [1, 2]], torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(draft_probs[2, [3, 4]], torch.tensor([0.60, 0.40]))
    assert torch.count_nonzero(draft_probs).item() == 6

    proposer.clear_draft_probs(draft_probs)

    assert torch.count_nonzero(draft_probs).item() == 0
