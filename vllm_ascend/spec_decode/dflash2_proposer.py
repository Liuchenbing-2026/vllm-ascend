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
"""Ascend DFlash2 proposer: DFlash parallel drafting + candidate selector.

Selected for ``method="dflash"`` when the draft architecture is
``DFlash2DraftModel`` (e.g. z-lab/Qwen3.8-27B-DFlash2). The backbone pass is
identical to DFlash; the parallel argmax tail is replaced by the candidate
selector: top-K per slot, adjacent-transition scoring, and a deterministic
greedy walk over the precomputed scores.

Correctness-first v1 notes:

- The walk is the deterministic greedy variant (upstream T=0 path). The
  T>0 Gumbel path walk and the realized-proposal distribution caching
  (upstream ``_sample_path`` / ``_cache_draft_logits``) are deferred to v2;
  until then T>0 requests draft through the same deterministic walk, which
  keeps verification lossless under the existing greedy-draft policy.
- The walk runs as ``num_speculative_tokens`` small torch ops instead of one
  Triton program per request (upstream vllm#52816). Acceptable for v1;
  a triton-ascend kernel is the perf follow-up.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_ascend.models._dflash2_math import selector_walk
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

logger = init_logger(__name__)


class AscendDflash2Proposer(AscendDflashProposer):
    """DFlash2: parallel drafting + lightweight candidate path selector."""

    # llm_base_proposer._run_merged_draft dispatches the sequential tail
    # through _sample_parallel_draft_tokens only when this flag is set. The
    # selector walk is exactly that: a sequential pass over the parallel
    # draft block (name is inherited from the DSpark hook).
    uses_markov_head = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        # The anchor (bonus) token sits at query offset 0 of each request in
        # the expanded input_ids layout ([batch, 1 + num_speculative_tokens]).
        self._anchor_indices = torch.arange(self.max_batch_size, device=device, dtype=torch.int64) * (
            1 + self.num_speculative_tokens
        )
        self._selector_tokens = torch.empty(
            (self.max_batch_size, self.num_speculative_tokens),
            dtype=torch.int64,
            device=device,
        )

    def _selector_walk(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Deterministic greedy walk over ``[B, steps, K, K]`` edge scores.

        ``scores[b, s, p, c]`` is the transition score from predecessor
        candidate ``p`` (anchor for the first step) to candidate ``c``.
        Ties break to the smallest candidate index, so the whole path is a
        deterministic function of the score bits (matches upstream T=0).
        """
        tokens = self._selector_tokens[:batch_size]
        return selector_walk(candidate_ids[:batch_size], scores[:batch_size], out=tokens)

    def _sample_parallel_draft_tokens(self, sample_hidden_states: torch.Tensor) -> torch.Tensor:
        """DFlash2 selector tail over the parallel draft block.

        Args:
            sample_hidden_states: [batch * num_speculative_tokens, hidden]
                hidden states of the mask query positions.

        Returns:
            [batch, num_speculative_tokens] draft token ids (target vocab).
        """
        num_spec = self.num_speculative_tokens
        # sample_hidden_states is [batch * num_spec, hidden]; the batch dim is
        # recovered by floordiv. Under dynamic shapes Dynamo cannot prove the
        # divisibility, so record it before the (B, S, *) views below.
        torch._check(sample_hidden_states.shape[0] % num_spec == 0)
        batch_size = sample_hidden_states.shape[0] // num_spec
        model = self.model

        candidate_ids, unary_logits = model.compute_candidates(sample_hidden_states)
        candidate_ids = candidate_ids.view(batch_size, num_spec, self.selector_top_k)
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_ids[self._anchor_indices[:batch_size]]
        hidden = sample_hidden_states.view(batch_size, num_spec, -1)

        scores = model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )
        return self._selector_walk(candidate_ids, scores, batch_size)


__all__ = ["AscendDflash2Proposer"]
