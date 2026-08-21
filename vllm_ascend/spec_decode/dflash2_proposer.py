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
selector: top-K per slot, adjacent-transition scoring, and a path walk over
the precomputed scores.

Implementation notes:

- ``draft_sample_method=greedy`` uses the deterministic argmax walk and does
  not allocate a dense proposal distribution. Explicit ``probabilistic`` mode
  uses inverse-CDF sampling and retains the realized sparse proposal
  distribution for lossless rejection sampling.
- The walk runs as ``num_speculative_tokens`` small torch ops instead of one
  Triton program per request (upstream vllm#52816). Acceptable for v1;
  a triton-ascend kernel is the perf follow-up.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import GREEDY_TEMPERATURE
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.models._dflash2_math import (
    densify_selector_probs,
    selector_sample_path,
    selector_walk,
)
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.utils import global_stream, npu_stream_switch


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
        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method == "standard"
            and self.speculative_config.draft_sample_method == "probabilistic"
        )
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
        self._selector_temperatures: torch.Tensor | None = None
        self._selector_greedy_mask: torch.Tensor | None = None
        self._selector_uniforms: torch.Tensor | None = None
        self._selector_candidate_ids: torch.Tensor | None = None
        self._selector_q_rows: torch.Tensor | None = None
        if self._enable_probabilistic_draft_probs:
            self._selector_temperatures = torch.ones(
                self.max_batch_size,
                dtype=torch.float32,
                device=device,
            )
            self._selector_greedy_mask = torch.ones(
                self.max_batch_size,
                dtype=torch.bool,
                device=device,
            )
            self._selector_uniforms = torch.zeros(
                (self.max_batch_size, self.num_speculative_tokens),
                dtype=torch.float32,
                device=device,
            )
            self._selector_candidate_ids = torch.empty(
                (
                    self.max_batch_size,
                    self.num_speculative_tokens,
                    self.selector_top_k,
                ),
                dtype=torch.int64,
                device=device,
            )
            self._selector_q_rows = torch.empty(
                (
                    self.max_batch_size,
                    self.num_speculative_tokens,
                    self.selector_top_k,
                ),
                dtype=torch.float32,
                device=device,
            )
        self._draft_probs: torch.Tensor | None = None
        self._active_draft_prob_candidate_ids: torch.Tensor | None = None
        self._selector_req_ids: tuple[str, ...] = ()

    def _stage_selector_sampling(
        self,
        sampling_metadata: SamplingMetadata,
        batch_size: int,
    ) -> None:
        """Refresh the persistent sampling inputs before draft graph replay."""
        assert self._enable_probabilistic_draft_probs
        assert self._selector_temperatures is not None
        assert self._selector_greedy_mask is not None
        assert self._selector_uniforms is not None
        temperatures = sampling_metadata.temperature
        if temperatures is None:
            self._selector_greedy_mask[:batch_size].fill_(True)
            self._selector_temperatures[:batch_size].fill_(1.0)
        else:
            temperatures = temperatures[:batch_size]
            greedy_mask = temperatures == GREEDY_TEMPERATURE
            self._selector_greedy_mask[:batch_size].copy_(greedy_mask)
            self._selector_temperatures[:batch_size].copy_(temperatures)
            self._selector_temperatures[:batch_size].masked_fill_(greedy_mask, 1.0)
            self._selector_temperatures[:batch_size].clamp_min_(1e-5)
        generators = sampling_metadata.generators
        # The uniforms buffer is reused by every captured draft-graph replay.
        # Order both sides of that reuse: the RNG stream must not overwrite it
        # while the previous replay is still reading it, and the next replay
        # must not read it until the refresh has finished.
        current_stream = torch.npu.current_stream()
        with npu_stream_switch(global_stream()):
            global_stream().wait_stream(current_stream)
            if len(generators) != batch_size:
                self._selector_uniforms[:batch_size].uniform_()
            for request_index, generator in generators.items():
                if request_index < batch_size:
                    self._selector_uniforms[request_index].uniform_(generator=generator)
        current_stream.wait_stream(global_stream())

    def _propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.batch_size()
        if self._enable_probabilistic_draft_probs:
            self._stage_selector_sampling(sampling_metadata, batch_size)
        draft_token_ids = super()._propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            target_model_batch_desc=target_model_batch_desc,
            sampling_metadata=sampling_metadata,
            **kwargs,
        )
        if self._enable_probabilistic_draft_probs:
            assert self.runner is not None
            self._selector_req_ids = tuple(self.runner.input_batch.req_ids[:batch_size])
        return draft_token_ids

    def _sample_selector_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Walk ``[B, steps, K, K]`` scores and retain the realized q rows.

        ``scores[b, s, p, c]`` is the transition score from predecessor
        candidate ``p`` (anchor for the first step) to candidate ``c``.
        Greedy rows take the argmax; random rows sample by inverse CDF.
        """
        if not self._enable_probabilistic_draft_probs:
            return selector_walk(
                candidate_ids[:batch_size],
                scores[:batch_size],
                out=self._selector_tokens[:batch_size],
            )

        assert self._selector_uniforms is not None
        assert self._selector_temperatures is not None
        assert self._selector_greedy_mask is not None
        assert self._selector_candidate_ids is not None
        assert self._selector_q_rows is not None
        tokens, q_rows = selector_sample_path(
            candidate_ids[:batch_size],
            scores[:batch_size],
            self._selector_uniforms[:batch_size],
            self._selector_temperatures[:batch_size],
            self._selector_greedy_mask[:batch_size],
        )
        self._selector_tokens[:batch_size].copy_(tokens)
        self._selector_candidate_ids[:batch_size].copy_(candidate_ids[:batch_size])
        self._selector_q_rows[:batch_size].copy_(q_rows)
        return self._selector_tokens[:batch_size]

    def prepare_draft_probs(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> torch.Tensor | None:
        """Densify the realized top-K q rows in verification token order."""
        if not self._enable_probabilistic_draft_probs:
            return None

        assert self._selector_candidate_ids is not None
        assert self._selector_q_rows is not None
        batch_size = len(spec_decode_metadata.num_draft_tokens)
        lengths = spec_decode_metadata.num_draft_tokens
        num_tokens = int(spec_decode_metadata.draft_token_ids.shape[0])
        if sum(lengths) != num_tokens:
            raise RuntimeError(
                "DFlash2 draft-token counts do not match verification tokens: "
                f"counts={sum(lengths)}, tokens={num_tokens}"
            )

        vocab_size = int(self.model.model.candidate_selector.predecessor_codebook.shape[0])
        capacity = self.vllm_config.scheduler_config.max_num_seqs * self.num_speculative_tokens
        if num_tokens > capacity:
            raise RuntimeError(
                "DFlash2 verification batch exceeds the proposal probability buffer: "
                f"tokens={num_tokens}, capacity={capacity}"
            )
        if self._draft_probs is None:
            self._draft_probs = torch.zeros(
                (capacity, vocab_size),
                dtype=torch.float32,
                device=self._selector_candidate_ids.device,
            )

        assert self.runner is not None
        current_req_ids = tuple(self.runner.input_batch.req_ids[:batch_size])
        full_stable_batch = current_req_ids == self._selector_req_ids[:batch_size] and all(
            length == self.num_speculative_tokens for length in lengths
        )
        if full_stable_batch:
            selected_candidate_ids = self._selector_candidate_ids[:batch_size].flatten(0, 1)
            selected_q_rows = self._selector_q_rows[:batch_size].flatten(0, 1)
        else:
            previous_rows = {req_id: row for row, req_id in enumerate(self._selector_req_ids)}
            row_indices: list[int] = []
            step_indices: list[int] = []
            for req_id, length in zip(current_req_ids, lengths):
                if not 0 <= length <= self.num_speculative_tokens:
                    raise RuntimeError(f"Invalid DFlash2 draft length: request={req_id}, length={length}")
                if length == 0:
                    continue
                previous_row = previous_rows.get(req_id)
                if previous_row is None:
                    raise RuntimeError(f"DFlash2 verification has no proposal state for request {req_id}")
                row_indices.extend([previous_row] * length)
                step_indices.extend(range(length))
            device = self._selector_candidate_ids.device
            row_index = torch.tensor(row_indices, dtype=torch.int64, device=device)
            step_index = torch.tensor(step_indices, dtype=torch.int64, device=device)
            selected_candidate_ids = self._selector_candidate_ids[row_index, step_index]
            selected_q_rows = self._selector_q_rows[row_index, step_index]

        draft_probs, candidate_ids = densify_selector_probs(
            selected_candidate_ids,
            selected_q_rows,
            self._draft_probs,
        )
        if int(draft_probs.shape[0]) != num_tokens:
            raise RuntimeError(
                "DFlash2 proposal probability rows do not match verification tokens: "
                f"rows={draft_probs.shape[0]}, tokens={num_tokens}"
            )
        self._active_draft_prob_candidate_ids = candidate_ids
        return draft_probs

    def clear_draft_probs(self, draft_probs: torch.Tensor) -> None:
        """Restore the reusable dense proposal buffer to all zeros."""
        candidate_ids = self._active_draft_prob_candidate_ids
        if candidate_ids is not None:
            draft_probs.scatter_(1, candidate_ids, 0.0)
            self._active_draft_prob_candidate_ids = None

    def _sample_parallel_draft_tokens(self, sample_hidden_states: torch.Tensor) -> torch.Tensor:
        """DFlash2 selector tail over the parallel draft block.

        Args:
            sample_hidden_states: [batch * num_speculative_tokens, hidden]
                hidden states of the mask query positions.

        Returns:
            [batch, num_speculative_tokens] draft token ids (target vocab).
        """
        num_spec = self.num_speculative_tokens
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
        return self._sample_selector_path(candidate_ids, scores, batch_size)


__all__ = ["AscendDflash2Proposer"]
