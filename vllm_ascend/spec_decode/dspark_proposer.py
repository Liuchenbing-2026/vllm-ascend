# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Clean-room DSpark proposer for vLLM-Ascend.
#
# DSpark is semi-autoregressive: it reuses the DFlash parallel backbone to
# produce base logits for a whole block of draft positions in one pass, then a
# low-rank Markov head refines each position sequentially. Only the sampling
# tail differs from DFlash, so this proposer inherits the DFlash input build,
# context-KV precompute, and graph machinery and overrides just the tail.
from typing import Any

import torch
from vllm.config import VllmConfig

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class AscendDsparkProposer(AscendDflashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner=runner)
        # Internally the backbone is identical to DFlash; reuse every base-class
        # branch keyed on "dflash". External routing keeps method == "dspark".
        self.method = "dflash"

        # The base inflates hidden_size by hc_mult (for MTP's pre-hc residual).
        # DSpark consumes aux (12288) and combines it to main_x (hidden=4096),
        # which is what the context-KV buffers carry, so restore the true width
        # and rebuild the affected buffers.
        draft_hidden = (
            vllm_config.speculative_config.draft_model_config.hf_config.hidden_size
        )
        if self.hidden_size != draft_hidden:
            self.hidden_size = draft_hidden
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, draft_hidden), dtype=self.dtype, device=device
            )
            self._dflash_hidden_states = torch.zeros(
                (self.max_num_tokens, draft_hidden), dtype=self.dtype, device=device
            )

        # Per-request seed for the Markov recurrence (the bonus / last accepted
        # token, in target-vocab ids). Captured each step in set_inputs_first_pass.
        self._dspark_anchor_ids: torch.Tensor | None = None

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        self._dspark_anchor_ids = next_token_ids
        return super().set_inputs_first_pass(
            target_token_ids, next_token_ids, *args, **kwargs
        )

    def _sample_sequential(
        self,
        sample_hidden_states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Serial Markov refinement over the block's base logits.

        Args:
            sample_hidden_states: ``[batch_size * num_spec, hidden]`` head hidden
                states, ordered (request, step).
            batch_size: number of requests.

        Returns:
            ``[batch_size, num_speculative_tokens]`` draft token ids in the
            target vocabulary.
        """
        n = self.num_speculative_tokens
        hidden = sample_hidden_states.view(batch_size * n, -1)
        base_logits = self.model.compute_draft_logits(hidden).view(batch_size, n, -1)

        prev_ids = self._dspark_anchor_ids[:batch_size]
        draft_ids = torch.empty(
            batch_size, n, dtype=torch.long, device=base_logits.device
        )
        for step in range(n):
            bias = self.model.markov_bias(self.model.markov_embed(prev_ids))
            logits = base_logits[:, step, :] + bias
            sampled = self.model.map_draft_to_target(logits.argmax(dim=-1))
            draft_ids[:, step] = sampled
            prev_ids = sampled
        return draft_ids

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        # DFlash-style single backbone pass: precompute context K/V, then run
        # the non-causal forward over the query block.
        model_kwargs: dict[str, Any] = self.build_model_inputs_first_pass(
            num_input_tokens
        )
        last_hidden_states = self.model(**model_kwargs)

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        return self._sample_sequential(sample_hidden_states, batch_size)
