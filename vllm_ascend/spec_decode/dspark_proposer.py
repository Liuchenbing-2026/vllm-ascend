from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class AscendDsparkProposer(AscendDflashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            runner=runner,
        )

        # Initialize and establish static address for graph mode
        blk = 1 + self.num_speculative_tokens
        checkpoint_block_size = getattr(
            self.speculative_config.draft_model_config.hf_config,
            "block_size",
            None,
        )
        if checkpoint_block_size is not None and checkpoint_block_size != blk:
            raise ValueError(
                "DSpark query block must contain one anchor plus all speculative "
                f"tokens: checkpoint block_size={checkpoint_block_size}, "
                f"num_speculative_tokens={self.num_speculative_tokens}"
            )
        self._dspark_query_tokens_per_req = blk
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
    ) -> dict[str, Any]:
        num_context = self._dflash_num_context
        context_states = self._dflash_hidden_states[:num_context]
        context_positions = self._context_positions_buffer[:num_context]
        context_slots = self._context_slot_mapping_buffer[:num_context]

        # Rejected verifier rows have no valid draft-cache destination. The
        # direct Ascend cache-update hook does not safely ignore negative
        # slots, so remove those rows before writing context K/V.
        valid_context = context_slots >= 0

        self.model.precompute_and_store_context_kv(
            context_states[valid_context].contiguous(),
            context_positions[valid_context].contiguous(),
            context_slots[valid_context].to(torch.int32).contiguous(),
        )

        return {
            "input_ids": self.input_ids[:num_input_tokens],
            "positions": self.positions[:num_input_tokens],
            "inputs_embeds": None,
        }

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # Dspark cross-attention: context K/V from target hidden states,
        # Q from query embeddings (next token + mask tokens).

        batch_size = cad.num_reqs

        # Position 0 is the anchor and positions 1..N are supervised draft slots.
        num_query_per_req = self._dspark_query_tokens_per_req
        num_query_total = batch_size * num_query_per_req

        # Newly added hidden_states, need to convert to KV Cache
        num_context = target_token_ids.shape[0]
        self._dflash_num_context = num_context
        if self.use_cuda_graph:
            self._dflash_hidden_states[: self.max_query_tokens].zero_()
            self._context_positions_buffer[: self.max_query_tokens].zero_()
            self._context_slot_mapping_buffer[: self.max_query_tokens].fill_(-1)
            self.input_ids.fill_(self.parallel_drafting_token_id)
            self.positions.zero_()
            self._slot_mapping_buffer.fill_(-1)
        else:
            self._context_positions_buffer[:num_context].fill_(-1)
            self._context_slot_mapping_buffer[:num_context].fill_(-1)
            self.input_ids[:num_query_total].fill_(self.parallel_drafting_token_id)
            self.positions[:num_query_total].zero_()
            self._slot_mapping_buffer[:num_query_total].fill_(-1)
        self._dflash_hidden_states[:num_context] = target_hidden_states

        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        if n < self._dspark_seed_buffer.shape[0]:
            self._dspark_seed_buffer[n:].fill_(0)

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        has_num_rejected = num_rejected_tokens_gpu is not None
        old_query_start_loc = cad.query_start_loc

        if target_positions.dim() != 1:
            raise NotImplementedError("DSpark proposer only supports 1-D positions")

        # Recompute context and query slots from absolute positions. Rejected
        # context rows are marked invalid and filtered before the cache write.
        copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            context_slot_mapping_ptr=cad.slot_mapping,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=old_query_start_loc,
            seq_lens_ptr=cad.seq_lens,
            num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.kernel_block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            batch_size=batch_size,
            HAS_NUM_REJECTED=has_num_rejected,
            SAMPLE_FROM_ANCHOR=False,
            RECOMPUTE_CONTEXT_SLOTS=True,
        )

        # Build attn_metadata
        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        last_position_indices = old_query_start_loc[1:] - 1
        if has_num_rejected:
            last_position_indices = last_position_indices - num_rejected_tokens_gpu
        last_position_indices = torch.maximum(
            last_position_indices,
            old_query_start_loc[:-1],
        )
        last_valid_positions = target_positions.index_select(
            0,
            last_position_indices.long(),
        ).to(torch.int32)
        new_seq_lens = last_valid_positions + num_query_per_req + 1

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = new_seq_lens
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)
        if getattr(cad, "_seq_lens_cpu", None) is not None:
            cad._seq_lens_cpu = new_seq_lens.detach().cpu()
        if getattr(cad, "seq_lens_cpu", None) is not None:
            cad.seq_lens_cpu = new_seq_lens.detach().cpu()

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.positions = self.positions[:num_query_total]
        if getattr(cad, "positions_cpu", None) is not None:
            cad.positions_cpu = cad.positions.detach().cpu()
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        # DSpark uses one anchor query plus one query per speculative token.
        num_query_per_req = self._dspark_query_tokens_per_req
        num_query_total = num_reqs * num_query_per_req
        num_query_tokens = min(num_query_total if num_reqs > 0 else num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        self.token_indices_to_sample.fill_(0)

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=[],
                    num_tokens=num_input_tokens,
                )
