import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, ForwardContext
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
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
        self._dspark_context_kv_graph: ACLGraphWrapper | None = None
        self._dspark_context_kv_precomputed = False

    def _use_dspark_context_kv_bucket_graph(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        return (
            self.use_cuda_graph
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and os.getenv(
                "VLLM_ASCEND_ENABLE_GLM_DSPARK_CONTEXT_KV_BUCKET_GRAPH",
                "0",
            ).lower()
            in ("1", "true", "yes", "on")
        )

    def _get_dspark_context_kv_bucket(self, num_context: int) -> int | None:
        if num_context <= 0 or num_context > self.max_query_tokens:
            return None
        bucket_width = self._dspark_query_tokens_per_req
        return min(
            self.max_query_tokens,
            ((num_context + bucket_width - 1) // bucket_width) * bucket_width,
        )

    def _precompute_padded_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
        )
        return context_states

    def _precompute_context_kv_bucket_graph(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        bucket = self._get_dspark_context_kv_bucket(self._dflash_num_context)
        if bucket is None:
            return False

        if self._dspark_context_kv_graph is None:
            self._dspark_context_kv_graph = ACLGraphWrapper(
                self._precompute_padded_context_kv,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )

        query_batch_descriptor = forward_context.batch_descriptor
        forward_context.batch_descriptor = BatchDescriptor(num_tokens=bucket)
        try:
            self._dspark_context_kv_graph(
                self._dflash_hidden_states[:bucket],
                self._context_positions_buffer[:bucket],
                self._context_slot_mapping_buffer[:bucket],
            )
        finally:
            forward_context.batch_descriptor = query_batch_descriptor

        self._dspark_context_kv_precomputed = True
        return True

    def _precompute_live_context_kv(self) -> None:
        num_context = self._dflash_num_context
        context_states = self._dflash_hidden_states[:num_context]
        context_positions = self._context_positions_buffer[:num_context]
        context_slots = self._context_slot_mapping_buffer[:num_context]

        # Rejected verifier rows have no valid draft-cache destination. Keep
        # this filtering outside the query ACL graph so graph replay preserves
        # the same cache semantics as the correct eager path.
        valid_context = context_slots >= 0
        self.model.precompute_and_store_context_kv(
            context_states[valid_context].contiguous(),
            context_positions[valid_context].contiguous(),
            context_slots[valid_context].to(torch.int32).contiguous(),
        )

    def _initialize_graph_padding(
        self,
        num_context: int,
        num_query_total: int,
    ) -> None:
        context_bucket = num_context
        if os.getenv(
            "VLLM_ASCEND_ENABLE_GLM_DSPARK_CONTEXT_KV_BUCKET_GRAPH",
            "0",
        ).lower() in ("1", "true", "yes", "on"):
            context_bucket = (
                self._get_dspark_context_kv_bucket(num_context) or num_context
            )

        if context_bucket > num_context:
            self._dflash_hidden_states[num_context:context_bucket].zero_()
            self._context_positions_buffer[num_context:context_bucket].zero_()
            self._context_slot_mapping_buffer[num_context:context_bucket].fill_(-1)

        if self.max_query_tokens > num_query_total:
            self.input_ids[num_query_total : self.max_query_tokens].fill_(
                self.parallel_drafting_token_id
            )
            self.positions[num_query_total : self.max_query_tokens].zero_()
            self._slot_mapping_buffer[num_query_total : self.max_query_tokens].fill_(-1)

    def prepare_dspark_context_kv_for_graph(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        if not self.use_cuda_graph or forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return False

        if self._use_dspark_context_kv_bucket_graph(forward_context):
            if self._precompute_context_kv_bucket_graph(forward_context):
                return True

        self._precompute_live_context_kv()
        self._dspark_context_kv_precomputed = True
        return True

    def _stabilize_padded_graph_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        actual_num_reqs: int,
        padded_num_reqs: int,
    ) -> None:
        if padded_num_reqs <= actual_num_reqs:
            return

        common_attn_metadata.seq_lens[actual_num_reqs:padded_num_reqs].fill_(1)
        for attr_name in ("_seq_lens_cpu", "seq_lens_cpu"):
            seq_lens_cpu = getattr(common_attn_metadata, attr_name, None)
            if seq_lens_cpu is None:
                continue
            seq_lens_cpu = self._adjust_tensor(seq_lens_cpu, padded_num_reqs)
            seq_lens_cpu[actual_num_reqs:padded_num_reqs].fill_(1)
            setattr(common_attn_metadata, attr_name, seq_lens_cpu)

        if hasattr(common_attn_metadata, "actual_seq_lengths_q"):
            common_attn_metadata.actual_seq_lengths_q = [
                self._dspark_query_tokens_per_req
            ] * padded_num_reqs

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
    ) -> dict[str, Any]:
        if not self._dspark_context_kv_precomputed:
            self._precompute_live_context_kv()

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
            self._initialize_graph_padding(num_context, num_query_total)
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
        # context rows remain invalid and are filtered before the context K/V
        # cache write, matching the eager path exactly.
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
