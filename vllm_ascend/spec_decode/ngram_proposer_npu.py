import os

import torch
import torch_npu

from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU
from vllm.v1.worker.gpu_input_batch import InputBatch

# Set VLLM_NGRAM_USE_ASCENDC=0 to fall back to PyTorch tensor ops for A/B perf comparison
_USE_ASCENDC = os.environ.get("VLLM_NGRAM_USE_ASCENDC", "1") != "0"


class AscendNgramProposerNPU(NgramProposerGPU):
    def __init__(self, vllm_config, device: torch.device, runner):
        self.runner = runner
        super().__init__(vllm_config, device=device)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        pass

    def update_token_ids_ngram(
        self,
        sampled_token_ids: torch.Tensor | list[list[int]],
        gpu_input_batch: InputBatch,
        token_ids_gpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _USE_ASCENDC:
            return super().update_token_ids_ngram(
                sampled_token_ids, gpu_input_batch, token_ids_gpu,
                num_tokens_no_spec, discard_request_mask,
            )

        num_reqs = gpu_input_batch.num_reqs

        if isinstance(sampled_token_ids, list):
            max_len = max(
                (len(sublist) for sublist in sampled_token_ids),
                default=0,
            )
            max_len = max(max_len, 1)
            padded_list = [
                sublist + [-1] * (max_len - len(sublist))
                for sublist in sampled_token_ids
            ]
            sampled_token_ids = torch.tensor(
                padded_list, dtype=torch.int32, device=self.device
            )

        # Slice to active requests and call AscendC fused operator
        next_token_ids, valid_count, valid_sampled = (
            torch.ops._C_ascend.npu_update_token_ids_ngram(
                sampled_token_ids,
                token_ids_gpu[:num_reqs],
                num_tokens_no_spec[:num_reqs],
                discard_request_mask[:num_reqs].to(torch.int8),
                gpu_input_batch.vocab_size,
            )
        )
        return next_token_ids, valid_count, valid_sampled

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len]
        valid_sampled_token_ids_gpu: torch.Tensor,  # [batch_size, num_spec_tokens + 1]
        valid_sampled_tokens_count: torch.Tensor,  # [batch_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _USE_ASCENDC:
            return super().propose(
                num_tokens_no_spec, token_ids_gpu,
                valid_sampled_token_ids_gpu, valid_sampled_tokens_count,
            )

        # Scatter newly sampled tokens into token_ids_gpu (same as parent)
        batch_size = num_tokens_no_spec.shape[0]
        max_seq_len = token_ids_gpu.shape[1]
        max_new_tokens = valid_sampled_token_ids_gpu.shape[1]

        offsets = torch.arange(max_new_tokens, device=self.device)
        write_positions = num_tokens_no_spec.unsqueeze(1) + offsets.unsqueeze(0)
        valid_write_mask = offsets.unsqueeze(0) < valid_sampled_tokens_count.unsqueeze(1)
        in_bounds = write_positions < max_seq_len
        scatter_mask = (
            valid_write_mask
            & (valid_sampled_token_ids_gpu != -1)
            & in_bounds
        )

        write_positions_long = write_positions.clamp(max=max_seq_len - 1).long()
        existing_values = token_ids_gpu.gather(1, write_positions_long)

        tokens_cast = valid_sampled_token_ids_gpu.to(token_ids_gpu.dtype)
        tokens_to_scatter = torch.where(scatter_mask, tokens_cast, existing_values)
        token_ids_gpu.scatter_(1, write_positions_long, tokens_to_scatter)

        # Compute updated lengths and validity mask
        num_tokens_tmp = num_tokens_no_spec + valid_sampled_tokens_count
        sampled_flags = valid_sampled_tokens_count > 0
        combined_mask = sampled_flags & (num_tokens_tmp >= self.min_n)

        # Call AscendC fused ngram match + extract operator
        draft_tokens, num_valid_draft_tokens = (
            torch.ops._C_ascend.npu_ngram_match_extract(
                token_ids_gpu,
                num_tokens_tmp.to(torch.int32),
                combined_mask.to(torch.int8),
                self.min_n,
                self.max_n,
                self.k,
            )
        )

        return draft_tokens, num_valid_draft_tokens


def copy_num_valid_draft_tokens_npu(
    num_valid_draft_tokens_cpu: torch.Tensor,
    num_valid_draft_tokens_copy_stream,
    num_valid_draft_tokens_event,
    num_valid_draft_tokens: torch.Tensor | None,
    batch_size: int,
) -> None:
    """Async D2H copy using NPU native stream API."""
    if num_valid_draft_tokens is None:
        return

    num_reqs_to_copy = min(batch_size, num_valid_draft_tokens.shape[0])
    if num_reqs_to_copy <= 0:
        return

    default_stream = torch_npu.npu.current_stream()
    with torch_npu.npu.stream(num_valid_draft_tokens_copy_stream):
        num_valid_draft_tokens_copy_stream.wait_stream(default_stream)
        num_valid_draft_tokens_cpu[:num_reqs_to_copy].copy_(
            num_valid_draft_tokens[:num_reqs_to_copy], non_blocking=True
        )
        num_valid_draft_tokens_event.record()
