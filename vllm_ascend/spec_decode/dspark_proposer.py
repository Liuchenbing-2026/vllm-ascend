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

import json
import os
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from vllm_ascend import envs
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.utils import lmhead_tp_enable

_DSPARK_LOGIT_DEBUG_TOP_K = 5


def _rank_for_token_ids(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(1, token_ids.unsqueeze(1))
    return (logits > selected).sum(dim=1) + 1


def _build_logit_debug_record(
    captured: dict[str, torch.Tensor | float],
    target_logits: torch.Tensor,
    num_draft_tokens: Sequence[int],
    draft_token_ids: torch.Tensor,
    record_index: int,
) -> dict[str, Any]:
    base_logits = captured["base_logits"]
    markov_bias = captured["markov_bias"]
    final_logits = captured["final_logits"]
    prev_token_ids = captured["prev_token_ids"]
    proposed_token_ids = captured["proposed_token_ids"]
    assert isinstance(base_logits, torch.Tensor)
    assert isinstance(markov_bias, torch.Tensor)
    assert isinstance(final_logits, torch.Tensor)
    assert isinstance(prev_token_ids, torch.Tensor)
    assert isinstance(proposed_token_ids, torch.Tensor)

    num_reqs, max_spec_len, vocab_size = base_logits.shape
    num_draft_tokens_tensor = torch.as_tensor(
        num_draft_tokens,
        device=base_logits.device,
        dtype=torch.long,
    )
    valid_mask = torch.arange(
        max_spec_len, device=base_logits.device
    ).unsqueeze(0) < num_draft_tokens_tensor.unsqueeze(1)
    base_rows = base_logits[valid_mask].float()
    markov_rows = markov_bias[valid_mask].float()
    final_rows = final_logits[valid_mask].float()
    prev_rows = prev_token_ids[valid_mask].long()
    proposed_rows = proposed_token_ids[valid_mask].long()

    num_tokens = base_rows.shape[0]
    target_rows = target_logits[:num_tokens].float()
    if target_rows.shape != base_rows.shape:
        raise ValueError(
            "DSpark logit debug requires matching full vocab logits: "
            f"target={tuple(target_rows.shape)} draft={tuple(base_rows.shape)}"
        )

    verified_draft_ids = draft_token_ids[:num_tokens].long()
    target_token_ids = target_rows.argmax(dim=-1)
    position_ids = torch.arange(max_spec_len, device=base_logits.device).expand(num_reqs, -1)[valid_mask]
    request_ids = torch.arange(num_reqs, device=base_logits.device).unsqueeze(1).expand(-1, max_spec_len)[valid_mask]

    top_k = min(_DSPARK_LOGIT_DEBUG_TOP_K, vocab_size)
    tensors = {
        "request_id": request_ids,
        "position": position_ids,
        "prev_token_id": prev_rows,
        "draft_token_id": verified_draft_ids,
        "captured_draft_token_id": proposed_rows,
        "target_token_id": target_token_ids,
        "accepted": verified_draft_ids == target_token_ids,
        "base_argmax": base_rows.argmax(dim=-1),
        "markov_argmax": markov_rows.argmax(dim=-1),
        "final_argmax": final_rows.argmax(dim=-1),
        "base_target_rank": _rank_for_token_ids(base_rows, target_token_ids),
        "markov_target_rank": _rank_for_token_ids(markov_rows, target_token_ids),
        "final_target_rank": _rank_for_token_ids(final_rows, target_token_ids),
        "base_draft_rank": _rank_for_token_ids(base_rows, verified_draft_ids),
        "markov_draft_rank": _rank_for_token_ids(markov_rows, verified_draft_ids),
        "base_top_ids": torch.topk(base_rows, top_k, dim=-1).indices,
        "markov_top_ids": torch.topk(markov_rows, top_k, dim=-1).indices,
        "final_top_ids": torch.topk(final_rows, top_k, dim=-1).indices,
        "target_top_ids": torch.topk(target_rows, top_k, dim=-1).indices,
    }
    cpu_values = {name: value.detach().cpu().tolist() for name, value in tensors.items()}
    rows = [
        {name: values[row_index] for name, values in cpu_values.items()}
        for row_index in range(num_tokens)
    ]
    return {
        "record": record_index,
        "num_draft_tokens": num_draft_tokens_tensor.cpu().tolist(),
        "markov_scale": float(captured["markov_scale"]),
        "rows": rows,
    }


class AscendDsparkProposer(AscendDflashProposer):
    """DSpark: DFlash parallel drafting + sequential Markov correction.

    The backbone pass is identical to DFlash (a bonus anchor plus N mask
    queries, non-causal within the block, context K/V precomputed from the
    target's aux hidden states). Sampling replaces the parallel argmax with
    a left-to-right loop that biases each position's base logits with a
    low-rank Markov head conditioned on the previously sampled token.

    Selected for ``method="dflash"`` when the draft architecture is
    ``Qwen3DSparkModel`` (e.g. GLM-5.2 DSpark speculators checkpoints).
    """

    uses_markov_head = True
    supports_glm_draft_aclgraph = True

    def _draft_aclgraph_disabled_reason(self) -> str | None:
        if envs.VLLM_ASCEND_DSPARK_REFERENCE_ATTENTION:
            return "reference-attention diagnostics perform Python/CPU work"
        if envs.VLLM_ASCEND_DSPARK_CAUSAL_DIAG:
            return "causal-attention diagnostics are enabled"
        if (
            envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH
            and envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS > 0
        ):
            return "logit/backbone diagnostics perform CPU snapshots and file I/O"
        if not self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            return "DSpark requires a cudagraph mode containing FULL graphs"
        if self.vllm_config.lora_config is not None:
            return "DSpark draft ACLGraph does not yet support LoRA"
        if self.pcp_size * self.dcp_size > 1:
            return "DSpark draft ACLGraph does not yet support PCP/DCP"
        if lmhead_tp_enable():
            return "DSpark drafting does not support LM-head tensor parallel"
        return None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        # The anchor (bonus) token sits at query offset 0 of each request in
        # the expanded input_ids layout ([batch, 1 + num_speculative_tokens]).
        self._anchor_indices = torch.arange(
            self.max_batch_size, device=device, dtype=torch.int64
        ) * (1 + self.num_speculative_tokens)
        self._markov_scale = envs.VLLM_ASCEND_DSPARK_MARKOV_SCALE
        self._last_logit_debug: dict[str, torch.Tensor | float] | None = None
        self._last_backbone_debug: dict[str, Any] | None = None
        self._logit_debug_records = 0
        self._dspark_enable_draft_aclgraph = self.use_cuda_graph
        self._dspark_graph_runnable_uses_buffers = False
        self._dspark_graph_model_inputs: dict[str, Any] | None = None
        self._dspark_draft_capture_sizes: list[int] = []
        self._dspark_logged_capture_keys: set[Any] = set()
        self._dspark_logged_replay_keys: set[Any] = set()

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        query_len = 1 + self.num_speculative_tokens
        graph_enabled = (
            self.use_cuda_graph
            and self._dspark_enable_draft_aclgraph
            and cudagraph_mode.has_full_cudagraphs()
        )

        draft_compilation_config = copy(self.vllm_config.compilation_config)
        capture_sizes = draft_compilation_config.cudagraph_capture_sizes
        if capture_sizes is not None:
            draft_compilation_config.cudagraph_capture_sizes = list(capture_sizes)
        dspark_capture_sizes: list[int] = []
        if graph_enabled:
            dspark_capture_sizes = self._derive_draft_cudagraph_capture_sizes(query_len)
            if dspark_capture_sizes:
                draft_compilation_config.cudagraph_capture_sizes = dspark_capture_sizes
                draft_compilation_config.max_cudagraph_capture_size = dspark_capture_sizes[-1]
            else:
                # The runner captures drafter graphs indirectly while walking
                # target FULL descriptors. Without a uniform target descriptor
                # there is no one-to-one capture trigger for a draft graph key;
                # do not leave a key that would lazily capture during serving.
                graph_enabled = False
                logger.warning(
                    "DSpark draft ACLGraph found no uniform target FULL "
                    "capture descriptors; keeping the drafter eager."
                )
        dispatcher_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY if graph_enabled else CUDAGraphMode.NONE
        )

        draft_vllm_config = replace(
            self.vllm_config,
            compilation_config=draft_compilation_config,
        )
        self.cudagraph_dispatcher = CudagraphDispatcher(draft_vllm_config)
        self.cudagraph_dispatcher.uniform_decode_query_len = query_len
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            dispatcher_mode,
            query_len,
        )
        self._dspark_draft_capture_sizes = sorted({
            descriptor.num_tokens
            for _, descriptors in self.cudagraph_dispatcher.get_capture_descs()
            for descriptor in descriptors
        })

        if self.use_cuda_graph and not graph_enabled:
            logger.warning(
                "DSpark draft ACLGraph requires resolved FULL graph support; "
                "keeping the drafter eager."
            )
            self.use_cuda_graph = False
            self._dspark_enable_draft_aclgraph = False

    def get_cudagraph_capture_sizes(self) -> list[int]:
        return list(self._dspark_draft_capture_sizes)

    def _draft_cudagraph_dispatcher(self) -> CudagraphDispatcher:
        return self.cudagraph_dispatcher

    def _draft_uniform_decode(self, target_model_batch_desc) -> bool:
        # DSpark always runs one fixed [anchor + N masks] query block per
        # request, independently of whether the verifier batch is mixed.
        return True

    def _sync_draft_cudagraph_mode_across_dp(self) -> bool:
        return True

    def _derive_draft_cudagraph_capture_sizes(self, query_len: int) -> list[int]:
        target_dispatcher = getattr(self.runner, "cudagraph_dispatcher", None)
        get_capture_descs = getattr(target_dispatcher, "get_capture_descs", None)
        if get_capture_descs is None:
            return []

        sizes: set[int] = set()
        for runtime_mode, descriptors in get_capture_descs():
            if runtime_mode != CUDAGraphMode.FULL:
                continue
            for descriptor in descriptors:
                num_reqs = getattr(descriptor, "num_reqs", None)
                if not getattr(descriptor, "uniform", False) or num_reqs is None:
                    continue
                size = int(num_reqs) * query_len
                if size > 0:
                    sizes.add(size)
        return sorted(sizes)

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()
        # The outer ACLGraphWrapper owns capture. Keep the inner draft model
        # eager to avoid nested torch.compile/ACLGraph state.
        draft_model_config = copy(draft_vllm_config.model_config)
        draft_model_config.enforce_eager = True
        draft_compilation_config = copy(draft_vllm_config.compilation_config)
        draft_compilation_config.mode = CompilationMode.NONE
        return replace(
            draft_vllm_config,
            model_config=draft_model_config,
            compilation_config=draft_compilation_config,
        )

    def load_model(self, model) -> None:
        enable_draft_aclgraph = self.use_cuda_graph and self._dspark_enable_draft_aclgraph
        if enable_draft_aclgraph:
            # The base loader wraps the complete merged draft flow. DSpark must
            # keep context-KV precompute and sequential Markov sampling outside
            # capture, so suppress that generic wrapper here.
            self.use_cuda_graph = False
        try:
            super().load_model(model)
        finally:
            self.use_cuda_graph = enable_draft_aclgraph

        if enable_draft_aclgraph:
            logger.info(
                "DSpark draft ACLGraph is enabled for the backbone forward; "
                "context-KV precompute and Markov sampling remain eager."
            )
            self.update_stream = torch.npu.Stream()
            self._dspark_backbone_runnable: ACLGraphWrapper | Callable = ACLGraphWrapper(
                self._run_dspark_model_from_graph_buffers,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                # This is a split backbone graph, not the merged EAGLE graph
                # whose wrapper may safely skip the replay ordering barrier.
                use_eagle=False,
                enable_enpu=self.enable_enpu,
                # _run_merged_draft synchronizes before graph-task update, so
                # the wrapper must not repeat the same host barrier after the
                # update.  ExternalEvent waits captured in the backbone graph
                # still order replay behind the update stream.
                caller_orders_graph_update=True,
            )
            self._dspark_graph_runnable_uses_buffers = True
        else:
            self._dspark_backbone_runnable = self._run_dspark_model
            self._dspark_graph_runnable_uses_buffers = False

        draft_backbone = getattr(getattr(self, "model", None), "model", None)
        if draft_backbone is not None:
            global_rank = dist.get_rank() if dist.is_initialized() else 0
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            draft_backbone._dspark_backbone_debug_enabled = (
                bool(envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH)
                and envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS > 0
                and global_rank % tp_size == 0
            )

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
        if (
            is_profile
            or not self.use_cuda_graph
            or not self._dspark_enable_draft_aclgraph
            or not aclgraph_runtime_mode.has_mode(CUDAGraphMode.FULL)
        ):
            return super().dummy_run(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=None,
                dummy_compute_logits=dummy_compute_logits,
                is_profile=is_profile,
                **kwargs,
            )

        query_len = 1 + self.num_speculative_tokens
        if num_reqs <= 0:
            num_reqs = max(1, min(num_tokens, self.max_query_tokens) // query_len)
        num_query_tokens = min(num_reqs * query_len, self.max_query_tokens)
        num_active_loras = len(self.runner.input_batch.lora_id_to_lora_request)
        has_lora = num_active_loras > 0
        aclgraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_query_tokens,
            uniform_decode=True,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            valid_modes={CUDAGraphMode.FULL},
        )
        num_input_tokens = batch_descriptor.num_tokens
        (
            num_input_tokens,
            num_tokens_across_dp,
            synced_cudagraph_mode,
        ) = self.runner._sync_metadata_across_dp(
            num_input_tokens,
            is_draft_model=True,
            cudagraph_mode=aclgraph_runtime_mode,
            allow_dp_padding=True,
        )
        if num_tokens_across_dp is not None:
            dp_rank = getattr(self.runner, "dp_rank", 0)
            num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
            aclgraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens,
                uniform_decode=True,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
                valid_modes={synced_cudagraph_mode},
            )
            num_input_tokens = batch_descriptor.num_tokens
            num_tokens_across_dp.fill_(num_input_tokens)

        graph_num_reqs = getattr(batch_descriptor, "num_reqs", None)
        if graph_num_reqs is None:
            graph_num_reqs = max(num_reqs, num_input_tokens // query_len)
        # DFlash dummy capture binds this exact persistent buffer into the
        # query KV update op. Dummy rows must never write cache slot zero.
        self._slot_mapping_buffer[:num_input_tokens].fill_(PADDING_SLOT_ID)
        self._context_slot_mapping_buffer[:num_input_tokens].fill_(PADDING_SLOT_ID)
        return super().dummy_run(
            num_tokens=num_input_tokens,
            num_reqs=int(graph_num_reqs),
            num_tokens_across_dp=num_tokens_across_dp,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            dummy_compute_logits=dummy_compute_logits,
            is_profile=is_profile,
            **kwargs,
        )

    def _pad_dspark_query_buffers(
        self,
        num_actual_tokens: int,
        num_input_tokens: int,
    ) -> None:
        if num_input_tokens <= num_actual_tokens:
            return
        self.input_ids[num_actual_tokens:num_input_tokens].fill_(
            self.parallel_drafting_token_id
        )
        self.positions[num_actual_tokens:num_input_tokens].zero_()
        self._slot_mapping_buffer[num_actual_tokens:num_input_tokens].fill_(
            PADDING_SLOT_ID
        )

    def _synchronize_before_dspark_graph_update(self) -> None:
        if not self.enable_enpu:
            # graph_task_update mutates the captured attention handles on a
            # side stream. Wait for the previous split-backbone replay (and
            # its eager Markov tail) before allowing the next update to start.
            torch.npu.current_stream().synchronize()

    def _run_dspark_model(self, **model_inputs: Any) -> torch.Tensor:
        return self.model(**model_inputs)

    def _run_dspark_model_from_graph_buffers(self) -> torch.Tensor:
        model_inputs = self._dspark_graph_model_inputs
        if model_inputs is None:
            raise RuntimeError(
                "DSpark draft ACLGraph inputs were not prepared before capture/replay"
            )
        return self._run_dspark_model(**model_inputs)

    def _run_prepared_dspark_model(
        self,
        run_model: Callable[..., torch.Tensor],
        model_inputs: dict[str, Any],
    ) -> torch.Tensor:
        if (
            self._dspark_graph_runnable_uses_buffers
            and run_model is self._dspark_backbone_runnable
        ):
            self._dspark_graph_model_inputs = model_inputs
            forward_context = get_forward_context()
            batch_descriptor = getattr(forward_context, "batch_descriptor", None)
            graph_entries = getattr(run_model, "concrete_aclgraph_entries", {})
            graph_entry = graph_entries.get(batch_descriptor)
            had_graph = graph_entry is not None and graph_entry.aclgraph is not None
            output = run_model()
            graph_entries = getattr(run_model, "concrete_aclgraph_entries", {})
            graph_entry = graph_entries.get(batch_descriptor)
            has_graph = graph_entry is not None and graph_entry.aclgraph is not None
            if has_graph and not had_graph and batch_descriptor not in self._dspark_logged_capture_keys:
                logger.info(
                    "Captured DSpark draft backbone ACLGraph for %s",
                    batch_descriptor,
                )
                self._dspark_logged_capture_keys.add(batch_descriptor)
            elif had_graph and batch_descriptor not in self._dspark_logged_replay_keys:
                logger.info(
                    "Replayed DSpark draft backbone ACLGraph for %s",
                    batch_descriptor,
                )
                self._dspark_logged_replay_keys.add(batch_descriptor)
            return output
        return run_model(**model_inputs)

    def _update_full_graph_params_if_needed(
        self,
        forward_context,
        num_input_tokens: int,
        multi_steps_attn_metadata: list[dict[str, Any]],
    ) -> None:
        if (
            self.use_cuda_graph
            and self._dspark_enable_draft_aclgraph
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            # The DSpark orchestrator updates graph tasks immediately before
            # backbone replay. Base's normal post-runnable update would happen
            # after eager Markov kernels have already been enqueued.
            return
        return super()._update_full_graph_params_if_needed(
            forward_context,
            num_input_tokens,
            multi_steps_attn_metadata,
        )

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
        forward_context = get_forward_context()
        use_backbone_aclgraph = (
            self.use_cuda_graph
            and self._dspark_enable_draft_aclgraph
            and forward_context is not None
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        )
        if not use_backbone_aclgraph:
            return super()._run_merged_draft(
                num_input_tokens=num_input_tokens,
                batch_size=batch_size,
                token_indices_to_sample=token_indices_to_sample,
                target_positions=target_positions,
                inputs_embeds=inputs_embeds,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=num_tokens,
                is_prefill=is_prefill,
            )

        if lmhead_tp_enable():
            raise NotImplementedError(
                "DSpark drafting does not support LM-head tensor parallel yet."
            )

        self._pad_dspark_query_buffers(num_tokens, num_input_tokens)
        self._synchronize_before_dspark_graph_update()
        # This call performs context-KV precompute eagerly on every iteration.
        # Only the fixed-shape query model forward below is captured.
        model_inputs = self.build_model_inputs_first_pass(num_input_tokens)
        self._update_full_graph_params(
            forward_context,
            num_input_tokens,
            multi_steps_attn_metadata,
        )
        hidden_states = self._run_prepared_dspark_model(
            self._dspark_backbone_runnable,
            model_inputs,
        )
        if self.model_returns_tuple():
            last_hidden_states, _ = hidden_states
        else:
            last_hidden_states = hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        return self._sample_parallel_draft_tokens(sample_hidden_states)

    def record_target_logit_debug(self, logits: torch.Tensor, metadata: Any) -> None:
        captured = self._last_logit_debug
        backbone_captured = self._last_backbone_debug
        self._last_logit_debug = None
        self._last_backbone_debug = None
        # Chunked prefill accumulates context/raw debug chunks across several
        # proposer calls; everything accumulated so far is already inside the
        # snapshot consumed above. Release them here even when max_records is
        # exhausted so a debug-enabled server cannot retain request-sized
        # tensors across verification steps.
        draft_backbone = getattr(getattr(self, "model", None), "model", None)
        if draft_backbone is not None:
            draft_backbone._dspark_context_debug_chunks = []
            draft_backbone._dspark_raw_context_debug_chunks = []
        debug_path = envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH
        max_records = max(0, envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS)
        if draft_backbone is not None:
            draft_backbone._dspark_backbone_debug_enabled = (
                bool(debug_path) and self._logit_debug_records < max_records
            )
        if not debug_path or captured is None or self._logit_debug_records >= max_records:
            return

        record_index = self._logit_debug_records
        self._logit_debug_records += 1
        if draft_backbone is not None:
            draft_backbone._dspark_backbone_debug_enabled = (
                self._logit_debug_records < max_records
            )
        try:
            global_rank = dist.get_rank() if dist.is_initialized() else 0
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if global_rank % tp_size != 0:
                return
            target_logits = logits[metadata.target_logits_indices]
            record = _build_logit_debug_record(
                captured,
                target_logits,
                metadata.num_draft_tokens,
                metadata.draft_token_ids,
                record_index,
            )
            record["global_rank"] = global_rank
            output_path = f"{debug_path}.rank{global_rank}.jsonl"
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            if backbone_captured is not None:
                backbone_captured["record"] = record_index
                backbone_captured["global_rank"] = global_rank
                torch.save(
                    backbone_captured,
                    f"{debug_path}.rank{global_rank}.record{record_index}.pt",
                )
        except Exception:
            logger.exception("Failed to write DSpark logit debug record")

    def _sample_parallel_draft_tokens(
        self, sample_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Sequential Markov sampling over the parallel draft block.

        Args:
            sample_hidden_states: [batch * num_speculative_tokens, hidden]
                hidden states of the mask query positions.

        Returns:
            [batch, num_speculative_tokens] draft token ids (target vocab).
        """
        num_spec = self.num_speculative_tokens
        batch_size = sample_hidden_states.shape[0] // num_spec
        model = self.model
        draft_backbone = getattr(model, "model", None)

        # One GEMM for all block positions, in draft-vocab space; the Markov
        # bias is added per step before the argmax.
        base_logits = model.compute_draft_logits(sample_hidden_states)
        base_logits = base_logits.view(batch_size, num_spec, -1)
        capture_debug = (
            bool(envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH)
            and self._logit_debug_records
            < max(0, envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS)
            and bool(
                getattr(draft_backbone, "_dspark_backbone_debug_enabled", False)
            )
        )
        markov_debug = torch.empty_like(base_logits) if capture_debug else None
        final_debug = torch.empty_like(base_logits) if capture_debug else None
        prev_debug = (
            torch.empty((batch_size, num_spec), dtype=torch.int64, device=base_logits.device)
            if capture_debug
            else None
        )

        prev_tokens = self.input_ids[self._anchor_indices[:batch_size]]
        draft_tokens = base_logits.new_empty(
            (batch_size, num_spec), dtype=torch.int64
        )
        for step in range(num_spec):
            markov_bias = model.markov_bias(model.markov_embed(prev_tokens))
            applied_markov_bias = markov_bias
            markov_scale = getattr(self, "_markov_scale", 1.0)
            if markov_scale != 1.0:
                applied_markov_bias = markov_bias * markov_scale
            step_logits = base_logits[:, step] + applied_markov_bias
            if capture_debug:
                assert markov_debug is not None
                assert final_debug is not None
                assert prev_debug is not None
                markov_debug[:, step].copy_(applied_markov_bias)
                final_debug[:, step].copy_(step_logits)
                prev_debug[:, step].copy_(prev_tokens)
            prev_tokens = model.map_draft_to_target(step_logits.argmax(dim=-1))
            draft_tokens[:, step] = prev_tokens
        if capture_debug:
            assert markov_debug is not None
            assert final_debug is not None
            assert prev_debug is not None
            self._last_logit_debug = {
                "base_logits": base_logits.detach(),
                "markov_bias": markov_debug.detach(),
                "final_logits": final_debug.detach(),
                "prev_token_ids": prev_debug.detach(),
                "proposed_token_ids": draft_tokens.detach(),
                "markov_scale": float(getattr(self, "_markov_scale", 1.0)),
            }
            self._last_backbone_debug = getattr(
                draft_backbone,
                "_last_dspark_backbone_debug",
                None,
            )
        # This sampling path also runs for intermediate chunked-prefill steps,
        # so the context/raw chunk accumulators must survive it; they are only
        # released in record_target_logit_debug, which runs at the real target
        # verification point. Only the per-forward snapshot is consumed here.
        if draft_backbone is not None:
            draft_backbone._last_dspark_backbone_debug = None
        return draft_tokens
