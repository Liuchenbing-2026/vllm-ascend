from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, replace
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_inputs_kernel_single_grid
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class AscendDflashProposer(AscendEagleProposer):
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

        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        self.input_ids = torch.zeros(
            max(self.max_num_tokens, self.max_query_tokens),
            dtype=torch.int32,
            device=device,
        )

        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        self.arange_dflash = torch.arange(self.max_positions + 1, device=device, dtype=torch.int32)

        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
        )

        self.parallel_drafting_hidden_state_tensor = None
        self._slot_mapping_buffers_by_gid: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._draft_kernel_block_size_by_gid: dict[int, int] = {}
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_input_slot_mappings: dict[int, torch.Tensor] = {}

    @property
    def dflash_config(self) -> dict[str, Any]:
        config = self.speculative_config.draft_model_config.hf_config
        return getattr(config, "dflash_config", None) or {}

    def allow_multiple_draft_kv_cache_groups(self) -> bool:
        return True

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self._draft_kernel_block_size_by_gid.clear()
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            configured_size = (
                kernel_block_sizes[gid]
                if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                else 0
            )
            self._draft_kernel_block_size_by_gid[gid] = int(
                configured_size
                or attn_group.get_metadata_builder().kv_cache_spec.block_size
            )
        self._ensure_slot_mapping_buffers()

    def _draft_kv_gids(self) -> list[int]:
        return self._draft_kv_cache_group_ids or [
            self.kv_cache_gid if self.kv_cache_gid >= 0 else 0
        ]

    def _ensure_slot_mapping_buffers(self) -> None:
        gids = self._draft_kv_gids()
        first_gid = gids[0]
        for gid in gids:
            if gid in self._slot_mapping_buffers_by_gid:
                continue
            if gid == first_gid:
                self._slot_mapping_buffers_by_gid[gid] = (
                    self._context_slot_mapping_buffer,
                    self._slot_mapping_buffer,
                )
            else:
                self._slot_mapping_buffers_by_gid[gid] = (
                    torch.zeros(
                        self.max_num_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    torch.zeros(
                        self.max_query_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                )

    def clear_per_group_attn_metadata(self) -> None:
        self._per_group_block_tables.clear()
        self._per_group_input_slot_mappings.clear()

    def set_per_group_attn_metadata(
        self,
        kv_cache_gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if kv_cache_gid not in self._draft_kv_gids():
            return
        self._per_group_block_tables[kv_cache_gid] = block_table
        self._per_group_input_slot_mappings[kv_cache_gid] = slot_mapping

    def _get_dflash_block_table(
        self,
        kv_cache_gid: int,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor:
        block_table = self._per_group_block_tables.get(kv_cache_gid)
        if block_table is not None:
            return block_table
        if kv_cache_gid == self.kv_cache_gid:
            return cad.block_table_tensor
        if self.runner is not None:
            return self.runner.input_batch.block_table[
                kv_cache_gid
            ].get_device_tensor()[: cad.num_reqs]
        raise RuntimeError(
            "Missing DFlash block table for draft KV cache group "
            f"{kv_cache_gid}."
        )

    def _get_dflash_input_slot_mapping(
        self,
        kv_cache_gid: int,
        cad: CommonAttentionMetadata,
    ) -> torch.Tensor:
        slot_mapping = self._per_group_input_slot_mappings.get(kv_cache_gid)
        if slot_mapping is not None:
            return slot_mapping
        if kv_cache_gid == self.kv_cache_gid:
            return cad.slot_mapping
        raise RuntimeError(
            "Missing DFlash context slot mapping for draft KV cache group "
            f"{kv_cache_gid}."
        )

    def _get_dflash_context_slot_mapping(
        self,
        num_context: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._ensure_slot_mapping_buffers()
        if not self._draft_layer_to_kv_cache_gid:
            return self._context_slot_mapping_buffer[:num_context]
        return {
            layer_name: self._slot_mapping_buffers_by_gid[
                self._draft_layer_to_kv_cache_gid[layer_name]
            ][0][:num_context]
            for layer_name in self._draft_attn_layer_names
        }

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                # The final full-attention DFlash layer is non-causal. Sliding
                # layers receive causal metadata on a per-layer basis below.
                use_non_causal=True,
            ),
        )

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        return self.dflash_config.get("use_aux_hidden_state", True)

    def _raise_if_query_window_exceeds_max_model_len(
        self,
        max_seq_len: int,
        num_query_per_req: int,
    ) -> None:
        """Fail closed before constructing DFlash query positions or slots."""
        runner = getattr(self, "runner", None)
        effective_max_model_len = getattr(
            runner, "effective_drafter_max_model_len", None
        )
        if effective_max_model_len is None:
            draft_model_config = getattr(self, "draft_model_config", None)
            effective_max_model_len = getattr(
                draft_model_config, "max_model_len", None
            )
        if effective_max_model_len is None:
            effective_max_model_len = self.max_model_len

        required_seq_len = int(max_seq_len) + int(num_query_per_req)
        effective_max_model_len = int(effective_max_model_len)
        if required_seq_len > effective_max_model_len:
            raise RuntimeError(
                "DFlash query window exceeds the drafter max model length: "
                f"max_seq_len={max_seq_len}, "
                f"query_tokens={num_query_per_req} "
                f"(1 bonus + {num_query_per_req - 1} masks), "
                f"required_seq_len={required_seq_len}, "
                "effective_drafter_max_model_len="
                f"{effective_max_model_len}. The model runner must skip "
                "drafting for this step."
            )

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
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.num_reqs
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # This is a second, fail-closed boundary behind the model-runner gate.
        # Without it, query positions and FIA seq_lens can exceed max_model_len,
        # while the slot-mapping kernel silently clamps the block-table column.
        self._raise_if_query_window_exceeds_max_model_len(
            cad.max_seq_len,
            num_query_per_req,
        )

        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        has_num_rejected = num_rejected_tokens_gpu is not None
        self._ensure_slot_mapping_buffers()
        draft_kv_group_ids = self._draft_kv_gids()
        for kv_cache_gid in draft_kv_group_ids:
            context_slot_mapping_buffer, query_slot_mapping_buffer = (
                self._slot_mapping_buffers_by_gid[kv_cache_gid]
            )
            block_table = self._get_dflash_block_table(kv_cache_gid, cad)
            copy_and_expand_dflash_inputs_kernel_single_grid[1,](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=self._get_dflash_input_slot_mapping(
                    kv_cache_gid, cad
                ),
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=context_slot_mapping_buffer,
                out_query_slot_mapping_ptr=query_slot_mapping_buffer,
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=block_table,
                block_table_stride=block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=(
                    num_rejected_tokens_gpu if has_num_rejected else 0
                ),
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=self._draft_kernel_block_size_by_gid.get(
                    kv_cache_gid, self.kernel_block_size
                ),
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
            )

        primary_kv_cache_gid = draft_kv_group_ids[0]
        query_slot_mapping = self._slot_mapping_buffers_by_gid[
            primary_kv_cache_gid
        ][1][:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.block_table_tensor = self._get_dflash_block_table(
            primary_kv_cache_gid, cad
        )
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    def _build_per_group_and_layer_attn_metadata(
        self,
        cad: CommonAttentionMetadata,
        draft_index: int = 0,
        *,
        for_graph_capture: bool = False,
    ) -> tuple[list[object], dict[str, object]]:
        self._ensure_slot_mapping_buffers()
        sliding_layer_names: set[str] = getattr(
            self.model, "sliding_attention_layer_names", set()
        )
        per_group: list[object] = []
        per_layer: dict[str, object] = {}

        for attn_group in self.draft_attn_groups:
            kv_cache_gid = attn_group.kv_cache_group_id
            group_cad = cad.replace(
                block_table_tensor=self._get_dflash_block_table(kv_cache_gid, cad),
                slot_mapping=self._slot_mapping_buffers_by_gid[kv_cache_gid][1][
                    : cad.num_actual_tokens
                ],
                causal=False,
            )
            builder = attn_group.get_metadata_builder()

            def build(group_metadata: CommonAttentionMetadata) -> object:
                if for_graph_capture:
                    return builder.build_for_graph_capture(
                        group_metadata,
                        AscendAttentionState.ChunkedPrefill,
                    )
                return builder.build_for_drafting(
                    common_attn_metadata=group_metadata,
                    draft_index=draft_index,
                )

            noncausal_metadata = build(group_cad)
            if hasattr(noncausal_metadata, "attn_mask"):
                noncausal_metadata.attn_mask = None
            if hasattr(noncausal_metadata, "attn_state"):
                noncausal_metadata.attn_state = AscendAttentionState.ChunkedPrefill
            per_group.append(noncausal_metadata)
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = noncausal_metadata

            causal_layers = sliding_layer_names & set(attn_group.layer_names)
            if causal_layers:
                causal_metadata = build(group_cad.replace(causal=True))
                if hasattr(causal_metadata, "attn_state"):
                    causal_metadata.attn_state = AscendAttentionState.ChunkedPrefill
                for layer_name in causal_layers:
                    per_layer[layer_name] = causal_metadata

        missing_layers = self._draft_attn_layer_names - set(per_layer)
        assert not missing_layers, (
            "DFlash attention metadata is missing draft layers: "
            f"{sorted(missing_layers)}"
        )
        for layer_name, attn_metadata in per_layer.items():
            expected_causal = layer_name in sliding_layer_names
            actual_causal = getattr(attn_metadata, "causal", None)
            assert actual_causal is expected_causal, (
                f"DFlash layer {layer_name} expected causal={expected_causal}, "
                f"got {actual_causal}."
            )
        return per_group, per_layer

    def build_per_group_and_layer_attn_metadata(
        self,
        cad: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        return self._build_per_group_and_layer_attn_metadata(
            cad,
            draft_index,
            for_graph_capture=False,
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
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            self._ensure_slot_mapping_buffers()
            primary_kv_cache_gid = self._draft_kv_gids()[0]
            dummy_seq_lens = self._prepare_dummy_seq_lens(num_reqs)
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.arange_dflash[: num_reqs + 1] * num_query_per_req,
                query_start_loc_cpu=torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * num_query_per_req,
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                seq_lens=dummy_seq_lens,
                num_reqs=num_reqs,
                num_actual_tokens=num_query_tokens,
                max_query_len=num_query_per_req,
                max_seq_len=0,
                slot_mapping=self._slot_mapping_buffers_by_gid[
                    primary_kv_cache_gid
                ][1][:num_query_total],
                attn_state=AscendAttentionState.ChunkedPrefill,
                causal=False,
                is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
                block_table_tensor=self.runner.input_batch.block_table[
                    primary_kv_cache_gid
                ].get_device_tensor()[:num_reqs],
            )

            _, per_layer_attn_metadata = (
                self._build_per_group_and_layer_attn_metadata(
                common_attn_metadata,
                    for_graph_capture=True,
                )
            )
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        self.token_indices_to_sample.fill_(0)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
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
                self.precompute_context_kv(write_cache=False)
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

    def _prepare_dummy_seq_lens(self, num_reqs: int) -> torch.Tensor:
        dummy_seq_lens = self.seq_lens_group[0][:num_reqs]
        dummy_seq_lens.copy_(self.runner.seq_lens[:num_reqs])
        dummy_seq_lens.clamp_min_(1)
        return dummy_seq_lens

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
    ) -> dict[str, Any]:
        return dict(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self.positions[:num_input_tokens],
            inputs_embeds=None,
        )

    def precompute_context_kv(self, *, write_cache: bool = True) -> None:
        num_context = self._dflash_num_context

        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            (
                self._get_dflash_context_slot_mapping(num_context)
                if write_cache
                else None
            ),
        )

    def _raise_if_multimodal(self):
        pass
