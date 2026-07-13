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
import logging
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from vllm.config import VllmConfig

from vllm_ascend import envs
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

logger = logging.getLogger(__name__)

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

    def load_model(self, model) -> None:
        super().load_model(model)
        draft_backbone = getattr(getattr(self, "model", None), "model", None)
        if draft_backbone is not None:
            global_rank = dist.get_rank() if dist.is_initialized() else 0
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            draft_backbone._dspark_backbone_debug_enabled = (
                bool(envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_PATH)
                and envs.VLLM_ASCEND_DSPARK_LOGIT_DEBUG_MAX_RECORDS > 0
                and global_rank % tp_size == 0
            )

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
