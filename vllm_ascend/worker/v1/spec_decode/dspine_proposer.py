# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# DSpine proposer for the vLLM-Ascend v1 runner (main branch).
#
# DSpine reuses the DFlash parallel backbone: one pass over the anchor-first
# query block builds the draft features, the code bridge refines the residual
# stream layer by layer, and only the *tail* differs. The tail redoes the last
# bridge write with the exact candidate token of the predecessor ("exact
# sender") for K candidates per row and walks a K-lane chain.
#
# The base proposer already exposes a custom-tail hook for DFlash-style drafts
# (``use_dflash2_selector`` + ``compute_draft_token_ids``), so this proposer
# reuses it instead of overriding ``_run_merged_draft``.
from __future__ import annotations

import time

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.models.dspine_math import build_serial_tables, serial_refine_greedy
from vllm_ascend.worker.v1.spec_decode.dflash_proposer import AscendDflashProposer

logger = init_logger(__name__)


def is_dspine_draft(speculative_config) -> bool:
    """A DSpine draft is a DFlash draft plus ``dflash_config.selected_prefix_code_bridge``."""
    draft = getattr(speculative_config, "draft_model_config", None)
    if draft is None:
        return False
    hf_config = getattr(draft, "hf_config", None)
    if hf_config is None:
        return False
    dflash = getattr(hf_config, "dflash_config", None)
    if not isinstance(dflash, dict):
        inner = getattr(hf_config, "model", None)
        dflash = getattr(inner, "dflash_config", None)
    return isinstance(dflash, dict) and isinstance(dflash.get("selected_prefix_code_bridge"), dict)


class AscendDspineProposer(AscendDflashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        draft_config = vllm_config.speculative_config.draft_model_config
        if getattr(draft_config, "quantization", None) is not None:
            raise ValueError("The DSpine draft is served in BF16; do not set quantization on the draft.")

        self.dspine_trained_block_size = int(getattr(draft_config.hf_config, "block_size", 0) or 0)
        if self.dspine_trained_block_size < 2:
            raise ValueError(f"DSpine draft config lacks a valid block_size: {self.dspine_trained_block_size}")
        self.dspine_block_size = self.num_speculative_tokens + 1
        if self.dspine_block_size > self.dspine_trained_block_size:
            raise ValueError(
                f"DSpine drafts a block of at most {self.dspine_trained_block_size} rows "
                f"(1 anchor + {self.dspine_trained_block_size - 1} proposals); "
                f"num_speculative_tokens={self.num_speculative_tokens} needs {self.dspine_block_size}"
            )
        if self.dspine_block_size != self.dspine_trained_block_size:
            logger.warning(
                "DSpine serves %d draft rows per block but was trained with %d; "
                "acceptance length may drop. Use num_speculative_tokens=%d for the full block.",
                self.dspine_block_size,
                self.dspine_trained_block_size,
                self.dspine_trained_block_size - 1,
            )
        sliding_window = getattr(draft_config.hf_config, "sliding_window", None)
        if getattr(draft_config.hf_config, "use_sliding_window", False) and sliding_window:
            logger.warning(
                "DSpine draft was trained with a %s-token sliding window on its block attention; "
                "the window is taken from the draft config (layer_types/sliding_window).",
                sliding_window,
            )

        # Reuse the DFlash custom-tail hook: the base proposer calls
        # ``compute_draft_token_ids`` for parallel drafting when this is set.
        self.use_dflash2_selector = True
        self._dspine_tg: torch.Tensor | None = None
        self._dspine_tsv: torch.Tensor | None = None
        self._dspine_topk = 0
        self._anchor_indices = (
            torch.arange(self.max_batch_size, device=device, dtype=torch.int64) * self.dspine_block_size
        )

    @property
    def dspine_bridge(self):
        return self.model.model.selected_prefix_bridge

    def _maybe_share_lm_head(self, model) -> None:
        if getattr(self.model, "draft_id_to_target_id", None) is not None:
            raise ValueError(
                "DSpine refinement indexes the full target-vocabulary head; reduced draft vocabularies are unsupported."
            )
        self.model.has_own_lm_head = False
        super()._maybe_share_lm_head(model)

    def load_model(self, model) -> None:
        super().load_model(model)
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise NotImplementedError(
                "The DSpine refinement tail reads the full draft head on every rank; "
                f"it requires tensor parallel size 1, got {tp_size}"
            )
        if get_pp_group().world_size != 1:
            raise NotImplementedError(
                "DSpine drafts do not carry their own embed_tokens/lm_head; they are shared "
                f"with the target model, which requires pipeline parallel size 1, got {get_pp_group().world_size}"
            )
        # The bridge reads whole request blocks, so the served block size must be
        # fixed before the first (possibly graph-captured) forward.
        self.model.model.set_block_size(self.dspine_block_size)
        self._dspine_build_tables()

    @torch.no_grad()
    def _dspine_build_tables(self) -> None:
        """Build the vocabulary-wide sender tables (TG/TSV) once, after weight loading."""
        bridge = self.dspine_bridge
        draft = self.model.model
        head_weight = self.model.lm_head.weight
        last = bridge.num_layers - 1
        started = time.time()
        tg, tsv = build_serial_tables(
            head_weight,
            draft.norm.weight,
            bridge.feature_write_weights[last].detach(),
            bridge.packed_weight[bridge.rank :].detach(),
            bridge.codebook,
        )
        self._dspine_tg = tg
        self._dspine_tsv = tsv
        self._dspine_topk = int(bridge.head_topk)
        logger.info(
            "DSpine refinement tables built: vocab=%d rank=%d topk=%d tg=%s tsv=%s in %.1fs",
            head_weight.shape[0],
            bridge.rank,
            self._dspine_topk,
            tuple(tg.shape),
            tuple(tsv.shape),
            time.time() - started,
        )

    def set_inputs_first_pass(self, *args, **kwargs):
        result = super().set_inputs_first_pass(*args, **kwargs)
        # DSpine checkpoints are trained with causal block attention
        # (draft_block_causal); the config patch sets is_causal so the attention
        # layers and this metadata agree.
        cad = result[2]
        cad.causal = True
        cad.attn_mask = None
        return result

    def compute_draft_token_ids(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """K-lane serial last-write refinement over one draft block per request."""
        return self._dspine_serial_refine(hidden_states), None

    @torch.inference_mode()
    def _dspine_serial_refine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._dspine_tg is None:
            raise RuntimeError("DSpine refinement tables were not built; load_model must run before the first draft")
        steps = self.num_speculative_tokens
        block = self.dspine_block_size
        total = hidden_states.shape[0]
        if total == 0:
            return hidden_states.new_empty((0,), dtype=torch.long)
        if total % steps:
            raise ValueError(
                f"DSpine expected hidden states divisible by num_speculative_tokens={steps}, got {total}."
            )
        batch = total // steps
        rows = hidden_states.view(batch, steps, -1)
        anchors = self.input_ids[self._anchor_indices[:batch]]
        stash = self.model.model.last_write_stash
        if stash is None:
            raise RuntimeError("DSpine tail ran without the draft forward's last-write stash")
        used = batch * block
        tokens = serial_refine_greedy(
            rows,
            anchors,
            stash["raw"][:used],
            stash["rms"][:used],
            stash["packed"][:used],
            self.model.lm_head.weight,
            self._dspine_tg,
            self._dspine_tsv,
            self.dspine_bridge.gate.bias,
            self.model.model.norm.weight,
            self._dspine_topk,
            block,
        )
        return tokens.reshape(-1)