#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/model_executor/models/qwen3_dspark.py
# (upstream vllm#46995 / vllm#47093, not present in vLLM v0.23.0).
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
"""Qwen3-shaped DSpark draft model for semi-autoregressive drafting.

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head. GLM-5.2 DSpark
speculators checkpoints resolve to this architecture.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    maybe_prefix,
    process_eagle_weight,
)


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    ``markov_w1[token]`` embeds the previously sampled token (target vocab);
    ``markov_w2`` projects it to a draft-vocab bias added to the base draft
    logits. The two vocab sizes coincide for full-vocab drafts such as
    RedHatAI/GLM-5.2-speculator.dspark.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size, markov_rank, prefix=maybe_prefix(prefix, "markov_w2")
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of ``token_ids`` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        """Draft-vocab transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model = Qwen3DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size, scale=logit_scale)
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Draft-vocab logits without the d2t scatter.

        The proposer adds the Markov bias in draft space and then remaps the
        sampled ids via ``map_draft_to_target``.
        """
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the local TP vocab shard without a full-vocab gather."""
        get_local_logits = getattr(self.logits_processor, "get_local_logits", None)
        if not callable(get_local_logits):
            raise RuntimeError("the active logits processor has no local-logits path")
        return get_local_logits(hidden_states, self.lm_head)

    def draft_vocab_start_index(self) -> int:
        """Global draft-vocab offset of this rank's local LM-head shard."""
        return int(self.lm_head.shard_indices.org_vocab_start_index)

    def supports_local_markov_argmax(self) -> bool:
        """Whether both vocab-parallel heads have identical local shards."""
        base = self.lm_head
        markov = self.model.markov_head.markov_w2
        base_indices = base.shard_indices
        markov_indices = markov.shard_indices
        base_group = getattr(base, "comm_group", None)
        markov_group = getattr(markov, "comm_group", None)
        return bool(
            callable(getattr(self.logits_processor, "get_local_logits", None))
            and base_group is not None
            and markov_group is not None
            and base.num_org_embeddings_per_partition
            == markov.num_org_embeddings_per_partition
            and base_indices.org_vocab_start_index
            == markov_indices.org_vocab_start_index
            and base_indices.org_vocab_end_index
            == markov_indices.org_vocab_end_index
            and base.num_added_embeddings_per_partition == 0
            and markov.num_added_embeddings_per_partition == 0
            and base_group.world_size == markov_group.world_size
            and base_group.rank_in_group == markov_group.rank_in_group
            and self.config.draft_vocab_size < 2**24
        )

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocab ids to target ids (identity for full-vocab drafts)."""
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def markov_local_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        """Compute the Markov head's matching local TP vocab shard."""
        get_local_logits = getattr(self.logits_processor, "get_local_logits", None)
        if not callable(get_local_logits):
            raise RuntimeError("the active logits processor has no local-logits path")
        markov_w2 = self.model.markov_head.markov_w2
        markov_start = int(markov_w2.shard_indices.org_vocab_start_index)
        draft_start = self.draft_vocab_start_index()
        if markov_start != draft_start:
            raise RuntimeError(
                "DSpark LM and Markov heads use different TP vocab partitions: "
                f"lm_head={draft_start}, markov_w2={markov_start}"
            )
        return get_local_logits(markov_embed, markov_w2)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        for name, loaded_weight in weights:
            # t2d is training-only; the draft remaps via d2t at sampling time.
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # confidence_head is not wired into inference; mask_embedding is
        # unused (DSpark masks via the mask_token_id vocab row). embed_tokens
        # and lm_head are optional: when omitted they are shared from the
        # target model by the proposer.
        skip_substrs = ["mask_embedding", "confidence_head"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        loader = AutoWeightsLoader(self, skip_prefixes=None, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
