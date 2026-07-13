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
"""Support DSpark speculators-format draft checkpoints (e.g. GLM-5.2 DSpark).

vLLM v0.23.0 predates upstream DSpark support (vllm#46995 / vllm#47093).
This patch backports the config-side pieces:

1. Registers the ``dspark`` speculators config updater so checkpoints such as
   RedHatAI/GLM-5.2-speculator.dspark resolve to the ``Qwen3DSparkModel``
   draft architecture, with the aux hidden state layer ids and mask token id
   mapped to where the v0.23.0 DFlash stack reads them.
2. Accepts ``method="dspark"`` and normalizes it to ``"dflash"``: DSpark uses
   DFlash's 1+N bonus-anchor drafting block, so the stock DFlash wiring
   (``parallel_drafting``, the N+1 scheduler lookahead slots and the
   hidden-states plumbing) applies unchanged. The DSpark drafter itself is
   selected by draft architecture in ``vllm_ascend.spec_decode``.
"""

import typing

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import VllmConfig
from vllm.config.speculative import SpeculativeConfig, SpeculativeMethod
from vllm.transformers_utils.configs.speculators.algos import register_speculator

_DSPARK_PASSTHROUGH_KEYS = (
    "markov_rank",
    "markov_head_type",
    "block_size",
    "enable_confidence_head",
    "confidence_head_with_markov",
)


@register_speculator("dspark")
def update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
    """Map a speculators-format DSpark config onto the draft model config.

    Mirrors upstream vllm#47093, except that ``target_layer_ids`` and
    ``mask_token_id`` are kept inside ``dflash_config`` where the v0.23.0
    DFlash model and proposer read them (upstream main moved them to
    top-level keys that v0.23.0 does not know about).
    """
    pre_trained_config["architectures"] = ["Qwen3DSparkModel"]
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]

    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    # For the target-side aux hidden state hook.
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    # DFlash configs use different indexing for the target layers; the DFlash
    # model derives the fc fusion width from len(target_layer_ids).
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
    }

    for key in _DSPARK_PASSTHROUGH_KEYS:
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]


# Accept method="dspark" through the pydantic Literal validation, then let
# __post_init__ normalize it onto the DFlash wiring.
SpeculativeConfig.__annotations__["method"] = (
    typing.Literal[SpeculativeMethod, "dspark"] | None
)
rebuild_dataclass(SpeculativeConfig, force=True)
rebuild_dataclass(VllmConfig, force=True)

_original_post_init = SpeculativeConfig.__post_init__


def _dspark_normalizing_post_init(self) -> None:
    is_dspark = self.method == "dspark" or (
        self.method is None
        and self.model is not None
        and "dspark" in self.model.lower()
    )
    if is_dspark:
        self.method = "dflash"
    _original_post_init(self)


SpeculativeConfig.__post_init__ = _dspark_normalizing_post_init
