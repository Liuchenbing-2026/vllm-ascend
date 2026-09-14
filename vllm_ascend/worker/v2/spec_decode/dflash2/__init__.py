# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from typing import Any

DFLASH2_DRAFT_ARCHITECTURE = "DFlash2DraftModel"


def is_dflash2_draft(vllm_config: Any) -> bool:
    """Whether the DFlash draft is a DFlash2 one, by its checkpoint architecture.

    Mirrors ``VllmConfig._is_dflash2_draft`` from vLLM #52816, which the
    supported vLLM 0.26 pin does not have.
    """
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.method != "dflash":
        return False
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None:
        return False
    return DFLASH2_DRAFT_ARCHITECTURE in (draft_model_config.architectures or [])
