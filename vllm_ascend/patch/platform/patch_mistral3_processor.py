# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import contextlib
from typing import Any

from transformers.models.pixtral import PixtralProcessor
from vllm.transformers_utils.processor import cached_processor_from_config


def _patched_get_hf_processor(self: Any, **kwargs: object) -> Any:
    return cached_processor_from_config(
        self.ctx.model_config,
        processor_cls=PixtralProcessor,
        **kwargs,
    )


def install_patch() -> None:
    from vllm.model_executor.models.mistral3 import Mistral3ProcessingInfo

    Mistral3ProcessingInfo.get_hf_processor = _patched_get_hf_processor


with contextlib.suppress(ImportError):
    install_patch()
