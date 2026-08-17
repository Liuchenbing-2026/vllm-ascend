# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

from vllm.model_executor.models.mistral3 import Mistral3ProcessingInfo

from vllm_ascend.patch.platform import patch_mistral3_processor


def test_patch_is_installed() -> None:
    assert Mistral3ProcessingInfo.get_hf_processor is patch_mistral3_processor._patched_get_hf_processor


def test_processor_uses_checkpoint_tokenizer() -> None:
    model_config = object()
    processing_info = SimpleNamespace(ctx=SimpleNamespace(model_config=model_config))
    expected_processor = object()

    with patch.object(
        patch_mistral3_processor,
        "cached_processor_from_config",
        return_value=expected_processor,
    ) as processor_factory:
        processor = patch_mistral3_processor._patched_get_hf_processor(
            processing_info,
            use_fast=True,
        )

    assert processor is expected_processor
    processor_factory.assert_called_once_with(
        model_config,
        processor_cls=patch_mistral3_processor.PixtralProcessor,
        use_fast=True,
    )
