# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url

from tests.e2e.conftest import VllmRunner
from vllm_ascend.patch.platform import patch_mistral3_processor  # noqa: F401

MODEL = os.getenv("VLLM_TEST_MISTRAL3_MODEL", "mistralai/Shieldstral-1.0-3B")


@pytest.mark.e2e_model(MODEL)
@patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_shieldstral_multimodal_processor() -> None:
    image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_url(image, format="PNG")},
                },
                {
                    "type": "text",
                    "text": (
                        "Does this image contain physical violence? "
                        "Answer yes or no."
                    ),
                },
            ],
        }
    ]

    with VllmRunner(
        MODEL,
        tokenizer_mode="mistral",
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        cudagraph_capture_sizes=[1],
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.8,
    ) as runner:
        outputs = runner.model.chat(
            messages,
            sampling_params=SamplingParams(temperature=0, max_tokens=2),
        )

    assert len(outputs) == 1
    assert outputs[0].outputs[0].text, (
        "Shieldstral should generate a non-empty safety decision"
    )
