# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer

from vllm_ascend.patch.platform import patch_renderer_vocab_guard as mod


class _DummyRenderer(BaseRenderer):
    def render_messages(self, messages: Any, params: Any) -> Any:
        raise NotImplementedError


def _make_renderer(*, model_vocab_size: int, is_multimodal: bool, tokenizer_max_id: int | None) -> BaseRenderer:
    renderer = _DummyRenderer.__new__(_DummyRenderer)
    renderer.model_config = SimpleNamespace(
        get_vocab_size=lambda: model_vocab_size,
        is_multimodal_model=is_multimodal,
    )
    renderer.tokenizer = None if tokenizer_max_id is None else SimpleNamespace(max_token_id=tokenizer_max_id)
    return renderer


def test_text_only_model_rejects_tokenizer_only_token_ids() -> None:
    mod.apply_patch()
    renderer = _make_renderer(model_vocab_size=100, is_multimodal=False, tokenizer_max_id=150)

    renderer.validate_token_ids([])
    renderer.validate_token_ids([0, 42, 99])

    with pytest.raises(VLLMValidationError, match="Token id 100 is out of vocabulary"):
        renderer.validate_token_ids([1, 100])

    with pytest.raises(VLLMValidationError, match="Token id -1 is out of vocabulary"):
        renderer.validate_token_ids([-1, 5])


def test_multimodal_model_allows_tokenizer_placeholder_token_ids() -> None:
    mod.apply_patch()
    renderer = _make_renderer(model_vocab_size=100, is_multimodal=True, tokenizer_max_id=150)

    renderer.validate_token_ids([0, 99, 150])

    with pytest.raises(VLLMValidationError, match="Token id 151 is out of vocabulary"):
        renderer.validate_token_ids([151])


def test_validate_token_ids_without_tokenizer_uses_model_vocab() -> None:
    mod.apply_patch()
    renderer = _make_renderer(model_vocab_size=64, is_multimodal=False, tokenizer_max_id=None)

    renderer.validate_token_ids([0, 63])
    with pytest.raises(VLLMValidationError, match="Token id 64 is out of vocabulary"):
        renderer.validate_token_ids([64])
