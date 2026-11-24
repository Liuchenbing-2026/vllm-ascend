# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer

from vllm_ascend.patch.platform import patch_renderer_vocab_guard as mod


class _DummyRenderer(BaseRenderer):
    def render_messages(self, messages: Any, params: Any) -> Any:
        raise NotImplementedError


def _make_renderer(
    *,
    model_vocab_size: int,
    is_multimodal: bool,
    tokenizer_max_id: int | None,
    supports_multimodal_inputs: bool | None = None,
) -> BaseRenderer:
    renderer = _DummyRenderer.__new__(_DummyRenderer)
    config_kwargs: dict[str, Any] = {
        "get_vocab_size": lambda: model_vocab_size,
        "is_multimodal_model": is_multimodal,
    }
    if supports_multimodal_inputs is not None:
        config_kwargs["supports_multimodal_inputs"] = supports_multimodal_inputs
    renderer.model_config = SimpleNamespace(**config_kwargs)
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


def test_apply_patch_strict_noop_post_upstream_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    def _native_fixed_validate(
        self: Any,
        token_ids: Sequence[int],
        *,
        parameter: str | None = None,
    ) -> None:
        max_id = self.model_config.get_vocab_size() - 1
        if self.tokenizer is not None and self.model_config.is_multimodal_model:
            max_id = max(self.tokenizer.max_token_id, max_id)
        if token_ids and max(token_ids) > max_id:
            raise VLLMValidationError("out of vocab", parameter=parameter)

    monkeypatch.setattr(BaseRenderer, "validate_token_ids", _native_fixed_validate)
    mod.apply_patch()
    assert BaseRenderer.validate_token_ids is _native_fixed_validate


def test_apply_patch_failsafe_on_compiled_or_raising_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raising_validate(self: Any, token_ids: Sequence[int], *, parameter: str | None = None) -> None:
        raise RuntimeError("unexpected internal signature")

    monkeypatch.setattr(BaseRenderer, "validate_token_ids", _raising_validate)
    mod.apply_patch()
    assert BaseRenderer.validate_token_ids is _raising_validate

    monkeypatch.setattr(BaseRenderer, "validate_token_ids", len)
    mod.apply_patch()
    assert BaseRenderer.validate_token_ids is len


def test_language_model_only_multimodal_weights_rejects_tokenizer_only_token_ids() -> None:
    mod.apply_patch()
    # `--language-model-only` / `--limit-mm-per-prompt 0` keeps is_multimodal_model=True
    # while supports_multimodal_inputs is False.
    text_only_mm = _make_renderer(
        model_vocab_size=100,
        is_multimodal=True,
        tokenizer_max_id=150,
        supports_multimodal_inputs=False,
    )
    text_only_mm.validate_token_ids([0, 99])
    with pytest.raises(VLLMValidationError, match="Token id 100 is out of vocabulary"):
        text_only_mm.validate_token_ids([1, 100])
    with pytest.raises(VLLMValidationError, match="Token id -1 is out of vocabulary"):
        text_only_mm.validate_token_ids([-1, 5])

    # EPD `--language-model-only --enable-mm-embeds` sets supports_multimodal_inputs=True.
    mm_embeds = _make_renderer(
        model_vocab_size=100,
        is_multimodal=True,
        tokenizer_max_id=150,
        supports_multimodal_inputs=True,
    )
    mm_embeds.validate_token_ids([0, 99, 150])
    with pytest.raises(VLLMValidationError, match="Token id 151 is out of vocabulary"):
        mm_embeds.validate_token_ids([151])


def test_probe_populates_supports_multimodal_inputs_for_upstream_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unfixed_reads_supports_mm(
        self: Any,
        token_ids: Sequence[int],
        *,
        parameter: str | None = None,
    ) -> None:
        _ = self.model_config.supports_multimodal_inputs
        max_id = max(self.tokenizer.max_token_id, self.model_config.get_vocab_size() - 1)
        if token_ids and max(token_ids) > max_id:
            raise VLLMValidationError("out of vocab", parameter=parameter)

    monkeypatch.setattr(BaseRenderer, "validate_token_ids", _unfixed_reads_supports_mm)
    mod.apply_patch()
    assert getattr(BaseRenderer.validate_token_ids, mod._VOCAB_GUARD_ATTR, False) is True

    def _fixed_reads_supports_mm(
        self: Any,
        token_ids: Sequence[int],
        *,
        parameter: str | None = None,
    ) -> None:
        max_id = self.model_config.get_vocab_size() - 1
        if self.tokenizer is not None and self.model_config.supports_multimodal_inputs:
            max_id = max(self.tokenizer.max_token_id, max_id)
        if token_ids and max(token_ids) > max_id:
            raise VLLMValidationError("out of vocab", parameter=parameter)

    monkeypatch.setattr(BaseRenderer, "validate_token_ids", _fixed_reads_supports_mm)
    mod.apply_patch()
    assert BaseRenderer.validate_token_ids is _fixed_reads_supports_mm
