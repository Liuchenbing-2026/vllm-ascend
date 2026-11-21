# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Patch BaseRenderer.validate_token_ids to reject tokenizer-only token IDs on text-only models."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Sequence
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer

_VOCAB_GUARD_ATTR = "_ascend_vocab_guard_patched"


def _upstream_checks_multimodal(fn: Any) -> bool:
    try:
        return "is_multimodal_model" in inspect.getsource(fn)
    except (OSError, TypeError):
        return False


def _validate_token_ids(
    self: BaseRenderer,
    token_ids: Sequence[int],
    *,
    parameter: str | None = None,
) -> None:
    """Reject out-of-vocab IDs while permitting tokenizer placeholders only for multimodal models."""
    tokenizer = self.tokenizer
    if not token_ids:
        return

    max_input_id = max(token_ids)
    min_input_id = min(token_ids)

    # Only multimodal placeholders may exceed the model vocab because they are
    # replaced before the embedding lookup. A text-only model embeds every ID.
    max_token_id = self.model_config.get_vocab_size() - 1
    if tokenizer is not None and self.model_config.is_multimodal_model:
        max_token_id = max(tokenizer.max_token_id, max_token_id)

    if min_input_id < 0:
        raise VLLMValidationError(
            f"Token id {min_input_id} is out of vocabulary",
            parameter=parameter,
        )
    if max_input_id > max_token_id:
        raise VLLMValidationError(
            f"Token id {max_input_id} is out of vocabulary",
            parameter=parameter,
        )


def apply_patch() -> None:
    """Apply the text-only token ID vocabulary bound check if upstream lacks it."""
    current = getattr(BaseRenderer, "validate_token_ids", None)
    if current is None or getattr(current, _VOCAB_GUARD_ATTR, False):
        return
    if _upstream_checks_multimodal(current):
        return

    patched = functools.wraps(current)(_validate_token_ids)
    setattr(patched, _VOCAB_GUARD_ATTR, True)
    BaseRenderer.validate_token_ids = patched  # type: ignore[method-assign]


apply_patch()
