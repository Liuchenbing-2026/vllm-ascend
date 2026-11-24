# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Patch BaseRenderer.validate_token_ids to enforce vocab bounds when MM inputs are disabled."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer

_VOCAB_GUARD_ATTR = "_ascend_vocab_guard_patched"


def _safe_param_names(fn: Any) -> frozenset[str] | None:
    """Return explicit parameter names for ``fn``, or ``None`` if uninspectable."""
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def _upstream_enforces_text_vocab_bound(fn: Any) -> bool:
    """Probe behaviorally whether ``fn`` already enforces text-only model vocab bounds.

    Returns True when already fixed post-upstream-pin or when probing is unsafe/incompatible
    so startup never fails.
    """
    param_names = _safe_param_names(fn)
    if param_names is None or "token_ids" not in param_names:
        return True

    probe = SimpleNamespace(
        model_config=SimpleNamespace(
            get_vocab_size=lambda: 8,
            is_multimodal_model=False,
            supports_multimodal_inputs=False,
        ),
        tokenizer=SimpleNamespace(max_token_id=16),
    )
    try:
        fn(probe, [8])
    except VLLMValidationError:
        return True
    except Exception:
        return True
    return False


def _make_validated_fn(orig_validate: Any) -> Any:
    @functools.wraps(orig_validate)
    def _validate_token_ids(
        self: BaseRenderer,
        token_ids: Sequence[int],
        *,
        parameter: str | None = None,
    ) -> None:
        """Reject out-of-vocab IDs while allowing tokenizer placeholders only when MM inputs are enabled."""
        supports_mm = getattr(
            self.model_config,
            "supports_multimodal_inputs",
            getattr(self.model_config, "is_multimodal_model", False),
        )
        if not token_ids or self.tokenizer is None or supports_mm:
            orig_validate(self, token_ids, parameter=parameter)
            return

        max_input_id = max(token_ids)
        min_input_id = min(token_ids)
        max_token_id = self.model_config.get_vocab_size() - 1
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

    return _validate_token_ids


def apply_patch() -> None:
    """Apply the text-only token ID vocabulary bound check if upstream lacks it."""
    current = getattr(BaseRenderer, "validate_token_ids", None)
    if current is None or getattr(current, _VOCAB_GUARD_ATTR, False):
        return
    if _upstream_enforces_text_vocab_bound(current):
        return

    patched = _make_validated_fn(current)
    setattr(patched, _VOCAB_GUARD_ATTR, True)
    BaseRenderer.validate_token_ids = patched  # type: ignore[method-assign]


apply_patch()
