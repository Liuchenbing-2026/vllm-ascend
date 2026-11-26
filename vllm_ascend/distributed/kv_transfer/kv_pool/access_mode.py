# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Cross-version access_mode helpers for KV offloading and AscendStore.

Supports vLLM's ``access_mode`` contract (``read`` | ``write`` | ``readwrite``)
on pinned releases prior to upstream inclusion as well as newer vLLM builds
that already expose ``vllm.v1.kv_offload.config.OffloadAccessMode``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any


class _FallbackOffloadAccessMode(Enum):
    """Direction(s) in which an offloading connector or tier may transfer data.

    Compatibility fallback for vLLM versions prior to upstream ``access_mode``
    support; can be removed once the pinned vLLM baseline includes it.
    """

    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"

    @classmethod
    def from_config(
        cls,
        value: str | _FallbackOffloadAccessMode | None,
    ) -> _FallbackOffloadAccessMode:
        """Parse a configured ``access_mode`` value, defaulting to ``readwrite``."""
        if value is None:
            return cls.READWRITE
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"access_mode must be a string, got {type(value).__name__}: {value!r}")
        try:
            return cls(value.strip().lower())
        except ValueError:
            valid = ", ".join(repr(access_mode.value) for access_mode in cls)
            raise ValueError(f"Invalid access_mode {value!r}; expected one of {valid}") from None

    @property
    def can_read(self) -> bool:
        """Whether the connector or tier may serve loads."""
        return self is not _FallbackOffloadAccessMode.WRITE

    @property
    def can_write(self) -> bool:
        """Whether the connector or tier may accept stores."""
        return self is not _FallbackOffloadAccessMode.READ


if TYPE_CHECKING:
    OffloadAccessMode = _FallbackOffloadAccessMode
else:
    try:
        from vllm.v1.kv_offload.config import OffloadAccessMode
    except ImportError:
        OffloadAccessMode = _FallbackOffloadAccessMode


def has_upstream_access_mode() -> bool:
    """Return True when the active vLLM runtime defines ``OffloadAccessMode``."""
    return OffloadAccessMode is not _FallbackOffloadAccessMode


def parse_access_mode(extra_config: Mapping[str, Any] | None) -> OffloadAccessMode:
    """Parse ``access_mode`` from a connector or tier configuration mapping."""
    if extra_config is None:
        return OffloadAccessMode.READWRITE
    return OffloadAccessMode.from_config(extra_config.get("access_mode"))
