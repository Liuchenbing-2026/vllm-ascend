# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from unittest.mock import Mock

import pytest

from vllm_ascend.compilation import acl_graph


def test_capture_guard_removed_by_core(monkeypatch):
    monkeypatch.delattr(acl_graph.monitor, "validate_cudagraph_capturing_enabled", raising=False)
    acl_graph.validate_cudagraph_capturing_enabled()


def test_legacy_capture_guard_allows_capture(monkeypatch):
    guard = Mock()
    monkeypatch.setattr(acl_graph.monitor, "validate_cudagraph_capturing_enabled", guard, raising=False)
    acl_graph.validate_cudagraph_capturing_enabled()
    guard.assert_called_once_with()


def test_legacy_capture_guard_rejection_propagates(monkeypatch):
    guard = Mock(side_effect=RuntimeError("capture disabled by legacy core"))
    monkeypatch.setattr(acl_graph.monitor, "validate_cudagraph_capturing_enabled", guard, raising=False)
    with pytest.raises(RuntimeError, match="capture disabled by legacy core"):
        acl_graph.validate_cudagraph_capturing_enabled()
