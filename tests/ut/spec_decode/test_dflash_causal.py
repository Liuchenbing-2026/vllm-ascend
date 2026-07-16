# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash draft-attention causality resolution and JetSpec config adaptation.

JetSpec heads are causal DFlash drafters; running them with the standard
non-causal draft attention silently collapses the acceptance length, so the
config-only branches here are worth pinning on CPU.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.spec_decode.dflash_proposer import resolve_dflash_draft_attn_causal

QWEN3_DFLASH_MODULE = "vllm.model_executor.models.qwen3_dflash"


def _draft_hf_config(num_hidden_layers=5, layer_types=None, dflash_config=None):
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
    )


@pytest.mark.parametrize(
    "dflash_config,layer_types,expected",
    [
        # JetSpec-style head: explicit causal override wins.
        ({"causal": True}, ["full_attention"] * 5, True),
        ({"causal": True}, None, True),
        # Explicit non-causal override.
        ({"causal": False}, None, False),
        # Standard DFlash diffusion head: no override, no layer_types
        # -> non-causal draft attention.
        (None, None, False),
        ({}, None, False),
        # SWA-derived per-layer causality: any full-attention layer makes
        # the draft require non-causal support; all-SWA drafters are causal.
        (None, ["sliding_attention", "full_attention"] + ["full_attention"] * 3, False),
        (None, ["sliding_attention"] * 5, True),
    ],
)
def test_resolve_draft_attn_causal(dflash_config, layer_types, expected):
    config = _draft_hf_config(layer_types=layer_types, dflash_config=dflash_config)
    assert resolve_dflash_draft_attn_causal(config) is expected


@pytest.mark.parametrize(
    "dflash_config,layer_types,expected",
    [
        ({"causal": True}, None, True),
        ({"causal": False}, ["sliding_attention"] * 5, False),
        ({}, None, False),
        (None, None, False),
        # The fallback must mirror the model-side per-layer resolution, not
        # collapse to a plain switch: all-SWA drafters stay causal.
        (None, ["sliding_attention"] * 5, True),
        (None, ["sliding_attention", "full_attention"], False),
    ],
)
def test_resolve_draft_attn_causal_legacy_fallback(dflash_config, layer_types, expected):
    """vLLM without ``dflash_has_any_non_causal`` (pre-#48167) must resolve
    to the same causality the draft layers were built with."""
    config = _draft_hf_config(layer_types=layer_types, dflash_config=dflash_config)
    stub = types.ModuleType(QWEN3_DFLASH_MODULE)  # lacks dflash_has_any_non_causal
    with patch.dict(sys.modules, {QWEN3_DFLASH_MODULE: stub}):
        assert resolve_dflash_draft_attn_causal(config) is expected


def _spec_vllm_config(method="dflash", num_speculative_tokens=15, dflash_config=None):
    hf_config = SimpleNamespace(dflash_config=dflash_config)
    draft_model_config = SimpleNamespace(hf_config=hf_config)
    speculative_config = SimpleNamespace(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        draft_model_config=draft_model_config,
    )
    return SimpleNamespace(speculative_config=speculative_config)


class TestAdaptJetspecDraftConfig:
    def test_translates_causal_head(self):
        dflash_config = {"causal_head": True, "block_size": 16, "mask_token_id": 151669}
        vllm_config = _spec_vllm_config(dflash_config=dflash_config)

        NPUPlatform._adapt_jetspec_draft_config(vllm_config)

        assert dflash_config["causal"] is True
        # The checkpoint's own key is preserved for hf_config fidelity.
        assert dflash_config["causal_head"] is True

    def test_existing_causal_key_wins(self):
        dflash_config = {"causal_head": True, "causal": False}
        vllm_config = _spec_vllm_config(dflash_config=dflash_config)

        NPUPlatform._adapt_jetspec_draft_config(vllm_config)

        assert dflash_config["causal"] is False
        assert dflash_config["causal_head"] is True

    def test_translation_is_idempotent(self):
        dflash_config = {"causal_head": True, "block_size": 16}
        vllm_config = _spec_vllm_config(dflash_config=dflash_config)

        NPUPlatform._adapt_jetspec_draft_config(vllm_config)
        NPUPlatform._adapt_jetspec_draft_config(vllm_config)

        assert dflash_config["causal"] is True
        assert dflash_config["causal_head"] is True

    def test_warns_on_unrecognized_causality_key(self):
        dflash_config = {"is_causal": True, "block_size": 16}
        vllm_config = _spec_vllm_config(dflash_config=dflash_config)

        with patch("vllm_ascend.platform.logger") as mock_logger:
            NPUPlatform._adapt_jetspec_draft_config(vllm_config)

        mock_logger.warning_once.assert_called_once()
        assert "causal" not in dflash_config

    @pytest.mark.parametrize(
        "vllm_config",
        [
            SimpleNamespace(speculative_config=None),
            _spec_vllm_config(method="eagle3"),
            _spec_vllm_config(dflash_config=None),
        ],
    )
    def test_noop_when_not_dflash_draft(self, vllm_config):
        NPUPlatform._adapt_jetspec_draft_config(vllm_config)  # must not raise

    def test_warns_when_query_slots_exceed_trained_block(self):
        dflash_config = {"causal": True, "block_size": 16}
        vllm_config = _spec_vllm_config(num_speculative_tokens=16, dflash_config=dflash_config)

        with patch("vllm_ascend.platform.logger") as mock_logger:
            NPUPlatform._adapt_jetspec_draft_config(vllm_config)

        mock_logger.warning_once.assert_called_once()

    def test_no_warning_at_trained_block_boundary(self):
        dflash_config = {"causal": True, "block_size": 16}
        vllm_config = _spec_vllm_config(num_speculative_tokens=15, dflash_config=dflash_config)

        with patch("vllm_ascend.platform.logger") as mock_logger:
            NPUPlatform._adapt_jetspec_draft_config(vllm_config)

        mock_logger.warning_once.assert_not_called()
