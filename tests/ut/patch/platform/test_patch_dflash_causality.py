from types import SimpleNamespace

import pytest
import vllm.model_executor.models.qwen3_dflash as qwen3_dflash

from vllm_ascend.patch.platform import patch_dflash_causality

SLIDING = "sliding_attention"
FULL = "full_attention"


def _config(num_hidden_layers, layer_types=None, causal_override=None, is_causal=None):
    dflash_config = None if causal_override is None else {"causal": causal_override}
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
        is_causal=is_causal,
    )


def test_patch_rebinds_the_module_helper() -> None:
    assert qwen3_dflash._dflash_layer_causal is patch_dflash_causality._dflash_layer_causal


def test_dflash2_checkpoint_layers_are_non_causal() -> None:
    """z-lab/Qwen3.8-27B-DFlash2: top-level is_causal=false over five sliding layers.

    vLLM 0.26 without the patch resolves every one of these layers as causal.
    """
    config = _config(5, layer_types=[SLIDING] * 5, is_causal=False)

    assert [qwen3_dflash._dflash_layer_causal(config, i) for i in range(5)] == [False] * 5
    assert qwen3_dflash.dflash_has_any_non_causal(config)


@pytest.mark.parametrize(
    ("config", "expected_any_non_causal"),
    [
        # Legacy override still applies when is_causal is absent.
        (_config(2, layer_types=[SLIDING] * 2, causal_override=False), True),
        # Top-level attention semantics.
        (_config(2, layer_types=[SLIDING] * 2, is_causal=False), True),
        (_config(2, layer_types=[FULL] * 2, is_causal=True), False),
        # SWA-derived defaults: full-attention layers are non-causal, all-sliding is causal.
        (_config(2, layer_types=[SLIDING, FULL]), True),
        (_config(2, layer_types=[SLIDING] * 2), False),
        # No layer types: non-causal.
        (_config(2), True),
    ],
)
def test_dflash_has_any_non_causal(config, expected_any_non_causal) -> None:
    assert qwen3_dflash.dflash_has_any_non_causal(config) is expected_any_non_causal


def test_top_level_is_causal_wins_over_dflash_config() -> None:
    config = _config(2, layer_types=[SLIDING, FULL], causal_override=True, is_causal=False)

    assert qwen3_dflash._dflash_layer_causal(config, 0) is False
    assert qwen3_dflash._dflash_layer_causal(config, 1) is False
