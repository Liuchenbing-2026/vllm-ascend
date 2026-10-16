# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import PretrainedConfig

from vllm_ascend.patch.platform.patch_speculative_config import hf_config_override


def _make_config(model_type, architectures, **kwargs):
    cfg = PretrainedConfig()
    cfg.model_type = model_type
    cfg.architectures = architectures
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.parametrize(
    "model_type,architectures,expected_architectures",
    [
        ("deepseek_v3", ["DeepseekV3ForCausalLM"], ["DeepSeekMTPModel"]),
        ("deepseek_v32", ["DeepseekV32ForCausalLM"], ["DeepSeekMTPModel"]),
        ("glm_moe_dsa", ["GlmMoeDsaForCausalLM"], ["DeepSeekMTPModel"]),
        ("deepseek_v4", ["DeepSeekV4ForCausalLM"], ["DeepSeekV4MTPModel"]),
    ],
)
def test_hf_config_override_rewrites_target_model_type(model_type, architectures, expected_architectures):
    cfg = _make_config(model_type, architectures, num_nextn_predict_layers=1)
    out = hf_config_override(cfg)
    assert out.model_type == "deepseek_mtp"
    assert out.architectures == expected_architectures


@pytest.mark.parametrize("model_type", ["deepseek_v3", "deepseek_v32", "glm_moe_dsa"])
def test_hf_config_override_exposes_n_predict_for_dense_mtp(model_type):
    cfg = _make_config(model_type, ["TargetForCausalLM"], num_nextn_predict_layers=3)
    out = hf_config_override(cfg)
    assert out.n_predict == 3


def test_hf_config_override_accepts_already_deepseek_mtp_config():
    """#18734: a config that is already normalized to deepseek_mtp must not
    hit an unbound `target_model_type` local variable."""
    cfg = _make_config("deepseek_mtp", ["DeepSeekMTPModel"], num_nextn_predict_layers=1)
    out = hf_config_override(cfg)
    assert out.model_type == "deepseek_mtp"
    assert out.architectures == ["DeepSeekMTPModel"]
    assert out.n_predict == 1


def test_hf_config_override_is_idempotent_for_mtp_configs():
    first = hf_config_override(_make_config("deepseek_v3", ["DeepseekV3ForCausalLM"], num_nextn_predict_layers=2))
    second = hf_config_override(first)
    assert second.model_type == "deepseek_mtp"
    assert second.architectures == ["DeepSeekMTPModel"]
    assert second.n_predict == 2


def test_hf_config_override_keeps_v4_mtp_architecture_when_already_normalized():
    cfg = _make_config("deepseek_v4", ["DeepSeekV4ForCausalLM"], num_nextn_predict_layers=1)
    normalized = hf_config_override(cfg)
    assert normalized.model_type == "deepseek_mtp"
    assert normalized.architectures == ["DeepSeekV4MTPModel"]
    assert normalized.n_predict == 1


def test_hf_config_override_has_no_implicit_original_model_type_capture():
    """Regression guard for #18734: `hf_config_override` must not depend on a
    local variable that is only bound on one branch. Asserting at source level
    keeps the check meaningful even when the runtime path is not exercised."""
    import ast
    import inspect

    from vllm_ascend.patch.platform import patch_speculative_config

    tree = ast.parse(inspect.getsource(patch_speculative_config.hf_config_override))
    loaded_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    assert "target_model_type" not in loaded_names
