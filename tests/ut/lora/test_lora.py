from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.lora import punica_npu
from vllm_ascend.patch.worker.patch_lora_vlm_prefix import _detect_prefix


def test_qwen35_models_auto_select_bmm_expand_slice(monkeypatch):
    for model_type in ("qwen3_5", "qwen3_5_moe"):
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
        )
        monkeypatch.setattr(
            punica_npu,
            "get_current_vllm_config_or_none",
            lambda config=config: config,
        )
        assert punica_npu._current_model_type() == model_type
        assert model_type in punica_npu._QWEN35_BMM_EXPAND_SLICE_MODEL_TYPES


def test_other_models_keep_fused_expand_slice(monkeypatch):
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="llama")))
    monkeypatch.setattr(punica_npu, "get_current_vllm_config_or_none", lambda: config)
    assert punica_npu._current_model_type() == "llama"
    assert "llama" not in punica_npu._QWEN35_BMM_EXPAND_SLICE_MODEL_TYPES


def test_enable_bmm_expand_slice_is_idempotent():
    wrapper = object.__new__(punica_npu.PunicaWrapperNPU)
    wrapper._use_bmm_expand_slice = False
    wrapper.enable_bmm_expand_slice("unit test")
    wrapper.enable_bmm_expand_slice("unit test repeated")
    assert wrapper._use_bmm_expand_slice is True


def test_vlm_prefix_detection_is_automatic():
    assert (
        _detect_prefix(
            ["model.layers.0.mlp.down_proj"],
            ["language_model.model.layers.0.mlp.down_proj"],
        )
        == "language_model."
    )
    assert _detect_prefix(["model.layers.0.mlp.down_proj"], ["model.layers.0.mlp.down_proj"]) == ""


def test_current_model_type_without_config(monkeypatch):
    monkeypatch.setattr(punica_npu, "get_current_vllm_config_or_none", MagicMock(return_value=None))
    assert punica_npu._current_model_type() == ""
