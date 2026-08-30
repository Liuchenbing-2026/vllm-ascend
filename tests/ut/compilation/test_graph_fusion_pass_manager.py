from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config import VllmConfig
from vllm_ascend import utils
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.compilation.passes import norm_quant_fusion_pass
from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager
from vllm_ascend.utils import AscendDeviceType


class TestGraphFusionPassManagerConfig(TestBase):
    def tearDown(self):
        clear_ascend_config()

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_configure_consumes_validated_ascend_compilation_config(self, mock_platform):
        vllm_config = VllmConfig()
        vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": "false",
                "fuse_qknorm_rope": "false",
                "fuse_muls_add": "false",
            }
        }
        init_ascend_config(vllm_config)

        manager = GraphFusionPassManager()
        manager.configure(vllm_config)

        self.assertFalse(manager.ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(manager.ascend_compilation_config.fuse_qknorm_rope)
        self.assertFalse(manager.ascend_compilation_config.fuse_muls_add)
        self.assertEqual(manager.passes, [])


@pytest.mark.parametrize(
    ("hf_config", "device_type", "expected_passes"),
    [
        (SimpleNamespace(model_type="mistral4"), AscendDeviceType.A2, 0),
        (
            SimpleNamespace(
                model_type="mistral3",
                text_config=SimpleNamespace(model_type="mistral4"),
            ),
            AscendDeviceType.A2,
            0,
        ),
        (SimpleNamespace(model_type="ministral3"), AscendDeviceType.A2, 0),
        (
            SimpleNamespace(
                model_type="mistral3",
                text_config=SimpleNamespace(model_type="ministral3"),
            ),
            AscendDeviceType.A2,
            0,
        ),
        (SimpleNamespace(model_type="mistral4"), AscendDeviceType.A3, 1),
        (SimpleNamespace(model_type="deepseek_v3"), AscendDeviceType.A2, 1),
    ],
)
def test_mistral_a2_skips_unsupported_add_rms_norm_quant_patterns(
    monkeypatch,
    hf_config,
    device_type,
    expected_passes,
):
    vllm_config = VllmConfig()
    vllm_config.additional_config = {
        "ascend_compilation_config": {
            "fuse_norm_quant": True,
            "fuse_qknorm_rope": False,
            "fuse_muls_add": False,
        }
    }
    vllm_config.model_config = SimpleNamespace(
        hf_config=hf_config,
        enforce_eager=True,
        is_deepseek_mla=False,
        get_total_num_kv_heads=lambda: 0,
    )
    init_ascend_config(vllm_config)

    pass_factory = MagicMock()
    monkeypatch.setattr(
        norm_quant_fusion_pass,
        "AddRMSNormQuantFusionPass",
        pass_factory,
    )
    monkeypatch.setattr(utils, "is_310p", lambda: False)
    monkeypatch.setattr(
        utils,
        "get_ascend_device_type",
        lambda: device_type,
    )

    manager = GraphFusionPassManager()
    manager.configure(vllm_config)

    assert len(manager.passes) == expected_passes
    if expected_passes:
        pass_factory.assert_called_once_with(vllm_config)
    else:
        pass_factory.assert_not_called()
