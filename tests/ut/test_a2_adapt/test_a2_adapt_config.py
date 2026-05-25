# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the A2 adaptation infrastructure.

These tests cover the pure-Python wiring (A2AdaptConfig dataclass + env
defaults). They do NOT touch torch_npu / NPU runtime and are safe to run
on machines without an NPU.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(_VLLM_AVAILABLE, "vllm not installed; A2AdaptConfig requires vllm logger")
class TestA2AdaptConfigDefaults(unittest.TestCase):
    """A2AdaptConfig honours additional_config > env > default precedence."""

    def setUp(self):
        for key in (
            "VLLM_ASCEND_A2_DSA_CP_DISABLE_ALL2ALL",
            "VLLM_ASCEND_A2_MOE_COMM",
            "VLLM_ASCEND_A2_DISPATCH_COMBINE_BS_MIN",
            "VLLM_ASCEND_A2_DISPATCH_COMBINE_BS_MAX",
        ):
            os.environ.pop(key, None)

    def _make(self, **kwargs):
        # Import lazily so that failures during import surface inside the test.
        from vllm_ascend.ascend_config import A2AdaptConfig

        return A2AdaptConfig(kwargs)

    def test_defaults(self):
        cfg = self._make()
        self.assertTrue(cfg.dsa_cp_disable_all2all)
        self.assertEqual(cfg.moe_comm, "auto")
        self.assertEqual(cfg.dispatch_combine_bs_min, 1)
        self.assertIsNone(cfg.dispatch_combine_bs_max)

    def test_additional_config_override(self):
        cfg = self._make(
            dsa_cp_disable_all2all=False,
            moe_comm="alltoall",
            dispatch_combine_bs_min=8,
            dispatch_combine_bs_max=512,
        )
        self.assertFalse(cfg.dsa_cp_disable_all2all)
        self.assertEqual(cfg.moe_comm, "alltoall")
        self.assertEqual(cfg.dispatch_combine_bs_min, 8)
        self.assertEqual(cfg.dispatch_combine_bs_max, 512)

    def test_env_override(self):
        os.environ["VLLM_ASCEND_A2_DSA_CP_DISABLE_ALL2ALL"] = "0"
        os.environ["VLLM_ASCEND_A2_MOE_COMM"] = "pp_fused"
        os.environ["VLLM_ASCEND_A2_DISPATCH_COMBINE_BS_MIN"] = "16"
        os.environ["VLLM_ASCEND_A2_DISPATCH_COMBINE_BS_MAX"] = "256"
        cfg = self._make()
        self.assertFalse(cfg.dsa_cp_disable_all2all)
        self.assertEqual(cfg.moe_comm, "pp_fused")
        self.assertEqual(cfg.dispatch_combine_bs_min, 16)
        self.assertEqual(cfg.dispatch_combine_bs_max, 256)

    def test_additional_config_beats_env(self):
        os.environ["VLLM_ASCEND_A2_MOE_COMM"] = "pp_fused"
        cfg = self._make(moe_comm="dispatch_combine")
        self.assertEqual(cfg.moe_comm, "dispatch_combine")

    def test_invalid_moe_comm_raises(self):
        with self.assertRaises(ValueError):
            self._make(moe_comm="bogus")

    def test_invalid_bs_bounds_raises(self):
        with self.assertRaises(ValueError):
            self._make(dispatch_combine_bs_min=10, dispatch_combine_bs_max=5)


if __name__ == "__main__":
    unittest.main()
