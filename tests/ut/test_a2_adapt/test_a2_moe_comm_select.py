# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the A2 branch of `select_moe_comm_method`.

The test suite monkey-patches `get_ascend_device_type` so it can run on
machines without an NPU. vllm itself is still required (we depend on
`vllm.config.VllmConfig` indirectly through the module under test).
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None


def _make_vllm_config(
    *,
    enable_expert_parallel: bool = True,
    world_size_across_dp: int = 8,
    pipeline_parallel_size: int = 1,
    num_experts: int = 256,
    num_experts_per_tok: int = 8,
):
    parallel_config = SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        world_size_across_dp=world_size_across_dp,
        pipeline_parallel_size=pipeline_parallel_size,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_experts_per_tok=num_experts_per_tok),
        get_num_experts=lambda: num_experts,
    )
    return SimpleNamespace(parallel_config=parallel_config, model_config=model_config)


@unittest.skipUnless(_VLLM_AVAILABLE, "vllm not installed; A2 select tests require vllm.config")
class TestA2AlltoallSelect(unittest.TestCase):
    """F2.1: A2 elif emits MoECommType.ALLTOALL when expected."""

    def _select(self, num_tokens, a2_moe, **kw):
        from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method
        from vllm_ascend.ascend_config import A2AdaptConfig
        from vllm_ascend.utils import AscendDeviceType

        a2_cfg = A2AdaptConfig({"moe_comm": a2_moe})

        cfg = _make_vllm_config(**kw)
        with (
            patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=512),
            patch("vllm_ascend.ascend_forward_context.get_ep_group") as ep_grp,
            patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
            patch(
                "vllm_ascend.ascend_forward_context.get_ascend_config",
                return_value=SimpleNamespace(a2_adapt_config=a2_cfg, enable_fused_mc2=0),
            ),
        ):
            ep_grp.return_value.world_size = (
                cfg.parallel_config.world_size_across_dp // cfg.parallel_config.pipeline_parallel_size
            )
            return select_moe_comm_method(num_tokens, cfg), MoECommType

    def test_force_alltoall(self):
        selected, MoECommType = self._select(num_tokens=128, a2_moe="alltoall")
        self.assertIs(selected, MoECommType.ALLTOALL)

    def test_auto_large_prefill_picks_alltoall(self):
        # num_tokens > mc2_capacity (512) + ep>=8 → ALLTOALL
        selected, MoECommType = self._select(num_tokens=2048, a2_moe="auto", world_size_across_dp=8)
        self.assertIs(selected, MoECommType.ALLTOALL)

    def test_auto_small_decode_does_not_pick_alltoall(self):
        # num_tokens <= mc2_capacity → should not return ALLTOALL via the new auto branch
        selected, MoECommType = self._select(num_tokens=64, a2_moe="auto")
        self.assertIsNot(selected, MoECommType.ALLTOALL)

    def test_none_falls_through_to_legacy(self):
        # a2_moe="none" → must follow the original A2 MC2/ALLGATHER logic
        selected, MoECommType = self._select(
            num_tokens=64, a2_moe="none", world_size_across_dp=16, num_experts=256
        )
        self.assertIn(selected, {MoECommType.MC2, MoECommType.ALLGATHER})


if __name__ == "__main__":
    unittest.main()
