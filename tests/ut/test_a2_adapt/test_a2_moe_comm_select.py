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


@unittest.skipUnless(_VLLM_AVAILABLE, "vllm not installed; A2 select tests require vllm.config")
class TestA2DispatchCombineSelect(unittest.TestCase):
    """F2.2: dispatch_combine force + auto-pick."""

    def _select(self, num_tokens, a2_moe, *, bs_max=None, is_draft_model=False, **kw):
        from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method
        from vllm_ascend.ascend_config import A2AdaptConfig
        from vllm_ascend.utils import AscendDeviceType

        user_cfg = {"moe_comm": a2_moe}
        if bs_max is not None:
            user_cfg["dispatch_combine_bs_max"] = bs_max
        a2_cfg = A2AdaptConfig(user_cfg)
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
            return select_moe_comm_method(num_tokens, cfg, is_draft_model=is_draft_model), MoECommType

    def test_force_dispatch_combine(self):
        selected, MoECommType = self._select(num_tokens=128, a2_moe="dispatch_combine")
        self.assertIs(selected, MoECommType.DISPATCH_COMBINE)

    def test_auto_mid_batch_picks_dispatch_combine(self):
        # 1 < num_tokens=256 <= mc2_capacity=512, ep=8 <= 32 → DISPATCH_COMBINE
        selected, MoECommType = self._select(num_tokens=256, a2_moe="auto", world_size_across_dp=8)
        self.assertIs(selected, MoECommType.DISPATCH_COMBINE)

    def test_auto_large_prefill_still_alltoall(self):
        # num_tokens > capacity falls past the dispatch_combine band into ALLTOALL.
        selected, MoECommType = self._select(num_tokens=2048, a2_moe="auto", world_size_across_dp=8)
        self.assertIs(selected, MoECommType.ALLTOALL)

    def test_draft_model_skips_dispatch_combine(self):
        selected, MoECommType = self._select(num_tokens=256, a2_moe="auto", is_draft_model=True)
        self.assertIsNot(selected, MoECommType.DISPATCH_COMBINE)

    def test_user_bs_max_caps_dispatch_combine(self):
        # If user sets bs_max=128, num_tokens=256 should NOT pick DISPATCH_COMBINE.
        selected, MoECommType = self._select(num_tokens=256, a2_moe="auto", bs_max=128)
        self.assertIsNot(selected, MoECommType.DISPATCH_COMBINE)


@unittest.skipUnless(_VLLM_AVAILABLE, "vllm not installed; A2 select tests require vllm.config")
class TestA2PpEpGatherSelect(unittest.TestCase):
    """F2.3: PP + ep_gather force + auto pickup + PP=1 fall-back."""

    def _select(self, num_tokens, a2_moe, pp=2, **kw):
        from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method
        from vllm_ascend.ascend_config import A2AdaptConfig
        from vllm_ascend.utils import AscendDeviceType

        a2_cfg = A2AdaptConfig({"moe_comm": a2_moe})
        cfg = _make_vllm_config(pipeline_parallel_size=pp, world_size_across_dp=16, **kw)
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

    def test_force_pp_ep_gather_pp2(self):
        selected, MoECommType = self._select(num_tokens=128, a2_moe="pp_ep_gather", pp=2)
        self.assertIs(selected, MoECommType.PP_EP_GATHER)

    def test_force_pp_ep_gather_pp1_falls_back(self):
        # pp=1 cannot honour the request; must fall back with a warning.
        selected, MoECommType = self._select(num_tokens=128, a2_moe="pp_ep_gather", pp=1)
        self.assertIs(selected, MoECommType.ALLGATHER)

    def test_auto_pp2_small_picks_pp_ep_gather(self):
        # pp=2 + num_tokens<=mc2_capacity + ep>1 → PP_EP_GATHER (before DISPATCH_COMBINE).
        selected, MoECommType = self._select(num_tokens=128, a2_moe="auto", pp=2)
        self.assertIs(selected, MoECommType.PP_EP_GATHER)

    def test_auto_pp1_falls_through(self):
        # pp=1 + auto → no PP path, fall through to DISPATCH_COMBINE / ALLTOALL / etc.
        selected, MoECommType = self._select(num_tokens=128, a2_moe="auto", pp=1)
        self.assertIsNot(selected, MoECommType.PP_EP_GATHER)


@unittest.skipUnless(_VLLM_AVAILABLE, "vllm not installed; A2 select tests require vllm.config")
class TestA2PpFusedMC2Select(unittest.TestCase):
    """F2.4: PP + fused MC2 force + auto pickup priority over PP_EP_GATHER."""

    def _select(self, num_tokens, a2_moe, pp=2, ep=8, **kw):
        from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method
        from vllm_ascend.ascend_config import A2AdaptConfig
        from vllm_ascend.utils import AscendDeviceType

        a2_cfg = A2AdaptConfig({"moe_comm": a2_moe})
        cfg = _make_vllm_config(pipeline_parallel_size=pp, world_size_across_dp=ep * pp, **kw)
        with (
            patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=512),
            patch("vllm_ascend.ascend_forward_context.get_ep_group") as ep_grp,
            patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
            patch(
                "vllm_ascend.ascend_forward_context.get_ascend_config",
                return_value=SimpleNamespace(a2_adapt_config=a2_cfg, enable_fused_mc2=1),
            ),
        ):
            ep_grp.return_value.world_size = (
                cfg.parallel_config.world_size_across_dp // cfg.parallel_config.pipeline_parallel_size
            )
            return select_moe_comm_method(num_tokens, cfg), MoECommType

    def test_force_pp_fused_inside_capacity(self):
        selected, MoECommType = self._select(num_tokens=128, a2_moe="pp_fused", pp=2, ep=8)
        self.assertIs(selected, MoECommType.PP_FUSED_MC2)

    def test_force_pp_fused_overflow_falls_back(self):
        # num_tokens > mc2_capacity → fall back to PP_EP_GATHER instead of PP_FUSED_MC2.
        selected, MoECommType = self._select(num_tokens=2048, a2_moe="pp_fused", pp=2, ep=8)
        self.assertIs(selected, MoECommType.PP_EP_GATHER)

    def test_force_pp_fused_pp1_falls_back(self):
        selected, MoECommType = self._select(num_tokens=128, a2_moe="pp_fused", pp=1, ep=8)
        self.assertIs(selected, MoECommType.ALLGATHER)

    def test_auto_pp_fused_priority_over_pp_ep_gather(self):
        # auto + pp>1 + ep<=32 + small batch → PP_FUSED_MC2 (NOT PP_EP_GATHER).
        selected, MoECommType = self._select(num_tokens=128, a2_moe="auto", pp=2, ep=8)
        self.assertIs(selected, MoECommType.PP_FUSED_MC2)

    def test_auto_pp_large_ep_falls_to_pp_ep_gather(self):
        # auto + pp>1 + ep>32 → PP_EP_GATHER (PP_FUSED's ep<=32 guard fails).
        selected, MoECommType = self._select(num_tokens=128, a2_moe="auto", pp=2, ep=64)
        self.assertIs(selected, MoECommType.PP_EP_GATHER)


if __name__ == "__main__":
    unittest.main()
