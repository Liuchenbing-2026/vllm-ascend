# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""A2 (NpuArch 220 / 910B) MoE communication implementations.

The classes in this module are emitted only by the A2 elif in
`vllm_ascend.ascend_forward_context.select_moe_comm_method`. On
A3 / A5 / 310P they are never instantiated.

Layout
------
- `DispatchCombineA2CommImpl`  ← feature F2.2
- `PpEpGatherA2CommImpl`       ← feature F2.3 (added in a later commit)
- `PpFusedMC2A2CommImpl`       ← feature F2.4 (added in a later commit)

All A2 impls subclass an existing `MoECommMethod` and rely on the
upstream `TokenDispatcherWithMC2` / `PrepareAndFinalizeWithMC2` building
blocks. The distinct subclasses keep `MoECommType` routing clean so the
caller can A/B test individual paths via
`additional_config.a2_adapt_config.moe_comm` without touching code.

When the A2-specific runtime primitive (e.g. ``torch_npu.npu_dispatch_ffn_combine``)
becomes available, the relevant subclass body can be replaced without
touching `select_moe_comm_method` or `setup_moe_comm_method`.
"""

from __future__ import annotations

from vllm_ascend.ops.fused_moe.moe_comm_method import MC2CommImpl


class DispatchCombineA2CommImpl(MC2CommImpl):
    """A2 standalone dispatch + combine path (feature F2.2).

    Mirrors the dispatch + combine half of A3's ``FUSED_MC2 mode 1``
    without the full fused-MC2 packaging. The actual A2 HCCL primitive
    (``torch_npu.npu_dispatch_ffn_combine`` or equivalent) lives inside
    ``TokenDispatcherWithMC2`` which is reused unchanged. Bring-up:

    1. ``additional_config.a2_adapt_config.moe_comm = "dispatch_combine"``
       forces this path unconditionally.
    2. ``"auto"`` picks it when ``bs_min < num_tokens <= bs_max`` and
       ``ep_world_size <= 32`` and ``not is_draft_model``.

    Known runtime unknowns (verify on real A2 hardware):

    - The A2 build of CANN may name the HCCL fused primitive differently
      from ``dispatch_ffn_combine``. If so, replace the dispatcher /
      prepare_finalize wiring here while keeping the class signature.
    - The dispatch + combine A2 kernel imposes a batch-size constraint.
      `select_moe_comm_method` validates ``num_tokens`` against
      ``a2_adapt_config.dispatch_combine_bs_max`` before emitting this
      class; any runtime mismatch surfaces as a CANN HCCL error.
    """

    # No overrides yet — the upstream MC2 path matches the dispatch +
    # combine semantics on A2 today. When the dedicated A2 fused primitive
    # ships, override `_get_prepare_finalize` to wire it in.
