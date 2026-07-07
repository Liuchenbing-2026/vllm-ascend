#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# ---------------------------------------------------------------------------
# DIAGNOSTIC: confirm/kill the "graph-replay resource accumulation" hypothesis
# for the intermittent FDO hang (bug.md ⑰ / ㉙ / ㉚).
#
# Lead: attention_v1.py forward does, at method-body level (NOT under an
# `if capturing:` guard):
#     graph_params.events[num_tokens].append(event)
#     graph_params.attn_params[num_tokens].append(attn_params)
# During normal aclgraph capture this appends ONCE per captured size (bounded).
# BUT if a decode step hits a num_tokens size that was NOT captured, the forward
# is re-entered and appends AGAIN -> events[num_tokens] / attn_params[num_tokens]
# grow across requests -> replay iterates an ever-longer zip -> mismatch /
# resource exhaustion -> hang AFTER a few requests (exactly the ㉙ pattern).
#
# This probe wraps update_graph_params (runs every decode replay) and logs, per
# call, the length of events/attn_params for the current size plus the TOTAL
# across all sizes. Run FDO, send ~10 requests, grep `[evprobe]`:
#   * total_events GROWS across requests  -> accumulation CONFIRMED. Fix: only
#     append during capture (guard the append), or clear per size before capture.
#   * total_events STAYS FLAT             -> not accumulation; wait for the
#     watchdog hang stack (patch_hang_watchdog) to locate the real stall.
#
# Enable with VLLM_ASCEND_EVPROBE=1 (no-op otherwise). Remove once resolved.
# ---------------------------------------------------------------------------
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_ASCEND_EVPROBE", "0") not in (
    "0", "", "false", "False")

if _ENABLED:
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.compilation.acl_graph import get_graph_params

    # Accessing a staticmethod via the class yields the plain function.
    _orig_update_graph_params = AscendAttentionBackendImpl.update_graph_params

    def _sizes(d, key):
        try:
            v = d.get(key)
            return len(v) if v is not None else None
        except Exception:  # noqa: BLE001
            return "err"

    def update_graph_params(*args, **kwargs):
        try:
            num_tokens = args[2] if len(args) > 2 else kwargs.get("num_tokens")
            gp = get_graph_params()
            total_events = sum(len(v) for v in gp.events.values())
            total_params = sum(len(v) for v in gp.attn_params.values())
            logger.warning(
                "[evprobe] num_tokens=%s events[nt]=%s attn_params[nt]=%s "
                "sizes=%d total_events=%d total_attn_params=%d",
                num_tokens, _sizes(gp.events, num_tokens),
                _sizes(gp.attn_params, num_tokens), len(gp.events),
                total_events, total_params)
        except Exception as e:  # noqa: BLE001 - never break the replay path
            logger.warning("[evprobe] read failed: %s", e)
        return _orig_update_graph_params(*args, **kwargs)

    AscendAttentionBackendImpl.update_graph_params = staticmethod(
        update_graph_params)
    logger.warning("[evprobe] update_graph_params wrapped "
                   "(VLLM_ASCEND_EVPROBE on) — watch total_events across "
                   "requests: growing => accumulation confirmed.")
