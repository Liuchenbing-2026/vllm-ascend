#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Helpers shared by the vLLM Ascend scheduler implementations."""

from vllm.config import SchedulerConfig


def effective_long_prefill_token_threshold(scheduler_config: SchedulerConfig, num_eligible_reqs: int) -> int:
    """Return the long-prefill token cap to use for the current scheduling step.

    The Ascend schedulers ship their own copy of the upstream
    ``Scheduler.schedule()`` body, so upstream changes to the
    ``long_prefill_token_threshold`` handling have to be mirrored here. vLLM
    v0.31 changed them:

    * The cap only exists to stop a long prefill from starving *other*
      requests. When a single request is eligible there is nobody to starve,
      so the cap is lifted (``0``, i.e. no cap).
    * With ``--long-prefill-token-threshold-adaptive`` the cap is floored at
      an equal share of the input budget, so it never cuts a request below
      ``max_num_batched_tokens / num_eligible_reqs``.

    ``num_eligible_reqs`` must be the number of requests that may consume the
    token budget on this step, i.e. ``num_running + num_waiting`` as returned
    by ``Scheduler.get_request_counts()``.

    On vLLM releases that predate the adaptive knob the legacy behaviour (the
    configured cap always applies) is kept, so both CI lanes stay correct.
    """
    threshold = scheduler_config.long_prefill_token_threshold
    adaptive = getattr(scheduler_config, "long_prefill_token_threshold_adaptive", None)
    if adaptive is None:
        # vLLM < 0.31 has no adaptive knob: keep the legacy semantics.
        return threshold
    if num_eligible_reqs <= 1:
        return 0
    if threshold > 0 and adaptive:
        threshold = max(threshold, scheduler_config.max_num_batched_tokens // num_eligible_reqs)
    return threshold
