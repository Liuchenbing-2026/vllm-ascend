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
"""Adaptive long-prefill token threshold on the Ascend schedulers.

Upstream vLLM v0.31 changed `long_prefill_token_threshold`:

* a single eligible request is never capped (there is nobody to starve), and
* `--long-prefill-token-threshold-adaptive` floors the cap at an equal share
  of the input budget.

The Ascend schedulers copy the upstream `schedule()` body, so they have to
apply the same rules through `effective_long_prefill_token_threshold`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.core.test_dyntra_lb_scheduler import (
    create_dyntra_lb_scheduler,
    make_dyntra_test_config,
)
from tests.ut.kv_offload.utils import create_request
from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBScheduler
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkScheduler
from vllm_ascend.core.scheduler_utils import effective_long_prefill_token_threshold
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

LOAD_PREFIX = "effective_long_prefill_token_threshold"
PATCH_BALANCE = "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config"


def _profiling_chunk_config():
    # Mirrors the real AscendConfig entry; the scheduler reads every value at
    # construction time, so the stand-in has to carry all of them.
    config = MagicMock()
    config.enabled = False
    config.smooth_factor = 0.8
    config.min_chunk = 256
    config.max_fit_chunk = 8192
    return config


def _ascend_config(balance_enabled=False):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_balance_scheduling=balance_enabled,
            profiling_chunk_config=_profiling_chunk_config(),
            short_request_first_config=SimpleNamespace(enabled=False, threshold=256, long_max_wait_ms=2000.0),
        )
    )


def _adaptive_config(threshold=8, adaptive=None, budget=1024):
    fields = {
        "long_prefill_token_threshold": threshold,
        "max_num_batched_tokens": budget,
    }
    if adaptive is not None:
        fields["long_prefill_token_threshold_adaptive"] = adaptive
    return SimpleNamespace(**fields)


def test_threshold_is_kept_on_releases_without_the_adaptive_knob():
    config = _adaptive_config()
    assert effective_long_prefill_token_threshold(config, 1) == 8
    assert effective_long_prefill_token_threshold(config, 4) == 8


@pytest.mark.parametrize("num_eligible_reqs", [0, 1])
def test_single_eligible_request_is_never_capped(num_eligible_reqs):
    assert effective_long_prefill_token_threshold(_adaptive_config(adaptive=False), num_eligible_reqs) == 0


def test_cap_applies_from_two_eligible_requests_on():
    config = _adaptive_config(adaptive=False)
    assert effective_long_prefill_token_threshold(config, 2) == 8
    assert effective_long_prefill_token_threshold(config, 4) == 8


def test_adaptive_flag_floors_the_cap_at_the_fair_share():
    config = _adaptive_config(threshold=8, adaptive=True, budget=1024)
    assert effective_long_prefill_token_threshold(config, 2) == 512
    assert effective_long_prefill_token_threshold(config, 4) == 256
    assert effective_long_prefill_token_threshold(config, 1) == 0


def test_adaptive_flag_never_lowers_a_generous_cap():
    config = _adaptive_config(threshold=900, adaptive=True, budget=1024)
    assert effective_long_prefill_token_threshold(config, 4) == 900


def test_adaptive_flag_does_not_resurrect_a_disabled_cap():
    config = _adaptive_config(threshold=0, adaptive=True, budget=1024)
    assert effective_long_prefill_token_threshold(config, 4) == 0


SCHEDULERS = [
    (RecomputeScheduler, "vllm_ascend.core.recompute_scheduler"),
    (DyntraLBScheduler, "vllm_ascend.core.dyntra_lb_scheduler"),
    (ProfilingChunkScheduler, "vllm_ascend.core.scheduler_profiling_chunk"),
    (BalanceScheduler, "vllm_ascend.patch.platform.patch_balance_schedule"),
]


def _make_scheduler(
    scheduler_cls,
    *,
    threshold=8,
    adaptive=None,
    budget=1024,
    num_requests=1,
    num_tokens=200,
):
    config = make_dyntra_test_config(max_num_batched_tokens=budget, block_size=128)
    config.scheduler_config.long_prefill_token_threshold = threshold
    if adaptive is not None:
        config.scheduler_config.long_prefill_token_threshold_adaptive = adaptive
    # BalanceScheduler only runs its copied schedule() body when DP balance
    # scheduling is on; when it is off the upstream body (which already
    # implements the adaptive threshold) is used instead.
    ascend_config = _ascend_config(balance_enabled=scheduler_cls is BalanceScheduler)
    with (
        patch("vllm_ascend.ascend_config.init_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config),
        patch(PATCH_BALANCE, return_value=ascend_config),
    ):
        scheduler = create_dyntra_lb_scheduler(config, scheduler_cls=scheduler_cls)
    for index in range(num_requests):
        scheduler.add_request(create_request(request_id=index + 1, num_tokens=num_tokens))
    return scheduler


@pytest.mark.parametrize(("scheduler_cls", "module"), SCHEDULERS)
def test_scheduler_passes_running_plus_waiting_to_the_helper(scheduler_cls, module):
    scheduler = _make_scheduler(scheduler_cls, num_requests=3)
    with patch(f"{module}.{LOAD_PREFIX}", return_value=8) as resolve:
        scheduler.schedule()
    config, num_eligible_reqs = resolve.call_args.args
    assert config is scheduler.scheduler_config
    assert num_eligible_reqs == 3


@pytest.mark.parametrize(("scheduler_cls", "_module"), SCHEDULERS)
def test_single_request_prefills_the_whole_prompt(scheduler_cls, _module):
    scheduler = _make_scheduler(scheduler_cls, adaptive=False, num_requests=1, num_tokens=200)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens.values()) == [200]


@pytest.mark.parametrize(("scheduler_cls", "_module"), SCHEDULERS)
def test_two_requests_are_still_capped(scheduler_cls, _module):
    scheduler = _make_scheduler(scheduler_cls, adaptive=False, num_requests=2, num_tokens=200)
    output = scheduler.schedule()
    assert sorted(output.num_scheduled_tokens.values()) == [8, 8]


@pytest.mark.parametrize(("scheduler_cls", "_module"), SCHEDULERS)
def test_adaptive_flag_raises_the_cap_to_the_fair_share(scheduler_cls, _module):
    # 2 requests, 1024-token budget -> floor 512 instead of the configured 8.
    scheduler = _make_scheduler(scheduler_cls, adaptive=True, budget=1024, num_requests=2, num_tokens=900)
    output = scheduler.schedule()
    assert sorted(output.num_scheduled_tokens.values()) == [512, 512]
