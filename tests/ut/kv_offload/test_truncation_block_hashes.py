# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""P-side prompt truncation must rebuild ``block_hashes``.

The request hashes its full prompt on construction. Dropping the last token
afterwards leaves stale hashes that describe the pre-truncation prompt, so
the local prefix-cache lookup keeps using a longer prompt than the one the
scheduler and the remote side agree on.
"""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    MooncakeConnectorScheduler as HybridScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorScheduler as LayerwiseScheduler,
)


class _HashingRequest:
    def __init__(self, num_tokens: int, params: dict | None = None):
        self.prompt_token_ids = list(range(num_tokens))
        self.prompt_embeds = None
        self._all_token_ids = list(range(num_tokens))
        self.num_prompt_tokens = num_tokens
        self.max_tokens = 8
        self.kv_transfer_params = {"do_remote_decode": True} if params is None else params
        # What a real Request computes from the full prompt at construction.
        self.block_hashes = [tuple(self._all_token_ids)]

    def update_block_hashes(self) -> None:
        self.block_hashes.append(tuple(self._all_token_ids))


def _hybrid_scheduler() -> HybridScheduler:
    scheduler = object.__new__(HybridScheduler)
    scheduler.need_truncate = True
    return scheduler


def _layerwise_scheduler() -> LayerwiseScheduler:
    scheduler = object.__new__(LayerwiseScheduler)
    scheduler.need_truncate = True
    return scheduler


@pytest.mark.parametrize(
    "make_scheduler,truncate_name",
    [
        (_hybrid_scheduler, "_truncate_request_for_prefill"),
        (_layerwise_scheduler, "_truncate_request_for_hybrid_prefill"),
    ],
)
def test_truncation_rebuilds_block_hashes(make_scheduler, truncate_name):
    request = _HashingRequest(5)
    getattr(make_scheduler(), truncate_name)(request)

    assert request.num_prompt_tokens == 4
    assert request._all_token_ids == [0, 1, 2, 3]
    assert request.kv_transfer_params["_p_side_truncated"] is True
    # The stale full-prompt hash must be replaced by the truncated prompt.
    assert request.block_hashes == [(0, 1, 2, 3)]


@pytest.mark.parametrize(
    "make_scheduler,truncate_name",
    [
        (_hybrid_scheduler, "_truncate_request_for_prefill"),
        (_layerwise_scheduler, "_truncate_request_for_hybrid_prefill"),
    ],
)
def test_repeated_truncation_does_not_rehash(make_scheduler, truncate_name):
    request = _HashingRequest(5)
    scheduler = make_scheduler()
    getattr(scheduler, truncate_name)(request)
    getattr(scheduler, truncate_name)(request)

    assert request.block_hashes == [(0, 1, 2, 3)]
    assert request.num_prompt_tokens == 4


@pytest.mark.parametrize(
    "make_scheduler,truncate_name",
    [
        (_hybrid_scheduler, "_truncate_request_for_prefill"),
        (_layerwise_scheduler, "_truncate_request_for_hybrid_prefill"),
    ],
)
def test_single_token_prompt_keeps_hashes(make_scheduler, truncate_name):
    request = _HashingRequest(1)
    getattr(make_scheduler(), truncate_name)(request)

    assert request.block_hashes == [(0,)]
    assert "_p_side_truncated" not in request.kv_transfer_params
