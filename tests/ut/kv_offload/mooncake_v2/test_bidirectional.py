# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.bidirectional import BidirectionalKVConfig

from .helpers import make_blocks, make_full_spec, make_mamba_spec, make_request
from .test_pull_scheduler import make_pull_scheduler, make_sending_thread
from .test_pull_worker import make_req_meta, make_thread


def config(**extra):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
    )


def caches(spec=None):
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec or make_full_spec())])


@pytest.mark.parametrize(
    "extra",
    [
        {"bidirectional_kv_xfer": "true"},
        {"bidirectional_kv_xfer": True, "decoder_kv_blocks_ttl": 0},
        {"bidirectional_kv_xfer": True, "decoder_kv_blocks_ttl": float("nan")},
        {"bidirectional_kv_xfer": True, "kv_recompute_threshold": -1},
        {"bidirectional_kv_xfer": True, "kv_recompute_threshold": True},
    ],
)
def test_invalid_config(extra):
    with pytest.raises(ValueError):
        BidirectionalKVConfig.from_config(config(**extra), caches())


def test_disabled_preserves_legacy_topologies_and_enabled_is_guarded():
    cfg = config()
    cfg.parallel_config.tensor_parallel_size = 8
    assert not BidirectionalKVConfig.from_config(cfg, caches(make_mamba_spec())).enabled
    cfg.kv_transfer_config.kv_connector_extra_config["bidirectional_kv_xfer"] = True
    with pytest.raises(ValueError, match="full attention"):
        BidirectionalKVConfig.from_config(cfg, caches())
    cfg.parallel_config.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="full attention"):
        BidirectionalKVConfig.from_config(cfg, caches(make_mamba_spec()))


def reverse_scheduler(*, producer=True):
    scheduler = make_pull_scheduler()
    scheduler.bidirectional = BidirectionalKVConfig(True, 480, 16)
    scheduler.block_size = 16
    scheduler.kv_transfer_config = SimpleNamespace(is_kv_producer=producer, is_kv_consumer=not producer)
    return scheduler


def retained_params():
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_retained": True,
        "remote_num_tokens": 48,
        "remote_num_prompt_tokens": 48,
        "remote_block_ids": ([20, 21, 22],),
        "remote_engine_id": "engine-d",
        "remote_request_id": "prior-turn",
        "remote_host": "127.0.0.1",
        "remote_port": 6001,
    }


def test_reverse_loads_computed_prefix_and_leaves_new_suffix():
    scheduler = reverse_scheduler()
    request = make_request(num_prompt_tokens=80, prompt_token_ids=list(range(80)), kv_transfer_params=retained_params())
    assert scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
    scheduler.update_state_after_alloc(request, make_blocks(), 32)
    metadata = scheduler.build_connector_meta(MagicMock())
    assert metadata.requests[request.request_id].num_external_tokens == 32
    assert metadata.requests[request.request_id].remote_retained is True
    assert request.kv_transfer_params["do_remote_decode"] is True
    assert scheduler._reqs_recv_info[request.request_id][-1] == request.request_id
    assert scheduler.get_num_new_matched_tokens(request, 16) == (0, False)


def test_threshold_and_local_hits_skip_reverse_dma():
    scheduler = reverse_scheduler()
    scheduler.bidirectional = BidirectionalKVConfig(True, 480, 64)
    request = make_request(num_prompt_tokens=80, kv_transfer_params=retained_params())
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.bidirectional = BidirectionalKVConfig(True, 480, 0)
    assert scheduler.get_num_new_matched_tokens(request, 48) == (0, False)


@pytest.mark.parametrize("status", [RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED])
def test_decoder_retains_only_full_computed_blocks(status):
    scheduler = reverse_scheduler(producer=False)
    scheduler._sending_thread = MagicMock()
    request = make_request(num_computed_tokens=47, status=status)
    delayed, metadata = scheduler.request_finished(request, ([10, 11, 12, 13],))
    assert delayed and metadata["remote_num_tokens"] == 32
    assert metadata["remote_block_ids"] == ([10, 11],)
    assert metadata["remote_retained"] is True
    assert metadata["do_remote_decode"] is True
    assert "last_token_id" not in metadata


def test_abort_does_not_advertise_retained_blocks():
    scheduler = reverse_scheduler(producer=False)
    request = make_request(num_computed_tokens=48, status=RequestStatus.FINISHED_ABORTED)
    assert scheduler.request_finished(request, ([10, 11, 12],)) == (False, None)


def test_expired_retention_cannot_be_claimed(monkeypatch):
    thread = make_sending_thread()
    thread.retained_ttl_secs = 10
    thread.add_delayed_request("old", 1)
    monkeypatch.setattr(pull_scheduler.time, "time", lambda: 12)
    assert not thread._claim_retained_request("old", "consumer-a")
    assert thread.get_and_clear_finished_requests() == {"old"}


def test_live_read_pins_blocks_and_only_owner_ack_releases(monkeypatch):
    thread = make_sending_thread()
    thread.retained_ttl_secs = 10
    thread.add_delayed_request("live", 1)
    monkeypatch.setattr(pull_scheduler.time, "time", lambda: 2)
    assert thread._claim_retained_request("live", "consumer-a")
    assert not thread._claim_retained_request("live", "consumer-b")
    monkeypatch.setattr(pull_scheduler.time, "time", lambda: 100)
    assert not thread.get_and_clear_finished_requests()
    thread._handle_finished_request("live", "consumer-b")
    thread._handle_finished_request("live")
    assert not thread.get_and_clear_finished_requests()
    thread._handle_finished_request("live", "consumer-a")
    assert thread.get_and_clear_finished_requests() == {"live"}
    assert not thread.active_reads


def test_failed_claim_never_submits_dma():
    thread = make_thread()
    thread._get_remote_metadata = MagicMock()
    thread._claim_retained_request = MagicMock(return_value=False)
    request = make_req_meta()
    request.remote_retained = True
    thread.executor = MagicMock()
    assert thread._handle_requests("engine-d", "127.0.0.1", 6001, {"next": request}) == {"next"}
    thread.executor.submit.assert_not_called()
