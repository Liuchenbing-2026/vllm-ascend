# SPDX-License-Identifier: Apache-2.0
"""Bidirectional KV transfer (D node -> P node readback) coverage.

The feature mirrors the upstream vLLM NIXL connector:

* ``bidirectional_kv_xfer`` in the KV transfer extra config turns one engine
  into ``kv_both``, i.e. a Prefill engine that also publishes the blocks it
  computed so the Decode node of the *next* turn can read them back.
* ``kv_recompute_threshold`` keeps a short readback local: transferring fewer
  tokens than the threshold costs more than recomputing them.
* ``decoder_kv_blocks_ttl`` pins the Decode blocks for a fixed time because the
  next turn is client-driven, and the exported deadline lets the Prefill worker
  decline a read that arrives once the blocks may already be gone.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    MooncakePullConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import MooncakeKVConnectorStats
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    BIDIRECTIONAL_KV_XFER_EXPIRY_SAFETY_MARGIN_SECONDS,
    read_bidirectional_kv_xfer_settings,
)

from .helpers import make_blocks, make_full_spec, make_request
from .test_pull_worker import make_req_meta, make_thread


def make_bidirectional_scheduler(
    *,
    enabled: bool = True,
    kv_recompute_threshold: int = 64,
    decoder_kv_blocks_ttl: float = 480.0,
) -> MooncakePullConnectorScheduler:
    scheduler = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    scheduler.need_truncate = False
    scheduler.num_speculative_tokens = 0
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
    scheduler.is_bidirectional_kv_xfer_enabled = enabled
    scheduler.kv_recompute_threshold = kv_recompute_threshold
    scheduler.decoder_kv_blocks_ttl = decoder_kv_blocks_ttl
    scheduler.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    scheduler.group_block_size = [16]
    scheduler.group_unique_specs = [[make_full_spec()]]
    scheduler.engine_id = "engine-d"
    scheduler.side_channel_host = "10.0.0.20"
    scheduler.side_channel_port = 7000
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_in_batch = set()
    scheduler._reqs_recv_info = {}
    scheduler._sending_thread = MagicMock()
    scheduler._recving_thread = MagicMock()
    return scheduler


def readback_params(**overrides: object) -> dict[str, object]:
    """KV transfer params of a turn-2 Prefill request whose peer is a Decode node."""
    params: dict[str, object] = {
        "do_remote_decode": True,
        "remote_block_ids": ([20, 21, 22],),
        "remote_num_tokens": 48,
        "remote_engine_id": "engine-d",
        "remote_request_id": "request-d",
        "remote_host": "10.0.0.20",
        "remote_port": 7000,
    }
    params.update(overrides)
    return params


def test_settings_are_read_from_the_kv_transfer_extra_config() -> None:
    config = SimpleNamespace(
        get_from_extra_config=lambda key, default: {
            "bidirectional_kv_xfer": True,
            "kv_recompute_threshold": 8,
            "decoder_kv_blocks_ttl": 30,
        }.get(key, default)
    )

    settings = read_bidirectional_kv_xfer_settings(config)

    assert settings.enabled is True
    assert settings.kv_recompute_threshold == 8
    assert settings.decoder_kv_blocks_ttl == 30.0


def test_settings_fall_back_to_upstream_defaults() -> None:
    settings = read_bidirectional_kv_xfer_settings(SimpleNamespace(get_from_extra_config=lambda key, default: default))

    # Same defaults as the upstream vLLM NIXL connector, so one KV transfer
    # config drives either backend.
    assert settings.enabled is False
    assert settings.kv_recompute_threshold == 64
    assert settings.decoder_kv_blocks_ttl == 480.0


def test_scheduler_metadata_samples_a_fresh_scheduler_clock() -> None:
    from .test_pull_scheduler import make_sending_thread

    thread = make_sending_thread()
    assert thread.transfer_metadata_groups.scheduler_clock == 0.0

    # The clock is stamped when the reply is served, not when the thread starts,
    # so a peer cannot translate a deadline with a stale sample.
    before = time.perf_counter()
    decoded = msgspec.msgpack.decode(thread.encode_metadata(), type=MooncakeTransferMetadataGroups)
    after = time.perf_counter()

    assert 0.0 < decoded.scheduler_clock <= time.perf_counter()
    assert before <= decoded.scheduler_clock <= after
    assert thread.transfer_metadata_groups.scheduler_clock == 0.0


def test_decode_node_publishes_computed_blocks_with_ttl_deadline() -> None:
    scheduler = make_bidirectional_scheduler(decoder_kv_blocks_ttl=120.0)
    # A Decode request has no ``do_remote_decode`` flag: it is the peer of a
    # Prefill request, so it now publishes everything it computed.
    request = make_request(
        request_id="request-d",
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={},
        num_computed_tokens=48,
        output_token_ids=[777],
    )

    delay_free, params = scheduler.request_finished(request, ([10, 11, 12],))

    assert delay_free is True
    assert params is not None
    # Three computed blocks cover 48 tokens; the Prefill node needs all of them.
    assert params["remote_block_ids"] == ([10, 11, 12],)
    assert params["remote_num_tokens"] == 48
    assert params["remote_num_prompt_tokens"] == 48
    assert params["remote_engine_id"] == "engine-d"
    assert params["remote_request_id"] == "request-d"
    assert params["remote_host"] == "10.0.0.20"
    assert params["remote_port"] == 7000
    assert params["last_token_id"] == 777
    # The reverse read of the next turn is client-driven, so the Decode blocks
    # are pinned by a fixed TTL and the deadline is exported to the peers.
    assert params["do_remote_decode"] is True
    assert params["do_remote_prefill"] is False
    expiry = params["remote_blocks_expiry_time"]
    assert expiry is not None
    assert time.perf_counter() < expiry <= time.perf_counter() + 120.0
    # The local lease is tracked on the wall clock, the exported deadline on the
    # monotonic clock so that the peer can translate it into its own domain.
    pinned = scheduler._sending_thread.add_delayed_request.call_args.args
    assert pinned[0] == "request-d"
    assert pinned[1] == pytest.approx(time.time(), abs=5.0)
    assert pinned[2] == 120.0


def test_decode_node_publishes_nothing_for_an_unsupported_finish_reason() -> None:
    scheduler = make_bidirectional_scheduler()
    request = make_request(
        request_id="request-d",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params={},
        num_computed_tokens=48,
    )

    assert scheduler.request_finished(request, ([10, 11, 12],)) == (False, None)
    scheduler._sending_thread.add_delayed_request.assert_not_called()


def test_decode_node_does_not_publish_without_bidirectional_kv_xfer() -> None:
    scheduler = make_bidirectional_scheduler(enabled=False)
    request = make_request(
        request_id="request-d",
        status=RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={},
        num_computed_tokens=48,
    )

    # Single-role behaviour is unchanged: only Prefill nodes publish KV.
    assert scheduler.request_finished(request, ([10, 11, 12],)) == (False, None)
    scheduler._sending_thread.add_delayed_request.assert_not_called()


def test_prefill_node_reads_back_the_decode_blocks() -> None:
    scheduler = make_bidirectional_scheduler(kv_recompute_threshold=32)
    request = make_request(
        request_id="request-p2",
        prompt_token_ids=list(range(64)),
        num_prompt_tokens=64,
        kv_transfer_params=readback_params(),
    )

    assert scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0) == (48, True)
    # The recorded local progress is forwarded so the worker can align slices.
    assert request.kv_transfer_params["num_computed_tokens"] == 0


def test_prefill_node_reads_back_only_the_missing_tokens() -> None:
    scheduler = make_bidirectional_scheduler(kv_recompute_threshold=32)
    request = make_request(
        request_id="request-p2",
        prompt_token_ids=list(range(64)),
        num_prompt_tokens=64,
        kv_transfer_params=readback_params(),
    )

    # 16 tokens already hit the local prefix cache, so 32 remain to pull.
    assert scheduler.get_num_new_matched_tokens(request, num_computed_tokens=16) == (32, True)


def test_short_readback_is_recomputed_locally() -> None:
    scheduler = make_bidirectional_scheduler(kv_recompute_threshold=64)
    request = make_request(
        request_id="request-p2",
        prompt_token_ids=list(range(64)),
        num_prompt_tokens=64,
        kv_transfer_params=readback_params(remote_num_tokens=32),
    )

    # 32 remote tokens cost more to transfer than to recompute.
    assert scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0) == (0, False)


def test_readback_is_not_skipped_when_the_threshold_is_disabled() -> None:
    scheduler = make_bidirectional_scheduler(kv_recompute_threshold=0)
    request = make_request(
        request_id="request-p2",
        prompt_token_ids=list(range(64)),
        num_prompt_tokens=64,
        kv_transfer_params=readback_params(remote_num_tokens=32),
    )

    assert scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0) == (32, True)


def test_readback_registers_one_receive_transfer() -> None:
    scheduler = make_bidirectional_scheduler()
    request = make_request(
        request_id="request-p2",
        prompt_token_ids=list(range(64)),
        num_prompt_tokens=64,
        kv_transfer_params=readback_params(),
    )

    scheduler.update_state_after_alloc(request, make_blocks(), num_external_tokens=48)

    assert list(scheduler._reqs_need_recv) == ["request-p2"]
    assert scheduler._reqs_need_recv["request-p2"][3] == 48
    assert scheduler._reqs_recv_info["request-p2"] == ("10.0.0.20", 7000, "request-d")
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert request.kv_transfer_params["_remote_blocks_processed"] is True

    # A request is registered once; a repeated alloc must not add a transfer.
    scheduler._reqs_need_recv.clear()
    scheduler.update_state_after_alloc(request, make_blocks(), num_external_tokens=48)
    assert scheduler._reqs_need_recv == {}


def test_zero_external_readback_only_acknowledges_the_remote_blocks() -> None:
    scheduler = make_bidirectional_scheduler()
    request = make_request(
        request_id="request-p2",
        prompt_token_ids=list(range(64)),
        num_prompt_tokens=64,
        kv_transfer_params=readback_params(remote_block_ids=([20],)),
    )

    scheduler.update_state_after_alloc(request, make_blocks(), num_external_tokens=0)

    assert scheduler._reqs_need_recv == {}
    scheduler._recving_thread.add_request.assert_called_once_with("10.0.0.20", 7000, "request-d")


def test_metadata_forwards_the_exported_block_deadline() -> None:
    meta = MooncakeConnectorMetadata()
    meta.add_new_req(
        request_id="request-p2",
        local_block_ids=([10],),
        local_full_block_ids=([10],),
        local_num_prompt_tokens=64,
        num_external_tokens=48,
        kv_transfer_params={
            "remote_block_ids": ([20, 21, 22],),
            "remote_engine_id": "engine-d",
            "remote_request_id": "request-d",
            "remote_host": "10.0.0.20",
            "remote_port": 7000,
            "remote_num_prompt_tokens": 48,
            "remote_blocks_expiry_time": 1234.5,
        },
    )

    assert meta.requests["request-p2"].remote_blocks_expiry_time == 1234.5


def test_metadata_defaults_to_no_deadline() -> None:
    meta = MooncakeConnectorMetadata()
    meta.add_new_req(
        request_id="request-p",
        local_block_ids=([10],),
        local_full_block_ids=([10],),
        local_num_prompt_tokens=64,
        num_external_tokens=48,
        kv_transfer_params={
            "remote_block_ids": ([20],),
            "remote_engine_id": "engine-p",
            "remote_request_id": "request-p",
            "remote_host": "10.0.0.1",
            "remote_port": 6000,
            "remote_num_prompt_tokens": 32,
        },
    )

    assert meta.requests["request-p"].remote_blocks_expiry_time is None


def test_expired_reverse_read_is_declined_and_counted() -> None:
    thread = make_thread(is_bidirectional_kv_xfer_enabled=True)
    thread.remote_clock_offsets["engine-d"] = 100.0
    request = make_req_meta(engine_id="engine-d", remote_blocks_expiry_time=time.perf_counter() + 200.0)
    metadata = SimpleNamespace(engine_id="engine-d")

    assert thread._is_turn2_read_expired(request, metadata) is False

    # The peer's clock runs 100 s ahead, so its own deadline is 100 s later than
    # the local one: a block that expires "later" on the peer is about to expire
    # here.
    request = make_req_meta(engine_id="engine-d", remote_blocks_expiry_time=time.perf_counter() + 100.0)
    assert thread._is_turn2_read_expired(request, metadata) is True

    remaining, expired = thread._drop_expired_reads("engine-d", {"request-p2": request}, metadata)

    assert remaining == {}
    assert expired == {"request-p2"}
    assert thread.num_expired_reads == 1
    assert thread.get_and_clear_num_expired_reads() == 1
    assert thread.get_and_clear_num_expired_reads() == 0


def test_reverse_read_without_a_translatable_deadline_is_kept() -> None:
    thread = make_thread(is_bidirectional_kv_xfer_enabled=True)
    metadata = SimpleNamespace(engine_id="engine-d")
    soon = time.perf_counter() - 100.0
    request = make_req_meta(engine_id="engine-d", remote_blocks_expiry_time=soon)

    # No clock sample for the peer yet, so the deadline cannot be translated and
    # the read is attempted: the peer publishes the deadline because it accepts
    # reads until then.
    assert thread._is_turn2_read_expired(request, metadata) is False

    # A request without remote blocks has nothing to read.
    assert (
        thread._is_turn2_read_expired(
            make_req_meta(engine_id="engine-d", local=((),), remote_blocks_expiry_time=soon),
            metadata,
        )
        is False
    )

    # Deadlines are only honoured where they can be: no deadline at all.
    thread.remote_clock_offsets["engine-d"] = 100.0
    assert thread._is_turn2_read_expired(make_req_meta(engine_id="engine-d"), metadata) is False


def test_expiry_is_ignored_without_bidirectional_kv_xfer() -> None:
    thread = make_thread()
    thread.remote_clock_offsets["engine-d"] = 100.0
    metadata = SimpleNamespace(engine_id="engine-d")
    request = make_req_meta(engine_id="engine-d", remote_blocks_expiry_time=time.perf_counter() + 100.0)

    remaining, expired = thread._drop_expired_reads("engine-d", {"request-p2": request}, metadata)

    assert remaining == {"request-p2": request}
    assert expired == set()
    assert thread.num_expired_reads == 0


def test_expiry_margin_declines_reads_inside_the_safety_window() -> None:
    """The tail of a lease is not read: the peer drops the blocks on its own clock."""
    thread = make_thread(is_bidirectional_kv_xfer_enabled=True)
    thread.remote_clock_offsets["engine-d"] = 100.0
    metadata = SimpleNamespace(engine_id="engine-d")
    margin = BIDIRECTIONAL_KV_XFER_EXPIRY_SAFETY_MARGIN_SECONDS
    now = time.perf_counter()

    inside_window = make_req_meta(engine_id="engine-d", remote_blocks_expiry_time=now + 100.0 + margin - 0.5)
    outside_window = make_req_meta(engine_id="engine-d", remote_blocks_expiry_time=now + 100.0 + margin + 0.5)

    assert thread._is_turn2_read_expired(inside_window, metadata) is True
    assert thread._is_turn2_read_expired(outside_window, metadata) is False


def test_peer_clock_offset_uses_the_metadata_round_trip_midpoint() -> None:
    thread = make_thread(is_bidirectional_kv_xfer_enabled=True)
    metadata = SimpleNamespace(engine_id="engine-d", scheduler_clock=150.0)

    thread._record_remote_clock("engine-d", metadata, send_time=100.0, recv_time=110.0)

    # The reply was stamped while it was in flight, at the round-trip midpoint.
    assert thread.remote_clock_offsets["engine-d"] == pytest.approx(45.0)

    # A peer that does not stamp its clock must not shift the local deadline.
    thread._record_remote_clock("engine-d", SimpleNamespace(scheduler_clock=0.0), 100.0, 110.0)
    assert thread.remote_clock_offsets["engine-d"] == pytest.approx(45.0)


def test_peer_clock_is_not_tracked_without_bidirectional_kv_xfer() -> None:
    thread = make_thread()
    metadata = SimpleNamespace(engine_id="engine-d", scheduler_clock=150.0)

    thread._record_remote_clock("engine-d", metadata, send_time=100.0, recv_time=110.0)

    assert thread.remote_clock_offsets == {}


def test_stats_count_expired_reads_apart_from_failures() -> None:
    stats = MooncakeKVConnectorStats(data={})
    stats.record_kv_expired_req()

    reduced = stats.reduce()

    assert reduced["Num expired requests"] == 1
    assert reduced["Num failed transfers"] == 0
    assert reduced["Num successful transfers"] == 0
    assert stats.is_empty() is False
    assert stats.aggregate(MooncakeKVConnectorStats(data={})) is stats


def test_worker_reports_expired_reads_through_the_connector_stats() -> None:
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
        MooncakePullConnectorWorker,
    )

    worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    worker._recving_thread = MagicMock()
    worker._recving_thread.get_and_clear_num_expired_reads.return_value = 2
    worker.xfer_stats = MooncakeKVConnectorStats(data={})

    reported = worker.get_kv_connector_stats()

    assert reported is not None
    assert reported.reduce()["Num expired requests"] == 2
    # The counter is drained, so the next interval starts from zero.
    assert worker.xfer_stats.is_empty()
