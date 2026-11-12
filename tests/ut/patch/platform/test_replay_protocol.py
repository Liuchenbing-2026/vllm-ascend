# SPDX-License-Identifier: Apache-2.0
import importlib.util
import pickle
from pathlib import Path

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus


def _load_patch():
    path = Path(__file__).parents[4] / "vllm_ascend/patch/platform/patch_swa_bounded_replay.py"
    spec = importlib.util.spec_from_file_location("replay_protocol_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_resume_protocol_does_not_mutate_cached_data():
    before_fields = dict(CachedRequestData.__dataclass_fields__)
    before_method = Scheduler._make_cached_request_data
    patch = _load_patch()
    assert not patch._needs_cached_replay_patch()
    assert CachedRequestData.__dataclass_fields__ == before_fields
    assert Scheduler._make_cached_request_data is before_method


@pytest.mark.parametrize("replay_start", [0, 64])
def test_resumed_new_request_preserves_replay_start_on_wire(replay_start):
    _load_patch()
    request = Request("resumed", [11, 12, 13], SamplingParams(max_tokens=1), None)
    request.status = RequestStatus.PREEMPTED
    request.replay_start = replay_start
    payload = NewRequestData.from_request(request, ([0],), request.prefill_token_ids)
    # The multiprocess executor sends SchedulerOutput with pickle, not
    # the engine-core MessagePack protocol. Exercise that payload round trip.
    decoded = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert decoded.replay_start == replay_start
    assert decoded.req_id == "resumed"
    assert list(decoded.prefill_token_ids) == [11, 12, 13]


def test_unknown_signature_still_raises(monkeypatch):
    patch = _load_patch()

    def changed(self, renamed):
        raise AssertionError("must not execute")

    monkeypatch.setattr(Scheduler, "_make_cached_request_data", changed)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch._needs_cached_replay_patch()


@pytest.mark.parametrize("missing", ["native_field", "cached_contract"])
def test_inconsistent_payload_contract_still_raises(monkeypatch, missing):
    patch = _load_patch()
    if missing == "native_field":
        fields = dict(NewRequestData.__dataclass_fields__)
        del fields["replay_start"]
        monkeypatch.setattr(NewRequestData, "__dataclass_fields__", fields)
    else:
        fields = dict(CachedRequestData.__dataclass_fields__)
        fields["resumed_req_ids"] = object()
        monkeypatch.setattr(CachedRequestData, "__dataclass_fields__", fields)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch._needs_cached_replay_patch()


def test_core_before_replay_feature_needs_no_patch(monkeypatch):
    patch = _load_patch()
    monkeypatch.delattr(Scheduler, "_mark_prefix_replay")
    assert not patch._needs_cached_replay_patch()
