# SPDX-License-Identifier: Apache-2.0
"""Run the real driver; isolate HTTP/model/snapshot IO at their boundaries."""

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest


@pytest.fixture
def driver(monkeypatch):
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "bidir_driver_under_test", tools / "test_mooncake_bidirectional_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_driver(driver, monkeypatch, tmp_path, *, wrong_output=False, broken_capture=False):
    fixture = {"first": [10, 11], "first_reference": [12, 13], "second": [10, 11, 12, 14], "second_reference": [15, 16]}
    fixture_file = tmp_path / "fixture.json"
    fixture_file.write_text(json.dumps(fixture))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "--fixture",
            str(fixture_file),
            "--prefill",
            "http://P",
            "--decode",
            "http://D",
            "--output",
            str(output),
            "--guard-status",
            str(tmp_path / "guard-status.json"),
            "--container-id",
            "a" * 64,
            "--fixture-sha256",
            hashlib.sha256(fixture_file.read_bytes()).hexdigest(),
        ],
    )
    events = []
    descriptor = {
        "remote_num_tokens": 3,
        "remote_block_ids": [[1]],
        "remote_engine_id": "epoch",
        "remote_request_id": "r",
    }

    def prefill(*args):
        events.append("prefill")
        return "r", {"kv_transfer_params": descriptor}

    def decode(*args):
        events.append("decode")
        if events.count("decode") == 1:
            return fixture["first_reference"], descriptor
        return ([99, 16] if wrong_output else fixture["second_reference"]), descriptor

    def capture(*args):
        phase = args[-1]
        events.append(phase)
        if broken_capture and phase == "P-imported":
            raise RuntimeError("injected snapshot error")
        return {"manifest": phase, "total_bytes": 1}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(driver.httpx, "Client", lambda **kwargs: Client())
    monkeypatch.setattr(driver, "prefill", prefill)
    monkeypatch.setattr(driver, "decode", decode)
    monkeypatch.setattr(driver, "capture", capture)
    monkeypatch.setattr(driver, "verify_transport_endpoints", lambda *args: {})
    monkeypatch.setattr(driver, "metric", lambda *args: sum(e == "prefill" for e in events))
    monkeypatch.setattr(
        driver,
        "compare_snapshots",
        lambda *args, **kwargs: {"all_bytes_exact": True, "results": [{"all_finite": True}]},
    )
    error = None
    try:
        driver.main()
    except (AssertionError, RuntimeError) as caught:
        error = caught
    return SimpleNamespace(events=events, error=error, report=json.loads((output / "result.json").read_text()))


def test_all_three_snapshots_precede_second_decode(driver, monkeypatch, tmp_path):
    result = run_driver(driver, monkeypatch, tmp_path)
    assert result.error is None
    assert result.events.index("P-recomputed") < [i for i, event in enumerate(result.events) if event == "decode"][1]
    assert result.report["acceptance_pass"]


def test_output_error_stops_without_reference_decode(driver, monkeypatch, tmp_path):
    result = run_driver(driver, monkeypatch, tmp_path, wrong_output=True)
    assert isinstance(result.error, AssertionError)
    assert result.events[-1] == "decode"
    assert result.events.count("decode") == 2
    assert result.report["reuse_output_ids"] == [99, 16]
    assert not result.report.get("acceptance_pass", False)
    assert (tmp_path / "STOP").exists()


def test_capture_error_stops_before_recompute_or_decode(driver, monkeypatch, tmp_path):
    result = run_driver(driver, monkeypatch, tmp_path, broken_capture=True)
    assert isinstance(result.error, RuntimeError)
    assert result.events[-1] == "P-imported"
    assert result.events.count("decode") == 1
    assert "P-recomputed" not in result.events


@pytest.mark.parametrize("nonfinite", [False, True])
def test_byte_or_nonfinite_mismatch_stops_before_second_decode(driver, monkeypatch, tmp_path, nonfinite):
    def compare(*args, **kwargs):
        return {"all_bytes_exact": nonfinite, "results": [{"all_finite": not nonfinite}]}

    original = driver.main

    def main():
        monkeypatch.setattr(driver, "compare_snapshots", compare)
        return original()

    monkeypatch.setattr(driver, "main", main)
    result = run_driver(driver, monkeypatch, tmp_path)
    assert isinstance(result.error, AssertionError)
    assert result.report["three_snapshots_captured"]
    assert not result.report.get("three_snapshots_validated", False)
    assert result.events.count("decode") == 1


def test_real_transport_rpc_and_guarded_client(driver, tmp_path):
    path = tmp_path / "guard-status.json"
    status = {
        "container_id": "a" * 64,
        "checked_at_epoch": time.time(),
        "stop_utc": "2099-01-01T00:00:00Z",
        "running": True,
        "required_listeners_observed": True,
        "endpoints": [{"local_ip": "10.0.0.9", "local_port": port, "state": "0A"} for port in (31001, 29520)],
    }
    path.write_text(json.dumps(status))
    args = SimpleNamespace(guard_status=path, container_id="a" * 64, prefill="http://P", decode="http://D")
    calls = []

    def respond(request):
        calls.append(json.loads(request.content)["method"])
        return httpx.Response(
            200,
            json={"results": [{"engine_id": "epoch", "host": "10.0.0.9", "rpc_port": 31001, "handshake_port": 29520}]},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        guarded = driver.GuardedClient(client, args)
        result = driver.verify_transport_endpoints(guarded, args)
        assert set(result) == {"P", "D"} and len(calls) == 2
        status["checked_at_epoch"] -= 10
        path.write_text(json.dumps(status))
        with pytest.raises(RuntimeError, match="freshly ready"):
            guarded.post("http://P/collective_rpc", json={})
        assert len(calls) == 2


def test_worker_endpoint_method_reads_actual_connector_fields(driver, monkeypatch):
    import mooncake_kv_snapshot

    dependency = ModuleType("vllm.distributed.kv_transfer")
    connector = SimpleNamespace(
        engine_id="epoch", side_channel_host="10.0.0.9", te_rpc_port=42019, handshake_port=29520
    )
    dependency.get_kv_transfer_group = lambda: SimpleNamespace(connector_worker=connector)
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    actual = mooncake_kv_snapshot.KVSnapshotWorkerExtension().mooncake_transport_endpoint()
    assert actual == {"engine_id": "epoch", "host": "10.0.0.9", "rpc_port": 42019, "handshake_port": 29520}
