# SPDX-License-Identifier: Apache-2.0
"""CPU policy/host-control tests; Docker and Linux proc IO are fake boundaries."""

import copy
import importlib.util
import json
import socket
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[3] / "tools/kv_diagnostic_readiness.py"
    spec = importlib.util.spec_from_file_location("kv_guard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def allocation(tmp_path):
    return {
        "status": "allocated",
        "lease_id": "cpu-example-not-a-real-lease",
        "devices": [2, 5],
        "start_utc": "2026-10-18T15:40:00Z",
        "no_start_after_utc": "2026-10-18T15:45:00Z",
        "stop_utc": "2026-10-18T16:04:00Z",
        "end_utc": "2026-10-18T16:05:00Z",
        "host_network_approved": True,
        "endpoint_policy_reviewed": True,
        "host_ip": "10.0.0.9",
        "allowed_ips": ["127.0.0.1", "10.0.0.9"],
        "interface": "eth0",
        "dynamic_ports": [1024, 65535],
        "static_ports": [29510, 29511, 29520, 29521, 29530, 29531, 29666],
        "required_listeners": [29510, 29511],
        "max_sockets": 64,
        "output_limit_bytes": 6 * 1024**3,
        "hccl_host_ports": [61000, 61031],
        "hccl_npu_ports": [62000, 62031],
        "output": str(tmp_path),
        "model": "/fixed/model",
        "ascend_source": "/fixed/ascend-a6",
        "core_source": "/fixed/core-6c",
        "runtime_work": "/fixed/existing-runtime",
        "diagnostic_source": "/fixed/diagnostic",
        "launch_script": "/fixed/launch.sh",
    }


def container(spec, *, running=True):
    devices = [
        *(f"/dev/davinci{card}" for card in spec["devices"]),
        "/dev/davinci_manager",
        "/dev/devmm_svm",
        "/dev/hisi_hdc",
    ]
    return {
        "Id": "a" * 64,
        "Config": {"Labels": {"task": "bidir-three-kv", "lease": spec["lease_id"]}},
        "HostConfig": {
            "Devices": [{"PathOnHost": path, "PathInContainer": path, "CgroupPermissions": "rw"} for path in devices],
            "Privileged": False,
            "RestartPolicy": {"Name": "no"},
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "NetworkMode": "host",
        },
        "State": {"Running": running, "Pid": 123 if running else 0, "Status": "running" if running else "created"},
    }


def endpoint(port=16383, address="10.0.0.9", **changes):
    return {
        "local_ip": address,
        "local_port": port,
        "remote_ip": "0.0.0.0",
        "remote_port": 0,
        "protocol": "tcp",
        "state": "0A",
        "inode": "99",
        **changes,
    }


class Docker:
    def __init__(self, spec, endpoints=(), stuck=False):
        self.current = container(spec)
        self.sockets = list(endpoints)
        self.calls = []
        self.stuck = stuck

    def inspect(self, _):
        return copy.deepcopy(self.current)

    def endpoints(self, _):
        return self.sockets

    def call(self, *argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == "stop" and self.stuck:
            raise subprocess.TimeoutExpired(argv, 15)
        self.current["State"].update(Running=False, Pid=0, Status="exited")


class Clock:
    def __init__(self, now, *, rollback=False):
        self.now, self.elapsed, self.rollback = now, 0, rollback

    def wall(self):
        return self.now - self.elapsed if self.rollback else self.now + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, _):
        self.elapsed += 1800


def test_plan_maps_only_assigned_devices_and_fixed_digest(module, allocation):
    plan = module.create_plan(allocation)
    mapped = [plan[i + 1] for i, value in enumerate(plan) if value == "--device"]
    assert mapped == [
        "/dev/davinci2:/dev/davinci2:rw",
        "/dev/davinci5:/dev/davinci5:rw",
        "/dev/davinci_manager:/dev/davinci_manager:rw",
        "/dev/devmm_svm:/dev/devmm_svm:rw",
        "/dev/hisi_hdc:/dev/hisi_hdc:rw",
    ]
    assert "--privileged" not in plan and "--pull" not in plan
    assert "--cap-drop" in plan and "no-new-privileges" in plan
    assert module.IMAGE in plan
    assert "--pull=never" in plan and "--read-only" in plan
    assert "HCCL_HOST_SOCKET_PORT_RANGE=61000-61031" in plan
    assert "HCCL_NPU_SOCKET_PORT_RANGE=62000-62031" in plan
    assert not any("site-packages/vllm" in value for value in plan)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("devices", [0, 0]),
        ("devices", list(range(8))),
        ("status", "proposal"),
        ("host_network_approved", False),
        ("endpoint_policy_reviewed", False),
        ("stop_utc", "2026-10-18T16:04:30Z"),
        ("allowed_ips", ["0.0.0.0", "127.0.0.1"]),
        ("output_limit_bytes", 7 * 1024**3),
    ],
)
def test_rejects_unapproved_or_unbounded_plan(module, allocation, key, value):
    allocation[key] = value
    with pytest.raises(ValueError):
        module.create_plan(allocation)


@pytest.mark.parametrize("rollback", [False, True])
def test_deadline_stops_exact_id_even_with_wall_clock_rollback(module, allocation, rollback):
    docker = Docker(allocation)
    clock = Clock(module.utc_timestamp(allocation["start_utc"]), rollback=rollback)
    assert (
        module.supervise(
            allocation,
            "a" * 64,
            allocation["output"],
            docker,
            wall=clock.wall,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        == 1
    )
    assert docker.calls == [("stop", "--time", "10", "a" * 64)]
    root = Path(allocation["output"])
    assert json.loads((root / "guard-ready.json").read_text())["container_id"] == "a" * 64
    assert "deadline" in json.loads((root / "guard-exit.json").read_text())["reason"]


def test_budget_stops_and_escalates_only_same_id(module, allocation, monkeypatch):
    docker = Docker(allocation, stuck=True)
    monkeypatch.setattr(module, "output_bytes", lambda _: 6 * 1024**3)
    now = module.utc_timestamp(allocation["start_utc"])
    assert module.supervise(allocation, "a" * 64, allocation["output"], docker, wall=lambda: now) == 1
    assert docker.calls == [("stop", "--time", "10", "a" * 64), ("kill", "a" * 64)]


def test_foreign_ownership_never_stopped(module, allocation):
    docker = Docker(allocation)
    docker.current["Config"]["Labels"]["lease"] = "someone-else"
    with pytest.raises(ValueError, match="ownership"):
        module.cleanup_owned(docker, "a" * 64, allocation)
    assert not docker.calls


def test_privileged_container_rejected(module, allocation):
    current = container(allocation)
    current["HostConfig"]["Privileged"] = True
    with pytest.raises(ValueError, match="unsafe"):
        module.inspect_owned(current, "a" * 64, allocation)


@pytest.mark.parametrize(
    "item",
    [
        endpoint(address="0.0.0.0"),
        endpoint(address="192.0.2.1"),
        endpoint(port=80),
        endpoint(port=29510),
        endpoint(remote_ip="192.0.2.1", remote_port=16384),
    ],
)
def test_dynamic_or_static_boundary_violation_rejected(module, allocation, item):
    with pytest.raises(ValueError):
        module.check_endpoints([item], allocation)


def test_arbitrary_dynamic_te_port_accepted_on_approved_ip(module, allocation):
    assert module.check_endpoints([endpoint(42019), endpoint(29510, "127.0.0.1")], allocation) == {42019, 29510}


def test_unrelated_host_listener_filtered_by_owned_inode(module):
    table = (
        "header\n 0: 00000000:0050 00000000:0000 0A 0:0 00:0 0 0 0 88\n"
        " 1: 0900000A:A423 00000000:0000 0A 0:0 00:0 0 0 0 99\n"
    )
    result = module.parse_socket_table(table, socket.AF_INET, "tcp", {"99"})
    assert len(result) == 1 and result[0]["local_ip"] == "10.0.0.9" and result[0]["local_port"] == 42019


def test_proc_ipv6_loopback_and_malformed_row(module):
    assert module.decode_address("00000000000000000000000001000000", socket.AF_INET6) == "::1"
    with pytest.raises(ValueError, match="unreadable"):
        module.parse_socket_table("header\nbad\n", socket.AF_INET, "tcp", set())


def test_output_size_and_symlink_fail_closed(module, tmp_path):
    (tmp_path / "dump").write_bytes(b"abcd")
    assert module.output_bytes(tmp_path) >= 4
    (tmp_path / "foreign").symlink_to(tmp_path / "dump")
    with pytest.raises(ValueError, match="symlink"):
        module.output_bytes(tmp_path)


def test_created_container_waits_then_stop_flag_cleans(module, allocation):
    docker = Docker(allocation)
    docker.current = container(allocation, running=False)
    clock = Clock(module.utc_timestamp(allocation["start_utc"]))

    def stop_after_first_poll(_):
        (Path(allocation["output"]) / "STOP").touch()

    assert (
        module.supervise(
            allocation,
            "a" * 64,
            allocation["output"],
            docker,
            wall=clock.wall,
            monotonic=clock.monotonic,
            sleep=stop_after_first_poll,
        )
        == 0
    )
    assert len(docker.calls) == 1


def test_port_violation_stops_before_ready_for_requests(module, allocation):
    docker = Docker(allocation, [endpoint(address="0.0.0.0")])
    clock = Clock(module.utc_timestamp(allocation["start_utc"]))
    assert module.supervise(allocation, "a" * 64, allocation["output"], docker, wall=clock.wall) == 1
    assert not (Path(allocation["output"]) / "guard-status.json").exists()
    assert len(docker.calls) == 1


def test_expired_proposed_window_never_arms(module, allocation):
    docker = Docker(allocation)
    with pytest.raises(ValueError, match="outside assigned"):
        module.supervise(
            allocation,
            "a" * 64,
            allocation["output"],
            docker,
            wall=lambda: module.utc_timestamp("2026-10-18T15:46:00Z"),
        )
    assert not docker.calls and not (Path(allocation["output"]) / "guard-ready.json").exists()


def test_owned_unsafe_configuration_still_gets_cleanup(module, allocation):
    docker = Docker(allocation)
    docker.current["HostConfig"]["Privileged"] = True
    assert (
        module.supervise(
            allocation,
            "a" * 64,
            allocation["output"],
            docker,
            wall=lambda: module.utc_timestamp(allocation["start_utc"]),
        )
        == 1
    )
    assert docker.calls == [("stop", "--time", "10", "a" * 64)]
    assert not (Path(allocation["output"]) / "guard-ready.json").exists()


def test_identity_change_before_kill_never_kills_foreign_owner(module, allocation):
    docker = Docker(allocation, stuck=True)
    original = docker.call

    def changed(*argv, **kwargs):
        try:
            original(*argv, **kwargs)
        finally:
            docker.current["Config"]["Labels"]["lease"] = "foreign-owner"

    docker.call = changed
    with pytest.raises(ValueError, match="ownership"):
        module.cleanup_owned(docker, "a" * 64, allocation)
    assert docker.calls == [("stop", "--time", "10", "a" * 64)]


def test_unobservable_proc_failure_stops_task(module, allocation):
    docker = Docker(allocation)

    def unavailable(_):
        raise PermissionError("proc unavailable")

    docker.endpoints = unavailable
    assert (
        module.supervise(
            allocation,
            "a" * 64,
            allocation["output"],
            docker,
            wall=lambda: module.utc_timestamp(allocation["start_utc"]),
        )
        == 1
    )
    assert len(docker.calls) == 1
    assert "proc unavailable" in (Path(allocation["output"]) / "guard-exit.json").read_text()


def test_duplicate_guard_is_rejected_without_touching_container(module, allocation):
    docker = Docker(allocation)
    (Path(allocation["output"]) / "guard-ready.json").write_text("{}")
    with pytest.raises(ValueError, match="do not duplicate"):
        module.supervise(
            allocation,
            "a" * 64,
            allocation["output"],
            docker,
            wall=lambda: module.utc_timestamp(allocation["start_utc"]),
        )
    assert not docker.calls


@pytest.mark.parametrize("condition", ["ready", "stale", "expired"])
def test_explicit_start_requires_fresh_guard_and_window(module, allocation, condition):
    docker = Docker(allocation)
    docker.current = container(allocation, running=False)
    now = module.utc_timestamp(allocation["start_utc"])
    status = {
        "container_id": "a" * 64,
        "lease_id": allocation["lease_id"],
        "guard_pid": 777,
        "stop_utc": allocation["stop_utc"],
        "checked_at_epoch": now - (10 if condition == "stale" else 0),
    }
    root = Path(allocation["output"])
    for filename in ("guard-ready.json", "guard-status.json"):
        (root / filename).write_text(json.dumps(status))
    clock = (lambda: now + 1000) if condition == "expired" else (lambda: now)
    if condition == "ready":
        result = module.start_owned(allocation, "a" * 64, docker, wall=clock, probe_pid=lambda *_: None)
        assert result["container_id"] == "a" * 64 and docker.calls == [("start", "a" * 64)]
    else:
        with pytest.raises((RuntimeError, ValueError)):
            module.start_owned(allocation, "a" * 64, docker, wall=clock, probe_pid=lambda *_: None)
        assert not docker.calls


def test_cleanup_error_receipt_matches_process_exit(module, allocation):
    docker = Docker(allocation)
    original = docker.call

    def changed_owner_after_stop(*args, **kwargs):
        original(*args, **kwargs)
        docker.current["Config"]["Labels"]["lease"] = "foreign-owner"

    docker.call = changed_owner_after_stop
    root = Path(allocation["output"])
    (root / "STOP").touch()
    exit_code = module.supervise(
        allocation, "a" * 64, allocation["output"], docker, wall=lambda: module.utc_timestamp(allocation["start_utc"])
    )
    receipt = json.loads((root / "guard-exit.json").read_text())
    assert exit_code == 1 and receipt["exit"] == exit_code
    assert "cleanup_error" in receipt
