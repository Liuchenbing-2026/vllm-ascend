# SPDX-License-Identifier: Apache-2.0
"""New entry boundaries only; no old component matrix or device execution."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return SimpleNamespace(
        entry=importlib.import_module("run_mooncake_kv_diagnostic"),
        runtime=importlib.import_module("kv_diagnostic_runtime"),
        snapshot=importlib.import_module("mooncake_kv_snapshot"),
    )


@pytest.fixture
def allocation(tmp_path):
    return {
        "status": "allocated",
        "lease_id": "cpu-only-not-an-allocation",
        "devices": [2, 5],
        "start_utc": "2026-10-19T01:00:00Z",
        "no_start_after_utc": "2026-10-19T01:05:00Z",
        "stop_utc": "2026-10-19T01:24:00Z",
        "end_utc": "2026-10-19T01:25:00Z",
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
        "output": str(tmp_path / "new-output"),
        "model": "/fixed/model",
        "ascend_source": "/fixed/ascend",
        "core_source": "/fixed/core",
        "runtime_work": "/fixed/runtime",
        "diagnostic_source": "/fixed/diagnostic",
        "launch_script": "/fixed/launch.sh",
    }


class Clock:
    def __init__(self, now):
        self.now, self.elapsed = now, 0

    def wall(self):
        return self.now + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, delay):
        self.elapsed += delay


class Docker:
    def __init__(self, module, spec, failure=None):
        self.module, self.spec, self.failure = module, spec, failure
        self.events, self.current, self.limits = [], None, []

    def call(self, *argv, timeout=5):
        self.events.append(argv[0])
        self.limits.append((argv[0], timeout))
        if argv[:2] == ("container", "ls"):
            return "b" * 64 if self.failure == "collision" else ""
        if argv[:2] == ("image", "inspect"):
            return json.dumps([{"Id": "image-id", "RepoDigests": [self.module.IMAGE]}])
        if argv[0] == "create":
            devices = [
                *(f"/dev/davinci{card}" for card in self.spec["devices"]),
                "/dev/davinci_manager",
                "/dev/devmm_svm",
                "/dev/hisi_hdc",
            ]
            self.current = {
                "Id": "a" * 64,
                "Image": "image-id",
                "Mounts": [],
                "Config": {"Labels": {"task": "bidir-three-kv", "lease": self.spec["lease_id"]}},
                "HostConfig": {
                    "Devices": [
                        {"PathOnHost": path, "PathInContainer": path, "CgroupPermissions": "rw"} for path in devices
                    ],
                    "Privileged": False,
                    "ReadonlyRootfs": True,
                    "RestartPolicy": {"Name": "no"},
                    "NetworkMode": "host",
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges"],
                },
                "State": {"Running": False, "Pid": 0, "Status": "created"},
            }
            if self.failure in ("timeout", "disconnect", "unknown-create"):
                if self.failure == "unknown-create":
                    self.current = None
                if self.failure == "disconnect":
                    raise InterruptedError("SSH disconnect")
                raise subprocess.TimeoutExpired(argv, timeout)
            return "a" * 64
        if argv[0] == "start":
            self.current["State"].update(Running=True, Pid=123, Status="running")
        if argv[0] == "exec":
            if "check-cache" in argv:
                self.events.append("cache-check")
                if self.failure == "cache":
                    raise subprocess.CalledProcessError(1, argv)
            else:
                self.events.append("diagnostic")
                if self.failure == "diagnostic":
                    raise subprocess.CalledProcessError(1, argv)
                root = Path(self.spec["output"]) / "original-case"
                root.mkdir()
                (root / "result.json").write_text('{"acceptance_pass":true}')
        if argv[0] in ("stop", "kill"):
            self.current["State"].update(Running=False, Pid=0, Status="exited")
        return ""

    def inspect(self, _):
        if self.current is None:
            raise subprocess.CalledProcessError(1, ["docker", "inspect"])
        return self.current


class Guard:
    def __init__(self):
        self.pid, self.returncode = 777, None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("guard", timeout)
        return self.returncode


def run_entry(modules, allocation, failure=None):
    entry = modules.entry
    clock = Clock(entry.utc_timestamp(allocation["start_utc"]))
    docker = Docker(entry, allocation, failure)
    guard = Guard()
    if failure == "guard-stuck":
        guard.terminate = lambda: None

    def kill_group(pid, _signal):
        assert pid == guard.pid
        docker.events.append("guard-group-kill")
        guard.kill()

    def spawn(root, _, container_id):
        if failure != "guard-not-ready":
            status = {
                "container_id": container_id,
                "lease_id": allocation["lease_id"],
                "guard_pid": guard.pid,
                "stop_utc": allocation["stop_utc"],
                "checked_at_epoch": clock.wall(),
            }
            for name in ("guard-ready.json", "guard-status.json"):
                (root / name).write_text(json.dumps(status))
        return guard

    fixture = b'{"synthetic":true}'
    source = Path(allocation["output"]).parent / "diagnostic"
    (source / "tools").mkdir(parents=True)
    files = [
        "run_mooncake_kv_diagnostic.py",
        "kv_diagnostic_runtime.py",
        "kv_diagnostic_readiness.py",
        "launch_mooncake_kv_diagnostic.sh",
        "mooncake_kv_snapshot.py",
        "test_mooncake_bidirectional_model.py",
    ]
    hashes = {}
    for filename in files:
        path = source / "tools" / filename
        path.write_bytes((Path(entry.__file__).parent / filename).read_bytes())
        if failure == "other-host-entry" and filename == "run_mooncake_kv_diagnostic.py":
            path.write_bytes(path.read_bytes() + b"\n# a different reviewed copy\n")
        hashes["tools/" + filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    allocation["diagnostic_source"] = str(source)
    allocation["launch_script"] = str(source / "tools/launch_mooncake_kv_diagnostic.sh")
    if failure == "source-hash":
        hashes["tools/test_mooncake_bidirectional_model.py"] = "0" * 64
    result, error = None, None
    try:
        result = entry.execute(
            allocation,
            fixture,
            hashlib.sha256(fixture).hexdigest(),
            json.dumps({"image_id": "image-id", "diagnostic_files": hashes}).encode(),
            docker=docker,
            guard_factory=spawn,
            wall=clock.wall,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            probe_pid=lambda *_: None,
            kill_group=kill_group,
        )
    except (InterruptedError, TimeoutError, subprocess.SubprocessError, RuntimeError, ValueError) as caught:
        error = caught
    receipt = json.loads((Path(allocation["output"]) / "controller-result.json").read_text())
    return SimpleNamespace(result=result, error=error, receipt=receipt, docker=docker, guard=guard)


@pytest.mark.parametrize("failure", ["timeout", "disconnect", "guard-not-ready"])
def test_create_or_guard_failure_never_starts_and_cleans_owned_id(modules, allocation, failure):
    result = run_entry(modules, allocation, failure)
    assert result.error is not None and "start" not in result.docker.events
    assert result.receipt["cleanup"]["running"] is False
    assert all(limit <= 10 for name, limit in result.docker.limits if name == "create")
    if failure == "guard-not-ready":
        assert result.receipt["guard_reaped"]


def test_unknown_daemon_create_is_not_claimed_reclaimed(modules, allocation):
    result = run_entry(modules, allocation, "unknown-create")
    assert result.error is not None and "start" not in result.docker.events
    assert result.receipt["create_outcome_unknown"]
    assert "cleanup" not in result.receipt


@pytest.mark.parametrize("failure", ["cache", "diagnostic", None])
def test_model_path_always_finishes_owned_cleanup(modules, allocation, failure):
    result = run_entry(modules, allocation, failure)
    assert result.receipt["cleanup"]["running"] is False and result.receipt["guard_reaped"]
    assert "cache-check" in result.docker.events
    if failure == "cache":
        assert "diagnostic" not in result.docker.events
    elif failure is None:
        assert result.result["complete"]
        assert result.docker.events.index("cache-check") < result.docker.events.index("diagnostic")


def test_local_hung_cli_is_killed_and_reaped(modules, tmp_path):
    pidfile = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        "import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)",
        str(pidfile),
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        modules.entry.bounded_command(command, 1, tmp_path / "stdout", tmp_path / "stderr")
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def runtime_fixture(tmp_path):
    names = [
        "vllm",
        "vllm_ascend",
        "torch",
        "torch_npu",
        "torch._C",
        "torch_npu._C",
        "vllm_ascend.vllm_ascend_C",
        "mooncake.engine",
    ]
    rules, loaded = {}, {}
    for name in names:
        native = name.endswith(("_C", "engine"))
        path = tmp_path / (name + (".so" if native else ".py"))
        path.write_bytes(b"\x7fELFcpu-fixture" if native else b"# cpu fixture\n")
        rules[name] = {"origin": str(path), "kind": "native" if native else "python"}
        loaded[name] = SimpleNamespace(__file__=str(path), __version__="cpu-fixture")
    loaded["vllm.config"] = SimpleNamespace(CacheConfig=lambda block_size: SimpleNamespace(block_size=block_size))
    return (
        {"image_id": "image-id", "modules": rules},
        {"image_id": "image-id", "readonly_rootfs": True, "mounts": []},
        loaded,
    )


@pytest.mark.parametrize("failure", ["hash", "origin", "unapproved-bind", None])
def test_actual_import_gate_runs_before_server_handoff(modules, tmp_path, failure):
    contract, observed, loaded = runtime_fixture(tmp_path)
    if failure == "hash":
        contract["modules"]["torch"]["sha256"] = "0" * 64
    if failure == "origin":
        contract["modules"]["torch"]["origin"] = "/unexpected/torch.py"
    if failure == "unapproved-bind":
        observed["mounts"] = [{"Destination": loaded["torch._C"].__file__, "Source": "/host/native.so", "RW": False}]
    for name, data in (("contract", contract), ("observed", observed)):
        (tmp_path / (name + ".json")).write_text(json.dumps(data))
    calls = []

    def invoke():
        return modules.runtime.serve(
            tmp_path / "contract.json",
            tmp_path / "observed.json",
            tmp_path / "receipt.json",
            ["--model", "/fixed/model", "--block-size", "128"],
            importer=loaded.__getitem__,
            runner=lambda *args, **kwargs: calls.append(args),
        )

    if failure:
        with pytest.raises(ValueError):
            invoke()
        assert not calls
    else:
        invoke()
        assert len(calls) == 1
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["server_handoff_allowed"] == (failure is None)
    assert "origin" in receipt["modules"]["torch"]


@pytest.mark.parametrize("arguments", [[], ["--block-size", "256"], ["--block-size", "128", "--block-size=128"]])
def test_block_contract_rejects_unfixed_or_conflicting_cli(modules, arguments):
    with pytest.raises(ValueError):
        modules.runtime.check_block_argument(arguments)


@pytest.mark.parametrize(("configured", "group", "valid"), [(128, 128, True), (256, 128, False), (128, 256, False)])
def test_effective_worker_cache_uses_real_extension_method(modules, monkeypatch, configured, group, valid):
    dependency = ModuleType("vllm.distributed.kv_transfer")
    connector = SimpleNamespace(
        engine_id="epoch",
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=configured)),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=group))]
        ),
    )
    dependency.get_kv_transfer_group = lambda: SimpleNamespace(connector_worker=connector)
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    result = modules.snapshot.KVSnapshotWorkerExtension().mooncake_diagnostic_cache_config()
    if valid:
        assert modules.runtime.verify_worker_cache([result])["engine_id"] == "epoch"
    else:
        with pytest.raises(ValueError):
            modules.runtime.verify_worker_cache([result])


def test_runtime_cli_keeps_gate_options_before_server_remainder(modules, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        sys, "argv", ["runtime", "serve", "--receipt", str(tmp_path / "receipt"), "--", "--block-size", "128"]
    )
    monkeypatch.setattr(modules.runtime, "serve", lambda *args, **kwargs: calls.append(args))
    modules.runtime.main()
    assert calls[0][2] == tmp_path / "receipt" and calls[0][3] == ["--block-size", "128"]


def test_stuck_guard_group_is_reaped_only_after_container_cleanup(modules, allocation):
    result = run_entry(modules, allocation, "guard-stuck")
    assert result.error is not None
    assert result.receipt["cleanup"]["running"] is False and result.receipt["guard_reaped"]
    assert result.docker.events.index("stop") < result.docker.events.index("guard-group-kill")


def test_existing_container_is_never_taken_over(modules, allocation):
    result = run_entry(modules, allocation, "collision")
    assert isinstance(result.error, ValueError)
    assert "create" not in result.docker.events and "stop" not in result.docker.events
    assert "cleanup" not in result.receipt


def test_disconnect_exception_reaps_real_local_cli(modules, monkeypatch, tmp_path):
    original_wait = subprocess.Popen.wait
    interrupted_pids = []

    def interrupted(process, timeout=None):
        if not interrupted_pids:
            interrupted_pids.append(process.pid)
            modules.entry.controller_signal(None, None)
        return original_wait(process, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupted)
    with pytest.raises(InterruptedError):
        modules.entry.bounded_command(
            [sys.executable, "-c", "import time;time.sleep(30)"], 1, tmp_path / "stdout", tmp_path / "stderr"
        )
    with pytest.raises(ProcessLookupError):
        os.kill(interrupted_pids[0], 0)


def test_same_interpreter_handoff_restores_argv(modules, tmp_path):
    contract, observed, loaded = runtime_fixture(tmp_path)
    for name, data in (("contract", contract), ("observed", observed)):
        (tmp_path / (name + ".json")).write_text(json.dumps(data))
    before = sys.argv
    seen = []
    modules.runtime.serve(
        tmp_path / "contract.json",
        tmp_path / "observed.json",
        tmp_path / "receipt.json",
        ["--block-size", "128"],
        importer=loaded.__getitem__,
        runner=lambda *args, **kwargs: seen.append(sys.argv[:]),
    )
    assert seen[0] == ["vllm.entrypoints.openai.api_server", "--block-size", "128"]
    assert sys.argv is before


def test_reviewed_bound_binary_has_explicit_image_copy_provenance(modules, tmp_path):
    contract, observed, loaded = runtime_fixture(tmp_path)
    path = Path(loaded["vllm_ascend.vllm_ascend_C"].__file__)
    observed["mounts"] = [{"Destination": str(path), "Source": "/host/image-copy.so", "RW": False}]
    contract["modules"]["vllm_ascend.vllm_ascend_C"].update(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), binary_source_image_id="image-id"
    )
    report = {}
    modules.runtime.verify_imports(contract, observed, report, importer=loaded.__getitem__)
    assert report["modules"]["vllm_ascend.vllm_ascend_C"]["provenance"] == "host-bind"
    assert report["imports_verified"]


@pytest.mark.parametrize("changed", [False, True])
def test_reviewed_entry_source_hash_binds_driver_before_create(modules, allocation, changed):
    result = run_entry(modules, allocation, "source-hash" if changed else None)
    if changed:
        assert isinstance(result.error, ValueError) and not result.docker.events
        assert "driver" not in result.receipt
    else:
        assert result.result["complete"]
        assert len(result.receipt["entry_source_sha256"]) == 6


def test_tmpfs_import_cannot_be_misclassified_as_image_binary(modules, tmp_path):
    contract, observed, loaded = runtime_fixture(tmp_path)
    observed["tmpfs"] = {str(tmp_path): "rw,size=256m"}
    with pytest.raises(ValueError, match="writable source"):
        modules.runtime.verify_imports(contract, observed, {}, importer=loaded.__getitem__)


@pytest.mark.parametrize("different", [False, True])
def test_actual_host_entry_fingerprint_matches_reviewed_copy(modules, allocation, different):
    result = run_entry(modules, allocation, "other-host-entry" if different else None)
    if different:
        assert isinstance(result.error, ValueError) and not result.docker.events
        assert "actual host entry" in result.receipt["error"]
    else:
        assert result.result["complete"]


def test_controller_deadline_reserves_cleanup_without_wall_clock_extension(modules, allocation):
    clock = Clock(modules.entry.utc_timestamp(allocation["start_utc"]))
    deadline = modules.entry.Deadline(allocation, wall=clock.wall, monotonic=clock.monotonic)
    clock.elapsed = 1439
    with pytest.raises(TimeoutError):
        deadline.limit(5)
    deadline.cleaning = True
    assert deadline.limit(100) == 59
    clock.now -= 3600
    clock.elapsed = 1500
    with pytest.raises(TimeoutError):
        deadline.limit(5)


def test_reviewed_contract_fingerprint_is_checked_before_allocation_execution(modules, monkeypatch, tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entry",
            "--allocation",
            str(tmp_path / "not-issued.json"),
            "--fixture",
            str(tmp_path / "missing-fixture"),
            "--fixture-sha256",
            "0" * 64,
            "--runtime-contract",
            str(contract),
            "--runtime-contract-sha256",
            "0" * 64,
        ],
    )
    with pytest.raises(SystemExit) as result:
        modules.entry.main()
    assert result.value.code == 2


def test_interpreter_identity_is_observed_in_running_namespace(modules, tmp_path):
    contract, observed, loaded = runtime_fixture(tmp_path)
    report = {}
    modules.runtime.verify_imports(contract, observed, report, importer=loaded.__getitem__)
    assert report["python"]["resolved"] == str(Path(sys.executable).resolve())
    assert report["python"]["sha256"] == modules.runtime.file_digest(Path(sys.executable).resolve())
