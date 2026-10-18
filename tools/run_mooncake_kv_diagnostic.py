# SPDX-License-Identifier: Apache-2.0
"""Explicit, bounded host entrypoint after a regulator-issued allocation only."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from kv_diagnostic_readiness import (
    IMAGE,
    TASK,
    HostDocker,
    cleanup_owned,
    create_plan,
    inspect_owned,
    start_owned,
    utc_timestamp,
    validate_allocation,
    write_json,
)

CLI_REAP_RESERVE_SECONDS = 2


class Deadline:
    def __init__(self, spec, *, wall=time.time, monotonic=time.monotonic):
        self.wall, self.monotonic = wall, monotonic
        self.stop, self.end = (utc_timestamp(spec[key]) for key in ("stop_utc", "end_utc"))
        now, elapsed = wall(), monotonic()
        self.mono_stop, self.mono_end = elapsed + self.stop - now, elapsed + self.end - now
        self.cleaning = False

    def limit(self, requested):
        end, mono_end = (self.end, self.mono_end) if self.cleaning else (self.stop, self.mono_stop)
        # Reserve two seconds for terminating/reaping a timed-out local CLI.
        remaining = min(requested, min(end - self.wall(), mono_end - self.monotonic()) - CLI_REAP_RESERVE_SECONDS)
        if remaining <= 0:
            raise TimeoutError("assigned absolute/monotonic deadline exhausted")
        return remaining


def bounded_command(argv, timeout, stdout_path, stderr_path):
    """Bound and reap the local CLI process group, not arbitrary host PIDs."""
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=CLI_REAP_RESERVE_SECONDS)
            raise
    if result:
        raise subprocess.CalledProcessError(result, argv)
    return stdout_path.read_text()


class BoundedDocker(HostDocker):
    def __init__(self, root, deadline):
        self.root, self.deadline, self.sequence = root, deadline, 0

    def call(self, *argv, timeout=5):
        self.sequence += 1
        stem = self.root / f"control-{self.sequence:04d}"
        limit = self.deadline.limit(timeout)
        record = {"argv": ["docker", *argv], "timeout": limit}
        try:
            output = bounded_command(record["argv"], limit, stem.with_suffix(".stdout"), stem.with_suffix(".stderr"))
            record["exit"] = 0
            return output
        except BaseException as error:
            record["error"] = repr(error)
            raise
        finally:
            write_json(stem.with_suffix(".json"), record, exclusive=True)


def spawn_guard(root, allocation, container_id):
    with (root / "guard-host.stdout").open("xb") as stdout, (root / "guard-host.stderr").open("xb") as stderr:
        return subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("kv_diagnostic_readiness.py")),
                "guard",
                "--allocation",
                str(allocation),
                "--container-id",
                container_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


def verify_entry_sources(spec, contract):
    required = {
        "tools/run_mooncake_kv_diagnostic.py",
        "tools/kv_diagnostic_runtime.py",
        "tools/kv_diagnostic_readiness.py",
        "tools/launch_mooncake_kv_diagnostic.sh",
        "tools/mooncake_kv_snapshot.py",
        "tools/test_mooncake_bidirectional_model.py",
    }
    expected = contract["diagnostic_files"]
    if not required <= expected.keys():
        raise ValueError("runtime contract omits a diagnostic entry source")
    for name in ("run_mooncake_kv_diagnostic.py", "kv_diagnostic_readiness.py"):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected["tools/" + name]:
            raise ValueError("actual host entry/guard differs from reviewed source: " + name)
    root = Path(spec["diagnostic_source"]).resolve(strict=True)
    if Path(spec["launch_script"]).resolve(strict=True) != root / "tools/launch_mooncake_kv_diagnostic.sh":
        raise ValueError("launcher is not the reviewed diagnostic entry")
    actual = {}
    for relative in sorted(required):
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("diagnostic source escapes the reviewed root")
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[relative] != expected[relative]:
            raise ValueError("diagnostic source differs from reviewed hash: " + relative)
    return actual


def execute(
    spec,
    fixture_bytes,
    fixture_hash,
    contract_bytes,
    *,
    docker=None,
    guard_factory=spawn_guard,
    wall=time.time,
    monotonic=time.monotonic,
    sleep=time.sleep,
    probe_pid=os.kill,
    kill_group=os.killpg,
    enter_cleanup=lambda: None,
):
    validate_allocation(spec)
    if not utc_timestamp(spec["start_utc"]) <= wall() <= utc_timestamp(spec["no_start_after_utc"]):
        raise ValueError("no current assigned start window")
    if hashlib.sha256(fixture_bytes).hexdigest() != fixture_hash:
        raise ValueError("reviewed fixture hash mismatch")
    contract = json.loads(contract_bytes)
    root = Path(spec["output"])
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    (root / "allocation.json").write_text(json.dumps(spec, indent=2) + "\n")
    (root / "fixed-inputs.json").write_bytes(fixture_bytes)
    (root / "runtime-contract.json").write_bytes(contract_bytes)
    for path in root.iterdir():
        path.chmod(0o600)
    deadline = Deadline(spec, wall=wall, monotonic=monotonic)
    docker = BoundedDocker(root, deadline) if docker is None else docker
    container_id, guard = None, None
    name = TASK + "-" + spec["lease_id"]
    receipt = {
        "complete": False,
        "lease_released": False,
        "container_name": name,
        "fixture_sha256": fixture_hash,
        "runtime_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
    }
    create_attempted = False
    try:
        receipt["entry_source_sha256"] = verify_entry_sources(spec, contract)
        # Name collision is a refusal, never permission to take over a container.
        existing = docker.call(
            "container", "ls", "--all", "--no-trunc", "--filter", "name=^/" + name + "$", "--quiet"
        ).strip()
        if existing:
            raise ValueError("container name already exists; no takeover")
        image = json.loads(docker.call("image", "inspect", IMAGE))[0]
        if image["Id"] != contract["image_id"] or IMAGE not in (image.get("RepoDigests") or []):
            raise ValueError("actual image differs from runtime contract")
        plan = create_plan(spec)
        plan[2:2] = ["--cidfile", str(root / "container.id")]
        create_attempted = True
        container_id = docker.call(*plan[1:], timeout=10).strip()
        current = docker.inspect(container_id)
        inspect_owned(current, container_id, spec)
        if current["Image"] != image["Id"] or not current["HostConfig"]["ReadonlyRootfs"]:
            raise ValueError("container image/readonly identity differs before start")
        write_json(
            root / "container-observed.json",
            {
                "image_id": current["Image"],
                "mounts": current["Mounts"],
                "tmpfs": current["HostConfig"].get("Tmpfs", {}),
                "readonly_rootfs": True,
            },
            exclusive=True,
        )
        guard = guard_factory(root, root / "allocation.json", container_id)
        write_json(root / "controller-guard.json", {"pid": guard.pid, "container_id": container_id}, exclusive=True)
        ready_deadline = monotonic() + deadline.limit(10)
        while not all((root / item).is_file() for item in ("guard-ready.json", "guard-status.json")):
            if guard.poll() is not None or monotonic() >= ready_deadline:
                raise TimeoutError("independent guard did not become ready; start forbidden")
            deadline.limit(1)
            sleep(0.1)
        start_owned(spec, container_id, docker, wall=wall, probe_pid=probe_pid)
        # Engine starts only after its in-process preload identity gate. Before
        # any model request, ask the actual worker for its effective cache size.
        docker.call(
            "exec",
            container_id,
            "/work/.venv-core-stream/bin/python",
            "/diag/tools/kv_diagnostic_runtime.py",
            "check-cache",
            "--container-id",
            container_id,
            timeout=deadline.limit(120),
        )
        docker.call(
            "exec",
            container_id,
            "/work/.venv-core-stream/bin/python",
            "/diag/tools/test_mooncake_bidirectional_model.py",
            "--fixture",
            "/output/fixed-inputs.json",
            "--fixture-sha256",
            fixture_hash,
            "--prefill",
            "http://127.0.0.1:29510",
            "--decode",
            "http://127.0.0.1:29511",
            "--guard-status",
            "/output/guard-status.json",
            "--container-id",
            container_id,
            "--output",
            "/output/original-case",
            timeout=deadline.limit(600),
        )
        result = json.loads((root / "original-case/result.json").read_text())
        if not result.get("acceptance_pass"):
            raise AssertionError("fixed-input diagnostic acceptance failed")
        receipt["complete"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        enter_cleanup()
        deadline.cleaning = True
        (root / "STOP").touch(mode=0o600, exist_ok=True)
        # A timed-out docker client does not cancel the daemon operation. Resolve
        # only the exact unique name/cidfile, then verify full ID and owner labels.
        # An unknown late create is never started and is never called reclaimed.
        try:
            if container_id is None and create_attempted:
                container_id = docker.inspect(name)["Id"]
            if container_id is not None:
                receipt["cleanup"] = cleanup_owned(docker, container_id, spec)
            elif create_attempted:
                receipt["create_outcome_unknown"] = True
        except BaseException as error:
            receipt["cleanup_error"] = repr(error)
            receipt["complete"] = False
            if create_attempted and container_id is None:
                receipt["create_outcome_unknown"] = True
        if guard is not None:
            try:
                if guard.poll() is None and "cleanup" in receipt:
                    guard.terminate()
                guard.wait(timeout=deadline.limit(5))
                receipt["guard_reaped"] = True
            except BaseException as error:
                receipt["guard_cleanup_error"] = repr(error)
                receipt["complete"] = False
                if guard.poll() is None and "cleanup" in receipt:
                    # spawn_guard starts its own session: escalation covers only
                    # that guard and its own possibly blocked docker CLI child.
                    with contextlib.suppress(ProcessLookupError):
                        kill_group(guard.pid, signal.SIGKILL)
                    try:
                        guard.wait(timeout=deadline.limit(2))
                        receipt["guard_reaped"] = True
                    except BaseException as reap_error:
                        receipt["guard_reap_error"] = repr(reap_error)
                elif guard.poll() is None:
                    receipt["guard_left_for_deadline"] = True
        write_json(root / "controller-result.json", receipt, exclusive=True)
    if not receipt["complete"]:
        raise RuntimeError("controller cleanup incomplete; see fixed receipt")
    return receipt


def controller_signal(_number, _frame):
    raise InterruptedError("controller signal/disconnect")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--runtime-contract-sha256", required=True)
    args = parser.parse_args()
    contract = args.runtime_contract.read_bytes()
    if hashlib.sha256(contract).hexdigest() != args.runtime_contract_sha256:
        parser.error("runtime contract differs from reviewed fingerprint")
    signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    previous = {number: signal.signal(number, controller_signal) for number in signals}
    try:
        execute(
            json.loads(args.allocation.read_text()),
            args.fixture.read_bytes(),
            args.fixture_sha256,
            contract,
            enter_cleanup=lambda: [signal.signal(number, signal.SIG_IGN) for number in signals],
        )
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
