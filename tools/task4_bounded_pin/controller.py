# SPDX-License-Identifier: Apache-2.0
"""Host controller. Default is zero-allocation dry-run; --execute needs a new lease."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from task4_bounded_pin import StageALimits, exclusive_json, validate_lease
from timing import PhaseTimer


def command(args, timeout=30):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()


def inspect_owned(container, lease_id):
    state = json.loads(command(["docker", "inspect", container]))[0]
    if state["Id"] != container or state["Config"]["Labels"].get("task4.pin.lease") != lease_id:
        raise ValueError("Refusing to control an unowned container")
    return state


def stop_owned(container, lease_id):
    state = inspect_owned(container, lease_id)
    if state["State"]["Running"]:
        command(["docker", "stop", "--time", "10", container], timeout=45)
    state = inspect_owned(container, lease_id)
    if state["State"]["Running"]:
        raise RuntimeError("Owned container still running; report resource release failure")
    return state["State"]


def reap(process):
    if process is not None:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def check_payload(root, lease):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["ascend_sha"] != lease["ascend_sha"]:
        raise ValueError("Lease and packaged source SHA differ")
    for name, digest in manifest["payload"].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Payload mismatch: " + name)


def validate_idle(output):
    marker = "No running processes found in NPU "
    idle = {int(line.split(marker, 1)[1].split()[0]) for line in output.splitlines() if marker in line}
    if idle != set(range(8)):
        raise RuntimeError("All eight devices must be independently observed idle")


def docker_create_args(root, lease):
    name = "task4-rc2-pin-" + hashlib.sha256(lease["lease_id"].encode()).hexdigest()[:12]
    args = [
        "docker",
        "create",
        "--name",
        name,
        "--label",
        "task4.pin.lease=" + lease["lease_id"],
        "--restart=no",
        "--network=none",
        "--ipc=host",
        "--privileged",
        "--entrypoint",
        "/bin/bash",
    ]
    for device in [
        *(f"/dev/davinci{rank}" for rank in range(8)),
        "/dev/davinci_manager",
        "/dev/devmm_svm",
        "/dev/hisi_hdc",
    ]:
        args += ["--device", device]
    for source, target in (
        (str(root), "/task4"),
        (lease["model_path"], "/models/dsv4:ro"),
        ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver:ro"),
        ("/usr/local/dcmi", "/usr/local/dcmi:ro"),
        ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi:ro"),
    ):
        args += ["-v", source + ":" + target]
    return args + [lease["image_id"], "/task4/entrypoint.sh"]


def monitor(root, lease, container, logger, watchdog, timer):
    output = root / "output"
    exclusive_json(output / "monitor.started.json", {"wall_epoch": time.time(), "monotonic": time.monotonic()})
    while True:
        # Preserve worker first failure before evaluating a secondary timeout.
        if (output / "FAILED.json").exists():
            raise RuntimeError("Worker first failure recorded")
        validate_lease(lease, time.time())
        if json.loads((root / "lease.json").read_text()) != lease:
            raise RuntimeError("Lease changed")
        complete = timer.check(output, time.monotonic())
        if logger.poll() is not None or watchdog.poll() is not None:
            raise RuntimeError("Logger or watchdog exited unexpectedly")
        if not inspect_owned(container, lease["lease_id"])["State"]["Running"]:
            raise RuntimeError("Container exited before complete evidence")
        if timer.pin_deadline is not None and not (output / "pin.permit.json").exists():
            exclusive_json(
                output / "pin.permit.json",
                {
                    "lease_id": lease["lease_id"],
                    "deadline_monotonic": timer.pin_deadline,
                },
            )
        if complete:
            return
        time.sleep(1)


def execute(root, lease):
    validate_lease(lease, time.time())
    timer = PhaseTimer(lease, time.time(), time.monotonic())
    check_payload(root, lease)
    if time.time() >= lease["start_epoch"] + 60:
        raise RuntimeError("Too late to begin a fresh 25 minute attempt")
    if lease["image_id"] != "sha256:0faf08fb90dd21807266cb719b4947b0111b3bd5f58c46424810a8ecabe82b98":
        raise ValueError("Only the fixed reviewed rc2 image is allowed")
    image = json.loads(command(["docker", "image", "inspect", lease["image_id"]]))[0]
    if lease["image"] not in image["RepoDigests"]:
        raise ValueError("Image digest mismatch")
    output = root / "output"
    output.mkdir(exist_ok=False)
    exclusive_json(output / "timing.anchor.json", timer.record())
    (output / "image.json").write_text(json.dumps(image))
    initial = command(["npu-smi", "info"])
    (output / "resources_before.log").write_text(initial)
    validate_idle(initial)
    exclusive_json(output / "controller.identity.json", {"pid": os.getpid(), "start": time.time(), "lease": lease})
    container, watchdog, logger = None, None, None
    status = 1
    with (output / "server.log").open("x") as server_log, (output / "watchdog.log").open("x") as watchdog_log:
        try:
            argv = docker_create_args(root, lease)
            exclusive_json(output / "create.command.json", argv)
            container = command(argv)
            inspect_owned(container, lease["lease_id"])
            (output / "container.id").write_text(container)
            watchdog = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "controller.py"),
                    "--watchdog",
                    container,
                    lease["lease_id"],
                    str(timer.hard_deadline),
                ],
                stdout=watchdog_log,
                stderr=subprocess.STDOUT,
            )
            exclusive_json(output / "watchdog.identity.json", {"pid": watchdog.pid})
            validate_lease(lease, time.time())
            timer.check(output, time.monotonic())
            command(["docker", "start", container])
            logger = subprocess.Popen(
                ["docker", "logs", "--follow", container], stdout=server_log, stderr=subprocess.STDOUT
            )
            exclusive_json(output / "logger.identity.json", {"pid": logger.pid})
            monitor(root, lease, container, logger, watchdog, timer)
            status = 0
        except BaseException as error:
            with suppress(FileExistsError):
                exclusive_json(
                    output / "FAILED.json",
                    {
                        "source": "controller",
                        "error": repr(error),
                        "wall_epoch": time.time(),
                        "monotonic": time.monotonic(),
                    },
                )
        finally:
            # Bounded kernel evidence, including allocation/driver failures.
            try:
                kernel = command(
                    ["journalctl", "-k", "--since", "@" + str(lease["start_epoch"]), "-n", "2000", "--no-pager"],
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError) as error:
                kernel = "Kernel log capture unavailable: " + repr(error)
            (output / "kernel.log").write_text(kernel)
            # Keep watchdog armed if cleanup fails. Never stop by a reused name.
            if container is not None:
                state = stop_owned(container, lease["lease_id"])
                (output / "container_state.json").write_text(json.dumps(state, indent=2))
            reap(logger)
            reap(watchdog)
            exclusive_json(
                output / "children_reaped.json",
                {
                    "logger": None if logger is None else logger.returncode,
                    "watchdog": None if watchdog is None else watchdog.returncode,
                },
            )
            (output / "resources_after.log").write_text(command(["npu-smi", "info"]))
            exclusive_json(
                output / "controller.exit.json",
                {"exit": status, "time": time.time(), "meaning": "probe only, not serving/performance/224GiB repair"},
            )
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--watchdog", nargs=3, metavar=("CONTAINER_ID", "LEASE_ID", "DEADLINE"))
    args = parser.parse_args()
    if args.watchdog:
        container, lease_id, deadline = args.watchdog
        if len(container) != 64 or any(char not in "0123456789abcdef" for char in container):
            raise ValueError("Watchdog requires immutable full container ID")
        time.sleep(max(0, float(deadline) - time.monotonic()))
        stop_owned(container, lease_id)
        return 0
    if args.lease is None:
        parser.error("--lease is required")
    root = args.lease.resolve().parent
    if args.lease.name != "lease.json":
        raise ValueError("Lease must be packaged as lease.json")
    lease = json.loads(args.lease.read_text())
    check_payload(root, lease)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "ZERO_ALLOCATION_DRY_RUN",
                    "lease_status": lease["status"],
                    "rank0_probe_bytes": StageALimits().total_bytes,
                    "other_ranks_probe_bytes": 0,
                    "container_command_preview": docker_create_args(root, lease),
                },
                indent=2,
            )
        )
        return 0

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("controller SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    return execute(root, lease)


if __name__ == "__main__":
    sys.exit(main())
