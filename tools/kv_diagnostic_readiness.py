# SPDX-License-Identifier: Apache-2.0
"""Plan a two-device container and supervise its immutable ID from the host.

No allocation API, automatic start, restart, image pull, or device reset.
The allocation JSON must first be issued/reviewed by the resource supervisor.
Socket polling detects violations; it is not a firewall or a DMA fence.
"""

import argparse
import contextlib
import datetime
import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

TASK = "bidir-three-kv"
IMAGE = "quay.io/ascend/vllm-ascend@sha256:6348403a0ebb19f52bc1b7497de1daeb6df0e4650f9051c169e47a011527a6c3"
OUTPUT_LIMIT = 6 * 1024**3
SUPPORT_DEVICES = ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc")


def utc_timestamp(value):
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != datetime.timedelta(0):
        raise ValueError("absolute UTC time required")
    return parsed.timestamp()


def validate_allocation(spec):
    if spec["status"] != "allocated" or not re.fullmatch(r"[A-Za-z0-9_-]+", spec["lease_id"]):
        raise ValueError("a proposal or queue receipt is not an allocation")
    cards = spec["devices"]
    if len(cards) != 2 or len(set(cards)) != 2 or any(type(card) is not int or card < 0 for card in cards):
        raise ValueError("exactly two distinct assigned physical device IDs required")
    start, latest, stop, end = (
        utc_timestamp(spec[key]) for key in ("start_utc", "no_start_after_utc", "stop_utc", "end_utc")
    )
    if not start <= latest < stop <= end - 60:
        raise ValueError("invalid start cutoff or missing 60-second cleanup reserve")
    if spec["host_network_approved"] is not True or spec["endpoint_policy_reviewed"] is not True:
        raise ValueError("host networking and endpoint policy require explicit review")
    addresses = {str(ipaddress.ip_address(value)) for value in spec["allowed_ips"]}
    if any(ipaddress.ip_address(value).is_unspecified for value in addresses):
        raise ValueError("wildcard binding is outside this template")
    if spec["host_ip"] not in addresses or "127.0.0.1" not in addresses:
        raise ValueError("approved host IP and IPv4 loopback required")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", spec["interface"]):
        raise ValueError("invalid explicitly selected interface")
    low, high = spec["dynamic_ports"]
    if not 1024 <= low <= high <= 65535 or not 1 <= spec["max_sockets"] <= 64:
        raise ValueError("invalid bounded dynamic endpoint policy")
    if spec["output_limit_bytes"] != OUTPUT_LIMIT:
        raise ValueError("this diagnosis has a fixed 6 GiB output budget")
    for key in ("hccl_host_ports", "hccl_npu_ports"):
        first, last = spec[key]
        if not 1024 <= first <= last <= 65535:
            raise ValueError("HCCL needs an explicitly reviewed nonprivileged port range")
    for key in (
        "output",
        "model",
        "ascend_source",
        "core_source",
        "diagnostic_source",
        "launch_script",
        "runtime_work",
    ):
        path = Path(spec[key])
        if not path.is_absolute() or ".." in path.parts or ":" in str(path) or "," in str(path):
            raise ValueError("mount paths must be absolute and unambiguous")
    return spec


def create_plan(spec):
    """Return argv only. docker create/start must be explicit, after allocation."""
    validate_allocation(spec)
    argv = [
        "docker",
        "create",
        "--pull=never",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=256m",
        "--tmpfs",
        "/run:rw,nosuid,nodev,size=16m",
        "--tmpfs",
        "/root:rw,nosuid,nodev,size=256m",
        "--name",
        TASK + "-" + spec["lease_id"],
        "--runtime",
        "runc",
        "--label",
        "task=" + TASK,
        "--label",
        "lease=" + spec["lease_id"],
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--restart",
        "no",
        "--network",
        "host",
        "--cpus",
        "12",
        "--memory",
        "64g",
        "--memory-swap",
        "64g",
        "--shm-size",
        "4g",
        "--ulimit",
        "core=0",
        "--log-driver",
        "none",
        "--env",
        "VLLM_HOST_IP=" + spec["host_ip"],
        "--env",
        "HCCL_IF_IP=" + spec["host_ip"],
        "--env",
        "HCCL_SOCKET_IFNAME=" + spec["interface"],
        "--env",
        "GLOO_SOCKET_IFNAME=" + spec["interface"],
        "--env",
        "TP_SOCKET_IFNAME=" + spec["interface"],
        "--env",
        "HCCL_HOST_SOCKET_PORT_RANGE=" + "-".join(map(str, spec["hccl_host_ports"])),
        "--env",
        "HCCL_NPU_SOCKET_PORT_RANGE=" + "-".join(map(str, spec["hccl_npu_ports"])),
    ]
    for path in [*(f"/dev/davinci{card}" for card in spec["devices"]), *SUPPORT_DEVICES]:
        argv.extend(["--device", f"{path}:{path}:rw"])
    mounts = [
        (spec["output"], "/output", "rw"),
        (spec["model"], "/models/Qwen3-8B", "ro"),
        (spec["ascend_source"], "/vllm-workspace/vllm-ascend", "ro"),
        (spec["runtime_work"] + "/.venv-core-stream", "/work/.venv-core-stream", "ro"),
        (spec["diagnostic_source"], "/diag", "ro"),
        (spec["launch_script"], "/launch.sh", "ro"),
        ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver", "ro"),
        ("/usr/local/dcmi", "/usr/local/dcmi", "ro"),
        ("/etc/ascend_install.info", "/etc/ascend_install.info", "ro"),
        ("/etc/hccn.conf", "/etc/hccn.conf", "ro"),
        ("/usr/local/sbin/npu-smi", "/usr/local/bin/npu-smi", "ro"),
    ]
    for relative in (
        "chat_completion/protocol.py",
        "chat_completion/serving.py",
        "completion/protocol.py",
        "completion/serving.py",
    ):
        suffix = "/vllm/entrypoints/openai/" + relative
        mounts.append((spec["core_source"] + suffix, "/vllm-workspace/vllm" + suffix, "ro"))
    for source, destination, access in mounts:
        argv.extend(["--volume", f"{source}:{destination}:{access}"])
    return [*argv, "--entrypoint", "bash", IMAGE, "/launch.sh"]


def verify_identity(container, container_id, spec):
    if not re.fullmatch(r"[0-9a-f]{64}", container_id) or container["Id"] != container_id:
        raise ValueError("full immutable container ID mismatch")
    labels = container["Config"].get("Labels") or {}
    if labels.get("task") != TASK or labels.get("lease") != spec["lease_id"]:
        raise ValueError("container ownership mismatch; do not stop a foreign container")


def inspect_owned(container, container_id, spec):
    verify_identity(container, container_id, spec)
    config = container["HostConfig"]
    devices = config.get("Devices") or []
    expected = {*(f"/dev/davinci{card}" for card in spec["devices"]), *SUPPORT_DEVICES}
    if len(devices) != len(expected) or {item["PathOnHost"] for item in devices} != expected:
        raise ValueError("container device mapping differs from allocation")
    if any(item["PathInContainer"] != item["PathOnHost"] or item["CgroupPermissions"] != "rw" for item in devices):
        raise ValueError("unexpected device alias or permission")
    if config.get("Privileged") or config["RestartPolicy"]["Name"] != "no" or config["NetworkMode"] != "host":
        raise ValueError("unsafe container lifecycle configuration")
    if "ALL" not in (config.get("CapDrop") or []) or "no-new-privileges" not in (config.get("SecurityOpt") or []):
        raise ValueError("missing privilege restriction")


def output_bytes(root):
    """No following symlinks; include allocated bytes as well as logical size."""
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in [*dirs, *files]:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("unexpected output symlink")
            stat = path.stat()
            total += max(stat.st_size, stat.st_blocks * 512)
    return total


def decode_address(hex_address, family):
    raw = bytes.fromhex(hex_address)
    # /proc/net encodes each 32-bit word in native (little-endian on target) order.
    raw = b"".join(raw[index : index + 4][::-1] for index in range(0, len(raw), 4))
    return socket.inet_ntop(family, raw)


def parse_socket_table(text, family, protocol, owned_inodes):
    result = []
    for row in text.splitlines()[1:]:
        fields = row.split()
        if len(fields) < 10:
            raise ValueError("unreadable proc socket row")
        if fields[9] not in owned_inodes:
            continue
        local_ip, local_port = fields[1].split(":")
        remote_ip, remote_port = fields[2].split(":")
        result.append(
            {
                "protocol": protocol,
                "state": fields[3],
                "inode": fields[9],
                "local_ip": decode_address(local_ip, family),
                "local_port": int(local_port, 16),
                "remote_ip": decode_address(remote_ip, family),
                "remote_port": int(remote_port, 16),
            }
        )
    return result


def check_endpoints(endpoints, spec):
    if len(endpoints) > spec["max_sockets"]:
        raise ValueError("owned socket count exceeded")
    low, high = spec["dynamic_ports"]
    static = {int(port) for port in spec["static_ports"]}
    observed = set()
    for item in endpoints:
        port = item["local_port"]
        if item["local_ip"] not in spec["allowed_ips"] or not (port in static or low <= port <= high):
            raise ValueError("local socket outside reviewed address/port boundary")
        if port in (29510, 29511) and item["local_ip"] != "127.0.0.1":
            raise ValueError("HTTP/development RPC must bind IPv4 loopback only")
        if item["remote_port"] and (
            item["remote_ip"] not in spec["allowed_ips"]
            or not (item["remote_port"] in static or low <= item["remote_port"] <= high)
        ):
            raise ValueError("peer endpoint outside same-host boundary")
        if item["state"] == "0A":
            observed.add(port)
    return observed


class HostDocker:
    """The guard only inspects/stops; explicit start has its own readiness gate."""

    def call(self, *argv, timeout=5):
        return subprocess.run(["docker", *argv], check=True, capture_output=True, text=True, timeout=timeout).stdout

    def inspect(self, container_id):
        return json.loads(self.call("inspect", container_id))[0]

    def endpoints(self, container_id):
        pids = self.call("top", container_id, "-eo", "pid").splitlines()[1:]
        inodes, namespaces = set(), {}
        for value in pids:
            pid = int(value.strip())
            root = Path("/proc") / str(pid)
            if container_id not in (root / "cgroup").read_text():
                raise ValueError("cannot prove proc PID still belongs to owned container")
            # PID exit/read failures stop this diagnostic, never silently omit it.
            for fd in (root / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue  # FD may close between list/read; next poll rechecks.
                match = re.fullmatch(r"socket:\[(\d+)\]", target)
                if match:
                    inodes.add(match[1])
            namespaces.setdefault(os.readlink(root / "ns/net"), root)
        if not namespaces:
            raise ValueError("running container has no observable owned PIDs")
        result = []
        for root in namespaces.values():
            for table, family, protocol in (
                ("tcp", socket.AF_INET, "tcp"),
                ("tcp6", socket.AF_INET6, "tcp"),
                ("udp", socket.AF_INET, "udp"),
                ("udp6", socket.AF_INET6, "udp"),
            ):
                result.extend(parse_socket_table((root / "net" / table).read_text(), family, protocol, inodes))
        return result


def write_json(path, value, *, exclusive=False):
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    else:
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    path.chmod(0o600)


def cleanup_owned(docker, container_id, spec):
    verify_identity(docker.inspect(container_id), container_id, spec)
    # Reinspect ownership before escalation, including after a stop timeout.
    with contextlib.suppress(subprocess.SubprocessError, OSError):
        docker.call("stop", "--time", "10", container_id, timeout=15)
    current = docker.inspect(container_id)
    verify_identity(current, container_id, spec)
    if current["State"]["Running"]:
        docker.call("kill", container_id)
        current = docker.inspect(container_id)
        verify_identity(current, container_id, spec)
    if (
        current["State"]["Running"]
        or current["State"]["Pid"] != 0
        or current["HostConfig"]["RestartPolicy"]["Name"] != "no"
    ):
        raise RuntimeError("owned container remains live; supervisor must verify")
    return {"container_id": container_id, "running": False, "pid": 0, "restart": "no", "lease_released": False}


def supervise(spec, container_id, root, docker, *, wall=time.time, monotonic=time.monotonic, sleep=time.sleep):
    validate_allocation(spec)
    now = wall()
    if not utc_timestamp(spec["start_utc"]) <= now <= utc_timestamp(spec["no_start_after_utc"]):
        raise ValueError("outside assigned start window; never auto extend")
    root = Path(root)
    if root.is_symlink() or root.resolve() != Path(spec["output"]).resolve():
        raise ValueError("guard output must be the assigned private output root")
    if any((root / name).exists() for name in ("guard-ready.json", "guard-exit.json")):
        raise ValueError("existing guard evidence; do not duplicate or take over a guard")
    cutoff = utc_timestamp(spec["stop_utc"])
    monotonic_cutoff = monotonic() + cutoff - now
    ready = {
        "container_id": container_id,
        "lease_id": spec["lease_id"],
        "guard_pid": os.getpid(),
        "stop_utc": spec["stop_utc"],
        "scope": "host process; no device reset or lease release",
    }
    reason, exit_code = "", 1
    try:
        inspect_owned(docker.inspect(container_id), container_id, spec)
        write_json(root / "guard-ready.json", ready, exclusive=True)
        while True:
            current = docker.inspect(container_id)
            inspect_owned(current, container_id, spec)
            if wall() >= cutoff or monotonic() >= monotonic_cutoff:
                raise TimeoutError("absolute or monotonic deadline")
            if (root / "STOP").exists():
                reason, exit_code = "owner stop flag", 0
                break
            size = output_bytes(root)
            if size >= spec["output_limit_bytes"]:
                raise RuntimeError("6 GiB output budget reached")
            endpoints, observed = [], set()
            if current["State"]["Running"]:
                endpoints = docker.endpoints(container_id)
                observed = check_endpoints(endpoints, spec)
            elif current["State"]["Status"] != "created":
                reason, exit_code = "container exited", 0
                break
            write_json(
                root / "guard-status.json",
                {
                    **ready,
                    "checked_at_epoch": wall(),
                    "output_bytes": size,
                    "running": current["State"]["Running"],
                    "endpoints": endpoints,
                    "required_listeners_observed": set(spec["required_listeners"]) <= observed,
                },
            )
            sleep(1)
    except BaseException as error:
        reason = repr(error)
    finally:
        receipt = {"reason": reason, "at_epoch": wall()}
        try:
            receipt["cleanup"] = cleanup_owned(docker, container_id, spec)
        except BaseException as error:
            receipt["cleanup_error"] = repr(error)
            exit_code = 1
        receipt["exit"] = exit_code
        write_json(root / "guard-exit.json", receipt, exclusive=True)
    return exit_code


def check_guard_status(path, container_id, *, now=None):
    status = json.loads(path.read_text())
    now = time.time() if now is None else now
    if (
        status["container_id"] != container_id
        or not re.fullmatch(r"[0-9a-f]{64}", container_id)
        or not 0 <= now - status["checked_at_epoch"] <= 3
        or now >= utc_timestamp(status["stop_utc"])
        or not status["running"]
        or not status["required_listeners_observed"]
        or (path.parent / "guard-exit.json").exists()
        or (path.parent / "STOP").exists()
    ):
        raise RuntimeError("guard not freshly ready for model requests")
    return status


def interrupt_guard(_number, _frame):
    raise InterruptedError("host guard signal")


def start_owned(spec, container_id, docker, *, wall=time.time, probe_pid=os.kill):
    """Explicit controller action, never called by the host guard itself."""
    validate_allocation(spec)
    root = Path(spec["output"])
    status = json.loads((root / "guard-status.json").read_text())
    ready = json.loads((root / "guard-ready.json").read_text())
    now = wall()
    if not utc_timestamp(spec["start_utc"]) <= now <= utc_timestamp(spec["no_start_after_utc"]):
        raise ValueError("outside assigned start window")
    if (
        status["container_id"] != container_id
        or status["lease_id"] != spec["lease_id"]
        or ready["container_id"] != container_id
        or ready["lease_id"] != spec["lease_id"]
        or ready["stop_utc"] != spec["stop_utc"]
        or ready["guard_pid"] != status["guard_pid"]
        or ready["guard_pid"] <= 1
        or not 0 <= now - status["checked_at_epoch"] <= 3
        or (root / "STOP").exists()
        or (root / "guard-exit.json").exists()
    ):
        raise RuntimeError("independent guard is not freshly armed for this allocation")
    probe_pid(ready["guard_pid"], 0)
    current = docker.inspect(container_id)
    inspect_owned(current, container_id, spec)
    if current["State"]["Running"] or current["State"]["Status"] != "created":
        raise RuntimeError("only one freshly created container may start")
    if wall() > utc_timestamp(spec["no_start_after_utc"]):
        raise ValueError("start cutoff reached during readiness checks")
    docker.call("start", container_id)
    if wall() > utc_timestamp(spec["no_start_after_utc"]):
        cleanup_owned(docker, container_id, spec)
        raise TimeoutError("docker start crossed cutoff; owned container stopped")
    return {"container_id": container_id, "lease_id": spec["lease_id"], "started_at_epoch": wall()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "guard", "start", "ready"))
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--container-id")
    args = parser.parse_args()
    spec = validate_allocation(json.loads(args.allocation.read_text()))
    if args.action == "plan":
        print(json.dumps(create_plan(spec)))
        return 0
    if args.action == "start":
        print(json.dumps(start_owned(spec, args.container_id, HostDocker())))
        return 0
    if args.action == "ready":
        status = check_guard_status(Path(spec["output"]) / "guard-status.json", args.container_id)
        if status["lease_id"] != spec["lease_id"]:
            raise RuntimeError("guard lease mismatch")
        print(json.dumps(status))
        return 0
    # Host-only limits, independent of the task container's resource budget.
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupt_guard)
    return supervise(spec, args.container_id, spec["output"], HostDocker())


if __name__ == "__main__":
    raise SystemExit(main())
