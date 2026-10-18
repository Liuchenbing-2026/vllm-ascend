# SPDX-License-Identifier: Apache-2.0
"""Task-only stage A diagnostic. This branch cannot serve inference requests.

Only rank zero allocates four bounded host buffers after all real models load.
No KV pool, transfer, retry, cache flush or device initialization lives here.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageALimits:
    world_size: int = 8
    allocator_rank: int = 0
    steps_bytes: tuple[int, ...] = (32 << 20, 128 << 20, 256 << 20, 1024 << 20)
    total_bytes: int = 1440 << 20
    host_reserve_bytes: int = 64 << 30
    window_seconds: int = 25 * 60
    cleanup_and_audit_seconds: int = 7 * 60


def validate_lease(lease, now):
    limits = StageALimits()
    expected = {
        "status": "APPROVED",
        "scope": "task4-rc2-stage-a-full-model-pin",
        "physical_devices": list(range(limits.world_size)),
        "allocator_ranks": [limits.allocator_rank],
        "steps_bytes": list(limits.steps_bytes),
        "rank_limit_bytes": limits.total_bytes,
        "total_limit_bytes": limits.total_bytes,
        "vllm_sha": "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "ascend_base_sha": "8fde6d4e335fe300e2b949aab0e31659cce0811d",
    }
    if any(lease.get(key) != value for key, value in expected.items()):
        raise ValueError("Lease does not authorize this fixed stage A scope")
    if not isinstance(lease.get("lease_id"), str) or not lease["lease_id"].startswith("lease_"):
        raise ValueError("Missing supervisor lease identity")
    for key in ("start_epoch", "end_epoch"):
        if type(lease.get(key)) is not int:
            raise ValueError("Lease times must be integer epochs")
    if lease["end_epoch"] - lease["start_epoch"] != limits.window_seconds:
        raise ValueError("Requires exactly the reviewed 25 minute window")
    if not lease["start_epoch"] <= now < lease["end_epoch"] - limits.cleanup_and_audit_seconds:
        raise ValueError("Inactive lease or cleanup interval reached")
    sha = lease.get("ascend_sha", "")
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("Require exact reviewed diagnostic SHA")
    return limits


def validate_config(config, dummy):
    parallel = config.parallel_config
    if dummy or config.load_config.load_format == "dummy":
        raise ValueError("Real complete model required")
    if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size) != (8, 1, 1):
        raise ValueError("Requires TP8/PP1/DP1")
    if config.cache_config.kv_offloading_size not in (None, 0):
        raise ValueError("Original CPU offload pool must be disabled")
    if config.kv_transfer_config is not None:
        raise ValueError("KV connectors are forbidden in this diagnostic")
    if not config.use_v2_model_runner:
        raise ValueError("Diagnostic hook requires the V2 runner")


def exclusive_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def host_snapshot():
    # Only consume an already imported backend; never initialize torch/NPU here.
    def read(path):
        return Path(path).read_text()

    backend = sys.modules.get("torch_npu")
    if backend is None:
        raise RuntimeError("Missing initialized backend for allocator capture")
    nodes = {str(path): read(path) for path in Path("/sys/devices/system/node").glob("node*/meminfo")}
    if not nodes:
        raise RuntimeError("NUMA capture unavailable")
    cgroups = {}
    for line in read("/proc/self/cgroup").splitlines():
        _, controllers, relative = line.split(":", 2)
        if ".." in Path(relative).parts:
            raise ValueError("Unexpected cgroup path")
        if controllers == "":
            root = Path("/sys/fs/cgroup")
            names = ("memory.current", "memory.max", "memory.high", "memory.events", "cpuset.mems.effective")
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            names = ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt")
        else:
            continue
        location = root / relative.lstrip("/")
        while True:
            for name in names:
                path = location / name
                if path.exists():
                    cgroups[str(path)] = read(path)
            if location == root:
                break
            location = location.parent
    return {
        "pid": os.getpid(),
        "time_epoch": time.time(),
        "meminfo": read("/proc/meminfo"),
        "numa": nodes,
        "numa_maps": read("/proc/self/numa_maps"),
        "buddyinfo": read("/proc/buddyinfo"),
        "status": read("/proc/self/status"),
        "limits": read("/proc/self/limits"),
        "rlimit_memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        "cgroups": cgroups,
        # Preserve negative or inconsistent raw counters, never infer free/cache.
        "allocator": backend.npu.host_memory_stats(),
    }


def check_headroom(snapshot, remaining):
    meminfo = dict(line.split(":", 1) for line in snapshot["meminfo"].splitlines())
    available = int(meminfo["MemAvailable"].split()[0]) * 1024
    if available < StageALimits().host_reserve_bytes + 2 * remaining:
        raise RuntimeError("Host headroom below conservative stage A gate")
    groups = snapshot["cgroups"]
    for key, value in groups.items():
        name = Path(key).name
        if name not in ("memory.max", "memory.limit_in_bytes") or value.strip() == "max":
            continue
        usage_name = "memory.current" if name == "memory.max" else "memory.usage_in_bytes"
        usage = int(groups[str(Path(key).parent / usage_name)])
        if int(value) - usage < 2 * remaining:
            raise RuntimeError("Cgroup headroom below stage A gate")


class StageAProbe:
    def __init__(self, lease_path, output, rank, world_size, snapshot=host_snapshot, clock=time.time, sleep=time.sleep):
        self.lease_path = Path(lease_path)
        self.lease_bytes = self.lease_path.read_bytes()
        self.lease = json.loads(self.lease_bytes)
        self.limits = validate_lease(self.lease, clock())
        if type(rank) is not int or rank not in range(self.limits.world_size) or world_size != self.limits.world_size:
            raise ValueError("Invalid diagnostic rank/world")
        self.output, self.rank = Path(output), rank
        self.snapshot, self.clock, self.sleep = snapshot, clock, sleep
        self.claimed, self.retained = [], []
        # The controller creates a fresh directory. Duplicate worker/restart fails.
        exclusive_json(
            self.output / f"rank-{rank}.identity.json", {"pid": os.getpid(), "lease": self.lease["lease_id"]}
        )

    @classmethod
    def before_model(cls, config, dummy):
        validate_config(config, dummy)
        # Already initialized by the worker. No init_process_group/device call.
        import torch.distributed as distributed

        probe = cls("/task4/lease.json", "/task4/output", distributed.get_rank(), distributed.get_world_size())
        probe.capture("before_model_load")
        return probe

    def check(self):
        if self.lease_path.read_bytes() != self.lease_bytes:
            raise RuntimeError("Lease changed during attempt")
        validate_lease(self.lease, self.clock())
        if (self.output / "FAILED.json").exists():
            raise RuntimeError("Peer or controller failed; no further allocations")

    def capture(self, phase, **details):
        sample = self.snapshot()
        with (self.output / f"rank-{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps({"phase": phase, "rank": self.rank, "details": details, **sample}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return sample

    def mark(self, name):
        exclusive_json(self.output / f"{name}.rank-{self.rank}.json", {"pid": os.getpid(), "time": self.clock()})

    def wait_all(self, name):
        while True:
            self.check()
            if all((self.output / f"{name}.rank-{rank}.json").exists() for rank in range(self.limits.world_size)):
                return
            self.sleep(0.05)

    def allocate_step(self, ordinal, size, allocate):
        self.check()
        if self.rank != self.limits.allocator_rank:
            raise ValueError("Only rank zero may allocate a probe")
        if (
            type(ordinal) is not int
            or ordinal != len(self.claimed)
            or ordinal not in range(len(self.limits.steps_bytes))
        ):
            raise ValueError("Repeated, skipped or unknown allocation step")
        if type(size) is not int or size != self.limits.steps_bytes[ordinal]:
            raise ValueError("Allocation size is not the reviewed fixed step")
        if sum(self.claimed) + size > self.limits.total_bytes:
            raise ValueError("Probe total exceeds 1.40625GiB")
        # Persist a claim before calling allocator. A failure cannot be retried.
        exclusive_json(self.output / f"allocation-{ordinal}.json", {"rank": self.rank, "requested_bytes": size})
        self.claimed.append(size)
        try:
            value = allocate(size)
        except BaseException as error:
            self.fail(error)
            raise
        self.retained.append(value)

    def run_steps(self, allocate):
        self.capture("model_loaded")
        self.mark("model_loaded")
        self.wait_all("model_loaded")
        for ordinal, size in enumerate(self.limits.steps_bytes):
            self.check()
            sample = self.capture("before_pin", ordinal=ordinal, requested_bytes=size, allocator_rank=0)
            check_headroom(sample, self.limits.total_bytes - sum(self.limits.steps_bytes[:ordinal]))
            self.mark(f"before-{ordinal}")
            self.wait_all(f"before-{ordinal}")
            if self.rank == self.limits.allocator_rank:
                self.allocate_step(ordinal, size, allocate)
                exclusive_json(self.output / f"allocated-{ordinal}.json", {"requested_bytes": size})
            while not (self.output / f"allocated-{ordinal}.json").exists():
                self.check()
                self.sleep(0.05)
            self.check()
            self.capture("after_pin", ordinal=ordinal, requested_bytes=size, allocator_rank=0)
            self.mark(f"after-{ordinal}")
            self.wait_all(f"after-{ordinal}")
        self.capture("probe_complete", retained_probe_bytes=sum(self.claimed))
        self.mark("probe_complete")

    def fail(self, error):
        try:
            self.capture("exception", error=repr(error), claimed_bytes=sum(self.claimed))
        finally:
            with suppress(FileExistsError):  # Preserve the first failure.
                exclusive_json(self.output / "FAILED.json", {"rank": self.rank, "error": repr(error)})

    def after_model(self):
        # Imported only inside the approved, already initialized real worker.
        import torch

        def allocate(size):
            value = torch.zeros((size,), dtype=torch.int8, device="cpu", pin_memory=True)
            if value.numel() * value.element_size() != size or not value.is_pinned():
                raise RuntimeError("Allocator did not return the requested pinned byte buffer")
            return value

        self.run_steps(allocate)
        # Never return to profile/KV allocation/connector/serving. Retain buffers
        # until the host controller stops this attempt's container by immutable ID.
        while True:
            self.check()
            self.sleep(0.1)
