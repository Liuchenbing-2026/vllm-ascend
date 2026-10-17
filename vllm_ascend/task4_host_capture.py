"""Diagnostic-only host snapshots. Never imports torch or initializes a device.

Install only in an explicitly scheduled reproduction, not a production patch.
"""
from __future__ import annotations

import datetime
import json
import os
import resource
import sys
from pathlib import Path


class HostCapture:
    def __init__(self, directory, stats_reader=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stats_reader = stats_reader

    @staticmethod
    def read(path):
        try:
            with open(path) as handle:
                return handle.read(65536)
        except OSError as error:
            return {'unavailable': str(error)}

    def host_stats(self):
        if self.stats_reader is not None:
            return self.stats_reader()
        module = sys.modules.get('torch_npu')
        if module is None:
            return {'unavailable': 'torch_npu not already loaded; no import attempted'}
        try:
            return module.npu.host_memory_stats()
        except Exception as error:
            return {'unavailable': repr(error)}

    def cgroups(self):
        # cgroup v2 plus common v1 mounts. Keep source paths in every record.
        records = {}
        raw = self.read('/proc/self/cgroup')
        if not isinstance(raw, str):
            return raw
        for line in raw.splitlines():
            _, controllers, relative = line.split(':', 2)
            if '..' in Path(relative).parts:
                continue
            if controllers == '':
                root = Path('/sys/fs/cgroup')
                names = ('memory.current', 'memory.max', 'memory.high', 'memory.events', 'memory.stat', 'cpuset.mems.effective')
            elif 'memory' in controllers.split(','):
                root = Path('/sys/fs/cgroup/memory')
                names = ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt', 'memory.stat')
            else:
                continue
            path = root / relative.lstrip('/')
            while True:
                for name in names:
                    records[str(path/name)] = self.read(path/name)
                if path == root:
                    break
                path = path.parent
        return records

    @staticmethod
    def tensor_geometry(tensor, cpu_blocks):
        storage = tensor.untyped_storage()
        element_size = tensor.element_size()
        elements_per_block = 1
        for size in tensor.shape[1:]:
            elements_per_block *= size
        requested = cpu_blocks * elements_per_block * element_size
        return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                    dtype=str(tensor.dtype), device=str(tensor.device),
                    storage_ptr=storage.data_ptr(), storage_bytes=storage.nbytes(),
                    offset_bytes=tensor.storage_offset()*element_size,
                    cpu_blocks=cpu_blocks, requested_bytes=requested,
                    rounded_bytes=(1 << (requested-1).bit_length()) if requested else 0)

    def capture(self, phase, **details):
        # Read the already initialized process group; never initialize one.
        dist = sys.modules.get('torch.distributed')
        rank = None
        if dist is not None:
            try:
                if dist.is_initialized():
                    rank = dist.get_rank()
            except Exception:
                pass
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status = self.read('/proc/self/status')
        snapshot = dict(phase=phase, time_utc=now, pid=os.getpid(), ppid=os.getppid(), rank=rank,
                        rank_environment={k:os.environ[k] for k in ('RANK','LOCAL_RANK','WORLD_SIZE') if k in os.environ},
                        rlimit_memlock=resource.getrlimit(resource.RLIMIT_MEMLOCK),
                        status=status, limits=self.read('/proc/self/limits'),
                        meminfo=self.read('/proc/meminfo'), buddyinfo=self.read('/proc/buddyinfo'),
                        numa={str(p):self.read(p) for p in Path('/sys/devices/system/node').glob('node*/meminfo')},
                        numa_maps=self.read('/proc/self/numa_maps'),
                        memory_pressure=self.read('/proc/pressure/memory'),
                        cgroup=self.cgroups(), host_allocator=self.host_stats(), details=details)
        # A single record per process prevents rank interleaving. Flush before allocation.
        path = self.directory / ('worker-%s.jsonl' % os.getpid())
        with path.open('a') as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False, default=str)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def allocate_mirrors(self, caches, cpu_blocks, pin_memory, zeros, pre_allocate=None):
        # Preserve original order, shapes, dtype, device and allocator. No retry,
        # empty-cache, synchronization, fallback, block reduction or NPU calls.
        geometry = {name:self.tensor_geometry(t,cpu_blocks) for name,t in caches.items()}
        self.capture('before_mirrors', regions=geometry, pin_memory=pin_memory)
        result = {}
        for ordinal, (name, tensor) in enumerate(caches.items()):
            self.capture('before_pin', ordinal=ordinal, name=name, geometry=geometry[name])
            try:
                if pre_allocate is not None:
                    pre_allocate()
                result[name] = zeros((cpu_blocks,)+tuple(tensor.shape[1:]),
                                     dtype=tensor.dtype, device='cpu', pin_memory=pin_memory)
            except BaseException as error:
                try:
                    self.capture('pin_failed', ordinal=ordinal, name=name,
                                 error=repr(error), successful_regions=len(result))
                except Exception as capture_error:
                    print('task4 failure snapshot unavailable: '+repr(capture_error), file=sys.stderr, flush=True)
                raise
            self.capture('after_pin', ordinal=ordinal, name=name, successful_regions=len(result))
        self.capture('mirrors_complete', successful_regions=len(result))
        return result


def validate_minimal_geometry(caches, blocks, rank, world_size, capacity_bytes):
    if blocks != 1 or world_size != 8 or rank not in range(8):
        raise ValueError('Probe requires exactly one block, TP8, valid rank')
    if len(caches) != 46 or not 0 < capacity_bytes <= 4*1024**2:
        raise ValueError('Probe requires 46 regions and <=4MiB rank capacity')
    geometry = {name:HostCapture.tensor_geometry(t,1) for name,t in caches.items()}
    if any(len(t.shape)!=2 or t.element_size()!=1 or t.stride()[1]!=1 or t.stride()[0]!=t.shape[1] for t in caches.values()):
        raise ValueError('Unexpected region geometry; no probe allocation permitted')
    requested=sum(g['requested_bytes'] for g in geometry.values())
    rounded=sum(g['rounded_bytes'] for g in geometry.values())
    if requested > 4*1024**2 or rounded > 4*1024**2:
        raise ValueError('Rank probe memory ceiling exceeded')
    return requested,rounded


def load_probe_lease(path="/ws/task4_minpin/active_lease.json", now=None):
    lease = json.loads(Path(path).read_text())
    stamp = datetime.datetime.now(datetime.timezone.utc).timestamp() if now is None else now
    if lease.get("status") != "APPROVED" or not lease.get("lease_id"):
        raise ValueError("No explicitly approved new reservation")
    if lease["lease_id"] == "lease_aa1a6ca9-0c57-4c7c-ace4-1b7d6d027344":
        raise ValueError("Previous reservation was revoked")
    start, end = lease["start_epoch"], lease["end_epoch"]
    if end-start < 20*60 or not start <= stamp < end-120:
        raise ValueError("Reservation inactive, too short, or in cleanup interval")
    if lease.get("physical_devices") != list(range(8)):
        raise ValueError("Requires explicit eight-card reservation")
    if lease.get("rank_limit_bytes") != 4*1024**2 or lease.get("total_limit_bytes") != 32*1024**2:
        raise ValueError("Cannot alter diagnostic capacity ceilings")
    output = Path(lease["output_dir"])
    if output.parent != Path("/ws/task4_minpin/attempts") or output.name in ("", ".", ".."):
        raise ValueError("Require a unique attempt under /ws/task4_minpin/attempts")
    return lease


def before_model_guard(config, lease_path="/ws/task4_minpin/active_lease.json"):
    if config.parallel_config.world_size != 8 or config.cache_config.kv_offloading_size != 0.03125:
        raise ValueError('DIAGNOSTIC GUARD: only 32MiB aggregate offload budget allowed')
    return load_probe_lease(lease_path)


def capture_model_phase(phase, **details):
    lease = load_probe_lease()
    HostCapture(lease["output_dir"]).capture(phase, lease_id=lease["lease_id"], **details)


def run_minimal_probe(caches, blocks, pin_memory, zeros, capacity_bytes, rank, world_size):
    import time
    lease=load_probe_lease()
    capture=HostCapture(lease["output_dir"])
    try:
        if not pin_memory:
            raise ValueError('Probe requires pinned allocation; no pageable fallback')
        requested,rounded=validate_minimal_geometry(caches,blocks,rank,world_size,capacity_bytes)
        reserve_quota(capture.directory,rank,requested,rounded)
        capture.capture('probe_guard_passed',requested=requested,rounded=rounded)
        mirrors=capture.allocate_mirrors(caches,1,True,zeros,pre_allocate=load_probe_lease)
        capture.capture('probe_complete',successful_regions=len(mirrors))
        (capture.directory/('rank-%s.complete'%rank)).write_text(str(os.getpid()))
        # Intentionally never return to connector stream/backend initialization.
        # Controller collects all 8 markers then stops ONLY this container.
        while datetime.datetime.now(datetime.timezone.utc).timestamp() < lease["end_epoch"]-120:
            time.sleep(1)
        raise RuntimeError('Probe lease deadline reached; no serving permitted')
    except BaseException as error:
        capture.capture('probe_failed',error=repr(error))
        (capture.directory/'FAILED').write_text(repr(error))
        raise


def reserve_quota(directory, rank, requested, rounded):
    import fcntl
    if type(rank) is not int or rank not in range(8):
        raise ValueError("Invalid rank")
    if not 0 < requested <= rounded <= 4*1024**2:
        raise ValueError("Invalid per-rank quota")
    quota=Path(directory)/'quota.json' 
    with (Path(directory)/'quota.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(quota.read_text()) if quota.exists() else {}
        if str(rank) in state:
            raise ValueError('Duplicate rank reservation; retries forbidden')
        if sum(r['requested'] for r in state.values())+requested > 32*1024**2 or sum(r['rounded'] for r in state.values())+rounded > 32*1024**2:
            raise ValueError('Global 32MiB ceiling exceeded')
        state[str(rank)]=dict(requested=requested,rounded=rounded,pid=os.getpid())
        quota.write_text(json.dumps(state))
