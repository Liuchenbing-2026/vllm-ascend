# Standard
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB

# Mirrors FileStorageConfig::local_buffer_size in Mooncake C++.
DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES = 1280 * 1024 * 1024

# Mirrors DirectIO alignment in Mooncake's AllocateBatch.
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT


# ----- Disk-offload helpers (mirror of vllm PR #42689) -----------------


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_disk_offload_staging_bytes(size_list: list[int]) -> int:
    """Estimate the DirectIO staging bytes for one mooncake key.

    Mirrors mooncake's AllocateBatch: data is rounded up to
    ``_DIRECT_IO_ALIGNMENT`` (Linux DirectIO requirement) plus a small
    padding budget to cover header/footer slack.
    """
    data_size = sum(size_list)
    return _align_up(data_size, _DIRECT_IO_ALIGNMENT) + _DIRECT_IO_PADDING_BYTES


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    return max(
        1,
        int(raw_budget_bytes * envs_ascend.VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO),
    )


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    """Split a GET into sub-batches that fit the owner's staging buffer.

    Returns ``(batches, oversize_key)``. Aborts with ``([], key)`` if any
    single key by itself exceeds ``raw_budget_bytes``; otherwise
    ``oversize_key`` is ``None``.
    """
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    batch_keys: list[str] = []
    batch_addrs: list[list[int]] = []
    batch_sizes: list[list[int]] = []
    batch_bytes = 0

    for key, addr, size in zip(keys, addrs, sizes, strict=True):
        key_bytes = _estimate_disk_offload_staging_bytes(size)
        if key_bytes > raw_budget_bytes:
            return [], key
        if key_bytes > usable_budget_bytes:
            # Key fits the raw budget but not under the usability ratio:
            # flush any in-flight batch and ship this key on its own.
            if batch_keys:
                batches.append((batch_keys, batch_addrs, batch_sizes))
                batch_keys, batch_addrs, batch_sizes = [], [], []
                batch_bytes = 0
            batches.append(([key], [addr], [size]))
            continue
        if batch_keys and batch_bytes + key_bytes > usable_budget_bytes:
            batches.append((batch_keys, batch_addrs, batch_sizes))
            batch_keys, batch_addrs, batch_sizes = [], [], []
            batch_bytes = 0
        batch_keys.append(key)
        batch_addrs.append(addr)
        batch_sizes.append(size)
        batch_bytes += key_bytes

    if batch_keys:
        batches.append((batch_keys, batch_addrs, batch_sizes))
    return batches, None


def _call_replica_predicate(replica_desc: Any, method_name: str) -> bool:
    method = getattr(replica_desc, method_name, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _classify_replica_tier(replica_descs: Any) -> str:
    if not replica_descs:
        return "unknown"
    try:
        replica_desc = replica_descs[0]
    except (IndexError, KeyError, TypeError):
        return "unknown"

    if _call_replica_predicate(replica_desc, "is_memory_replica"):
        return "memory"
    if _call_replica_predicate(replica_desc, "is_disk_replica") or _call_replica_predicate(
        replica_desc, "is_local_disk_replica"
    ):
        return "disk"
    return "unknown"


def _get_replica_tiers_by_key(store: Any, keys: list[str]) -> dict[str, str]:
    tiers_by_key = {key: "unknown" for key in keys}
    try:
        replica_descs_by_key = store.batch_get_replica_desc(keys)
    except Exception as e:
        logger.warning(
            "Failed to get Mooncake replica descriptors for tier logging "
            "(batch_keys=%d, error=%s); marking tiers unknown",
            len(keys),
            e,
        )
        return tiers_by_key

    for key in keys:
        if hasattr(replica_descs_by_key, "get"):
            replica_descs = replica_descs_by_key.get(key)
        else:
            try:
                replica_descs = replica_descs_by_key[key]
            except (KeyError, TypeError):
                replica_descs = None
        tiers_by_key[key] = _classify_replica_tier(replica_descs)
    return tiers_by_key


def _log_mooncake_load_tier_summary(
    batch_keys: list[str],
    load_results: list[int],
    tiers_by_key: dict[str, str],
) -> None:
    tier_counts = {"memory": 0, "disk": 0, "unknown": 0}
    bytes_by_tier = {"memory": 0, "disk": 0, "unknown": 0}
    success_keys = 0
    failed_keys = 0

    for index, key in enumerate(batch_keys):
        tier = tiers_by_key.get(key, "unknown")
        if tier not in tier_counts:
            tier = "unknown"
        tier_counts[tier] += 1

        value = load_results[index] if index < len(load_results) else -1
        if value >= 0:
            success_keys += 1
            bytes_by_tier[tier] += int(value)
        else:
            failed_keys += 1

    logger.info(
        "Mooncake load tier summary: batch_keys=%d "
        "memory_keys=%d disk_keys=%d unknown_keys=%d "
        "success_keys=%d failed_keys=%d bytes_by_tier=%s",
        len(batch_keys),
        tier_counts["memory"],
        tier_counts["disk"],
        tier_counts["unknown"],
        success_keys,
        failed_keys,
        bytes_by_tier,
    )


class MooncakeBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False):
        self.config = MooncakeStoreConfig.load_from_env()
        if self.config.protocol != "ascend":
            raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")

        self.store: Any | None = None
        self.local_seg: str | None = None
        self._use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        self._lazy_init = lazy_init and self._use_fabric_mem
        self._store_initialized = False
        self._store_init_lock = threading.Lock()

        # Disk-offload wiring (vllm PR #42689).
        #
        # `MOONCAKE_PREFERRED_SEGMENT` is the upstream-PR "host:port" pin
        # that takes precedence over the upstream-#7820 bool flag
        # (`MooncakeStoreConfig.preferred_segment`) when both are set.
        # The bool flag still drives the per-put `ReplicateConfig` build
        # done by `_build_replicate_config()`; this override only swaps
        # in a different segment string at put time.
        self._preferred_segment_override = envs_ascend.MOONCAKE_PREFERRED_SEGMENT
        if self._preferred_segment_override is not None and not self._preferred_segment_override.strip():
            self._preferred_segment_override = None

        self.disk_offload_buffer_budget_bytes: int | None = (
            DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES if self.config.enable_offload else None
        )
        self.usable_disk_offload_buffer_budget_bytes: int | None = (
            None
            if self.disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(self.disk_offload_buffer_budget_bytes)
        )

        # Mode/offload consistency warnings — fail-soft, not fail-fast,
        # because operators may intentionally run mixed configurations.
        logger.info(
            "Mooncake mode=%s enable_offload=%s preferred_segment_override=%s",
            self.config.mode,
            self.config.enable_offload,
            self._preferred_segment_override or "<none>",
        )
        if self.config.mode == "embedded":
            if self.config.enable_offload and self._preferred_segment_override is None:
                logger.warning(
                    "enable_offload is set in embedded mode without "
                    "MOONCAKE_PREFERRED_SEGMENT; SSD tier will only see "
                    "puts that happen to land on the owner segment."
                )
            if self._preferred_segment_override is not None:
                logger.warning(
                    "MOONCAKE_PREFERRED_SEGMENT=%s with mode=embedded: rank-contributed segments will be idle.",
                    self._preferred_segment_override,
                )
        elif self.config.mode == "standalone-store" and not self.config.enable_offload:
            logger.warning(
                "standalone-store mode without enable_offload: large prefills may exceed the owner DirectIO budget."
            )

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    def _ensure_initialized(self):
        if self._store_initialized:
            return

        with self._store_init_lock:
            if self._store_initialized:
                return

            logger.info("Initializing Mooncake store on first put.")
            self.store = self._setup_store()
            self._store_initialized = True

    def _setup_store(self):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        store = MooncakeDistributedStore()
        local_hostname = get_ip()
        # ASCEND_ENABLE_USE_FABRIC_MEM: Enable unified memory address direct transmission scheme
        # and only can be used for 800 I/T A3 series.
        # Required supporting hardware versions are as follows:
        if not self._use_fabric_mem:
            transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
            self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size,
                local_buffer_size=self.config.local_buffer_size,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                engine=transfer_engine.get_engine(),
            )
        else:
            self.local_seg = local_hostname
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size,
                local_buffer_size=0,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
            )

        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(msg)
            raise RuntimeError(msg)
        return store

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        if not self._use_fabric_mem:
            local_hostname = get_ip()
            global_te.get_transfer_engine(local_hostname, device_name=None)
            global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MooncakeBackend.exists called before store initialization; treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        try:
            self._ensure_initialized()
            assert self.store is not None
            config = ReplicateConfig()
            # Upstream #7820: bool flag tells whether to pin to this rank's
            # local_seg as the preferred segment.
            if self.config.preferred_segment:
                config.preferred_segment = self.local_seg
            # vllm PR #42689: explicit owner pin via env var ("host:port"),
            # takes precedence over the bool flag because it's typically
            # used in standalone-store mode where local_seg is not the
            # owner.
            if self._preferred_segment_override is not None:
                config.preferred_segment = self._preferred_segment_override
            config.prefer_alloc_in_same_node = self.config.prefer_alloc_in_same_node
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            for value in res:
                if value < 0:
                    logger.error("Failed to put key %s,res:%s", keys, res)
                    if self._lazy_init:
                        logger.error("If this is the first DSV4(compress) request, this failure is expected.")
        except Exception as e:
            logger.error("Failed to put key %s,error:%s", keys, e)
            if self._lazy_init:
                logger.error("If this is the first DSV4(compress) request, this failure is expected.")

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        if self._lazy_init and not self._store_initialized:
            logger.error("MooncakeBackend.get called before store initialization, keys=%s", keys)
            return
        assert self.store is not None
        logger.debug(
            "MooncakeBackend.get enter keys=%d sample_keys=%s",
            len(keys),
            keys[:3],
        )

        # Compute sub-batches when disk-offload is enabled. With offload OFF
        # the budget is None and we issue exactly one batch, matching the
        # pre-#42689 behavior byte-for-byte.
        load_batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = [(keys, addrs, sizes)]
        if self.usable_disk_offload_buffer_budget_bytes is not None:
            total_staging_bytes = sum(_estimate_disk_offload_staging_bytes(size) for size in sizes)
            if total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes:
                assert self.disk_offload_buffer_budget_bytes is not None
                load_batches, oversized_key = _split_disk_offload_load_batches(
                    keys,
                    addrs,
                    sizes,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    oversized_key_index = keys.index(oversized_key)
                    oversized_key_bytes = _estimate_disk_offload_staging_bytes(sizes[oversized_key_index])
                    logger.error(
                        "Skipping Mooncake load batch because key %s requires %d staging bytes, exceeding budget %d",
                        oversized_key,
                        oversized_key_bytes,
                        self.disk_offload_buffer_budget_bytes,
                    )
                    return

        try:
            for batch_keys, batch_addrs, batch_sizes in load_batches:
                tiers_by_key: dict[str, str] | None = None
                if envs_ascend.VLLM_MOONCAKE_STORE_TIER_LOG:
                    tiers_by_key = _get_replica_tiers_by_key(self.store, batch_keys)
                res = self.store.batch_get_into_multi_buffers(batch_keys, batch_addrs, batch_sizes)
                res_list = list(res)
                if tiers_by_key is not None:
                    _log_mooncake_load_tier_summary(batch_keys, res_list, tiers_by_key)
                logger.debug(
                    "MooncakeBackend.get sub-batch keys=%d result_sample=%s negative_count=%d",
                    len(batch_keys),
                    res_list[:12],
                    sum(1 for value in res_list if value < 0),
                )
                failed = [(k, v) for k, v in zip(batch_keys, res_list, strict=True) if v < 0]
                if failed:
                    logger.error(
                        "Failed to get %d Mooncake keys from sub-batch (batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        failed[:3],
                    )
                    # Mirror upstream: on first sub-batch failure abort the
                    # rest so the caller can retry from the top.
                    break
        except Exception as e:
            logger.error("Failed to get key %s, error:%s", keys, e)


@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    preferred_segment: bool
    prefer_alloc_in_same_node: bool
    # Disk-offload extensions (mirror of vllm PR #42689). Appended at the
    # end with defaults so positional-arg call sites stay valid.
    mode: str = "embedded"
    enable_offload: bool = False

    def __post_init__(self) -> None:
        # Soft validation: only catch obvious mode/segment inconsistencies.
        # Skips fields the legacy ascend `from_file` may leave as None to
        # avoid breaking historical configs that didn't set these.
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"unknown Mooncake mode: {self.mode!r}")
        if self.mode == "standalone-store" and self.global_segment_size not in (0, "0"):
            raise ValueError("standalone-store mode requires global_segment_size == 0")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        master_server_address = os.getenv("MOONCAKE_MASTER", None)
        global_segment_size_env = os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=_parse_global_segment_size(
                global_segment_size_env
                if global_segment_size_env is not None
                else config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            master_server_address=master_server_address
            if master_server_address is not None
            else config.get("master_server_address"),
            preferred_segment=config.get("preferred_segment", False),
            prefer_alloc_in_same_node=config.get("prefer_alloc_in_same_node", True),
            mode=config.get("mode", "embedded"),
            enable_offload=bool(config.get("enable_offload", False)),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)


def _parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: GB, MB, KB, B

    Args:
        value: Input value (int, str, or other convertible types)

    Returns:
        int: Size in bytes

    Raises:
        ValueError: For invalid format, missing number, or negative values
        TypeError: For unsupported input types
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    UNIT_MULTIPLIERS = {
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"

    multiplier = UNIT_MULTIPLIERS[unit]
    return _convert_to_bytes(number_str, multiplier, value)


def _convert_to_bytes(number_str: str, multiplier: int, original_input: str) -> int:
    """
    Convert numeric string to byte count

    Args:
        number_str: Numeric portion of input
        multiplier: Unit conversion factor
        original_input: Original input string (for error messages)

    Returns:
        int: Byte count

    Raises:
        ValueError: For invalid numbers or negative results
    """
    try:
        numeric_value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{original_input}'")
    # Calculate byte count
    try:
        byte_count = int(numeric_value * multiplier)
    except OverflowError:
        raise ValueError(f"Storage size too large: '{original_input}'")
    return byte_count
