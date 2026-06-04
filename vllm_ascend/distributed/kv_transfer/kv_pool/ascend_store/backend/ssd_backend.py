#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import ctypes
import hashlib
import os
import tempfile
import threading
from typing import Any, BinaryIO

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend


class SSDBackend(Backend):
    """Local filesystem backend for Ascend KV Pool.

    AscendStore passes NPU device pointers for each KV segment. This backend
    stages bytes through CANN ACL host memory and persists one blob per KV Pool
    key under ``ssd_root_dir``.
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        extra_config: dict[str, Any] | None = None,
    ):
        extra_config = extra_config or {}
        self.root_dir = str(
            extra_config.get(
                "ssd_root_dir",
                extra_config.get(
                    "root_dir",
                    os.getenv(
                        "ASCEND_KV_POOL_SSD_ROOT",
                        os.path.join(tempfile.gettempdir(), "vllm_ascend_kv_pool_ssd"),
                    ),
                ),
            )
        )
        self._local_rank = getattr(parallel_config, "rank", 0)
        self._acl: Any | None = None
        os.makedirs(self.root_dir, exist_ok=True)

    def _get_acl(self):
        if self._acl is None:
            try:
                import acl  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "The Ascend SSD KV Pool backend requires the CANN ACL "
                    "Python runtime package."
                ) from exc
            self._acl = acl
        return self._acl

    @staticmethod
    def _ret_code(ret: Any) -> int:
        if isinstance(ret, tuple):
            return int(ret[-1])
        return int(ret)

    def _malloc_host(self, size: int) -> int:
        result = self._get_acl().rt.malloc_host(size)
        if isinstance(result, tuple):
            ptr, ret = result
        else:
            ptr, ret = result, 0
        ret_code = self._ret_code(ret)
        if ret_code != 0:
            raise RuntimeError(f"acl.rt.malloc_host({size}) failed: {ret_code}")
        return int(ptr)

    def _free_host(self, ptr: int) -> None:
        ret_code = self._ret_code(self._get_acl().rt.free_host(ptr))
        if ret_code != 0:
            logger.warning("acl.rt.free_host(%s) failed: %s", ptr, ret_code)

    def _memcpy(self, dst: int, dst_max: int, src: int, size: int, kind: int) -> None:
        ret_code = self._ret_code(
            self._get_acl().rt.memcpy(dst, dst_max, src, size, kind)
        )
        if ret_code != 0:
            raise RuntimeError(f"acl.rt.memcpy failed: {ret_code}")

    def _path_for_key(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root_dir, digest[:3], digest[3:5], f"{digest}.bin")

    @staticmethod
    def _total_size(sizes: list[int]) -> int:
        return sum(int(size) for size in sizes)

    def set_device(self):
        try:
            self._local_rank = get_world_group().local_rank
        except Exception:
            pass
        torch.npu.set_device(torch.device(f"npu:{self._local_rank}"))

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        # ACL memcpy consumes raw device pointers directly.
        return

    def exists(self, keys: list[str]) -> list[int]:
        return [1 if os.path.exists(self._path_for_key(key)) else 0 for key in keys]

    def _copy_device_segments_to_file(
        self, file_obj: BinaryIO, addrs: list[int], sizes: list[int]
    ) -> int:
        acl = self._get_acl()
        d2h_kind = getattr(acl, "ACL_MEMCPY_DEVICE_TO_HOST", 2)
        copied = 0
        for addr, size in zip(addrs, sizes):
            size = int(size)
            if size <= 0:
                continue
            host_ptr = self._malloc_host(size)
            try:
                self._memcpy(host_ptr, size, int(addr), size, d2h_kind)
                file_obj.write(ctypes.string_at(host_ptr, size))
            finally:
                self._free_host(host_ptr)
            copied += size
        return copied

    def _copy_file_to_device_segments(
        self, file_obj: BinaryIO, addrs: list[int], sizes: list[int]
    ) -> int:
        acl = self._get_acl()
        h2d_kind = getattr(acl, "ACL_MEMCPY_HOST_TO_DEVICE", 1)
        copied = 0
        for addr, size in zip(addrs, sizes):
            size = int(size)
            if size <= 0:
                continue
            chunk = file_obj.read(size)
            if len(chunk) != size:
                raise OSError(
                    f"Short SSD KV blob: expected {size} bytes, got {len(chunk)}"
                )
            host_ptr = self._malloc_host(size)
            try:
                ctypes.memmove(host_ptr, chunk, size)
                self._memcpy(int(addr), size, host_ptr, size, h2d_kind)
            finally:
                self._free_host(host_ptr)
            copied += size
        if file_obj.read(1):
            raise OSError("SSD KV blob has trailing data")
        return copied

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        ret: list[int] = []
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes):
            path = self._path_for_key(key)
            if os.path.exists(path):
                ret.append(0)
                continue
            tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                expected = self._total_size(key_sizes)
                with open(tmp_path, "wb") as f:
                    copied = self._copy_device_segments_to_file(
                        f, key_addrs, key_sizes
                    )
                if copied != expected:
                    raise OSError(
                        f"Short device read: expected {expected} bytes, got {copied}"
                    )
                os.replace(tmp_path, path)
                ret.append(0)
            except Exception as exc:
                logger.error("Failed to put SSD KV key %s: %s", key, exc)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                ret.append(1)
        return ret

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        ret: list[int] = []
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes):
            path = self._path_for_key(key)
            try:
                expected = self._total_size(key_sizes)
                with open(path, "rb") as f:
                    copied = self._copy_file_to_device_segments(
                        f, key_addrs, key_sizes
                    )
                if copied != expected:
                    raise OSError(
                        f"Invalid SSD KV blob size: expected {expected}, got {copied}"
                    )
                ret.append(0)
            except Exception as exc:
                logger.error("Failed to get SSD KV key %s: %s", key, exc)
                ret.append(1)
        return ret
