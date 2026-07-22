from collections import defaultdict, deque
from collections.abc import Iterator
from functools import partial

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import (
    CpuGpuOffloadingHandlers,
    SingleDirectionOffloadingHandler,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


def _swap_blocks_batch(src, dst, sizes, *, direction, **_kwargs):
    torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, direction)


class _NPUSingleDirectionOffloadingHandler(SingleDirectionOffloadingHandler):
    def __init__(
        self,
        gpu_tensors,
        cpu_tensors,
        block_size_factor,
        kv_cache_groups_data_refs,
        gpu_to_cpu,
        mmap_region=None,
    ):
        self.src_tensors = gpu_tensors if gpu_to_cpu else cpu_tensors
        self.dst_tensors = cpu_tensors if gpu_to_cpu else gpu_tensors
        self.gpu_to_cpu = gpu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self._swap_blocks_batch = partial(
            _swap_blocks_batch, direction=1 if gpu_to_cpu else 0
        )
        self.src_block_size_factor = 1 if gpu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if gpu_to_cpu else 1
        self.transfer_type = ("NPU", "CPU") if gpu_to_cpu else ("CPU", "NPU")
        self._mmap_region = mmap_region
        self._transfer_events = {}
        self._transfers = deque()
        self._stream_pool = []
        self._event_pool = []
        self._buffer_pool = []


class NPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.num_cpu_blocks = int(self.extra_config.get("num_cpu_blocks", 0))
        if self.num_cpu_blocks <= 0:
            raise ValueError("num_cpu_blocks must be positive")

        self._manager: OffloadingManager | None = None
        self._handlers: CpuGpuOffloadingHandlers | None = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_cpu_blocks,
                enable_events=enable_events,
            )
        return self._manager

    def canonicalize_kv_caches(self, kv_caches) -> CanonicalKVCaches:
        if not all(isinstance(cache, tuple) for cache in kv_caches.values()):
            raise NotImplementedError(
                "NPUOffloadingSpec supports split Attention KV caches"
            )

        num_blocks = self.kv_cache_config.num_blocks
        block_tensors = []
        refs_by_layer = defaultdict(list)
        for cache_tensor in self.kv_cache_config.kv_cache_tensors:
            layer_names = [
                name for name in cache_tensor.shared_by if name in kv_caches
            ]
            if not layer_names:
                continue

            first_cache = kv_caches[layer_names[0]]
            for component_idx, component in enumerate(first_cache):
                tensor = component.view(torch.int8).view(num_blocks, -1)
                tensor_idx = len(block_tensors)
                page_size = tensor.shape[1]
                block_tensors.append(CanonicalKVCacheTensor(tensor, page_size))
                for layer_name in layer_names:
                    assert (
                        kv_caches[layer_name][component_idx].data_ptr()
                        == component.data_ptr()
                    )
                    refs_by_layer[layer_name].append(
                        CanonicalKVCacheRef(tensor_idx, page_size)
                    )

        group_refs = [
            [
                ref
                for layer_name in group.layer_names
                for ref in refs_by_layer[layer_name]
            ]
            for group in self.kv_cache_config.kv_cache_groups
        ]
        return CanonicalKVCaches(block_tensors, group_refs)

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            original_handler = gpu_worker.SingleDirectionOffloadingHandler
            gpu_worker.SingleDirectionOffloadingHandler = (
                _NPUSingleDirectionOffloadingHandler
            )
            gpu_worker.torch.Event = torch.npu.Event
            try:
                self._handlers = CpuGpuOffloadingHandlers(
                    kv_caches=kv_caches,
                    block_size_factor=self.block_size_factor,
                    num_cpu_blocks=self.num_cpu_blocks,
                )
            finally:
                gpu_worker.SingleDirectionOffloadingHandler = original_handler

        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler


_original_register_kv_caches = OffloadingConnectorWorker.register_kv_caches


def _register_kv_caches(self, kv_caches):
    if isinstance(self.spec, NPUOffloadingSpec):
        self._register_handlers(self.spec.canonicalize_kv_caches(kv_caches))
    else:
        _original_register_kv_caches(self, kv_caches)


OffloadingConnectorWorker.register_kv_caches = _register_kv_caches
