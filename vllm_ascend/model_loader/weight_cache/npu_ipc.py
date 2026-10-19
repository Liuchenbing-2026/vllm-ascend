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
#
"""Ascend NPU IPC implementation for vLLM's fast-restart weight cache."""

import os
import sys
from dataclasses import dataclass, field

import torch
import torch.multiprocessing.reductions as torch_reductions
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    UnsupportedPlatformForIPCError,
    UnsupportedQuantForIPCError,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

_NPU_IPC_ARGS_LEN = 15
# torch_npu's reduce_tensor tuple stores the local device index at this
# position. The exact tuple ABI is validated before indexing it.
_NPU_IPC_DEVICE_INDEX = 6


@dataclass(frozen=True)
class CachedTensorSpec:
    """Serializable tensor invariants checked before any live mutation."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    layout: str
    device_type: str

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "CachedTensorSpec":
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            layout=str(tensor.layout),
            device_type=tensor.device.type,
        )

    def mismatch_fields(self, tensor: torch.Tensor) -> tuple[str, ...]:
        actual = type(self).from_tensor(tensor)
        fields = ("shape", "stride", "dtype", "layout", "device_type")
        return tuple(name for name in fields if getattr(self, name) != getattr(actual, name))


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def _reduce_tensor(tensor: torch.Tensor) -> tuple:
    # torch_npu replaces this module attribute during initialization. Resolve it
    # at call time so plugin import order cannot capture the CPU/CUDA fallback.
    return torch_reductions.reduce_tensor(tensor)


def _validate_npu_ipc_args(ipc_args: tuple) -> None:
    if len(ipc_args) != _NPU_IPC_ARGS_LEN:
        raise RuntimeError(
            "torch_npu returned an unexpected reduce_tensor tuple: "
            f"expected {_NPU_IPC_ARGS_LEN} elements, got {len(ipc_args)}"
        )


@dataclass
class NPUTensorEntry(TensorEntry):
    """TensorEntry variant that exports NPU storage through torch_npu IPC."""

    npu_ipc_args: tuple[object, ...] | None = None
    spec: CachedTensorSpec | None = None
    _npu_source_tensor: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, kind: str) -> "NPUTensorEntry":
        tensor = tensor.detach()
        spec = CachedTensorSpec.from_tensor(tensor)
        if _is_npu_tensor(tensor):
            _, ipc_args = _reduce_tensor(tensor)
            _validate_npu_ipc_args(ipc_args)
            return cls(
                kind=kind,
                npu_ipc_args=ipc_args,
                spec=spec,
                _npu_source_tensor=tensor,
            )
        # Keep every entry on the replacement class. Pickle otherwise resolves
        # the rebound protocol symbol to a class that no longer matches the
        # original dataclass object used by the producer.
        if tensor.is_cuda:
            _, ipc_args = _reduce_tensor(tensor)
            return cls(kind=kind, ipc_args=ipc_args, spec=spec)
        return cls(kind=kind, cpu_tensor=tensor.cpu().clone(), spec=spec)

    def rebuild(self, device_index: int) -> torch.Tensor:
        if self.npu_ipc_args is None:
            tensor = super().rebuild(device_index)
        else:
            from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

            _validate_npu_ipc_args(self.npu_ipc_args)
            args = list(self.npu_ipc_args)
            args[_NPU_IPC_DEVICE_INDEX] = device_index
            tensor = rebuild_npu_tensor(*args)
        if self.spec is not None:
            mismatch = self.spec.mismatch_fields(tensor)
            if mismatch:
                raise RuntimeError(f"cached tensor spec changed after IPC rebuild: {', '.join(mismatch)}")
        return tensor

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # The producer keeps this strong reference for the IPC allocation's
        # lifetime. Sending it would copy full weights into the engine.
        state["_npu_source_tensor"] = None
        return state


def _is_empty_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor carries no data that needs IPC transfer."""

    return tensor.numel() == 0


def _tensor_mismatch_fields(
    target: torch.Tensor,
    source: torch.Tensor,
) -> list[str]:
    target_device = target.device.type
    source_device = source.device.type
    fields = (
        ("shape", tuple(target.shape), tuple(source.shape)),
        ("stride", tuple(target.stride()), tuple(source.stride())),
        ("dtype", target.dtype, source.dtype),
        ("layout", target.layout, source.layout),
        ("device", target_device, source_device),
    )
    return [
        f"{name}: target={target_value!r} source={source_value!r}"
        for name, target_value, source_value in fields
        if target_value != source_value
    ]


class NPUIpcModelLoader(IpcModelLoader):
    """Weight-cache loader with fail-closed Ascend cache validation."""

    def _build_model(
        self,
        vllm_config,
        model_config: ModelConfig,
        prefix: str,
        entries: dict[str, TensorEntry],
        aliases: dict[str, str],
    ) -> torch.nn.Module:
        model = super()._build_model(vllm_config, model_config, prefix, entries, aliases)
        # IPC constructs the model on meta before replacing tensors. Rope keeps
        # hot-path aliases in module globals, so refresh those aliases once the
        # complete model, including post-load buffers, is materialized.
        from vllm_ascend.ops.rotary_embedding import rebind_cos_sin_cache_from_model

        rebind_cos_sin_cache_from_model(model)
        return model

    def _apply_entries(
        self,
        model: torch.nn.Module,
        entries: dict[str, TensorEntry],
        aliases: dict[str, str],
        device_index: int,
    ) -> None:
        modules = dict(model.named_modules(remove_duplicate=False))
        params = dict(model.named_parameters(remove_duplicate=False))
        buffers = dict(model.named_buffers(remove_duplicate=False))
        targets = {**buffers, **params}
        unknown = sorted(set(entries) - set(targets))
        if unknown:
            raise RuntimeError(f"cached tensors have no matching model entries: {unknown[:8]}")

        rebuilt: dict[str, torch.Tensor] = {}
        for name, entry in entries.items():
            expected_kind = "param" if name in params else "buffer"
            if entry.kind != expected_kind:
                raise RuntimeError(f"cached tensor {name!r} kind {entry.kind!r} != {expected_kind!r}")
            rebuilt[name] = entry.rebuild(device_index)

        for alias_name, canonical_name in aliases.items():
            if alias_name not in targets:
                raise RuntimeError(f"cached alias {alias_name!r} has no matching model entry")
            if canonical_name not in rebuilt:
                raise RuntimeError(
                    f"cached alias {alias_name!r} references missing canonical tensor {canonical_name!r}"
                )

        non_persistent_buffers = {name: module._non_persistent_buffers_set.copy() for name, module in modules.items()}
        registered: dict[str, torch.Tensor] = {}

        def _register(name: str, tensor: torch.Tensor, is_param: bool) -> None:
            module_name, _, leaf = name.rpartition(".")
            module = modules.get(module_name)
            if module is None:
                raise RuntimeError(f"cached tensor {name!r} has no matching module")
            module._parameters.pop(leaf, None)
            module._buffers.pop(leaf, None)
            if is_param:
                obj: torch.Tensor = (
                    tensor
                    if isinstance(tensor, torch.nn.Parameter)
                    else torch.nn.Parameter(tensor, requires_grad=False)
                )
                module.register_parameter(leaf, obj)
            else:
                obj = tensor
                module.register_buffer(leaf, obj)
            registered[name] = obj

        for name, tensor in rebuilt.items():
            if self.mode == "copy":
                tensor = tensor.clone()
            _register(name, tensor, name in params)

        for alias_name, canonical_name in aliases.items():
            canonical = registered[canonical_name]
            _register(alias_name, canonical, alias_name in params)

        for name, module in modules.items():
            module._non_persistent_buffers_set = non_persistent_buffers.get(name, set())

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        """Transactionally validate and reload a live model in place."""

        entries, aliases = self._fetch_entries(model_config)
        device_index = torch.accelerator.current_device_index()
        params = dict(model.named_parameters(remove_duplicate=False))
        buffers = dict(model.named_buffers(remove_duplicate=False))
        targets = {**buffers, **params}

        cached_names = set(entries) | set(aliases)
        missing = sorted(
            name for name, tensor in targets.items() if name not in cached_names and not _is_empty_tensor(tensor)
        )
        unexpected = sorted(cached_names - set(targets))
        if missing or unexpected:
            raise RuntimeError(
                f"cached tensor set does not match the live model: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )

        aliases_by_canonical: dict[str, list[str]] = {}
        for alias_name, canonical_name in aliases.items():
            if canonical_name not in entries:
                raise RuntimeError(
                    f"cached alias {alias_name!r} references missing canonical tensor {canonical_name!r}"
                )
            alias_target = targets.get(alias_name)
            canonical_target = targets.get(canonical_name)
            if alias_target is None or canonical_target is None:
                raise RuntimeError(f"cached alias {alias_name!r} -> {canonical_name!r} is incomplete")
            if alias_target is not canonical_target:
                raise RuntimeError(f"cached alias {alias_name!r} is not tied to {canonical_name!r} in the live model")
            aliases_by_canonical.setdefault(canonical_name, []).append(alias_name)

        staged: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for name, entry in entries.items():
            target = targets[name]
            expected_kind = "param" if name in params else "buffer"
            if entry.kind != expected_kind:
                raise RuntimeError(f"cached tensor {name!r} kind {entry.kind!r} != {expected_kind!r}")
            source = entry.rebuild(device_index)
            mismatch = _tensor_mismatch_fields(target, source)
            if mismatch:
                raise RuntimeError(f"cached tensor {name!r} is incompatible: {'; '.join(mismatch)}")
            for alias_name in aliases_by_canonical.get(name, []):
                alias_target = targets[alias_name]
                if alias_target.data_ptr() != target.data_ptr():
                    raise RuntimeError(f"cached alias {alias_name!r} changed the graph-visible data pointer")
            staged.append((name, target, source))

        data_ptrs = {name: target.data_ptr() for name, target, _ in staged}
        backups = [(target, target.detach().clone()) for _, target, _ in staged]

        def _restore() -> list[str]:
            failures: list[str] = []
            with torch.no_grad():
                for (name, target, _), (backup_target, backup) in zip(
                    staged,
                    backups,
                    strict=True,
                ):
                    try:
                        if target is not backup_target:
                            raise RuntimeError("target identity changed during reload")
                        target.copy_(backup)
                    except Exception as exc:
                        failures.append(f"{name}: {type(exc).__name__}: {exc}")
            return failures

        try:
            with torch.no_grad():
                for _, target, source in staged:
                    target.copy_(source)

            moved = [name for name, target, _ in staged if target.data_ptr() != data_ptrs[name]]
            if moved:
                raise RuntimeError(
                    f"in-place IPC reload moved parameter storage and invalidated captured graphs: {moved[:8]}"
                )
        except Exception as exc:
            failures = _restore()
            if failures:
                raise RuntimeError(
                    "IPC reload failed and rollback was incomplete: "
                    f"reload_error={exc!r}, rollback_errors={failures[:8]}"
                ) from exc
            raise


def check_ipc_platform_support() -> None:
    """Allow IPC weight sharing on Ascend NPU and CUDA-alike devices."""

    if current_platform.device_type == "npu" or current_platform.is_cuda_alike():
        return
    raise UnsupportedPlatformForIPCError(
        f"platform {current_platform.device_name!r} does not support IPC "
        "weight sharing; only CUDA, ROCm, and Ascend NPU are supported. "
        "Use the default --load-format for this platform."
    )


def check_ipc_quant_support(model: torch.nn.Module) -> None:
    """Reject quantization methods whose post-load transforms are not idempotent."""

    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase) and not quant_method.supports_pre_processed_weights:
            raise UnsupportedQuantForIPCError(
                f"layer {name or '<root>'}: {type(quant_method).__name__} "
                "does not support loading from pre-processed weights."
            )


def _install_daemon_model_lifecycle(module: object) -> None:
    """Initialize per-process Ascend state before the daemon creates a model.

    The weight-cache daemon builds its model without instantiating an
    ``NPUWorker``. Ascend config is process-local and is therefore absent in
    the spawned daemon even though the parent initialized it while creating the
    engine config. The spawn helper executes the daemon module as
    ``__mp_main__``, so wrapping ``get_daemon_model`` cannot cover the class
    method whose globals belong to that module. Wrap the class lifecycle hook
    instead.
    """

    daemon_cls = getattr(module, "WeightCacheDaemon", None)
    load_model = getattr(daemon_cls, "load_model", None)
    if load_model is None or getattr(
        load_model,
        "_vllm_ascend_lifecycle_wrapped",
        False,
    ):
        return

    def _wrapped(self):
        from vllm_ascend.ascend_config import init_ascend_config

        init_ascend_config(self.vllm_config)
        return load_model(self)

    _wrapped._vllm_ascend_lifecycle_wrapped = True
    daemon_cls.load_model = _wrapped


def _loaded_modules_with_spec_name(
    module_name: str,
    canonical_module: object | None = None,
) -> list[object]:
    """Return every live module loaded from the same source file.

    Python's spawn start method may load a module as ``__mp_main__`` instead of
    its canonical name, and the alias can retain ``__spec__.name ==
    "__mp_main__"``. Patch the execution alias as well as the canonical module
    so deferred function globals resolve to the Ascend IPC entry type.
    """

    modules: list[object] = []
    seen: set[int] = set()
    canonical_file = getattr(canonical_module, "__file__", None)
    canonical_file = os.path.realpath(canonical_file) if canonical_file else None
    # Plugin loading can import modules while this scan is running.
    for module_key, module in tuple(sys.modules.items()):
        if module is None:
            continue
        spec = getattr(module, "__spec__", None)
        module_file = getattr(module, "__file__", None)
        namespace = vars(module)
        same_source = (
            canonical_file is not None and module_file is not None and os.path.realpath(module_file) == canonical_file
        )
        spawn_alias = (
            module_key in {"__main__", "__mp_main__"}
            and callable(namespace.get("export_entries"))
            and "WeightCacheDaemon" in namespace
        )
        if getattr(spec, "name", None) != module_name and not same_source and not spawn_alias:
            continue
        module_id = id(module)
        if module_id not in seen:
            seen.add(module_id)
            modules.append(module)
    return modules


def _install_tensor_entry_factory(protocol_module: object) -> None:
    """Route the protocol's original factory to ``NPUTensorEntry``.

    The daemon may execute from a spawn alias and retain an un-rebound
    ``TensorEntry`` name in that alias's globals. All aliases nevertheless
    share the class object imported from ``protocol.py``. Patching the shared
    factory guarantees NPU tensors are exported through IPC before pickle,
    instead of copying the full model into the socket message.
    """

    tensor_entry_cls = protocol_module.TensorEntry
    if getattr(tensor_entry_cls, "_vllm_ascend_npu_factory", False):
        return

    def _from_tensor(cls, tensor: torch.Tensor, kind: str) -> NPUTensorEntry:
        del cls
        return NPUTensorEntry.from_tensor(tensor, kind)

    tensor_entry_cls.from_tensor = classmethod(_from_tensor)
    tensor_entry_cls._vllm_ascend_npu_factory = True


def install_npu_ipc() -> None:
    """Install NPU IPC symbols and register the Ascend ``ipc_cache`` loader."""

    from vllm.model_executor.model_loader.weight_cache import daemon as daemon_module
    from vllm.model_executor.model_loader.weight_cache import protocol

    protocol.check_ipc_platform_support = check_ipc_platform_support
    protocol.check_ipc_quant_support = check_ipc_quant_support
    _install_tensor_entry_factory(protocol)

    current_loader = type(get_model_loader(LoadConfig(load_format="ipc_cache")))
    if current_loader is not NPUIpcModelLoader:
        register_model_loader("ipc_cache")(NPUIpcModelLoader)

    early_bindings = (
        (
            "vllm.model_executor.model_loader.weight_cache.daemon",
            "TensorEntry",
            NPUTensorEntry,
        ),
        (
            "vllm.model_executor.model_loader.weight_cache.daemon",
            "check_ipc_platform_support",
            check_ipc_platform_support,
        ),
        (
            "vllm.model_executor.model_loader.weight_cache.daemon",
            "check_ipc_quant_support",
            check_ipc_quant_support,
        ),
        (
            "vllm.model_executor.model_loader.weight_cache.ipc_loader",
            "TensorEntry",
            NPUTensorEntry,
        ),
        (
            "vllm.model_executor.model_loader.weight_cache.ipc_loader",
            "check_ipc_platform_support",
            check_ipc_platform_support,
        ),
        (
            "vllm.model_executor.model_loader.weight_cache.ipc_loader",
            "check_ipc_quant_support",
            check_ipc_quant_support,
        ),
        (
            "vllm.entrypoints.cli.preload",
            "check_ipc_platform_support",
            check_ipc_platform_support,
        ),
    )
    daemon_modules = _loaded_modules_with_spec_name(
        "vllm.model_executor.model_loader.weight_cache.daemon",
        daemon_module,
    )
    if daemon_module not in daemon_modules:
        daemon_modules.append(daemon_module)
    for module in daemon_modules:
        _install_daemon_model_lifecycle(module)

    for module_name, symbol_name, value in early_bindings:
        canonical_module = sys.modules.get(module_name)
        modules = _loaded_modules_with_spec_name(module_name, canonical_module)
        if canonical_module is not None and canonical_module not in modules:
            modules.append(canonical_module)
        for module in modules:
            if hasattr(module, symbol_name):
                setattr(module, symbol_name, value)
