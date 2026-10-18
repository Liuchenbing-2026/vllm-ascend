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

import pickle
import sys
from types import SimpleNamespace

import pytest
import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    UnsupportedQuantForIPCError,
)

from vllm_ascend.model_loader.weight_cache import npu_ipc
from vllm_ascend.model_loader.weight_cache.npu_ipc import (
    NPUIpcModelLoader,
    NPUTensorEntry,
    _install_daemon_model_lifecycle,
    check_ipc_quant_support,
    install_npu_ipc,
)


def _ipc_args(device_index: int = 0) -> tuple:
    args = [None] * npu_ipc._NPU_IPC_ARGS_LEN
    args[npu_ipc._NPU_IPC_DEVICE_INDEX] = device_index
    return tuple(args)


def _cpu_entry(kind: str, values: list[float]) -> NPUTensorEntry:
    return NPUTensorEntry(
        kind=kind,
        cpu_tensor=torch.tensor(values, dtype=torch.float32),
    )


def _loader() -> NPUIpcModelLoader:
    loader = object.__new__(NPUIpcModelLoader)
    loader.mode = "copy"
    return loader


def test_npu_entry_exports_ipc_and_pickle_drops_source_tensor(monkeypatch):
    source = torch.arange(4, dtype=torch.float32)
    ipc_args = _ipc_args(device_index=3)
    monkeypatch.setattr(npu_ipc, "_is_npu_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        npu_ipc,
        "_reduce_tensor",
        lambda tensor: ("rebuild_npu_tensor", ipc_args),
    )

    entry = NPUTensorEntry.from_tensor(source, "param")

    assert entry.ipc_args is None
    assert entry.cpu_tensor is None
    assert entry.npu_ipc_args == ipc_args
    assert entry.spec == npu_ipc.CachedTensorSpec.from_tensor(source)
    assert entry._npu_source_tensor is not None
    assert entry._npu_source_tensor.data_ptr() == source.data_ptr()

    restored = pickle.loads(pickle.dumps(entry))
    assert restored.npu_ipc_args == ipc_args
    assert restored._npu_source_tensor is None


@pytest.mark.parametrize("tuple_length", [0, 14, 16])
def test_group_rejects_unexpected_npu_tuple_length(monkeypatch, tuple_length):
    monkeypatch.setattr(npu_ipc, "_is_npu_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        npu_ipc,
        "_reduce_tensor",
        lambda _tensor: ("rebuild_npu_tensor", (None,) * tuple_length),
    )

    with pytest.raises(RuntimeError, match="unexpected reduce_tensor tuple"):
        NPUTensorEntry.from_tensor(torch.ones(1), "param")


def test_npu_rebuild_retargets_local_device(monkeypatch):
    observed: list[tuple] = []

    def rebuild(*args):
        observed.append(args)
        return torch.empty(2)

    reductions = SimpleNamespace(rebuild_npu_tensor=rebuild)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.multiprocessing.reductions",
        reductions,
    )
    entry = NPUTensorEntry(kind="param", npu_ipc_args=_ipc_args(device_index=7))

    tensor = entry.rebuild(device_index=2)

    assert tensor.shape == (2,)
    assert observed[0][npu_ipc._NPU_IPC_DEVICE_INDEX] == 2


def test_cpu_entry_is_snapshotted_and_validates_spec():
    source = torch.arange(6, dtype=torch.float32).view(2, 3).t()
    entry = NPUTensorEntry.from_tensor(source, "buffer")

    assert entry.cpu_tensor is not None
    assert entry.cpu_tensor.data_ptr() != source.data_ptr()
    assert entry.rebuild(0).stride() == source.stride()
    assert entry.spec is not None

    incompatible = torch.arange(6, dtype=torch.float32).view(3, 2)
    assert entry.spec.mismatch_fields(incompatible) == ("stride",)


@pytest.mark.parametrize("mode", ["copy", "zero_copy"])
def test_apply_entries_reestablishes_tied_alias(mode):
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            weight = torch.nn.Parameter(torch.zeros(2))
            self.block.weight = weight
            self.block.alias = weight
            self.block.register_buffer("rope_cache", torch.zeros(2), persistent=False)

    model = TiedModel()
    old_tied = model.block.weight
    loader = _loader()
    loader.mode = mode
    entries = {
        "block.weight": _cpu_entry("param", [3.0, 4.0]),
        "block.rope_cache": _cpu_entry("buffer", [5.0, 6.0]),
    }

    loader._apply_entries(
        model,
        entries,
        {"block.alias": "block.weight"},
        device_index=0,
    )

    assert model.block.weight is model.block.alias
    if mode == "copy":
        assert model.block.weight is not old_tied
    torch.testing.assert_close(model.block.weight, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(model.block.rope_cache, torch.tensor([5.0, 6.0]))
    assert "rope_cache" in model.block._non_persistent_buffers_set


def test_load_weights_rejects_missing_and_unexpected_names(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    loader = _loader()
    monkeypatch.setattr(
        loader,
        "_fetch_entries",
        lambda _model_config: ({"unexpected": _cpu_entry("param", [1.0, 2.0])}, {}),
    )

    with pytest.raises(RuntimeError, match="missing=.*unexpected="):
        loader.load_weights(Model(), object())


def test_load_weights_allows_uncached_empty_tensor(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("empty_placeholder", torch.empty(0))

    loader = _loader()
    monkeypatch.setattr(
        loader,
        "_fetch_entries",
        lambda _model_config: ({"weight": _cpu_entry("param", [1.0, 2.0])}, {}),
    )
    monkeypatch.setattr(
        npu_ipc.torch.accelerator,
        "current_device_index",
        lambda: 0,
    )
    model = Model()

    loader.load_weights(model, object())

    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0]))
    assert model.empty_placeholder.numel() == 0


def test_load_weights_rolls_back_partial_copy_failure(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.second = torch.nn.Parameter(torch.tensor([3.0, 4.0]))

    model = Model()
    first_before = model.first.detach().clone()
    second_before = model.second.detach().clone()
    loader = _loader()
    monkeypatch.setattr(
        loader,
        "_fetch_entries",
        lambda _model_config: (
            {
                "first": _cpu_entry("param", [10.0, 20.0]),
                "second": _cpu_entry("param", [30.0, 40.0]),
            },
            {},
        ),
    )
    monkeypatch.setattr(
        npu_ipc.torch.accelerator,
        "current_device_index",
        lambda: 0,
    )
    copy_ = torch.Tensor.copy_
    failed = False

    def fail_once(target, source, *args, **kwargs):
        nonlocal failed
        if target is model.second and not failed:
            failed = True
            raise RuntimeError("injected copy failure")
        return copy_(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_once)

    with pytest.raises(RuntimeError, match="injected copy failure"):
        loader.load_weights(model, object())

    torch.testing.assert_close(model.first, first_before)
    torch.testing.assert_close(model.second, second_before)


def test_load_weights_is_in_place_and_preserves_non_persistent_buffer(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("rope_cache", torch.zeros(2), persistent=False)

    model = Model()
    data_ptrs = {
        "weight": model.weight.data_ptr(),
        "rope_cache": model.rope_cache.data_ptr(),
    }
    loader = _loader()
    monkeypatch.setattr(
        loader,
        "_fetch_entries",
        lambda _model_config: (
            {
                "weight": _cpu_entry("param", [1.0, 2.0]),
                "rope_cache": _cpu_entry("buffer", [3.0, 4.0]),
            },
            {},
        ),
    )
    monkeypatch.setattr(
        npu_ipc.torch.accelerator,
        "current_device_index",
        lambda: 0,
    )

    loader.load_weights(model, object())

    assert model.weight.data_ptr() == data_ptrs["weight"]
    assert model.rope_cache.data_ptr() == data_ptrs["rope_cache"]
    assert "rope_cache" in model._non_persistent_buffers_set
    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.rope_cache, torch.tensor([3.0, 4.0]))


def test_live_reload_rolls_back_when_graph_visible_pointer_moves(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    model = Model()
    original = model.weight.detach().clone()
    loader = _loader()
    monkeypatch.setattr(
        loader,
        "_fetch_entries",
        lambda _model_config: (
            {"weight": _cpu_entry("param", [10.0, 20.0])},
            {},
        ),
    )
    monkeypatch.setattr(
        npu_ipc.torch.accelerator,
        "current_device_index",
        lambda: 0,
    )

    original_copy = torch.Tensor.copy_
    original_data_ptr = torch.Tensor.data_ptr
    reloaded = False

    def copy_(target, source, *args, **kwargs):
        nonlocal reloaded
        result = original_copy(target, source, *args, **kwargs)
        if target is model.weight and torch.equal(
            target.detach(),
            torch.tensor([10.0, 20.0], device=target.device),
        ):
            reloaded = True
        return result

    def data_ptr(tensor):
        pointer = original_data_ptr(tensor)
        if reloaded and tensor is model.weight:
            return pointer + 1
        return pointer

    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    monkeypatch.setattr(torch.Tensor, "data_ptr", data_ptr)

    with pytest.raises(RuntimeError, match="moved parameter storage"):
        loader.load_weights(model, object())

    torch.testing.assert_close(model.weight, original)


def test_live_reload_rejects_alias_that_is_not_tied(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.alias = torch.nn.Parameter(torch.zeros(2))

    loader = _loader()
    monkeypatch.setattr(
        loader,
        "_fetch_entries",
        lambda _model_config: (
            {"weight": _cpu_entry("param", [1.0, 2.0])},
            {"alias": "weight"},
        ),
    )

    with pytest.raises(RuntimeError, match="is not tied"):
        loader.load_weights(Model(), object())


class _UnsupportedQuantMethod(QuantizeMethodBase):
    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError


class _SupportedQuantMethod(_UnsupportedQuantMethod):
    supports_pre_processed_weights = True


def test_quantization_support_is_fail_closed():
    model = torch.nn.Module()
    model.quant_method = _UnsupportedQuantMethod()

    with pytest.raises(UnsupportedQuantForIPCError):
        check_ipc_quant_support(model)

    model.quant_method = _SupportedQuantMethod()
    check_ipc_quant_support(model)


def test_install_npu_ipc_is_idempotent():
    install_npu_ipc()
    install_npu_ipc()

    assert type(get_model_loader(LoadConfig(load_format="ipc_cache"))) is NPUIpcModelLoader
    daemon = sys.modules["vllm.model_executor.model_loader.weight_cache.daemon"]
    assert daemon.WeightCacheDaemon.load_model._vllm_ascend_lifecycle_wrapped


def test_install_npu_ipc_patches_shared_tensor_entry_factory(monkeypatch):
    monkeypatch.setattr(npu_ipc, "_is_npu_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        npu_ipc,
        "_reduce_tensor",
        lambda _tensor: ("rebuild_npu_tensor", _ipc_args()),
    )

    install_npu_ipc()
    entry = TensorEntry.from_tensor(torch.ones(1), "param")

    assert isinstance(entry, NPUTensorEntry)
    assert entry.cpu_tensor is None
    assert entry.npu_ipc_args == _ipc_args()


def test_daemon_lifecycle_initializes_ascend_config_once_per_load(monkeypatch):
    calls: list[object] = []
    config = object()

    class WeightCacheDaemon:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def load_model(self):
            calls.append("load")
            return "model"

    module = SimpleNamespace(WeightCacheDaemon=WeightCacheDaemon)
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.init_ascend_config",
        lambda value: calls.append(("init", value)),
    )

    _install_daemon_model_lifecycle(module)
    _install_daemon_model_lifecycle(module)

    daemon = WeightCacheDaemon(config)
    assert daemon.load_model() == "model"
    daemon.load_model()
    assert calls == [
        ("init", config),
        "load",
        ("init", config),
        "load",
    ]


def test_install_npu_ipc_patches_daemon_run_as_main(monkeypatch):
    daemon_name = "vllm.model_executor.model_loader.weight_cache.daemon"
    main_module = SimpleNamespace(
        __spec__=SimpleNamespace(name=daemon_name),
        TensorEntry=object(),
        check_ipc_platform_support=lambda: None,
        check_ipc_quant_support=lambda _model: None,
        WeightCacheDaemon=type(
            "WeightCacheDaemon",
            (),
            {"load_model": lambda self: "model"},
        ),
    )
    monkeypatch.setitem(sys.modules, "__main__", main_module)

    install_npu_ipc()

    assert main_module.TensorEntry is NPUTensorEntry
    assert main_module.check_ipc_platform_support is npu_ipc.check_ipc_platform_support
    assert main_module.check_ipc_quant_support is check_ipc_quant_support
    assert main_module.WeightCacheDaemon.load_model._vllm_ascend_lifecycle_wrapped


def test_install_npu_ipc_patches_daemon_spawn_alias(monkeypatch):
    daemon_name = "vllm.model_executor.model_loader.weight_cache.daemon"
    canonical_module = sys.modules[daemon_name]
    spawn_module = SimpleNamespace(
        __spec__=SimpleNamespace(name="__mp_main__"),
        TensorEntry=object(),
        check_ipc_platform_support=lambda: None,
        check_ipc_quant_support=lambda _model: None,
        export_entries=lambda _model: None,
        WeightCacheDaemon=type(
            "WeightCacheDaemon",
            (),
            {"load_model": lambda self: "model"},
        ),
    )
    monkeypatch.setitem(sys.modules, "__mp_main__", spawn_module)

    install_npu_ipc()

    assert spawn_module.TensorEntry is NPUTensorEntry
    assert spawn_module.check_ipc_platform_support is npu_ipc.check_ipc_platform_support
    assert spawn_module.check_ipc_quant_support is check_ipc_quant_support
    assert spawn_module.WeightCacheDaemon.load_model._vllm_ascend_lifecycle_wrapped
