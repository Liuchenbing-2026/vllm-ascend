# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_patch(monkeypatch, copy_buffers_cls, enabled):
    # Only upstream/API and device dispatch dependencies are isolated. Execute
    # the complete patch source against real torch dtypes, not an AST copy.
    core = ModuleType("vllm.v1.worker.mamba_utils")
    if copy_buffers_cls is not None:
        core.MambaCopyBuffers = copy_buffers_cls
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        parent = ModuleType(name)
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    modules = {
        "vllm.v1.worker.mamba_utils": core,
        "vllm_ascend.device.hardware_profile": SimpleNamespace(
            HardwareCapability=SimpleNamespace(TRITON_BATCH_MEMCPY="batch"),
            get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: enabled),
        ),
        "vllm_ascend.ops.triton.batch_memcpy": SimpleNamespace(batch_memcpy_kernel=object()),
        "vllm_ascend.ops.triton.mamba.postprocess": SimpleNamespace(postprocess_mamba_fused_kernel=object()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_mamba_utils.py"
    spec = importlib.util.spec_from_file_location("mamba_patch_contract", source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return core, result, modules


@pytest.mark.parametrize("enabled", [False, True])
def test_new_core_without_legacy_buffers_keeps_kernel_dispatch(monkeypatch, enabled):
    core, patch, deps = _load_patch(monkeypatch, None, enabled)
    assert not hasattr(core, "MambaCopyBuffers")
    assert core._TEMPORAL_TILES == 1
    if enabled:
        assert core.batch_memcpy is patch._batch_memcpy_triton
        assert core.batch_memcpy_kernel is deps["vllm_ascend.ops.triton.batch_memcpy"].batch_memcpy_kernel
        assert (
            core.postprocess_mamba_fused_kernel
            is deps["vllm_ascend.ops.triton.mamba.postprocess"].postprocess_mamba_fused_kernel
        )
    else:
        assert not hasattr(core, "batch_memcpy")


@pytest.mark.parametrize(
    "dtype, expected", [(torch.uint64, torch.int64), (torch.int64, torch.int64), (torch.int32, torch.int32)]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_buffer_dtype_and_arguments_preserved(monkeypatch, dtype, expected, enabled):
    calls = []

    class Legacy:
        @classmethod
        def create(cls, count, config, funcs, make_buffer):
            calls.append((cls, count, config, funcs))
            return make_buffer(count, dtype=dtype)

    _load_patch(monkeypatch, Legacy, enabled)
    config, funcs = object(), object()
    result = Legacy.create(7, config, funcs, lambda n, dtype: torch.zeros(n, dtype=dtype))
    assert result.dtype == expected
    assert result.shape == (7,)
    assert calls == [(Legacy, 7, config, funcs)]


def test_legacy_create_failure_is_not_silenced(monkeypatch):
    class Legacy:
        @classmethod
        def create(cls, *_args):
            raise RuntimeError("allocation failed")

    _load_patch(monkeypatch, Legacy, False)
    with pytest.raises(RuntimeError, match="allocation failed"):
        Legacy.create(1, None, None, None)
