#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

wc_protocol = pytest.importorskip(
    "vllm.model_executor.model_loader.weight_cache.protocol",
    reason="installed vllm has no weight cache (needs >= 0.31)",
)
import vllm_ascend.patch.platform.patch_weight_cache as pwc  # noqa: E402


def _fake_torch(*, ipc_capable: bool):
    attrs = {"_share_npu_": staticmethod(lambda: None)}
    if ipc_capable:
        attrs["_new_shared_npu"] = staticmethod(lambda: None)
    storage_cls = type("FakeUntypedStorage", (), attrs)
    return SimpleNamespace(UntypedStorage=storage_cls)


class TestCheckIpcPlatformSupport:
    def test_allows_npu_with_ipc_capable_torch_npu(self):
        with (
            patch.object(pwc, "current_platform") as platform,
            patch.object(pwc, "torch", _fake_torch(ipc_capable=True)),
        ):
            platform.device_type = "npu"
            pwc._check_ipc_platform_support_npu()

    def test_rejects_npu_without_ipc_capable_torch_npu(self):
        with (
            patch.object(pwc, "current_platform") as platform,
            patch.object(pwc, "torch", _fake_torch(ipc_capable=False)),
        ):
            platform.device_type = "npu"
            with pytest.raises(wc_protocol.UnsupportedPlatformForIPCError):
                pwc._check_ipc_platform_support_npu()

    def test_consumers_bind_the_npu_aware_check(self):
        assert wc_protocol.check_ipc_platform_support is pwc._check_ipc_platform_support_npu
        for module_name in (
            "vllm.model_executor.model_loader.weight_cache.ipc_loader",
            "vllm.entrypoints.cli.preload",
        ):
            module = sys.modules.get(module_name)
            if module is not None:
                assert module.check_ipc_platform_support is pwc._check_ipc_platform_support_npu


class TestTensorEntry:
    def test_from_tensor_npu_uses_registered_reducer(self):
        tensor = MagicMock()
        tensor.device.type = "npu"
        tensor.detach.return_value = tensor
        with patch.object(
            torch.multiprocessing.reductions,
            "reduce_tensor",
            return_value=(MagicMock(name="rebuild_npu_tensor"), ("fake", "args")),
        ) as reduce_mock:
            entry = wc_protocol.TensorEntry.from_tensor(tensor, "param")
        reduce_mock.assert_called_once_with(tensor)
        assert entry.ipc_args == ("fake", "args")
        assert entry.cpu_tensor is None

    def test_from_tensor_non_npu_falls_back_to_cpu_value(self):
        entry = wc_protocol.TensorEntry.from_tensor(torch.zeros(2, 2), "param")
        assert entry.ipc_args is None
        assert entry.cpu_tensor is not None

    def test_rebuild_retargets_device_index_and_casts_format(self, monkeypatch):
        fake_reductions = types.ModuleType("torch_npu.multiprocessing.reductions")
        rebuilt = MagicMock(name="rebuilt_tensor")
        fake_reductions.rebuild_npu_tensor = MagicMock(return_value=rebuilt)
        fake_mp = types.ModuleType("torch_npu.multiprocessing")
        fake_mp.reductions = fake_reductions
        monkeypatch.setitem(sys.modules, "torch_npu.multiprocessing", fake_mp)
        monkeypatch.setitem(sys.modules, "torch_npu.multiprocessing.reductions", fake_reductions)

        entry = wc_protocol.TensorEntry(kind="param", ipc_args=(0, 1, 2, 3, 4, 5, 6, 7))
        entry.npu_format = 29  # ACL_FORMAT_FRACTAL_NZ
        with patch.object(pwc, "_cast_npu_format", return_value="casted") as cast_mock:
            result = entry.rebuild(device_index=3)
        fake_reductions.rebuild_npu_tensor.assert_called_once_with(0, 1, 2, 3, 4, 5, 3, 7)
        cast_mock.assert_called_once_with(rebuilt, 29)
        assert result == "casted"

    def test_rebuild_cpu_tensor_passthrough(self):
        tensor = torch.ones(2)
        entry = wc_protocol.TensorEntry(kind="buffer", cpu_tensor=tensor)
        assert entry.rebuild(device_index=0) is tensor


class TestDaemonFormatNormalization:
    def test_normalize_converts_non_nd_and_records_format(self):
        model = torch.nn.Linear(4, 4)
        casted = torch.zeros(4, 4)
        with (
            patch.object(pwc, "get_npu_format_int", side_effect=[29, None, None]),
            patch.object(pwc, "_cast_npu_format", return_value=casted) as cast_mock,
        ):
            pwc._normalize_model_formats_for_ipc(model)
        cast_mock.assert_called_once()
        formats = getattr(model, pwc._IPC_FORMATS_ATTR)
        assert formats == {"weight": 29}
        assert model.weight.data is casted

    def test_normalize_all_nd_records_nothing(self):
        model = torch.nn.Linear(4, 4)
        with patch.object(pwc, "get_npu_format_int", return_value=None):
            pwc._normalize_model_formats_for_ipc(model)
        assert getattr(model, pwc._IPC_FORMATS_ATTR) == {}

    def test_export_entries_annotates_recorded_formats(self):
        model = torch.nn.Linear(4, 4)
        setattr(model, pwc._IPC_FORMATS_ATTR, {"weight": 29})
        entry = wc_protocol.TensorEntry(kind="param", cpu_tensor=torch.zeros(4, 4))
        with patch.object(pwc, "_orig_export_entries", return_value=({"weight": entry}, {})):
            entries, aliases = pwc._export_entries_with_npu_formats(model)
        assert aliases == {}
        assert entries["weight"].npu_format == 29


class TestDraftLoadRouting:
    def _vllm_config(self, draft_model_config):
        return SimpleNamespace(
            load_config=SimpleNamespace(load_format="ipc_cache"),
            speculative_config=SimpleNamespace(draft_model_config=draft_model_config),
        )

    def test_reroutes_draft_load_to_draft_load_config(self):
        draft_config = object()
        vllm_config = self._vllm_config(draft_config)
        with (
            patch.object(pwc, "get_draft_load_config", return_value="DRAFT_LC") as draft_lc,
            patch.object(pwc, "_orig_get_model", return_value="model") as get_model,
        ):
            result = pwc._get_model_with_draft_cache_routing(
                vllm_config=vllm_config, model_config=draft_config, load_config=None
            )
        draft_lc.assert_called_once_with(vllm_config)
        assert get_model.call_args.kwargs["load_config"] == "DRAFT_LC"
        assert result == "model"

    def test_target_load_passes_through(self):
        vllm_config = self._vllm_config(object())
        with (
            patch.object(pwc, "get_draft_load_config") as draft_lc,
            patch.object(pwc, "_orig_get_model", return_value="model") as get_model,
        ):
            pwc._get_model_with_draft_cache_routing(vllm_config=vllm_config, model_config=object(), load_config=None)
        draft_lc.assert_not_called()
        assert get_model.call_args.kwargs["load_config"] is None

    def test_explicit_load_config_passes_through(self):
        draft_config = object()
        vllm_config = self._vllm_config(draft_config)
        with (
            patch.object(pwc, "get_draft_load_config") as draft_lc,
            patch.object(pwc, "_orig_get_model", return_value="model") as get_model,
        ):
            pwc._get_model_with_draft_cache_routing(
                vllm_config=vllm_config, model_config=draft_config, load_config="EXPLICIT"
            )
        draft_lc.assert_not_called()
        assert get_model.call_args.kwargs["load_config"] == "EXPLICIT"


class TestSnapshotGate:
    def test_snapshot_create_restore_rejected_on_npu(self):
        snapshot_runtime = pytest.importorskip("vllm.snapshot.runtime", reason="installed vllm has no snapshot support")
        with pytest.raises(RuntimeError, match="cuda-checkpoint"):
            snapshot_runtime.LocalSnapshotTools.preflight(MagicMock(), "create", MagicMock())
        with pytest.raises(RuntimeError, match="cuda-checkpoint"):
            snapshot_runtime.LocalSnapshotTools.preflight(MagicMock(), "restore", MagicMock())
