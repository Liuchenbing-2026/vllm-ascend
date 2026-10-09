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
"""Enable the vLLM 0.31+ weight cache (``vllm preload`` /
``--load-format ipc_cache``) on Ascend NPUs, and gate ``vllm snapshot``.

Upstream vLLM 0.31 ships a weight cache daemon that keeps post-quantized,
TP-sharded weights resident in device memory and serves them to restarting
engines over Unix sockets as device IPC handles. Upstream only allows
CUDA/ROCm platforms and only exports CUDA IPC handles. On Ascend this patch
provides the missing pieces:

1. ``check_ipc_platform_support`` accepts the NPU platform when the installed
   torch_npu supports NPU IPC storage sharing (``_share_npu_`` /
   ``_new_shared_npu``). torch_npu registers NPU-aware multiprocessing
   reducers on import, so ``torch.multiprocessing.reductions.reduce_tensor``
   already produces real NPU device IPC handles; ``TensorEntry`` is extended
   to use them, mirroring the CUDA path (device index is args[6] for both).

2. torch_npu's ``_share_npu_`` only exports base-format (ND/NCHW/...) storages,
   while Ascend ``process_weights_after_loading`` transmutes many weights to
   FRACTAL_NZ. After the daemon finishes loading, its resident tensors are
   normalized to ND once (in place, so the daemon keeps ~1x weights), and the
   original format is recorded per tensor name. ``TensorEntry`` carries that
   format to the engine, which casts ND back to NZ after mapping. The NZ<->ND
   round trip is lossless (layout-only) and matches the netloader's
   HCCL-transfer convention, including the W4A8 packed int32-NZ detour
   through an int8 view.

3. Ascend's speculative-decode proposer loads the draft via
   ``get_model(load_config=None)``, which would hit the target daemon group
   and fall back to disk. A narrow ``get_model`` wrapper reroutes draft loads
   through upstream ``get_draft_load_config`` so the daemon's draft group is
   used when ``--load-format ipc_cache``.

4. ``vllm snapshot create/restore`` requires CRIU plus the NVIDIA-only
   ``cuda-checkpoint`` plugin to freeze/restore device state; CANN has no
   equivalent, so snapshot creation/restoration is rejected with a clear
   error instead of failing deep inside the CRIU workflow.
"""

import sys
from itertools import chain

import torch
import torch.multiprocessing.reductions
import torch_npu  # noqa: F401  # importing registers NPU multiprocessing reducers
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, get_npu_format_int

logger = init_logger(__name__)

try:
    from vllm.model_executor.model_loader.weight_cache import daemon as _wc_daemon
    from vllm.model_executor.model_loader.weight_cache import protocol as _wc_protocol
except ImportError:
    # vllm < 0.31 has no weight cache; nothing to patch.
    _wc_daemon = None
    _wc_protocol = None

# Attribute on the daemon's model: {tensor_name: original aclFormat} for every
# weight that was normalized to ND for IPC sharing.
_IPC_FORMATS_ATTR = "_vllm_ascend_ipc_npu_formats"


def _cast_npu_format(tensor: torch.Tensor, acl_format: int) -> torch.Tensor:
    """Cast an NPU tensor to ``acl_format`` (ND <-> NZ).

    Mirrors ``vllm_ascend.model_loader.netloader.executor.elastic_load``:
    W4A8 packs int8 NZ tiles into int32 via ``Tensor.view`` and TransData
    cannot cast that int32 NZ view directly, so the cast detours through the
    underlying int8 layout.
    """
    contiguous = tensor if tensor.is_contiguous() else tensor.contiguous()
    if contiguous.dtype == torch.int32 and contiguous.numel() > 0 and contiguous.shape[-1] > 0:
        int8_tensor = contiguous.view(torch.int8).contiguous().clone()
        casted_int8 = torch_npu.npu_format_cast(int8_tensor, acl_format)
        packed = casted_int8.view(torch.int32)
        if acl_format == ACL_FORMAT_FRACTAL_ND and not packed.is_contiguous():
            packed = packed.contiguous()
        return packed
    return torch_npu.npu_format_cast(contiguous, acl_format)


def _normalize_model_formats_for_ipc(model: torch.nn.Module) -> None:
    """Convert the daemon's non-ND (e.g. FRACTAL_NZ) weights to ND in place.

    torch_npu's ``_share_npu_`` rejects non-base-format storages. The daemon
    never runs inference, so its resident tensors can live in ND layout at no
    cost; the engine casts back after mapping. Normalizing once here (instead
    of per export) keeps the daemon at ~1x weight memory and every later
    export zero-copy.
    """
    formats: dict[str, int] = {}
    seen: set[int] = set()
    named_tensors = chain(
        model.named_parameters(remove_duplicate=False),
        model.named_buffers(remove_duplicate=False),
    )
    for name, tensor in named_tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        npu_format = get_npu_format_int(tensor)
        if npu_format is None or npu_format == ACL_FORMAT_FRACTAL_ND:
            continue
        tensor.data = _cast_npu_format(tensor, ACL_FORMAT_FRACTAL_ND)
        formats[name] = npu_format
    if formats:
        logger.info(
            "Weight cache daemon: normalized %d non-ND weight tensor(s) to ND for NPU IPC sharing",
            len(formats),
        )
    setattr(model, _IPC_FORMATS_ATTR, formats)


if _wc_protocol is not None:
    # ------------------------------------------------------------------
    # 1. Platform gate: allow NPU when torch_npu supports IPC sharing.
    # ------------------------------------------------------------------
    _orig_check_ipc_platform_support = _wc_protocol.check_ipc_platform_support

    def _check_ipc_platform_support_npu() -> None:
        if current_platform.device_type != "npu":
            _orig_check_ipc_platform_support()
            return
        if not (hasattr(torch.UntypedStorage, "_share_npu_") and hasattr(torch.UntypedStorage, "_new_shared_npu")):
            raise _wc_protocol.UnsupportedPlatformForIPCError(
                "the installed torch_npu does not support NPU IPC storage "
                "sharing (_share_npu_/_new_shared_npu); upgrade torch_npu or "
                "use the default --load-format"
            )

    _wc_protocol.check_ipc_platform_support = _check_ipc_platform_support_npu

    # NPU device uuids are torch_npu._C._CUuuid objects, not str; the socket
    # path derivation and the daemon/engine uuid handshake expect text.
    _orig_get_current_device_uuid = _wc_protocol.get_current_device_uuid

    def _get_current_device_uuid_str() -> str:
        return str(_orig_get_current_device_uuid())

    _wc_protocol.get_current_device_uuid = _get_current_device_uuid_str
    for _uuid_consumer in (
        "vllm.model_executor.model_loader.weight_cache.daemon",
        "vllm.model_executor.model_loader.weight_cache.ipc_loader",
    ):
        _consumer = sys.modules.get(_uuid_consumer)
        if _consumer is not None:
            _consumer.get_current_device_uuid = _get_current_device_uuid_str

    # preload.py and ipc_loader.py bind the function by value at import time;
    # rebind the references in any consumer module that is already imported
    # (e.g. `vllm preload` imports its CLI module before the platform plugin).
    for _consumer_module_name in (
        "vllm.model_executor.model_loader.weight_cache.ipc_loader",
        "vllm.entrypoints.cli.preload",
    ):
        _consumer_module = sys.modules.get(_consumer_module_name)
        if _consumer_module is not None:
            _consumer_module.check_ipc_platform_support = _check_ipc_platform_support_npu

    # ------------------------------------------------------------------
    # 2. TensorEntry: NPU IPC reduce/rebuild + format metadata.
    # ------------------------------------------------------------------
    _TensorEntry = _wc_protocol.TensorEntry
    _orig_from_tensor = _TensorEntry.from_tensor.__func__
    _orig_rebuild = _TensorEntry.rebuild

    def _from_tensor(cls, tensor: torch.Tensor, kind: str):
        if tensor.device.type != "npu":
            return _orig_from_tensor(cls, tensor, kind)
        tensor = tensor.detach()
        # torch_npu replaces torch.multiprocessing.reductions.reduce_tensor
        # with an NPU-aware reducer on import; for NPU storages it returns
        # (rebuild_npu_tensor, args) carrying a real device IPC handle.
        _, ipc_args = torch.multiprocessing.reductions.reduce_tensor(tensor)
        return cls(kind=kind, ipc_args=ipc_args)

    def _rebuild(self, device_index: int) -> torch.Tensor:
        if self.ipc_args is None:
            return _orig_rebuild(self, device_index)
        args = list(self.ipc_args)
        # The device index is args[6] for both CUDA and NPU reduction args;
        # retarget it to the local index since the daemon and the engine may
        # have different ASCEND_RT_VISIBLE_DEVICES mappings.
        args[6] = device_index
        # Imported lazily: torch_npu.multiprocessing is unavailable in CPU
        # environments, but this path only ever runs against NPU IPC args.
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        tensor = rebuild_npu_tensor(*args)
        target_format = getattr(self, "npu_format", None)
        if target_format is not None and target_format != ACL_FORMAT_FRACTAL_ND:
            tensor = _cast_npu_format(tensor, target_format)
        return tensor

    _TensorEntry.from_tensor = classmethod(_from_tensor)
    _TensorEntry.rebuild = _rebuild

    # ------------------------------------------------------------------
    # 3. Daemon: normalize formats once after loading; annotate exports.
    # ------------------------------------------------------------------
    # export_entries is looked up at call time in the daemon child, after
    # the platform plugin has been imported there (lazily via
    # get_attn_backend_cls), so wrapping it covers spawned children. The
    # NZ->ND normalization happens here rather than in a get_model wrapper
    # for the same reason: get_model dispatch happens before this patch is
    # imported in the child.
    _orig_export_entries = _wc_daemon.export_entries

    def _export_entries_with_npu_formats(model: torch.nn.Module):
        if getattr(model, _IPC_FORMATS_ATTR, None) is None:
            _normalize_model_formats_for_ipc(model)
        entries, aliases = _orig_export_entries(model)
        formats = getattr(model, _IPC_FORMATS_ATTR, None)
        if formats:
            for name, npu_format in formats.items():
                entry = entries.get(name)
                if entry is not None:
                    entry.npu_format = npu_format
        return entries, aliases

    _wc_daemon.export_entries = _export_entries_with_npu_formats

    # ------------------------------------------------------------------
    # 4. Route speculative-draft loads to the daemon's draft group.
    #
    # Upstream proposers call get_draft_load_config() explicitly; Ascend's
    # proposer passes load_config=None, which would resolve to the target
    # load config and hit the wrong daemon group. Patch get_model narrowly:
    # only reroute when the loaded model is the draft model, no explicit
    # load config was given, and ipc_cache is in use.
    # ------------------------------------------------------------------
    import vllm.model_executor.model_loader as _model_loader
    from vllm.model_executor.model_loader.utils import get_draft_load_config

    _orig_get_model = _model_loader.get_model

    def _get_model_with_draft_cache_routing(
        *,
        vllm_config,
        model_config=None,
        prefix="",
        load_config=None,
    ):
        if load_config is None and vllm_config.load_config.load_format == "ipc_cache":
            spec = vllm_config.speculative_config
            if spec is not None and model_config is spec.draft_model_config:
                load_config = get_draft_load_config(vllm_config)
        return _orig_get_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
            load_config=load_config,
        )

    _model_loader.get_model = _get_model_with_draft_cache_routing


# ----------------------------------------------------------------------
# 5. vllm snapshot: reject cleanly on Ascend (CRIU + cuda-checkpoint are
#    NVIDIA-only; CANN has no device-state checkpoint equivalent).
# ----------------------------------------------------------------------
def _gate_snapshot_on_npu() -> None:
    try:
        from vllm.snapshot.runtime import LocalSnapshotTools
    except Exception:
        # vllm < 0.31 has no snapshot support; nothing to gate.
        return

    # Both create_snapshot and restore_snapshot start with
    # `LocalSnapshotTools.preflight(...)`; gating the class method is
    # independent of CLI import timing and fires before any state change.
    def _unsupported_preflight(self, action: str, artifact) -> None:
        raise RuntimeError(
            f"vllm snapshot {action} is not supported on Ascend NPU: it "
            "requires CRIU plus the NVIDIA-only cuda-checkpoint plugin to "
            "freeze and restore device state, and CANN provides no "
            "equivalent. Use `vllm preload` (weight cache daemon) for "
            "fast engine restarts instead."
        )

    LocalSnapshotTools.preflight = _unsupported_preflight


_gate_snapshot_on_npu()
