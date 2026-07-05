#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import vllm_ascend.logger  # noqa: F401


def _ensure_decode_context_world_size_alias():
    """Restore the ``get_decode_context_model_parallel_world_size`` symbol.

    vLLM 0.23 dropped ``get_decode_context_model_parallel_world_size`` from
    ``vllm.distributed`` (only ``get_dcp_group`` / ``get_pcp_group`` remain).
    Several vllm-ascend modules still ``from vllm.distributed import`` the old
    name at module import time, so the ascend plugin dies before the model is
    built. Injecting the alias here (executed via ``vllm_ascend/__init__.py``
    at plugin load, before those imports run) covers all of them with one
    compat shim. Never fatal: on older vLLM the symbol exists (no-op); if the
    dcp group accessor is missing we skip rather than crash plugin load.
    """
    try:
        import vllm.distributed as distributed

        if hasattr(distributed, "get_decode_context_model_parallel_world_size"):
            return

        from vllm.distributed import get_dcp_group

        def get_decode_context_model_parallel_world_size():
            return get_dcp_group().world_size

        distributed.get_decode_context_model_parallel_world_size = (
            get_decode_context_model_parallel_world_size)
    except Exception as e:  # noqa: BLE001 - best-effort compat, never fatal
        from vllm.logger import init_logger
        init_logger(__name__).warning(
            "get_decode_context_model_parallel_world_size alias skipped (%s); "
            "relying on native vllm.distributed exports.", e)


_ensure_decode_context_world_size_alias()

_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. E2E test
    conftest hooks do not run there, so global patches that affect scheduler
    and engine code must also be applied through these plugin entry points.
    """
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector

    register_connector()


def register_model_loader():
    _ensure_global_patch()

    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    from .models import register_model

    register_model()
