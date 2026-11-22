#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Enable core lookahead block hashing on vLLM v0.33.x for supporting KV connectors."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import vllm.v1.core.kv_cache_coordinator as vllm_kv_cache_coordinator
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

_PATCH_MARKER = "_ascend_lookahead_patched"
_CHILD_LOOKAHEAD_ATTR = "_ascend_num_hash_lookahead_tokens"


def _resolve_connector_class(
    kv_transfer_config: Any,
    *,
    factory_cls: Any = KVConnectorFactory,
    load_plugins_fn: Callable[[], None] | None = None,
) -> Any:
    """Resolve a KV connector class, loading general plugins once if unregistered."""
    get_cls = getattr(factory_cls, "get_connector_class", None)
    if not callable(get_cls):
        return None
    try:
        return get_cls(kv_transfer_config)
    except ValueError:
        if load_plugins_fn is None:
            from vllm.plugins import load_general_plugins

            load_plugins_fn = load_general_plugins
        try:
            load_plugins_fn()
            return get_cls(kv_transfer_config)
        except Exception:
            return None
    except Exception:
        return None


def _call_supports_lookahead(connector_cls: Any, kv_transfer_config: Any) -> bool:
    supports = getattr(connector_cls, "supports_lookahead_block_hashes", None)
    if not callable(supports):
        return False
    try:
        return bool(supports(kv_transfer_config))
    except TypeError:
        # Tolerate test doubles whose classmethod omits kv_transfer_config.
        if len(inspect.signature(supports).parameters) == 0:
            return bool(supports())
        raise


def _patch_connector_capabilities(
    kv_connector_base_cls: Any,
    multi_connector_cls: Any,
    *,
    factory_cls: Any = KVConnectorFactory,
    kv_transfer_config_cls: Any = KVTransferConfig,
    load_plugins_fn: Callable[[], None] | None = None,
) -> bool:
    changed = False
    if kv_connector_base_cls is not None and not hasattr(kv_connector_base_cls, "supports_lookahead_block_hashes"):

        def _base_supports_lookahead(cls: type[Any], kv_transfer_config: Any) -> bool:
            """Whether the connector supports core lookahead hashing."""
            return False

        _base_supports_lookahead.__name__ = "supports_lookahead_block_hashes"
        _base_supports_lookahead.__qualname__ = f"{kv_connector_base_cls.__name__}.supports_lookahead_block_hashes"
        kv_connector_base_cls.supports_lookahead_block_hashes = classmethod(_base_supports_lookahead)
        changed = True

    if multi_connector_cls is not None and "supports_lookahead_block_hashes" not in getattr(
        multi_connector_cls, "__dict__", {}
    ):

        def _multi_supports_lookahead(cls: type[Any], kv_transfer_config: Any) -> bool:
            """Return True only if every configured child connector supports lookahead hashes."""
            if kv_transfer_config is None:
                return False
            extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
            connectors_config = extra_config.get("connectors", [])
            if not connectors_config:
                return False
            engine_id = getattr(kv_transfer_config, "engine_id", None)
            for conn_config in connectors_config:
                child_config = kv_transfer_config_cls(**{"engine_id": engine_id, **conn_config})
                connector_cls = _resolve_connector_class(
                    child_config,
                    factory_cls=factory_cls,
                    load_plugins_fn=load_plugins_fn,
                )
                if not _call_supports_lookahead(connector_cls, child_config):
                    return False
            return True

        _multi_supports_lookahead.__name__ = "supports_lookahead_block_hashes"
        _multi_supports_lookahead.__qualname__ = f"{multi_connector_cls.__name__}.supports_lookahead_block_hashes"
        multi_connector_cls.supports_lookahead_block_hashes = classmethod(_multi_supports_lookahead)
        changed = True

    return changed


def _wrap_multi_connector_child_configs(multi_connector_cls: Any) -> bool:
    if multi_connector_cls is None:
        return False
    orig = getattr(multi_connector_cls, "__dict__", {}).get("_get_connector_classes_and_configs")
    orig_fn = orig.__func__ if isinstance(orig, classmethod) else orig
    if not callable(orig_fn) or getattr(orig_fn, _PATCH_MARKER, False):
        return False

    @functools.wraps(orig_fn)
    def _get_connector_classes_and_configs(cls: type[Any], vllm_config: Any) -> list[tuple[Any, Any]]:
        # Propagate the parent's resolved lookahead window so mixed
        # MultiConnector children cannot re-evaluate to a wider window on v0.33.
        parent_lookahead = getattr(vllm_config, "num_hash_lookahead_tokens", 0)
        result = orig_fn(cls, vllm_config)
        for _, child_vllm_config in result:
            child_dict = getattr(child_vllm_config, "__dict__", None)
            if isinstance(child_dict, dict):
                child_dict[_CHILD_LOOKAHEAD_ATTR] = parent_lookahead
        return result

    setattr(_get_connector_classes_and_configs, _PATCH_MARKER, True)
    multi_connector_cls._get_connector_classes_and_configs = classmethod(_get_connector_classes_and_configs)
    return True


def _patch_vllm_config_lookahead(
    vllm_config_cls: Any,
    multi_connector_cls: Any,
    *,
    factory_cls: Any = KVConnectorFactory,
    load_plugins_fn: Callable[[], None] | None = None,
) -> bool:
    if vllm_config_cls is None or hasattr(vllm_config_cls, "_resolve_num_hash_lookahead_tokens"):
        return False
    existing_prop = getattr(vllm_config_cls, "num_hash_lookahead_tokens", None)
    if not isinstance(existing_prop, property) or existing_prop.fget is None:
        return False
    if getattr(existing_prop.fget, _PATCH_MARKER, False):
        return False

    def _num_hash_lookahead_tokens(self: Any) -> int:
        """Tokens after each block that its hash covers; 0 keeps the legacy hashes."""
        instance_dict = getattr(self, "__dict__", None)
        if isinstance(instance_dict, dict) and _CHILD_LOOKAHEAD_ATTR in instance_dict:
            override = instance_dict[_CHILD_LOOKAHEAD_ATTR]
            if isinstance(override, int) and not isinstance(override, bool):
                return max(0, override)

        speculative_config = getattr(self, "speculative_config", None)
        if (
            speculative_config is None
            or not speculative_config.use_eagle_block_drop()
            or speculative_config.disable_lookahead_block_hashes
        ):
            return 0

        kv_transfer_config = getattr(self, "kv_transfer_config", None)
        if kv_transfer_config is not None:
            connector_cls = _resolve_connector_class(
                kv_transfer_config,
                factory_cls=factory_cls,
                load_plugins_fn=load_plugins_fn,
            )
            if not _call_supports_lookahead(connector_cls, kv_transfer_config):
                return 0
        else:
            cache_config = getattr(self, "cache_config", None)
            if cache_config is None or not cache_config.enable_prefix_caching:
                return 0

        return int(self.num_prefill_lookahead_tokens)

    setattr(_num_hash_lookahead_tokens, _PATCH_MARKER, True)
    vllm_config_cls.num_hash_lookahead_tokens = property(_num_hash_lookahead_tokens)
    _wrap_multi_connector_child_configs(multi_connector_cls)
    return True


def _patch_replay_boundaries(coordinator_module: Any) -> bool:
    if coordinator_module is None:
        return False
    orig_replay = getattr(coordinator_module, "replay_boundaries", None)
    if not callable(orig_replay) or getattr(orig_replay, _PATCH_MARKER, False):
        return False
    if "num_hash_lookahead_tokens" in inspect.signature(orig_replay).parameters:
        return False

    @functools.wraps(orig_replay)
    def _replay_boundaries(
        num_prompt_tokens: int,
        scheduler_block_size: int,
        use_eagle: bool,
        num_hash_lookahead_tokens: int = 0,
    ) -> tuple[int, ...]:
        if not use_eagle and num_hash_lookahead_tokens > 1:
            return (num_prompt_tokens - num_hash_lookahead_tokens,)
        return orig_replay(num_prompt_tokens, scheduler_block_size, use_eagle)

    setattr(_replay_boundaries, _PATCH_MARKER, True)
    coordinator_module.replay_boundaries = _replay_boundaries

    coordinator_cls = getattr(coordinator_module, "KVCacheCoordinator", None)
    if isinstance(coordinator_cls, type):
        orig_cache_blocks = getattr(coordinator_cls, "__dict__", {}).get("cache_blocks")
    else:
        orig_cache_blocks = None
    if (
        isinstance(coordinator_cls, type)
        and callable(orig_cache_blocks)
        and not getattr(orig_cache_blocks, _PATCH_MARKER, False)
    ):

        @functools.wraps(orig_cache_blocks)
        def _cache_blocks(self: Any, request: Any, num_computed_tokens: int) -> None:
            lookahead = getattr(getattr(self, "block_pool", None), "num_hash_lookahead_tokens", 0)
            if getattr(self, "drop_eagle_group_ids", None) or lookahead <= 1:
                orig_cache_blocks(self, request, num_computed_tokens)
                return
            boundaries = coordinator_module.replay_boundaries(
                request.num_prompt_tokens,
                self.scheduler_block_size,
                use_eagle=False,
                num_hash_lookahead_tokens=lookahead,
            )
            for group_id, manager in enumerate(self.single_type_managers):
                if not manager.enable_caching:
                    continue
                manager.cache_blocks(
                    request,
                    self.get_num_cacheable_tokens(num_computed_tokens, group_id),
                    retention_interval=self.retention_interval,
                    replay_boundaries=boundaries,
                )

        setattr(_cache_blocks, _PATCH_MARKER, True)
        target_coordinator_cls: Any = coordinator_cls
        target_coordinator_cls.cache_blocks = _cache_blocks

    return True


def apply_lookahead_block_hashes_patch(
    *,
    vllm_config_cls: Any = VllmConfig,
    kv_connector_base_cls: Any = KVConnectorBase_V1,
    multi_connector_cls: Any = MultiConnector,
    factory_cls: Any = KVConnectorFactory,
    kv_transfer_config_cls: Any = KVTransferConfig,
    coordinator_module: Any = vllm_kv_cache_coordinator,
    load_plugins_fn: Callable[[], None] | None = None,
) -> bool:
    """Apply lookahead block-hash capability negotiation when upstream lacks vllm#64096."""
    changed = _patch_connector_capabilities(
        kv_connector_base_cls,
        multi_connector_cls,
        factory_cls=factory_cls,
        kv_transfer_config_cls=kv_transfer_config_cls,
        load_plugins_fn=load_plugins_fn,
    )
    if _patch_vllm_config_lookahead(
        vllm_config_cls,
        multi_connector_cls,
        factory_cls=factory_cls,
        load_plugins_fn=load_plugins_fn,
    ):
        changed = True
    if _patch_replay_boundaries(coordinator_module):
        changed = True
    return changed


apply_lookahead_block_hashes_patch()
