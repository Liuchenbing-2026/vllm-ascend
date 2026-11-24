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

"""Enable lookahead-aligned replay boundaries on vLLM v0.34.0."""

import functools
from typing import Any

import vllm.v1.core.kv_cache_coordinator as vllm_kv_cache_coordinator

from vllm_ascend.utils import vllm_version_is

_PATCH_MARKER = "_ascend_lookahead_patched"


def _patch_replay_boundaries(coordinator_module: Any) -> bool:
    if not vllm_version_is("0.34.0") or coordinator_module is None:
        return False
    orig_replay = getattr(coordinator_module, "replay_boundaries", None)
    if not callable(orig_replay) or getattr(orig_replay, _PATCH_MARKER, False):
        return False

    @functools.wraps(orig_replay)
    def _replay_boundaries(
        num_prompt_tokens: int,
        scheduler_block_size: int,
        use_eagle: bool,
        num_hash_lookahead_tokens: int = 0,
    ) -> tuple[int, ...]:
        if not use_eagle and num_hash_lookahead_tokens > 1:
            return (max(0, num_prompt_tokens - num_hash_lookahead_tokens),)
        return orig_replay(num_prompt_tokens, scheduler_block_size, use_eagle)

    _replay_boundaries.__dict__.pop("__wrapped__", None)
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
    coordinator_module: Any = vllm_kv_cache_coordinator,
) -> bool:
    """Apply lookahead-aligned replay_boundaries wrapper on vLLM v0.34.0."""
    return _patch_replay_boundaries(coordinator_module)


apply_lookahead_block_hashes_patch()
