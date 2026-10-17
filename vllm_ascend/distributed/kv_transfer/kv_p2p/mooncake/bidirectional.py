# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Configuration for opt-in, full-attention multi-turn KV reuse."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import FullAttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class BidirectionalKVConfig:
    enabled: bool = False
    decoder_ttl_secs: float = 480
    recompute_threshold: int = 64

    @classmethod
    def from_config(cls, config: "VllmConfig", caches: "KVCacheConfig") -> "BidirectionalKVConfig":
        assert config.kv_transfer_config is not None
        extra = config.kv_transfer_config.kv_connector_extra_config
        enabled = extra.get("bidirectional_kv_xfer", False)
        if not isinstance(enabled, bool):
            raise ValueError("bidirectional_kv_xfer must be a boolean")
        if not enabled:
            return cls()
        ttl = extra.get("decoder_kv_blocks_ttl", 480)
        threshold = extra.get("kv_recompute_threshold", 64)
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("decoder_kv_blocks_ttl must be finite and positive")
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
            raise ValueError("kv_recompute_threshold must be a nonnegative integer")
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or config.speculative_config is not None
            or not caches.kv_cache_groups
            or any(type(group.kv_cache_spec) is not FullAttentionSpec for group in caches.kv_cache_groups)
        ):
            raise ValueError("bidirectional Mooncake requires TP/PP/DP/PCP/DCP=1, full attention, and no speculation")
        return cls(True, float(ttl), threshold)
