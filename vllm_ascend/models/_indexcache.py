# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""IndexCache layer-level decision helpers for DeepSeek V4 / SFA path.

Pure functions, no torch / no vllm deps; safe to unit-test without NPU.

Mirrors vllm/model_executor/models/deepseek_v2.py (>=0d4d33) decision logic
that backs the hf_overrides interface introduced by vllm-ascend PR #8398:

    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
    --hf-overrides '{"use_index_cache": true, "index_topk_pattern": "FSFSFS..."}'
"""


def compute_skip_topk(
    layer_idx: int,
    has_indexer: bool,
    use_index_cache: bool,
    index_topk_freq: int = 1,
    index_topk_pattern: str | None = None,
) -> bool:
    """Decide whether a DSA layer should reuse cached topk indices.

    Args:
        layer_idx: 0-based attention layer index.
        has_indexer: True iff this layer constructed an Indexer (sparse /
            v32 layer with compress_ratio==4).
        use_index_cache: hf_config.use_index_cache; master switch.
        index_topk_freq: hf_config.index_topk_freq; reuse cache 1-of-N layers
            when no explicit pattern is given. freq=1 -> never skip.
        index_topk_pattern: hf_config.index_topk_pattern; per-layer mask
            string; 'S' = skip / reuse cache, anything else = full top-k.

    Returns:
        True iff sfa_v1.AscendSFAImpl should read cached topk_indices on this
        layer; False to take the full indexer_select_post_process path.
    """
    if not has_indexer:
        return False
    if not use_index_cache:
        return False
    if index_topk_pattern is None:
        return max(layer_idx - 1, 0) % index_topk_freq != 0
    if 0 <= layer_idx < len(index_topk_pattern):
        return index_topk_pattern[layer_idx] == "S"
    return False
