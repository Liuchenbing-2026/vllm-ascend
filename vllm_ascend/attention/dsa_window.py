# SPDX-License-Identifier: Apache-2.0

from typing import Any

# SparseAttnSharedkvMetadata on A2 only has a verified, process-safe contract
# for the legacy 127/0 band.  With explicit ori sparse indices the attention
# kernel derives the real visible length from the slot list, while metadata is
# only used to partition work in 512-token S2 blocks.
DSPARK_SPARSE_SAS_METADATA_WIN_LEFT = 127
DSPARK_SPARSE_SAS_METADATA_WIN_RIGHT = 0
DSPARK_SPARSE_SAS_METADATA_S2_CAPACITY = 512


def validate_dspark_sparse_sas_metadata_capacity(block_size: int, window_size: int) -> None:
    visible_len_upper = int(window_size) + int(block_size)
    if visible_len_upper > DSPARK_SPARSE_SAS_METADATA_S2_CAPACITY:
        raise ValueError(
            "DSpark explicit sparse visibility exceeds the compatibility "
            "metadata schedule: "
            f"window_size + block_size={visible_len_upper}, "
            f"capacity={DSPARK_SPARSE_SAS_METADATA_S2_CAPACITY}. "
            "Use the PTA fallback or a metadata operator that supports the "
            "expanded window."
        )


def _get_dspark_draft_hf_config(vllm_config: Any) -> Any | None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    return getattr(draft_model_config, "hf_config", None)


def get_dspark_query_block_size(vllm_config: Any) -> int:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = getattr(speculative_config, "num_speculative_tokens", None)
    if num_speculative_tokens:
        return int(num_speculative_tokens)

    draft_hf_config = _get_dspark_draft_hf_config(vllm_config)
    return int(getattr(draft_hf_config, "dspark_block_size", 0) or 0)


def is_dspark_noncausal_draft(vllm_config: Any, common_attn_metadata: Any) -> bool:
    if getattr(common_attn_metadata, "causal", True):
        return False

    speculative_config = getattr(vllm_config, "speculative_config", None)
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark):
        return bool(use_dspark())

    draft_hf_config = _get_dspark_draft_hf_config(vllm_config)
    return bool(getattr(draft_hf_config, "dspark_block_size", 0))


def get_draft_swa_window(
    vllm_config: Any,
    common_attn_metadata: Any,
) -> tuple[int, int]:
    del common_attn_metadata
    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    # DSpark full-draft-block visibility is expressed by explicit sparse slot
    # indices. Do not encode it as a band window on the full paged KV cache.
    return window_size - 1, 0


def get_dspark_sparse_sas_window(
    vllm_config: Any,
    common_attn_metadata: Any,
) -> tuple[int, int]:
    """A2 operator window for DSpark PA_ND + explicit sparse indices.

    ``ori_sparse_indices`` is authoritative for token visibility. The A2
    SparseAttnSharedkv tiler requires the legacy 127/0 attributes for both
    metadata generation and attention; an expanded band poisons the device
    stream with AICPU error 22007 before Python fallback can run.
    """
    if not is_dspark_noncausal_draft(vllm_config, common_attn_metadata):
        return get_draft_swa_window(vllm_config, common_attn_metadata)

    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    block_size = get_dspark_query_block_size(vllm_config)
    if block_size <= 0:
        return window_size - 1, 0
    validate_dspark_sparse_sas_metadata_capacity(block_size, window_size)
    return DSPARK_SPARSE_SAS_METADATA_WIN_LEFT, DSPARK_SPARSE_SAS_METADATA_WIN_RIGHT


def get_dspark_sparse_sas_metadata_window(
    vllm_config: Any,
    common_attn_metadata: Any,
) -> tuple[int, int]:
    """A2-compatible scheduling window for explicit ori sparse indices."""
    if not is_dspark_noncausal_draft(vllm_config, common_attn_metadata):
        return get_draft_swa_window(vllm_config, common_attn_metadata)

    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    block_size = get_dspark_query_block_size(vllm_config)
    if block_size <= 0:
        return window_size - 1, 0
    validate_dspark_sparse_sas_metadata_capacity(block_size, window_size)
    return DSPARK_SPARSE_SAS_METADATA_WIN_LEFT, DSPARK_SPARSE_SAS_METADATA_WIN_RIGHT
