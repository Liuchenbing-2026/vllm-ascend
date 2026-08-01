# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/attn_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import vllm
from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup, extract_layer_index

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.utils import AscendDeviceType, calc_split_factor, get_ascend_device_type

_ATTENTION_MASK_BUILDER = None

_V2_UNSUPPORTED_HINT = "Run this model with VLLM_USE_V2_MODEL_RUNNER=0."

# Specs whose page stores one latent vector per token instead of a K/V pair.
# The two dimensions this module reports for them are the nope/rope split of
# that single vector, so a caller must never treat them as independent K and V
# head sizes: that budgets two pages where one was reserved.
# Upstream splits the family across two branches -- MLAAttentionSpec derives
# from FullAttentionSpec, SlidingWindowMLASpec from SlidingWindowSpec -- so no
# single base class covers both. Every Ascend subclass (AscendMLAAttentionSpec,
# AscendSlidingWindowMLASpec, and DeepSeek V4's SWA and compressor-state caches
# through it) inherits one of the two. Extend this tuple, never the individual
# isinstance checks below.
# Membership is narrower than "one vector per token": it also requires that the
# two dimensions BE the nope/rope split, which is what an MLAAttention layer can
# name and what the callers below spend them on. AscendSFAIndexerCacheSpec pages
# also hold one vector per token, but paired with a separate quantization scale
# rather than split into nope and rope, so it is refused up front by
# get_kv_cache_spec instead of being described here.
_SINGLE_LATENT_VECTOR_SPECS: tuple[type[KVCacheSpec], ...] = (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _stores_single_latent_vector(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether one page of this spec holds one latent vector, not a K/V pair."""
    return isinstance(kv_cache_spec, _SINGLE_LATENT_VECTOR_SPECS)


def _uses_dsv4_dsa_layout(kv_cache_spec: KVCacheSpec) -> bool:
    """Whether a spec uses Ascend DSA's single page-backed cache views."""
    return isinstance(kv_cache_spec, (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec)) and (
        kv_cache_spec.model_version == "deepseek_v4"
    )


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Build Ascend-specific KV cache specs for v2 worker patching."""
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = AttentionLayerBase
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)

    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            continue
        if isinstance(attn_module, Attention):
            if spec := attn_module.get_kv_cache_spec(vllm_config):
                kv_cache_spec[layer_name] = spec
            continue
        if isinstance(attn_module, MLAAttention):
            spec = attn_module.get_kv_cache_spec(vllm_config)
            if spec is None:
                continue
            if getattr(attn_module.impl, "fa_quant_layer", False):
                head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                dtype, cache_dtype_str = attn_module.impl.dtype, None
            else:
                head_size = spec.head_size
                dtype = spec.dtype
                cache_dtype_str = spec.cache_dtype_str
            kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                cache_dtype_str=cache_dtype_str,
            )
            continue
        if isinstance(attn_module, MambaBase):
            # A MambaSpec is not an AttentionSpec, so it trips the assert in
            # _allocate_kv_cache below instead of getting the recurrent-state
            # page layout the layer needs.
            raise NotImplementedError(
                f"Recurrent-state layer {layer_name} ({type(attn_module).__name__}) is not supported by the "
                f"v2 model runner on Ascend. {_V2_UNSUPPORTED_HINT}"
            )
        if isinstance(attn_module, CacheOnlyAttentionLayer):
            # HiddenStateCacheSpec pages hold one hidden-state vector per token,
            # while _allocate_kv_cache / _reshape_kv_cache_v2 always carve a
            # page into a K/V pair. v1 reshapes this layer through its own
            # per-role path (_reshape_kv_cache_tensors), which v2 has no
            # equivalent of.
            raise NotImplementedError(
                f"Hidden-state cache layer {layer_name} ({type(attn_module).__name__}) is not supported by the "
                f"v2 model runner on Ascend; this covers the extract_hidden_states drafter. {_V2_UNSUPPORTED_HINT}"
            )
        # Attention layers that are neither Attention nor MLAAttention still own
        # KV state -- DeepSeek V4's DSA contributes DSAAttention plus its SWA,
        # compressor-state and indexer caches, and every one of them must appear
        # here or the model reaches _allocate_kv_cache with no KV cache group.
        # Ask the layer for its own spec, mirroring the v1 runner's fallback
        # branch, and let the layout rules below decide what is buildable.
        spec = attn_module.get_kv_cache_spec(vllm_config)
        if not spec:
            continue
        if isinstance(spec, AscendSFAIndexerCacheSpec):
            # One packed vector per token plus its own scale accounting, not a
            # K/V pair: _allocate_kv_cache splits every page in two, which is
            # a layout this spec cannot describe. Keyed on the spec rather than
            # the layer class so any layer emitting this layout is covered; the
            # only one today is Ascend's AscendMiniMaxM3IndexerCache. DeepSeek
            # V3.2 does NOT arrive here -- upstream DeepseekV32IndexerCache
            # returns a plain MLAAttentionSpec (only the v1 runner rewrites it
            # to AscendSFAIndexerCacheSpec), so it is refused later, by
            # _get_attention_kv_cache_dims during KV cache allocation.
            raise NotImplementedError(
                f"Sparse-attention indexer layer {layer_name} ({type(attn_module).__name__}) is not supported "
                f"by the v2 model runner on Ascend; this covers layers reporting an "
                f"{AscendSFAIndexerCacheSpec.__name__} (MiniMax-M3 today). {_V2_UNSUPPORTED_HINT}"
            )
        kv_cache_spec[layer_name] = spec

    return kv_cache_spec


def get_attn_mask_builder(device: torch.device):
    """Get attention mask builder which only have one instance."""
    global _ATTENTION_MASK_BUILDER
    if _ATTENTION_MASK_BUILDER is None:
        _ATTENTION_MASK_BUILDER = AttentionMaskBuilder(device)
    return _ATTENTION_MASK_BUILDER


def _build_dsa_extra_kwargs(
    *,
    attn_group: AttentionGroup,
    num_reqs_actual: int,
    prefill_ratio_to_sas_metadata: dict[Any, Any],
    decode_ratio_to_sas_metadata: dict[Any, Any],
    common_ratio_to_sas_metadata: dict[Any, Any],
    for_cudagraph_capture: bool,
) -> dict[str, Any]:
    """Collect the kwargs AscendDSAMetadataBuilder.build requires."""
    if for_cudagraph_capture:
        # Capture runs on synthetic batches; give it a throwaway memo so it
        # cannot hand its tensors to the real batches that follow.
        prefill_ratio_to_sas_metadata = {}
        decode_ratio_to_sas_metadata = {}
        common_ratio_to_sas_metadata = {}
    return {
        # The padded request rows have stale block ids, and the builder needs
        # the unpadded count to zero them.
        "num_reqs_actual": num_reqs_actual,
        "prefill_ratio_to_sas_metadata": prefill_ratio_to_sas_metadata,
        "decode_ratio_to_sas_metadata": decode_ratio_to_sas_metadata,
        "common_ratio_to_sas_metadata": common_ratio_to_sas_metadata,
        # The group's spec, not the builder's: create_metadata_builders hands
        # the builder a clone rewritten to the kernel block size.
        "block_size": attn_group.kv_cache_spec.block_size,
    }


def build_attn_metadata(
    *,
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    dcp_local_seq_lens: torch.Tensor | None = None,
    # extra attributes for ascend npus.
    seq_lens_np: np.ndarray | None = None,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    num_computed_tokens_cpu: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    attn_state: Any | None = None,
    graph_pad_size: int = -1,
    num_input_tokens: int = 0,
    num_reqs_actual: int | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    causal: bool | Mapping[int, bool] = True,
) -> dict[str, Any]:
    """Build attention metadata for Ascend NPUs."""
    # TODO(Ronald1995): optimize AscendCommonAttentionMetadata.

    # Sparse-attention builders index by num_input_tokens: DSA, DSA-CP and SFA
    # all slice positions and slot mappings with it, and AscendSFAImpl falls
    # back to it for the topk token count. Leaving it at 0 empties those
    # slices, so it defaults to the token count this call is building for.
    # Only the two upstream draft-metadata callers reach that default --
    # AutoRegressiveSpeculator._build_draft_attn_metadata and DFlash's
    # _prepare_dflash_inputs_to_capture, both of which reach this function
    # through the module-level rebinds in build_attn_metadata_wrapper and
    # patch_dflash_speculator -- and both pass a padded token count.
    # AscendModelState.prepare_attn never reaches it: it always supplies
    # num_input_tokens explicitly, which is what keeps the count padded there,
    # because its num_tokens is the UNPADDED one outside FULL cudagraph mode.
    if num_input_tokens <= 0:
        num_input_tokens = num_tokens

    # seq_lens_np is used for ascend npus, it maybe None in spec_decode case,
    # we fill it with max_seq_len in case `attn_metadata_builder.build` raise
    # an error.
    if seq_lens_np is None:
        seq_lens_np = np.full(num_reqs, max_seq_len, dtype=np.int32)
    seq_lens_cpu = torch.from_numpy(seq_lens_np)[:num_reqs]
    if seq_lens_cpu_upper_bound is None:
        seq_lens_cpu_upper_bound = seq_lens_cpu

    # A DSA model spreads its layers over several KV cache groups, one per
    # compression ratio, whose builders share this scratch: the first builder to
    # run fills it in and the others reuse the decode/prefill split, positions,
    # rope tables and sparse-attention handles instead of recomputing them.
    # It has to be rebuilt on every call because the tensors it holds belong to
    # the batch currently being built.
    prefill_ratio_to_sas_metadata: dict[Any, Any] = {}
    decode_ratio_to_sas_metadata: dict[Any, Any] = {}
    common_ratio_to_sas_metadata: dict[Any, Any] = {}

    attn_metadata: dict[str, Any] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for i, kv_cache_spec in enumerate(kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        # Hybrid drafters can configure causality per KV cache group.
        group_causal = causal if isinstance(causal, bool) else causal.get(i, True)

        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            seq_lens=seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            positions=positions,
            attn_state=attn_state,
            graph_pad_size=graph_pad_size,
            num_input_tokens=num_input_tokens,
            max_seq_len=max_seq_len,
            causal=group_causal,
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(0)
            # Duck-typed: importing the DSA builders here would drag the DSA
            # attention backend and its device ops into the first metadata build
            # of every v2 model, sparse attention or not.
            is_dsa_builder = getattr(attn_metadata_builder, "requires_sparse_attention_kwargs", False)
            # DSA's builder requires kwargs that build_for_cudagraph_capture
            # cannot pass, so it builds the capture metadata the normal way.
            if for_cudagraph_capture and not is_dsa_builder:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                if is_dsa_builder:
                    attn_metadata_extra_kwargs = {
                        **attn_metadata_extra_kwargs,
                        **_build_dsa_extra_kwargs(
                            attn_group=attn_group,
                            num_reqs_actual=num_reqs if num_reqs_actual is None else num_reqs_actual,
                            prefill_ratio_to_sas_metadata=prefill_ratio_to_sas_metadata,
                            decode_ratio_to_sas_metadata=decode_ratio_to_sas_metadata,
                            common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                            for_cudagraph_capture=for_cudagraph_capture,
                        ),
                    }
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata


def build_attn_state(
    vllm_config: VllmConfig,
    seq_lens_np: np.ndarray,
    num_reqs,
    num_scheduled_tokens,
    num_valid_tokens,
):
    """Build attention state for npu's attention backend."""
    if vllm_config.model_config.runner_type == "pooling":
        # An attention-free model has no KV cache group at all, which is the
        # same "nothing was cached" situation as an encoder-only one.
        kv_cache_groups = vllm_config.kv_cache_config.kv_cache_groups
        if not kv_cache_groups or isinstance(kv_cache_groups[0].kv_cache_spec, EncoderOnlyAttentionSpec):
            attn_state = AscendAttentionState.PrefillNoCache
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
    elif np.array_equal(seq_lens_np[:num_reqs], num_scheduled_tokens):
        attn_state = AscendAttentionState.PrefillNoCache
    # We assume it is the decode stage, where prefill occurs
    # but only one token is not hit in cache.
    elif np.all(num_scheduled_tokens == 1):
        attn_state = AscendAttentionState.DecodeOnly
        if vllm_config.speculative_config and vllm_config.speculative_config.method == "mtp":
            # SpecDecoding now supports seq_len=1 and seq_len=2
            # In Prefilling Decoding Disaggregation scenario, SpecDecoding
            # need to supports seq_len=1
            attn_state = AscendAttentionState.SpecDecoding
    # Speculative decoding.
    elif np.all(num_valid_tokens == 1):
        if vllm_config.speculative_config and vllm_config.speculative_config.method == "mtp":
            attn_state = AscendAttentionState.SpecDecoding
        else:
            attn_state = AscendAttentionState.ChunkedPrefill
    # splitfuse
    elif vllm_config.scheduler_config.enable_chunked_prefill:
        attn_state = AscendAttentionState.ChunkedPrefill
    else:
        attn_state = AscendAttentionState.PrefillCacheHit
    return attn_state


def _get_layer_kv_cache_specs(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
    for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
        group_spec = group_kv_cache_spec.kv_cache_spec
        for layer_name in group_kv_cache_spec.layer_names:
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
            else:
                layer_kv_cache_spec[layer_name] = group_spec
    return layer_kv_cache_spec


def _get_attention_kv_cache_dims(layer_name: str, kv_cache_spec: AttentionSpec) -> tuple[int, int]:
    if _stores_single_latent_vector(kv_cache_spec):
        attn_layers = get_layers_from_vllm_config(get_current_vllm_config(), AttentionLayerBase, [layer_name])
        attn_layer = attn_layers[layer_name]
        if isinstance(attn_layer, MLAAttention):
            # MLA stores one latent cache whose two halves are the nope (K) and
            # rope (V) parts; only the layer knows how the head size splits.
            return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim
        # Only an MLAAttention layer can name the nope/rope split, and the pair
        # returned here is spent as two independent K and V extents: callers
        # budget one page per element and split each raw tensor accordingly.
        # There is no fallback that keeps that sound for a spec whose page holds
        # a single latent vector, so any non-MLAAttention layer holding one has
        # to be refused rather than routed to the generic tail below. Reached by
        # DeepSeek V4's DSAAttention, its SWA and compressor-state caches, and
        # DeepSeek V3.2's DeepseekV32IndexerCache -- per-role cache tuples the
        # v2 allocator cannot build either way.
        raise NotImplementedError(
            f"KV cache layer {layer_name} ({type(attn_layer).__name__}) reports a single-latent-vector "
            f"{type(kv_cache_spec).__name__} without being an MLAAttention layer, which the v2 model runner "
            f"on Ascend cannot allocate. {_V2_UNSUPPORTED_HINT}"
        )

    head_size_v = kv_cache_spec.head_size_v if hasattr(kv_cache_spec, "head_size_v") else kv_cache_spec.head_size
    return kv_cache_spec.head_size, head_size_v


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    shared_layers: dict[str, str],
    device: torch.device,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """
    Initialize the KV cache buffer with the correct size. The buffer needs to be
    reshaped to the desired shape before being used by the models.

    NOTE: To support prefill disaggregation, we need to split kvcache tensor
    into k_cache and v_cache, and the addr of both are aligned by 2M.

    Args:
        kv_cache_config: The KV cache config
        device: The device
    Returns:
        dict[str, tuple[torch.Tensor, torch.Tensor]]: A map between layer names
            to their corresponding memory buffer for K cache and V cache
    """
    vllm_config = get_current_vllm_config()

    # init kv cache tensors
    kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    # prefill disaggregation need the addr of cache tensor be aligned with 2M
    alignment = 2 * 1024 * 1024
    layer_kv_cache_spec = _get_layer_kv_cache_specs(kv_cache_config)
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if len(kv_cache_tensor.shared_by) == 0:
            continue

        if getattr(kv_cache_tensor, "block_stride", 0) > 0:
            raise NotImplementedError(
                "Packed block-stride KV cache tensors are not supported by Ascend's split K/V allocator."
            )

        # NOTE: We need to init k_cache tensor (nope cache tensor in mla) and
        # v_cache tensor (rope cache tensor in mla) separately to support
        # prefill disaggregation, as it only supports the 0-dim of kv_cache is
        # `num_blocks`.
        # For deepseek mla, we need to spilt cache tensor accrodding to the nope
        # head dim and rope head dim.
        example_layer_name = kv_cache_tensor.shared_by[0]
        example_kv_cache_spec = layer_kv_cache_spec[example_layer_name]
        assert isinstance(example_kv_cache_spec, AttentionSpec)

        dsa_layout_flags = [
            _uses_dsv4_dsa_layout(layer_kv_cache_spec[layer_name]) for layer_name in kv_cache_tensor.shared_by
        ]
        if any(dsa_layout_flags):
            assert all(dsa_layout_flags), "A shared KV cache tensor cannot mix DSV4 DSA and conventional K/V layouts."
            if vllm_config.kv_transfer_config is None:
                tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
            else:
                tensor = torch.zeros(kv_cache_tensor.size + alignment, dtype=torch.int8, device=device)
                tensor = _align_memory(tensor, alignment)[: kv_cache_tensor.size]
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor
            continue

        k_dim, v_dim = _get_attention_kv_cache_dims(example_layer_name, example_kv_cache_spec)
        assert k_dim > 0 and v_dim > 0
        kv_head_dim_list = [k_dim, v_dim]
        if enable_fa_quant(vllm_config):
            k_tensor_split_factor, v_tensor_split_factor = vllm_config.quant_config.get_kv_quant_split_factor(
                example_layer_name, kv_head_dim_list
            )
        else:
            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor(kv_head_dim_list)
        k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
        v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)

        if vllm_config.kv_transfer_config is None:
            k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=device)
            v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=device)
        else:
            k_tensor = torch.zeros(k_tensor_size + alignment, dtype=torch.int8, device=device)
            v_tensor = torch.zeros(v_tensor_size + alignment, dtype=torch.int8, device=device)
            k_tensor = _align_memory(k_tensor, alignment)[:k_tensor_size]
            v_tensor = _align_memory(v_tensor, alignment)[:v_tensor_size]
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == (kv_cache_raw_tensors.keys() | shared_layers.keys()), (
        "Some layers are not correctly initialized"
    )

    return kv_cache_raw_tensors


def _reshape_dsv4_dsa_cache(
    raw_tensor: torch.Tensor,
    kv_cache_spec: AttentionSpec,
    backend: Any,
) -> list[torch.Tensor]:
    """Build DSA's page-strided data/scale views from one byte backing."""
    assert raw_tensor.dtype == torch.int8
    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
    cache_shape = backend.get_kv_cache_shape(
        num_blocks,
        kv_cache_spec.block_size,
        kv_cache_spec.num_kv_heads,
        kv_cache_spec.head_size,
    )
    shapes = [cache_shape]
    dtypes = [kv_cache_spec.dtype]
    overlap_full_cache = False

    scale_dim = getattr(kv_cache_spec, "scale_dim", 0)
    if scale_dim:
        scale_dtype = kv_cache_spec.scale_dtype
        scale_shape = backend.get_kv_cache_shape(
            num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            scale_dim,
        )
        shapes.append(scale_shape)
        dtypes.append(scale_dtype)
        if get_ascend_device_type() is AscendDeviceType.A5:
            full_head_size = kv_cache_spec.head_size + scale_dim * get_dtype_size(scale_dtype)
            shapes.append(
                backend.get_kv_cache_shape(
                    num_blocks,
                    kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads,
                    full_head_size,
                )
            )
            dtypes.append(kv_cache_spec.dtype)
            overlap_full_cache = True

    views: list[torch.Tensor] = []
    base_offset_bytes = raw_tensor.storage_offset() * raw_tensor.element_size()
    storage_offset_bytes = base_offset_bytes
    for index, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        if overlap_full_cache and index == 2:
            storage_offset_bytes = base_offset_bytes
        dtype_size = get_dtype_size(dtype)
        assert kv_cache_spec.page_size_bytes % dtype_size == 0
        assert storage_offset_bytes % dtype_size == 0
        contiguous_stride = torch.empty(shape, device="meta").stride()
        views.append(
            torch.as_strided(
                raw_tensor.view(dtype),
                size=shape,
                stride=(kv_cache_spec.page_size_bytes // dtype_size, *contiguous_stride[1:]),
                storage_offset=storage_offset_bytes // dtype_size,
            )
        )
        storage_offset_bytes += contiguous_stride[0] * dtype_size
    return views


def _reshape_kv_cache_v2(
    attn_groups: Sequence[AttentionGroup],
    kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
    kv_cache_config: "KVCacheConfig | None" = None,
) -> dict[str, Any]:
    vllm_config = get_current_vllm_config()
    is_kv_consumer = (
        vllm_config.kv_transfer_config.is_kv_consumer if vllm_config.kv_transfer_config is not None else False
    )

    kv_caches: dict[str, Any] = {}
    for group in attn_groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        kv_cache_spec = group.kv_cache_spec
        if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
            kernel_block_size = kv_cache_spec.storage_block_size
        else:
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]

        for layer_name in group.layer_names:
            if layer_name in shared_kv_cache_layers:
                continue

            assert isinstance(kv_cache_spec, AttentionSpec)

            raw_cache = kv_cache_raw_tensors[layer_name]
            if _uses_dsv4_dsa_layout(kv_cache_spec):
                assert isinstance(raw_cache, torch.Tensor)
                kv_caches[layer_name] = _reshape_dsv4_dsa_cache(raw_cache, kv_cache_spec, group.backend)
                continue

            assert isinstance(raw_cache, tuple) and len(raw_cache) == 2
            raw_k_tensor, raw_v_tensor = raw_cache
            assert raw_k_tensor is not None
            assert raw_v_tensor is not None
            sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
            assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
            num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes

            num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
            kernel_num_blocks = num_blocks * num_blocks_per_kv_block

            kv_cache_shape = group.backend.get_kv_cache_shape(
                kernel_num_blocks,
                kernel_block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype,
            )

            if not _stores_single_latent_vector(kv_cache_spec):
                k_shape = kv_cache_shape[1:]
                if hasattr(kv_cache_spec, "head_size_v"):
                    v_shape = (*kv_cache_shape[1:-1], kv_cache_spec.head_size_v)
                else:
                    v_shape = k_shape
            else:
                mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                k_dim, v_dim = _get_attention_kv_cache_dims(layer_name, kv_cache_spec)
                k_shape = (mla_num_blocks, mla_block_size, num_kv_heads, k_dim)
                v_shape = (mla_num_blocks, mla_block_size, num_kv_heads, v_dim)

            k_cache_dtype = v_cache_dtype = kv_cache_spec.dtype
            if is_kv_consumer and enable_fa_quant(vllm_config):
                k_cache_dtype, v_cache_dtype = vllm_config.quant_config.get_kv_quant_dtype(
                    layer_name, kv_cache_spec.dtype, vllm_config.model_config
                )

            k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
            v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)
            kv_caches[layer_name] = (k_cache, v_cache)

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    return kv_caches


def bind_kv_cache(
    kv_caches: dict[str, Any],
    forward_context: dict[str, Any],
    runner_kv_caches: list[Any],
    num_attn_module: int = 1,
) -> None:
    """Bind every cache-only module, including multiple modules per layer."""
    assert len(runner_kv_caches) == 0

    index_to_names: defaultdict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index_to_names[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index_to_names):
        for layer_name in sorted(index_to_names[layer_index]):
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        layer = forward_context[layer_name]
        bind = getattr(layer, "bind_kv_cache", None)
        if callable(bind):
            # Current vLLM lets layers unpack raw cache allocations here.
            bind(kv_cache)
        else:
            # vLLM 0.26 binds the already-shaped cache by direct assignment;
            # AttentionLayerBase did not provide bind_kv_cache yet.
            layer.kv_cache = kv_cache


_BUILD_ATTN_METADATA_MODULE = vllm.v1.worker.gpu.spec_decode.speculator


@contextmanager
def build_attn_metadata_wrapper():
    """Context manager to override attention metadata building for Ascend NPUs."""
    original_func = _BUILD_ATTN_METADATA_MODULE.build_attn_metadata
    try:
        _BUILD_ATTN_METADATA_MODULE.build_attn_metadata = build_attn_metadata
        yield
    finally:
        _BUILD_ATTN_METADATA_MODULE.build_attn_metadata = original_func
