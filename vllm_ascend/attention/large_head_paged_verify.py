# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Paged verify and decode backend for head_dim=512 speculative attention.

On A2/A3 hardware, ``npu_fused_infer_attention_score`` in TND layout does not
support ``head_dim == 512``, while ``torch_npu._npu_paged_attention`` assumes a
single query token per batch row and does not support multi-query speculative
verification natively without per-token row expansion.

For decoder self-attention layers with ``head_dim == 512`` and
``sliding_window is None`` when speculative decoding is enabled on devices
with ``HardwareCapability.PAGED_ATTENTION``, this backend:

* In ``FULL_DECODE_ONLY`` (and eager) mode, allocates an Hkv-major
  (``[num_blocks, num_kv_heads, block_size, head_dim]``) contiguous KV cache
  and runs uniform verify/decode steps through
  ``npu_fused_infer_attention_score_v2`` in BNSD paged layout with per-request
  ``block_table``, explicit ``[B, 1, qlen, W]`` causal bool mask, and
  ``sparse_mode=0``.
* In ``CUDAGraphMode.FULL`` mode, keeps the natural LBNHC KV cache layout and
  expands multi-query rows into one-query-per-row metadata for
  ``full_graph_pa``.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch_npu
import vllm.envs as envs_vllm

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import (
    FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE,
    expand_pa_per_token_rows,
    in_full_cudagraph_mode,
)
from vllm_ascend.compilation.updatable_graph import (
    get_capture_resource,
    register_task,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

_FIA_V2_WORKSPACE_KEY = "npu_fused_infer_attention_score_v2.workspace"
_MAX_MASK_CACHE_ENTRIES = 8


def _supports_large_head_paged_path() -> bool:
    """Return True on devices that use A2/A3 paged attention for 512-dim heads."""
    return get_current_hardware_profile().supports(HardwareCapability.PAGED_ATTENTION)


def _uniform_query_len(actual_seq_lengths_q: list[int] | None) -> int | None:
    """Return the per-request query length if uniform and positive across the batch."""
    if not actual_seq_lengths_q:
        return None
    prev = 0
    query_len = actual_seq_lengths_q[0]
    if query_len <= 0:
        return None
    for end in actual_seq_lengths_q:
        if end - prev != query_len:
            return None
        prev = end
    return query_len


def build_paged_verify_mask(
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    block_size: int,
    device: torch.device,
    *,
    kpos: torch.Tensor | None = None,
    qpos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the ``[B, 1, k+1, W]`` bool verify mask for BNSD paged attention.

    Query row ``j`` (``0..k``) of request ``b`` attends key positions
    ``0..seq_lens[b] - (k + 1) + j`` (inclusive), where ``True`` means masked
    out. Real rows satisfy ``seq_lens[b] >= k + 1``; ``clamp_min(0)`` keeps key
    slot 0 unmasked on padded/dummy rows with ``seq_lens[b] < k + 1`` so
    softmax never observes an all-masked row.
    """
    qlen = num_speculative_tokens + 1
    batch_size = seq_lens.shape[0]
    width = block_table.shape[1] * block_size
    if kpos is None:
        kpos = torch.arange(width, dtype=torch.int32, device=device).view(1, 1, 1, width)
    if qpos is None:
        qpos = torch.arange(qlen, dtype=torch.int32, device=device).view(1, 1, qlen, 1)
    seqv = seq_lens.to(device=device, dtype=torch.int32).view(batch_size, 1, 1, 1)
    last_allowed = (seqv - qlen + qpos).clamp_min(0)
    return kpos > last_allowed


def _can_use_large_head_bnsd_verify(impl: Any, attn_metadata: Any, num_tokens: int) -> bool:
    """Return True when the layer geometry and query count fit BNSD paged verify/decode."""
    if (
        getattr(impl, "head_size", None) != FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
        or getattr(impl, "sliding_window", None) is not None
        or not _supports_large_head_paged_path()
    ):
        return False
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None or seq_lens.shape[0] <= 0 or num_tokens % seq_lens.shape[0] != 0:
        return False
    query_len = num_tokens // seq_lens.shape[0]
    spec_cfg = getattr(getattr(impl, "vllm_config", None), "speculative_config", None)
    verify_len = spec_cfg.num_speculative_tokens + 1 if spec_cfg is not None else None
    if _EXTRA_CTX.is_draft_model:
        return query_len == 1 or (verify_len is not None and query_len == verify_len)
    return verify_len is not None and query_len == verify_len


@dataclass
class AscendLargeHeadPagedMetadata(AscendMetadata):
    """Per-layer metadata carrying the BNSD verify mask or FULL PA toggle."""

    paged_verify_mask: torch.Tensor | None = None
    use_pa_graph: bool = False


def large_head_verify_graph_params(
    layer_name: str | None,
    metadata: AscendLargeHeadPagedMetadata,
) -> dict[str, Any]:
    """Return replay-time graph update kwargs for a head_dim=512 verify/draft task."""
    if getattr(metadata, "use_pa_graph", False):
        # FULL mode may have expanded block_tables to one row per query token.
        num_pa_rows = metadata.seq_lens.shape[0] if metadata.seq_lens is not None else 0
        block_table = (
            metadata.block_tables[:num_pa_rows]
            if num_pa_rows > 0 and metadata.block_tables is not None
            else metadata.block_tables
        )
        return {
            "block_table": block_table,
            "context_lens": metadata.seq_lens,
        }
    num_reqs = len(metadata.actual_seq_lengths_q) if metadata.actual_seq_lengths_q else 0
    block_table = (
        metadata.block_tables[:num_reqs]
        if num_reqs > 0 and metadata.block_tables is not None
        else metadata.block_tables
    )
    query_len = int(metadata.actual_seq_lengths_q[0]) if metadata.actual_seq_lengths_q else 0
    seq_lens_list = (
        metadata.seq_lens_list[:num_reqs]
        if num_reqs > 0 and metadata.seq_lens_list is not None
        else metadata.seq_lens_list
    )
    params: dict[str, Any] = {
        "block_table": block_table,
        "actual_seq_kvlen": seq_lens_list,
    }
    verify_mask = metadata.paged_verify_mask
    if verify_mask is not None and verify_mask.shape[2] == query_len:
        params["atten_mask"] = verify_mask
    elif query_len > 1:
        raise RuntimeError(
            "Large-head paged verify replay requires a matching paged_verify_mask in the runtime metadata of "
            f"layer {layer_name!r} (query_len={query_len}); uniform k+1-query steps must attach it during build()."
        )
    return params


def large_head_decode_graph_params(
    metadata: AscendLargeHeadPagedMetadata,
    seq_lens: list[int],
) -> dict[str, Any]:
    """Return replay-time graph update kwargs for a draft head_dim=512 decode step."""
    num_reqs = len(seq_lens)
    block_table = (
        metadata.block_tables[:num_reqs]
        if num_reqs > 0 and metadata.block_tables is not None
        else metadata.block_tables
    )
    if getattr(metadata, "use_pa_graph", False):
        return {
            "block_table": block_table,
            "context_lens": torch.tensor(seq_lens, dtype=torch.int32, device=metadata.seq_lens.device),
        }
    return {
        "block_table": block_table,
        "actual_seq_kvlen": seq_lens,
    }


@dataclass(frozen=True, slots=True)
class LargeHeadPagedVerifyParamProvider:
    """Graph parameter provider keyed solely by layer name for SharedSource matching."""

    layer_name: str | None

    def resolve(self, attn_metadata: dict[str, AscendLargeHeadPagedMetadata]) -> dict[str, Any]:
        return large_head_verify_graph_params(self.layer_name, attn_metadata[self.layer_name])  # type: ignore[index]


class AscendLargeHeadPagedMetadataBuilder(AscendAttentionMetadataBuilder):
    """Metadata builder for head_dim=512 non-sliding speculative attention layers."""

    metadata_cls: type[AscendLargeHeadPagedMetadata] = AscendLargeHeadPagedMetadata

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._kpos_cache: dict[int, torch.Tensor] = {}
        self._qpos_cache: dict[int, torch.Tensor] = {}

    def _get_cached_kpos(self, width: int) -> torch.Tensor:
        kpos = self._kpos_cache.get(width)
        if kpos is None:
            if len(self._kpos_cache) >= _MAX_MASK_CACHE_ENTRIES:
                self._kpos_cache.clear()
            kpos = torch.arange(width, dtype=torch.int32, device=self.device).view(1, 1, 1, width)
            self._kpos_cache[width] = kpos
        return kpos

    def _get_cached_qpos(self, qlen: int) -> torch.Tensor:
        qpos = self._qpos_cache.get(qlen)
        if qpos is None:
            if len(self._qpos_cache) >= _MAX_MASK_CACHE_ENTRIES:
                self._qpos_cache.clear()
            qpos = torch.arange(qlen, dtype=torch.int32, device=self.device).view(1, 1, qlen, 1)
            self._qpos_cache[qlen] = qpos
        return qpos

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> AscendLargeHeadPagedMetadata:
        metadata = cast(
            AscendLargeHeadPagedMetadata,
            super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build),
        )
        if getattr(self.kv_cache_spec, "head_size", None) != FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE:
            return metadata
        if in_full_cudagraph_mode(self.vllm_config):
            metadata.use_pa_graph = True
            expand_pa_per_token_rows(metadata)
            return metadata
        if self.speculative_config is None or metadata.block_tables is None:
            return metadata
        verify_len = self.speculative_config.num_speculative_tokens + 1
        if _uniform_query_len(metadata.actual_seq_lengths_q) != verify_len:
            return metadata
        num_reqs = len(metadata.actual_seq_lengths_q)
        block_table = metadata.block_tables[:num_reqs]
        width = block_table.shape[1] * self.kv_cache_spec.block_size
        metadata.paged_verify_mask = build_paged_verify_mask(
            metadata.seq_lens[:num_reqs],
            block_table,
            self.speculative_config.num_speculative_tokens,
            self.kv_cache_spec.block_size,
            self.device,
            kpos=self._get_cached_kpos(width),
            qpos=self._get_cached_qpos(verify_len),
        )
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs) -> AscendLargeHeadPagedMetadata:
        # Skip mask construction on the capture stream because it needs a
        # seq_lens H2D copy; graph capture binds an all-False placeholder instead.
        metadata = cast(
            AscendLargeHeadPagedMetadata,
            AscendAttentionMetadataBuilder.build(
                self,
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            ),
        )
        if (
            getattr(self.kv_cache_spec, "head_size", None) == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            and in_full_cudagraph_mode(self.vllm_config)
        ):
            metadata.use_pa_graph = True
            expand_pa_per_token_rows(metadata)
        return metadata


class AscendLargeHeadPagedAttentionBackendImpl(AscendAttentionBackendImpl):
    """Ascend attention impl adding BNSD FIA-v2 paged verify/decode for 512-dim heads."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.full_cudagraph = in_full_cudagraph_mode(getattr(self, "vllm_config", None))
        if (
            not self.full_cudagraph
            and not self.use_bnsd_kv_cache
            and self.head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            and _supports_large_head_paged_path()
        ):
            self.use_bnsd_kv_cache = True

    def _should_capture_bnsd_verify(self, attn_metadata: Any, num_tokens: int) -> bool:
        return (
            bool(_EXTRA_CTX.capturing)
            and not self.full_cudagraph
            and _can_use_large_head_bnsd_verify(self, attn_metadata, num_tokens)
        )

    def _maybe_init_kv_cache(self, kv_cache: Any = None) -> None:
        if self.key_cache is None and kv_cache is not None and (
            (isinstance(kv_cache, torch.Tensor) and kv_cache.dim() > 0 and kv_cache.shape[0] == 2)
            or (isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2)
        ):
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

    def _should_eager_bnsd_verify(self, attn_metadata: Any, num_tokens: int, kv_cache: Any = None) -> bool:
        self._maybe_init_kv_cache(kv_cache)
        if (
            _EXTRA_CTX.capturing
            or self.full_cudagraph
            or not self.use_bnsd_kv_cache
            or self.key_cache is None
            or self.value_cache is None
            or not (self.key_cache.is_contiguous() and self.value_cache.is_contiguous())
        ):
            return False
        if (
            not getattr(attn_metadata, "causal", True)
            or getattr(attn_metadata, "attn_state", None) == AscendAttentionState.PrefillNoCache
            or getattr(attn_metadata, "num_prefills", 0) > 0
            or getattr(attn_metadata, "block_tables", None) is None
        ):
            return False
        if not _can_use_large_head_bnsd_verify(self, attn_metadata, num_tokens):
            return False
        actual_seq_lengths_q = getattr(attn_metadata, "actual_seq_lengths_q", None)
        qlen = _uniform_query_len(actual_seq_lengths_q)
        if qlen is None or actual_seq_lengths_q is None or num_tokens != qlen * len(actual_seq_lengths_q):
            return False
        num_reqs = len(actual_seq_lengths_q)
        if attn_metadata.block_tables.shape[0] < num_reqs:
            return False
        seq_lens_list = getattr(attn_metadata, "seq_lens_list", None) or []
        if len(seq_lens_list) < num_reqs or any(int(s) <= 0 for s in seq_lens_list[:num_reqs]):
            return False
        if qlen > 1:
            verify_mask = getattr(attn_metadata, "paged_verify_mask", None)
            if verify_mask is None or verify_mask.shape[0] != num_reqs or verify_mask.shape[2] != qlen:
                return False
        return True

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        self._maybe_init_kv_cache(kv_cache)
        if self._should_capture_bnsd_verify(attn_metadata, query.shape[0]):
            return self.full_graph_fia_v2_paged_verify(query, attn_metadata, output)
        if (
            _EXTRA_CTX.capturing
            and self.head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            and self.sliding_window is None
            and _supports_large_head_paged_path()
        ):
            return self.full_graph_pa(query, attn_metadata, output)
        if self._should_eager_bnsd_verify(attn_metadata, query.shape[0], kv_cache):
            return self._eager_fia_v2_paged_verify(query, attn_metadata, output)
        return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._should_capture_bnsd_verify(attn_metadata, query.shape[0]):
            assert output is not None
            return self.full_graph_fia_v2_paged_verify(query, attn_metadata, output)
        if self._should_eager_bnsd_verify(attn_metadata, query.shape[0]):
            assert output is not None
            return self._eager_fia_v2_paged_verify(query, attn_metadata, output)
        return super().forward_paged_attention(query, attn_metadata, output)

    def _prepare_bnsd_paged_inputs(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> tuple[int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_bnsd_kv_cache:
            raise RuntimeError(
                "Large-head paged verify requires BNSD KV cache layout "
                "(AscendLargeHeadPagedBackend.get_kv_cache_shape)."
            )
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("Large-head paged verify requires initialized KV caches.")
        if not (self.key_cache.is_contiguous() and self.value_cache.is_contiguous()):
            raise RuntimeError("Large-head paged verify requires contiguous BNSD KV cache tensors.")

        num_tokens = query.shape[0]
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        num_reqs = len(actual_seq_lengths_q) if actual_seq_lengths_q else 0
        if num_reqs == 0 or num_tokens % num_reqs != 0:
            raise RuntimeError(
                f"Large-head paged verify requires num_tokens divisible by request count; "
                f"got {num_tokens} tokens for {num_reqs} requests."
            )
        qlen = num_tokens // num_reqs
        if [qlen * (i + 1) for i in range(num_reqs)] != actual_seq_lengths_q:
            raise RuntimeError(
                f"Large-head paged verify requires uniform query lengths, got {actual_seq_lengths_q}."
            )

        _, _, block_size, _ = self.key_cache.shape
        if attn_metadata.block_tables is None:
            raise RuntimeError("Large-head paged verify requires non-None block_tables.")
        block_table = attn_metadata.block_tables[:num_reqs]
        q_bnsd = (
            query[:num_tokens]
            .view(num_reqs, qlen, self.num_heads, self.head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        verify_mask = getattr(attn_metadata, "paged_verify_mask", None)
        if verify_mask is None or verify_mask.shape[0] != num_reqs or verify_mask.shape[2] != qlen:
            verify_mask = torch.zeros(
                (num_reqs, 1, qlen, block_table.shape[1] * block_size),
                dtype=torch.bool,
                device=query.device,
            )
        return num_tokens, num_reqs, qlen, block_size, q_bnsd, block_table, verify_mask

    def full_graph_fia_v2_paged_verify(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Capture a 512-dim paged verify or draft decode step as FIA-v2 BNSD."""
        num_tokens, num_reqs, qlen, block_size, q_bnsd, block_table, verify_mask = self._prepare_bnsd_paged_inputs(
            query, attn_metadata
        )
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        attn_output = torch.empty(
            num_reqs,
            self.num_heads,
            qlen,
            self.head_size,
            dtype=query.dtype,
            device=query.device,
        )
        actual_seq_qlen = [qlen] * num_reqs
        actual_seq_kvlen = attn_metadata.seq_lens_list[:num_reqs]

        workspace = get_capture_resource(
            _FIA_V2_WORKSPACE_KEY,
            lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=q_bnsd,
                key=self.key_cache,
                value=self.value_cache,
                atten_mask=verify_mask,
                block_table=block_table,
                input_layout="BNSD",
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=0,
            ),
            use_max_workspace=True,
        )
        register_task(
            torch_npu.npu_fused_infer_attention_score_v2.out,
            {
                "query": q_bnsd,
                "key": self.key_cache,
                "value": self.value_cache,
                "atten_mask": verify_mask,
                "block_table": block_table,
                "input_layout": "BNSD",
                "block_size": block_size,
                "actual_seq_qlen": actual_seq_qlen,
                "actual_seq_kvlen": actual_seq_kvlen,
                "num_key_value_heads": self.num_kv_heads,
                "num_query_heads": self.num_heads,
                "sparse_mode": 0,
                "softmax_scale": self.scale,
                "workspace": workspace,
                "out": [attn_output, softmax_lse],
            },
            LargeHeadPagedVerifyParamProvider(self._layer_name),
        )
        output_view = output[:num_tokens]
        output_view.copy_(attn_output.permute(0, 2, 1, 3).reshape(output_view.shape))
        return output

    def _eager_fia_v2_paged_verify(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Execute a non-captured uniform 512-dim verify/decode step via FIA-v2 BNSD."""
        num_tokens, num_reqs, qlen, block_size, q_bnsd, block_table, verify_mask = self._prepare_bnsd_paged_inputs(
            query, attn_metadata
        )
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=q_bnsd,
            key=self.key_cache,
            value=self.value_cache,
            atten_mask=verify_mask,
            block_table=block_table,
            input_layout="BNSD",
            block_size=block_size,
            actual_seq_qlen=[qlen] * num_reqs,
            actual_seq_kvlen=attn_metadata.seq_lens_list[:num_reqs],
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=0,
            softmax_scale=self.scale,
        )
        output_view = output[:num_tokens]
        output_view.copy_(attn_output.permute(0, 2, 1, 3).reshape(output_view.shape))
        return output


class AscendLargeHeadPagedBackend(AscendAttentionBackend):
    """Backend for head_dim=512 non-sliding layers under speculative decoding."""

    @staticmethod
    def get_impl_cls() -> type[AscendLargeHeadPagedAttentionBackendImpl]:
        return AscendLargeHeadPagedAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendLargeHeadPagedMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        if (
            head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            and _supports_large_head_paged_path()
            and not in_full_cudagraph_mode()
        ):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        if envs_vllm.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)


__all__ = [
    "AscendLargeHeadPagedAttentionBackendImpl",
    "AscendLargeHeadPagedBackend",
    "AscendLargeHeadPagedMetadata",
    "AscendLargeHeadPagedMetadataBuilder",
    "LargeHeadPagedVerifyParamProvider",
    "build_paged_verify_mask",
    "large_head_decode_graph_params",
    "large_head_verify_graph_params",
]
