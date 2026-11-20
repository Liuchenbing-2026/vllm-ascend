# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import _causal_conv1d_update_kernel
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "causal_conv1d_update_npu", "causal_conv1d_update_varlen", "extract_last_width"]


def causal_conv1d_update_varlen(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    null_block_id: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update GDN's dim-last state using device query boundaries.

    The model input may contain graph-padding tokens beyond the final live
    boundary. Use the upstream strided Triton kernel, with CUDA PDL disabled,
    instead of the native convolution's padded-input path. Keep state/weight
    layouts as views: [blocks, history, dim] and [width, dim], respectively.
    """
    if x.ndim != 2 or weight.ndim != 2 or conv_state.ndim != 3:
        raise ValueError("GDN convolution requires 2D inputs/weights and 3D states")
    dim = x.shape[1]
    width = weight.shape[0]
    state_len = width - 1 + max_query_len - 1
    if weight.shape[1] != dim or conv_state.shape[2] != dim:
        raise ValueError("GDN convolution requires token-major inputs and dim-last weights/states")
    if not 2 <= width <= 6 or max_query_len < 1 or conv_state.shape[1] < state_len:
        raise ValueError("Unsupported GDN convolution width or insufficient speculative state history")
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported GDN convolution activation: {activation}")
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("GDN convolution output must match the input shape, dtype and device")
    batch = conv_state_indices.numel()
    if query_start_loc.numel() != batch + 1 or num_accepted_tokens.numel() != batch:
        raise ValueError("GDN convolution request boundaries and accepted counts must match cache indices")
    _causal_conv1d_update_kernel[(batch, triton.cdiv(dim, 256))](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        out,
        batch,
        dim,
        max_query_len,
        state_len,
        conv_state.shape[0],
        0,
        x.stride(1),
        x.stride(0),
        weight.stride(1),
        weight.stride(0),
        conv_state.stride(0),
        conv_state.stride(2),
        conv_state.stride(1),
        conv_state_indices.stride(0),
        0,
        out.stride(1),
        out.stride(0),
        null_block_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        IS_VARLEN=True,
        IS_SPEC_DECODING=True,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        HAS_NULL_BLOCK=True,
        BLOCK_N=256,
        launch_pdl=False,
    )
    return out


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int | None = NULL_BLOCK_ID,
    validate_data: bool = False,
    out: torch.Tensor | None = None,
    *,
    pad_slot_id: int | None = None,
) -> torch.Tensor:
    """Ascend NPU Triton implementation of vLLM's ``causal_conv1d_update``.

    Launches the shared strided kernel with ``launch_pdl=False`` and no ``.item()``
    host synchronizations so ACLGraph decode capture proceeds without stalling.
    """
    if validate_data:
        assert x.stride(1) == 1
        if conv_state_indices is not None:
            effective_sentinel = pad_slot_id if pad_slot_id is not None else null_block_id
            assert effective_sentinel is not None

    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported causal_conv1d activation: {activation}")

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    if out is None:
        out = x
    else:
        if out.shape != x.shape:
            raise ValueError(f"`out` shape {tuple(out.shape)} must match `x` shape {tuple(x.shape)}.")
        if out.dtype != original_x_dtype or out.device != x.device:
            raise ValueError("`out` must have the same dtype and device as the input `x`.")

    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
        out = out.unsqueeze(-1)

    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        if conv_state_indices is None:
            raise ValueError("conv_state_indices is required when query_start_loc is provided")
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
        if seqlen < 1:
            raise ValueError(f"max_query_len must be >= 1 when query_start_loc is provided, got {seqlen}")

    if weight.shape[0] != dim and weight.shape[1] == dim:
        weight = weight.transpose(0, 1)
    _, width = weight.shape
    if not 2 <= width <= 7:
        raise ValueError(f"causal_conv1d_update_npu supports widths 2 to 7, got {width}")

    if conv_state.shape[1] != dim and conv_state.shape[2] == dim:
        conv_state = conv_state.transpose(1, 2)
    num_cache_lines, _, state_len = conv_state.size()

    if conv_state_indices is None:
        conv_state_indices = torch.arange(batch, dtype=torch.int32, device=x.device)
        has_null_block = False
        kernel_null_block_id = -1
    else:
        effective_null_block_id = pad_slot_id if pad_slot_id is not None else null_block_id
        has_null_block = effective_null_block_id is not None
        kernel_null_block_id = effective_null_block_id if effective_null_block_id is not None else -1

    if validate_data:
        assert dim == weight.size(0)
        assert state_len >= width - 1
        assert dim == conv_state.size(1)
        assert batch == conv_state_indices.shape[0], (
            f"ERROR: conv_state_indices should have shape ({batch},*) but got {conv_state_indices.shape}"
        )
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1

    if batch == 0:
        if unsqueeze:
            out = out.squeeze(-1)
        return out.to(original_x_dtype)

    stride_w_dim, stride_w_width = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)
    effective_state_len = width - 1 + (seqlen - 1) if num_accepted_tokens is not None else width - 1
    np2_statelen = triton.next_power_of_2(effective_state_len)

    _causal_conv1d_update_kernel[(batch, triton.cdiv(dim, 256))](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        out,
        batch,
        dim,
        seqlen,
        effective_state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        kernel_null_block_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        IS_VARLEN=query_start_loc is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        HAS_NULL_BLOCK=has_null_block,
        BLOCK_N=256,
        launch_pdl=False,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)
