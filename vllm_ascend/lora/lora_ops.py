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

import torch


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset, slice_size
    )


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    return torch.ops._C_ascend.sgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling
    )


def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )


def fused_moe_lora(
    inputs: torch.Tensor,
    lora_a_combined: torch.Tensor,
    lora_b_combined: torch.Tensor,
    indices: torch.Tensor,
    output_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    scale: float = 1.0,
):
    """Fused MoE-LoRA: shrink + expand_slice + accumulate in one AscendC kernel.

    Semantics (per row i, indices[i] = vid):
        if vid < 0: skip (no contribution)
        rank_buf = scale * (x[i] @ lora_a_combined[vid].T)
        output[i, slice_offset:slice_offset+slice_size] += rank_buf @ lora_b_combined[vid].T

    Args:
        inputs:           [batch_size, input_hidden_dim] bf16/half.
        lora_a_combined:  [num_loras_total, lora_rank, input_hidden_dim] bf16/half.
        lora_b_combined:  [num_loras_total, slice_size, lora_rank] bf16/half.
        indices:          [batch_size] int64, -1 means skip.
        output_tensor:    [batch_size, output_full_dim] bf16/half, inout.
        slice_offset:     where to start writing in output_tensor[:, ...].
        slice_size:       width of the destination slice.
        scale:            shrink scaling factor, fused into rank_buf.
    """
    return torch.ops._C_ascend.fused_moe_lora(
        inputs, lora_a_combined, lora_b_combined, indices, output_tensor,
        slice_offset, slice_size, scale,
    )
